from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from emr_logic import (
    load_all_tables,
    load_patient_data,
    detect_lab_abnormalities,
    detect_vital_abnormalities,
    summarize_medications,
    suggest_tests_and_conditions,
    build_overall_summary,
    build_progress_note,
    build_alerts,
    compute_severity_score,
    compare_today_yesterday,
    delta_text_from_df,
    answer_question,
    deceased_patient_deathtime,
    Tables,
)

st.set_page_config(page_title="EMR Assistant Demo", layout="wide")

st.title("🩺 EMR Assistant (Cloud Demo)")
st.caption("Streamlit UI for your EMR agent logic (MIMIC-IV demo).")


# ============================================================
# Helpers: dataset handling (cloud-friendly)
# ============================================================

DATA_DIR = Path(st.secrets.get("DATA_DIR", "/tmp/emr_demo_data"))  # Streamlit Cloud: /tmp is writable
DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_zip_to_dir(zip_bytes: bytes, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(out_dir)

    # Find mimic root folder inside extracted content
    # Expect mimic-iv-clinical-database-demo-2.2/
    for candidate in out_dir.rglob("mimic-iv-clinical-database-demo-2.2"):
        if (candidate / "hosp").exists() and (candidate / "icu").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find folder 'mimic-iv-clinical-database-demo-2.2' with 'hosp' and 'icu' inside the uploaded zip."
    )


@st.cache_data(show_spinner=True)
def cached_load_tables(base_path_str: str) -> Tables:
    return load_all_tables(Path(base_path_str))


# ============================================================
# Sidebar: dataset + patient selection
# ============================================================

st.sidebar.header("Dataset")
st.sidebar.write(
    "For free cloud deployment, upload a zip containing:\n"
    "`mimic-iv-clinical-database-demo-2.2/hosp` and `mimic-iv-clinical-database-demo-2.2/icu`"
)

uploaded = st.sidebar.file_uploader("Upload MIMIC-IV demo zip", type=["zip"])

if uploaded is not None:
    try:
        mimic_root = extract_zip_to_dir(uploaded.getvalue(), DATA_DIR)
        st.session_state["mimic_root"] = str(mimic_root)
        st.sidebar.success(f"Dataset ready: {mimic_root}")
    except Exception as e:
        st.sidebar.error(str(e))

mimic_root_str = st.session_state.get("mimic_root")

if not mimic_root_str:
    st.info("Upload a MIMIC-IV demo zip in the sidebar to start.")
    st.stop()

# Load tables (cached)
try:
    tables = cached_load_tables(mimic_root_str)
except Exception as e:
    st.error(f"Failed to load tables: {e}")
    st.stop()

st.sidebar.divider()
st.sidebar.header("Load Patient")

# Helpful: show a few subject_ids
some_ids = sorted(tables.admissions["subject_id"].dropna().astype(int).unique().tolist())
default_id = some_ids[0] if some_ids else 10000000

pid = st.sidebar.number_input("subject_id", min_value=0, value=int(st.session_state.get("pid", default_id)), step=1)
if st.sidebar.button("Load"):
    st.session_state["pid"] = int(pid)

if "pid" not in st.session_state:
    st.info("Click **Load** in the sidebar after choosing a patient.")
    st.stop()

pid = int(st.session_state["pid"])

data = load_patient_data(tables, pid)
if data is None:
    st.error(f"No admissions found for subject_id {pid}.")
    st.stop()

p_adm = data["admissions"]
death_time = deceased_patient_deathtime(p_adm)


# ============================================================
# Deceased patient mode (raw only)
# ============================================================

if death_time is not None:
    st.error(f"⚠️ PATIENT DECEASED — recorded death time: {death_time}")
    st.warning("Clinical summaries, alerts, trends, and risk scoring are disabled. Raw data only.")

    tabA, tabB, tabC, tabD = st.tabs(["Admissions", "Labs", "Vitals", "Medications"])
    with tabA:
        st.dataframe(p_adm)
    with tabB:
        st.dataframe(data["labs"].tail(200))
    with tabC:
        st.dataframe(data["vitals"].tail(200))
    with tabD:
        st.dataframe(data["meds"].tail(200))

    st.stop()


# ============================================================
# Living patient analysis
# ============================================================

labs = data["labs"]
vitals = data["vitals"]
meds = data["meds"]
dx = data["dx"]

lab_sentences, lab_flags = detect_lab_abnormalities(labs)
vital_sentences, vital_flags = detect_vital_abnormalities(vitals)
med_summary, meds_list, med_flags, med_interactions = summarize_medications(meds)
tests, conditions = suggest_tests_and_conditions(lab_flags, vital_flags, med_flags)

if not dx.empty and "long_title" in dx.columns:
    dx_list = dx["long_title"].dropna().unique().tolist()
    primary_dx = dx_list[0] if dx_list else "no clearly documented primary diagnosis"
else:
    primary_dx = "no clearly documented primary diagnosis"

overall = build_overall_summary(data, primary_dx)
progress = build_progress_note(primary_dx, lab_flags, vital_flags, med_flags, tests, conditions)
high_alerts, moderate_alerts, mild_alerts = build_alerts(lab_flags, vital_flags, med_flags, med_interactions)
severity_score, severity_level = compute_severity_score(lab_flags, vital_flags, med_flags, med_interactions)

analysis = {
    "lab_sentences": lab_sentences,
    "vital_sentences": vital_sentences,
    "med_summary": med_summary,
    "med_interactions": med_interactions,
    "lab_flags": lab_flags,
    "vital_flags": vital_flags,
    "med_flags": med_flags,
}

# ============================================================
# UI Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Summary", "Alerts", "Labs", "Vitals", "Medications", "Trends", "Delta (Day)", "Q&A"][:7]
)

with tab1:
    st.subheader("Overall summary")
    st.text(overall)
    st.subheader("Progress note")
    st.text(progress)

with tab2:
    st.subheader("Severity & Alerts")
    st.metric("Severity Score", severity_score, severity_level)

    if high_alerts:
        st.error("High-risk alerts:\n" + "\n".join(f"- {x}" for x in high_alerts))
    if moderate_alerts:
        st.warning("Moderate concerns:\n" + "\n".join(f"- {x}" for x in moderate_alerts))
    if mild_alerts:
        st.info("\n".join(mild_alerts))

    if med_interactions:
        st.warning("Medication interaction flags:\n" + "\n".join(f"- {x}" for x in med_interactions))

with tab3:
    st.subheader("Labs (tail)")
    st.dataframe(labs.tail(300))
    st.download_button(
        "Download labs CSV",
        data=labs.to_csv(index=False).encode("utf-8"),
        file_name=f"labs_subject_{pid}.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader("Vitals (tail)")
    st.dataframe(vitals.tail(300))
    st.download_button(
        "Download vitals CSV",
        data=vitals.to_csv(index=False).encode("utf-8"),
        file_name=f"vitals_subject_{pid}.csv",
        mime="text/csv",
    )

with tab5:
    st.subheader("Medications (tail)")
    st.write(med_summary)
    if meds_list:
        with st.expander("Medication list (unique)"):
            st.write("\n".join(f"- {m}" for m in meds_list))
    st.dataframe(meds.tail(300))
    st.download_button(
        "Download meds CSV",
        data=meds.to_csv(index=False).encode("utf-8"),
        file_name=f"meds_subject_{pid}.csv",
        mime="text/csv",
    )

with tab6:
    st.subheader("Trends")
    if labs.empty:
        st.info("No lab data for this patient.")
    else:
        itemids = labs["itemid"].dropna().astype(int).unique().tolist()
        itemid = st.selectbox("Choose lab itemid", options=sorted(itemids))

        df = labs[labs["itemid"] == itemid].dropna(subset=["charttime", "valuenum"]).sort_values("charttime")
        label = tables.lab_lookup.get(int(itemid), f"item {itemid}")
        st.caption(f"{label} (itemid {itemid})")

        if df.empty:
            st.info("No usable values for this item.")
        else:
            series = df.set_index("charttime")["valuenum"]
            st.line_chart(series)

with tab7:
    st.subheader("Delta (Today vs Yesterday)")
    comp = compare_today_yesterday(labs)
    st.text(delta_text_from_df(comp, tables.lab_lookup))
    if comp is not None and not comp.empty:
        with st.expander("Delta table"):
            st.dataframe(comp)

st.divider()

st.subheader("Q&A")
q = st.text_input("Ask about this patient (labs, vitals, meds, infection, renal, etc.)", key="qa_input")
if q:
    ans = answer_question(q, data, analysis, tables.lab_lookup)
    st.text(ans)
