import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import your original 1500+ line code WITHOUT modifying it.
# This module contains all logic: loading, analysis, alerts, trends, Q&A, etc.
import emr_core as core


st.title("EMR Assistant")

base_path = st.text_input(
    "Path to MIMIC-IV demo folder",
    value=""
)

if st.button("Initialize data"):
    core.initialize_data(base_path)
    st.success("Data loaded successfully")

def reset_patient_state():
    """Clear all patient-specific Streamlit state."""
    for k in [
        "analysis",
        "qa_answer",
        "qa_question_input",
    ]:
        if k in st.session_state:
            del st.session_state[k]

# -----------------------------
# Streamlit page config + light "desktop agent" feel
# -----------------------------
st.set_page_config(page_title="EMR Assistant (Web)", layout="wide")

st.markdown(
    """
    <style>
      /* Make it feel like a compact "agent" panel */
      .block-container { padding-top: 0.75rem; padding-bottom: 2rem; }
      [data-testid="stSidebar"] { min-width: 340px; width: 340px; }
      .mono textarea, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .smallcap { color: #666; font-size: 0.9rem; }
      .section-title { margin-top: 0.25rem; margin-bottom: 0.25rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("EMR Assistant")
st.markdown('<div class="smallcap">On-demand summary and Q&A from structured EMR data (MIMIC-IV demo).</div>', unsafe_allow_html=True)


# -----------------------------
# Session State
# -----------------------------
if "patient_id" not in st.session_state:
    st.session_state.patient_id = None
if "data" not in st.session_state:
    st.session_state.data = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None


# -----------------------------
# Helper: factual-only Q&A (used when patient is deceased)
# This replicates the method in your Tkinter class without changing emr_core.py.
# -----------------------------
def factual_lookup_only(question: str, data: dict) -> str:
    q = (question or "").lower().strip()
    if not q:
        return "Please enter a question."

    if data is None:
        return "No patient loaded."

    # Show recent labs
    if "lab" in q or "labs" in q or "value" in q:
        df = data.get("labs")
        if df is None or df.empty:
            return "No lab data available."
        return "Recent labs (last 20 rows):\n" + df.tail(20).to_string()

    # Show vitals
    if "vital" in q or "bp" in q or "heart rate" in q or "spo2" in q or "oxygen" in q:
        df = data.get("vitals")
        if df is None or df.empty:
            return "No vital sign data available."
        return "Recent vitals (last 20 rows):\n" + df.tail(20).to_string()

    # Show medications
    if "med" in q or "drug" in q or "medication" in q:
        df = data.get("meds")
        if df is None or df.empty:
            return "No medication data available."
        meds = sorted(df["drug"].dropna().unique().tolist()) if "drug" in df.columns else []
        if not meds:
            return "No medication names recorded."
        return "Medications recorded:\n" + "\n".join(f"- {m}" for m in meds)

    # Diagnoses
    if "diagnosis" in q or "dx" in q:
        df = data.get("dx")
        if df is None or df.empty:
            return "No diagnoses recorded."
        if "long_title" in df.columns:
            dx_list = df["long_title"].dropna().unique().tolist()
        else:
            dx_list = df["icd_code"].dropna().unique().tolist() if "icd_code" in df.columns else []
        if not dx_list:
            return "No diagnoses titles found."
        return "Diagnoses recorded:\n" + "\n".join(f"- {d}" for d in dx_list)

    return (
        "This patient is deceased. Clinical reasoning is disabled.\n\n"
        "You can ask factual questions like:\n"
        "- show recent labs\n"
        "- show recent vitals\n"
        "- list medications\n"
        "- what diagnoses are recorded?\n"
    )


# -----------------------------
# Sidebar: loader (desktop agent feel)
# -----------------------------
with st.sidebar:
    st.subheader("Patient Loader")
    pid_text = st.text_input("Enter subject_id", key="patient_input", value="" if st.session_state.patient_id is None else str(st.session_state.patient_id))
    load = st.button("Load", use_container_width=True)

    st.divider()
    st.caption("Exports (available after load)")
    export_summary = st.button("Export Summary TXT", use_container_width=True)
    export_labs = st.button("Export Labs CSV", use_container_width=True)
    export_vitals = st.button("Export Vitals CSV", use_container_width=True)
    export_meds = st.button("Export Meds CSV", use_container_width=True)


# -----------------------------
# Load patient (calls core logic, unchanged)
# -----------------------------
if load:
    if not pid_text.strip():
        st.sidebar.error("Please enter a subject_id.")
    else:
        try:
            pid = int(pid_text.strip())
            data = core.load_patient_data(pid)
        except ValueError:
            st.sidebar.error("Patient ID must be an integer.")
            data = None
        except FileNotFoundError as e:
            st.sidebar.error(str(e))
            data = None

        if data is None:
            st.sidebar.error(f"No admissions found for subject_id {pid_text.strip()}.")
        else:
            st.session_state.patient_id = pid
            st.session_state.data = data
            \

            # ---- DEATH CHECK (same behavior) ----
            p_adm = data["admissions"]
            death_time = None
            if "deathtime" in p_adm.columns:
                dt_series = p_adm["deathtime"].dropna()
                if not dt_series.empty:
                    death_time = dt_series.max()

            if death_time is not None:
                # Deceased patient mode: no clinical reasoning, only factual lookup
                st.session_state.analysis = {
                    "death_mode": True,
                    "death_time": death_time,
                    "labs": data["labs"],
                    "vitals": data["vitals"],
                    "meds": data["meds"],
                    "dx": data["dx"],
                }
            else:
                # Compute analysis for living patients (same functions, same results)
                p_labs = data["labs"]
                p_vitals = data["vitals"]
                p_meds = data["meds"]
                p_dx = data["dx"]

                lab_sentences, lab_flags = core.detect_lab_abnormalities(p_labs)
                vital_sentences, vital_flags = core.detect_vital_abnormalities(p_vitals)
                med_summary, meds_list, med_flags, med_interactions = core.summarize_medications(p_meds)
                tests, conditions = core.suggest_tests_and_conditions(lab_flags, vital_flags, med_flags)

                if not p_dx.empty and "long_title" in p_dx.columns:
                    dx_list = p_dx["long_title"].dropna().unique().tolist()
                    primary_dx = dx_list[0] if dx_list else "no clearly documented primary diagnosis"
                else:
                    primary_dx = "no clearly documented primary diagnosis"

                st.session_state.analysis = {
                    "death_mode": False,
                    "lab_sentences": lab_sentences,
                    "vital_sentences": vital_sentences,
                    "med_summary": med_summary,
                    "meds_list": meds_list,
                    "med_flags": med_flags,
                    "lab_flags": lab_flags,
                    "vital_flags": vital_flags,
                    "med_interactions": med_interactions,
                    "tests": tests,
                    "conditions": conditions,
                    "primary_dx": primary_dx,
                }

# -----------------------------
# Convenience refs
# -----------------------------
data = st.session_state.data
analysis = st.session_state.analysis


# -----------------------------
# Exports (mirror Tkinter export buttons)
# -----------------------------
def current_summary_text() -> str:
    if not data or not analysis:
        return ""
    if analysis.get("death_mode", False):
        dt = analysis.get("death_time")
        return (
            "⚠️ PATIENT DECEASED ⚠️\n\n"
            f"Recorded death time: {dt}\n\n"
            "Clinical summaries, alerts, trends, and risk scoring are disabled for deceased patients.\n\n"
            "You may still:\n"
            "- View raw Labs, Vitals, and Medications tables\n"
            "- Ask factual questions (e.g., 'show recent labs', 'what meds were given?')\n"
        )

    overall = core.build_overall_summary(data, analysis["primary_dx"])
    progress_note = core.build_progress_note(
        analysis["primary_dx"],
        analysis["lab_flags"],
        analysis["vital_flags"],
        analysis["med_flags"],
        analysis["tests"],
        analysis["conditions"],
    )
    high, moderate, mild = core.build_alerts(
        analysis["lab_flags"],
        analysis["vital_flags"],
        analysis["med_flags"],
        analysis["med_interactions"],
    )
    severity_score, severity_level = core.compute_severity_score(
        analysis["lab_flags"],
        analysis["vital_flags"],
        analysis["med_flags"],
        analysis["med_interactions"],
    )

    severity_bar = "[" + "#" * min(severity_score, 10) + "-" * max(0, 10 - severity_score) + "]"
    alerts_text = f"Overall severity score: {severity_score} ({severity_level} risk) {severity_bar}\n\n"
    if high:
        alerts_text += "High-risk alerts:\n" + "\n".join(f"- {x}" for x in high) + "\n\n"
    if moderate:
        alerts_text += "Moderate concerns:\n" + "\n".join(f"- {x}" for x in moderate) + "\n\n"
    if not high and not moderate:
        alerts_text += "No strong red-flag patterns detected from structured data alone.\n\n"
    if mild:
        alerts_text += "\n".join(mild)

    return overall + "\n\n" + progress_note + "\n\n" + alerts_text


if export_summary and data and analysis:
    txt = current_summary_text()
    st.sidebar.download_button(
        "Download Summary TXT",
        data=txt.encode("utf-8"),
        file_name=f"patient_{st.session_state.patient_id}_summary.txt",
        mime="text/plain",
        use_container_width=True
    )

if export_labs and data:
    df = data["labs"]
    st.sidebar.download_button(
        "Download Labs CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"patient_{st.session_state.patient_id}_labs.csv",
        mime="text/csv",
        use_container_width=True
    )

if export_vitals and data:
    df = data["vitals"]
    st.sidebar.download_button(
        "Download Vitals CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"patient_{st.session_state.patient_id}_vitals.csv",
        mime="text/csv",
        use_container_width=True
    )

if export_meds and data:
    df = data["meds"]
    st.sidebar.download_button(
        "Download Meds CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"patient_{st.session_state.patient_id}_meds.csv",
        mime="text/csv",
        use_container_width=True
    )


# -----------------------------
# Main UI: Tabs matching Tkinter
# -----------------------------
tabs = st.tabs(["Summary", "Labs", "Vitals", "Medications", "Trends", "Delta (Day)", "Q&A"])

# Summary tab
with tabs[0]:
    if not data or not analysis:
        st.info("Enter a subject_id in the left panel and click Load.")
    else:
        if analysis.get("death_mode", False):
            st.error("⚠️ PATIENT DECEASED ⚠️")
            st.write(f"Recorded death time: **{analysis.get('death_time')}**")
            st.text_area("Summary", current_summary_text(), height=260, key="death_summary", help="Clinical reasoning disabled for deceased patients.")
        else:
            overall = core.build_overall_summary(data, analysis["primary_dx"])
            progress_note = core.build_progress_note(
                analysis["primary_dx"],
                analysis["lab_flags"],
                analysis["vital_flags"],
                analysis["med_flags"],
                analysis["tests"],
                analysis["conditions"],
            )
            high, moderate, mild = core.build_alerts(
                analysis["lab_flags"],
                analysis["vital_flags"],
                analysis["med_flags"],
                analysis["med_interactions"],
            )
            severity_score, severity_level = core.compute_severity_score(
                analysis["lab_flags"],
                analysis["vital_flags"],
                analysis["med_flags"],
                analysis["med_interactions"],
            )

            st.text_area("Overview", overall, height=160, key="overview", help="Same text as Tkinter Summary tab.")
            st.text_area("Assessment & Plan", progress_note, height=260, key="progress_note")

            st.subheader("Alerts")
            st.caption(f"Overall severity score: {severity_score} ({severity_level})")
            if high:
                for h in high:
                    st.error(h)
            if moderate:
                for m in moderate:
                    st.warning(m)
            if not high and not moderate and mild:
                for x in mild:
                    st.info(x)
            elif mild:
                for x in mild:
                    st.info(x)

# ---------- LABS ----------
with tabs[1]:
    if not data or not analysis:
        st.info("Load a patient to view labs.")
    else:
        if analysis.get("death_mode", False):
            st.warning("Use 'View Full Lab Table' to inspect raw labs.")
        else:
            st.markdown("**Key recent lab findings:**")
            lab_text = "\n".join(f"- {s}" for s in analysis["lab_sentences"])
            st.text_area(
                "Lab Summary",
                lab_text,
                height=220,
                key="labs_summary",
            )

        # Button to show full table (Tkinter equivalent popup)
        with st.expander("📊 View Full Lab Table"):
            st.dataframe(data["labs"], use_container_width=True, height=500)


# ---------- VITALS ----------
with tabs[2]:
    if not data or not analysis:
        st.info("Load a patient to view vitals.")
    else:
        if analysis.get("death_mode", False):
            st.warning("Use 'View Full Vitals Table' to inspect raw vitals.")
        else:
            st.markdown("**Key recent vital sign findings:**")
            vitals_text = "\n".join(f"- {s}" for s in analysis["vital_sentences"])
            st.text_area(
                "Vitals Summary",
                vitals_text,
                height=220,
                key="vitals_summary",
            )

        # Button to show full table (Tkinter equivalent popup)
        with st.expander("📊 View Full Vitals Table"):
            st.dataframe(data["vitals"], use_container_width=True, height=500)

# ---------- MEDICATIONS ----------
with tabs[3]:
    if not data or not analysis:
        st.info("Load a patient to view medications.")
    else:
        if analysis.get("death_mode", False):
            st.warning("Use 'View Full Meds Table' to inspect raw medications.")
        else:
            # Medication overview (same text as Tkinter)
            st.markdown("**Medication overview:**")
            st.text_area(
                "Medication Summary",
                analysis["med_summary"],
                height=80,
                key="med_summary_text",
            )

            # Compressed medication list
            meds_list = analysis.get("meds_list", [])
            if meds_list:
                st.markdown("**Medication list (compressed):**")
                st.text_area(
                    "Medication List",
                    "\n".join(f"- {m}" for m in meds_list),
                    height=220,
                    key="med_list_text",
                )

            # Medication interaction warnings
            med_interactions = analysis.get("med_interactions", [])
            if med_interactions:
                st.markdown("**Potential medication concerns:**")
                for mi in med_interactions:
                    st.warning(mi)

        # Full table on demand (Tkinter "View Full Meds Table")
        with st.expander("📊 View Full Medications Table"):
            st.dataframe(data["meds"], use_container_width=True, height=500)


# Trends tab
with tabs[4]:
    if not data or not analysis:
        st.info("Load a patient to plot trends.")
    elif analysis.get("death_mode", False):
        st.warning("Trend plotting is disabled for deceased patients.")
    else:
        p_labs = data["labs"]
        if p_labs.empty:
            st.info("No lab data available for this patient.")
        else:
            # Mimic dropdown: "itemid – label" (same as Tkinter)
            itemids = p_labs["itemid"].value_counts().index.tolist()
            labels = [f"{it} – {core.LAB_LOOKUP.get(it, f'item {it}')}" for it in itemids[:50]]
            choice = st.selectbox("Select lab itemid", labels)
            itemid = int(choice.split("–")[0].strip())

            df = p_labs[p_labs["itemid"] == itemid].dropna(subset=["charttime", "valuenum"]).sort_values("charttime")
            if df.empty:
                st.info("No data points for that lab item.")
            else:
                label = core.LAB_LOOKUP.get(itemid, f"item {itemid}")
                fig, ax = plt.subplots(figsize=(7, 3.5))
                ax.plot(df["charttime"], df["valuenum"], marker="o", linestyle="-")
                ax.set_title(f"{label} (item {itemid}) over time")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                fig.autofmt_xdate()
                st.pyplot(fig, use_container_width=True)

# Delta tab
with tabs[5]:
    if not data or not analysis:
        st.info("Load a patient to see day-to-day delta.")
    elif analysis.get("death_mode", False):
        st.warning("Delta (day-to-day) analysis is disabled for deceased patients.")
    else:
        delta_df = core.compare_today_yesterday(data["labs"])
        delta_text = core.delta_text_from_df(delta_df)
        st.text_area("Delta (Today vs Yesterday)", delta_text, height=420, key="delta_text")

# Q&A tab
with tabs[6]:
    if not data or not analysis:
        st.info("Load a patient to use Q&A.")
    else:
        st.write("Ask about this patient (same rules + optional RAG as your desktop app).")
        q = st.text_input("Question", value="", key="qa_question")
        ask = st.button("Ask", key="qa_ask_btn")

        if ask:
            if analysis.get("death_mode", False):
                ans = factual_lookup_only(q, data)
            else:
                ans = core.answer_question(q, data, analysis)
            st.text_area("Answer", ans, height=320, key="qa_answer")
