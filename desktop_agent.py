import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import timedelta

import pandas as pd

# Optional: scikit-learn for RAG Q&A
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional: matplotlib for trend plots (embedded in Tkinter)
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ============================================================
# CONFIG – update this path to your MIMIC-IV demo folder
# ============================================================

BASE = Path("../mimic-iv-clinical-database-demo-2.2")
HOSP = BASE / "hosp"
ICU = BASE / "icu"


# ============================================================
# DATA LOADING
# ============================================================

def load_all_tables():
    if not BASE.exists():
        raise FileNotFoundError(
            f"Base folder {BASE} not found. Adjust BASE path in the code."
        )

    admissions = pd.read_csv(
        HOSP / "admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime"],   # <-- include deathtime
    )
    labs = pd.read_csv(
        HOSP / "labevents.csv.gz",
        parse_dates=["charttime"],
    )
    meds = pd.read_csv(
        HOSP / "prescriptions.csv.gz",
        parse_dates=["starttime", "stoptime"],
    )
    vitals = pd.read_csv(
        ICU / "chartevents.csv.gz",
        parse_dates=["charttime"],
    )
    diag = pd.read_csv(HOSP / "diagnoses_icd.csv.gz")
    icd = pd.read_csv(HOSP / "d_icd_diagnoses.csv.gz")
    labitems = pd.read_csv(HOSP / "d_labitems.csv.gz")

    lab_lookup = dict(zip(labitems["itemid"], labitems["label"]))

    return admissions, labs, meds, vitals, diag, icd, lab_lookup


ADMISSIONS, LABS, MEDS, VITALS, DIAG, ICD, LAB_LOOKUP = load_all_tables()


def load_patient_data(subject_id: int) -> Dict[str, Any] | None:
    p_adm = ADMISSIONS[ADMISSIONS["subject_id"] == subject_id].copy()
    if p_adm.empty:
        return None

    p_labs = LABS[LABS["subject_id"] == subject_id].copy()
    p_meds = MEDS[MEDS["subject_id"] == subject_id].copy()
    p_vitals = VITALS[VITALS["subject_id"] == subject_id].copy()

    p_dx = DIAG[DIAG["subject_id"] == subject_id].copy()
    if not p_dx.empty:
        p_dx = p_dx.merge(ICD, on="icd_code", how="left")

    return {
        "patient_id": subject_id,
        "admissions": p_adm,
        "labs": p_labs,
        "meds": p_meds,
        "vitals": p_vitals,
        "dx": p_dx,
    }


# ============================================================
# LAB/VITAL/MED ABNORMALITY DETECTION
# ============================================================

def detect_lab_abnormalities(p_labs: pd.DataFrame) -> Tuple[List[str], Dict[str, bool]]:
    sentences: List[str] = []
    flags = {
        "renal_issue": False,
        "electrolyte_issue": False,
        "bleeding_risk": False,
        "infection_risk": False,
        "metabolic_stress": False,
    }

    if p_labs.empty or "valuenum" not in p_labs.columns:
        return ["No recent laboratory values available."], flags

    latest = (
        p_labs.dropna(subset=["charttime", "valuenum"])
        .sort_values("charttime")
        .groupby("itemid")
        .tail(1)
    )

    creat_ids = [50912]                # Creatinine
    na_ids = [50983]                   # Sodium
    k_ids = [50822, 50971]             # Potassium
    lact_ids = [50813]                 # Lactate
    inr_ids = [51237, 51274]           # INR
    wbc_ids = [51300, 51301, 51302]    # WBC
    hb_ids = [50811, 51222, 51221]     # Hemoglobin

    def get_latest(ids):
        sub = latest[latest["itemid"].isin(ids)]
        if sub.empty:
            return None
        return sub.iloc[-1]["valuenum"]

    creat = get_latest(creat_ids)
    if creat is not None and creat > 1.5:
        flags["renal_issue"] = True
        sentences.append(
            f"Creatinine is elevated at {creat:.2f}, suggesting renal stress or impaired clearance."
        )

    na = get_latest(na_ids)
    if na is not None:
        if na < 130:
            flags["electrolyte_issue"] = True
            sentences.append(f"Sodium is low at {na:.1f}, consistent with hyponatremia.")
        elif na > 145:
            flags["electrolyte_issue"] = True
            sentences.append(f"Sodium is elevated at {na:.1f}, consistent with hypernatremia.")

    k = get_latest(k_ids)
    if k is not None:
        if k < 3.0:
            flags["electrolyte_issue"] = True
            sentences.append(f"Potassium is low at {k:.1f}, which may increase arrhythmia risk.")
        elif k > 5.5:
            flags["electrolyte_issue"] = True
            sentences.append(f"Potassium is elevated at {k:.1f}, which may increase arrhythmia risk.")

    lact = get_latest(lact_ids)
    if lact is not None:
        if lact > 4:
            flags["metabolic_stress"] = True
            sentences.append(
                f"Lactate is critically elevated at {lact:.1f}, suggesting significant hypoperfusion or severe stress."
            )
        elif lact > 2:
            flags["metabolic_stress"] = True
            sentences.append(
                f"Lactate is elevated at {lact:.1f}, suggesting increased metabolic demand or impaired clearance."
            )

    inr = get_latest(inr_ids)
    if inr is not None and inr > 1.5:
        flags["bleeding_risk"] = True
        sentences.append(f"INR is elevated at {inr:.2f}, indicating increased bleeding risk.")

    wbc = get_latest(wbc_ids)
    if wbc is not None:
        if wbc > 12:
            flags["infection_risk"] = True
            sentences.append(
                f"White blood cell count is elevated at {wbc:.1f}, which may indicate infection or inflammation."
            )
        elif wbc < 4:
            flags["infection_risk"] = True
            sentences.append(
                f"White blood cell count is low at {wbc:.1f}, which may reflect marrow suppression or severe infection."
            )

    hb = get_latest(hb_ids)
    if hb is not None and hb < 8:
        flags["bleeding_risk"] = True
        sentences.append(f"Hemoglobin is low at {hb:.1f}, consistent with significant anemia.")

    if not sentences:
        sentences.append("No major laboratory abnormalities detected based on core parameters.")

    return sentences, flags


def detect_vital_abnormalities(p_vitals: pd.DataFrame) -> Tuple[List[str], Dict[str, bool]]:
    sentences: List[str] = []
    flags = {
        "hemodynamic_issue": False,
        "resp_issue": False,
        "fever": False,
    }

    if p_vitals.empty or "valuenum" not in p_vitals.columns:
        sentences.append("No recent ICU vital signs available.")
        return sentences, flags

    df = (
        p_vitals.dropna(subset=["charttime", "valuenum"])
        .sort_values("charttime")
        .groupby("itemid")
        .tail(1)
    )

    HR_ID = 220045
    SBP_ID = 220050
    MAP_ID = 220052
    RR_IDS = [220210, 224690]
    SPO2_ID = 220277
    TEMP_ID = 223761

    def get_val(iids):
        sub = df[df["itemid"].isin([iids] if isinstance(iids, int) else iids)]
        if sub.empty:
            return None
        return sub.iloc[-1]["valuenum"]

    hr = get_val(HR_ID)
    sbp = get_val(SBP_ID)
    map_val = get_val(MAP_ID)
    rr = get_val(RR_IDS)
    spo2 = get_val(SPO2_ID)
    temp = get_val(TEMP_ID)

    if hr is not None and hr > 100:
        flags["hemodynamic_issue"] = True
        sentences.append(f"Heart rate is elevated at around {hr:.0f} beats per minute.")
    if sbp is not None and sbp < 90:
        flags["hemodynamic_issue"] = True
        sentences.append(f"Systolic blood pressure is low at about {sbp:.0f} mmHg.")
    if map_val is not None and map_val < 65:
        flags["hemodynamic_issue"] = True
        sentences.append(f"Mean arterial pressure is low at about {map_val:.0f} mmHg.")

    if rr is not None and rr > 24:
        flags["resp_issue"] = True
        sentences.append(f"Respiratory rate is elevated at about {rr:.0f} breaths per minute.")
    if spo2 is not None and spo2 < 92:
        flags["resp_issue"] = True
        sentences.append(f"Oxygen saturation has been as low as {spo2:.0f}%, below typical targets.")
    if temp is not None and temp > 38:
        flags["fever"] = True
        sentences.append(f"Temperature is elevated at approximately {temp:.1f} °C, consistent with fever.")

    if not sentences:
        sentences.append("Recent vital signs appear stable without major abnormalities.")

    return sentences, flags


def summarize_medications(p_meds: pd.DataFrame):
    flags = {
        "antibiotic_use": False,
        "anticoagulant_use": False,
        "cardiac_meds": False,
        "sedatives": False,
    }

    if p_meds.empty or "drug" not in p_meds.columns:
        return "No medications recorded for this patient.", [], flags, []

    meds = p_meds["drug"].dropna().astype(str).tolist()
    meds_lower = [m.lower() for m in meds]

    antibiotics_kw = ["cillin", "mycin", "cef", "metro", "vanco", "clavulanate"]
    anticoag_kw = ["heparin", "warfarin", "enoxaparin", "apixaban", "rivaroxaban"]
    cardiac_kw = ["metoprolol", "digoxin", "nitro", "hydral", "captopril", "atorva", "amiodarone"]
    sedative_kw = ["lorazepam", "midazolam", "propofol", "trazodone", "morphine", "dilaudid"]

    def any_match(keywords):
        return any(any(k in m for k in keywords) for m in meds_lower)

    if any_match(antibiotics_kw):
        flags["antibiotic_use"] = True
    if any_match(anticoag_kw):
        flags["anticoagulant_use"] = True
    if any_match(cardiac_kw):
        flags["cardiac_meds"] = True
    if any_match(sedative_kw):
        flags["sedatives"] = True

    interactions = []
    if flags["anticoagulant_use"] and any("aspirin" in m or "clopidogrel" in m for m in meds_lower):
        interactions.append("Combination of anticoagulant and antiplatelet therapy increases bleeding risk.")

    summary_bits = []
    if flags["antibiotic_use"]:

        summary_bits.append("antibiotics")
    if flags["anticoagulant_use"]:
        summary_bits.append("anticoagulants/antiplatelets")
    if flags["cardiac_meds"]:
        summary_bits.append("cardiovascular agents")
    if flags["sedatives"]:
        summary_bits.append("sedatives/analgesics")

    if summary_bits:
        summary = "Current therapy includes " + ", ".join(summary_bits) + ", among other agents."
    else:
        summary = "Current medications include several agents without prominent high-risk combinations detected."

    meds_unique = sorted(set(meds))

    return summary, meds_unique, flags, interactions


def suggest_tests_and_conditions(
    lab_flags: Dict[str, bool],
    vital_flags: Dict[str, bool],
    med_flags: Dict[str, bool],
):
    tests = set()
    conditions = set()

    if lab_flags["renal_issue"]:
        conditions.add("Acute or chronic kidney dysfunction.")
        tests.update(["Trend creatinine and BUN.", "Assess urine output and volume status."])
    if lab_flags["electrolyte_issue"]:
        conditions.add("Electrolyte disturbance (sodium/potassium).")
        tests.update(["Repeat basic metabolic panel.", "Consider ECG if potassium is abnormal."])
    if lab_flags["bleeding_risk"] or med_flags["anticoagulant_use"]:
        conditions.add("Increased bleeding risk.")
        tests.update(["Check hemoglobin and coagulation profile.", "Review indications and dosing for anticoagulation."])
    if lab_flags["infection_risk"] or vital_flags["fever"]:
        conditions.add("Possible infection or systemic inflammation.")
        tests.update(["CBC with differential.", "Blood/urine cultures as appropriate.", "Consider chest imaging."])
    if lab_flags["metabolic_stress"]:
        conditions.add("Global metabolic stress or hypoperfusion.")
        tests.update(["Repeat lactate.", "Assess hemodynamics and volume status."])

    if vital_flags["hemodynamic_issue"]:
        conditions.add("Hemodynamic instability.")
        tests.update(["Frequent blood pressure and MAP monitoring.", "Consider fluid status and vasoactive support."])
    if vital_flags["resp_issue"]:
        conditions.add("Respiratory compromise.")
        tests.update(["Arterial blood gas if available.", "Chest imaging.", "Review oxygen/ventilator settings."])

    if med_flags["antibiotic_use"]:
        tests.add("Review culture data and antibiotic spectrum/duration.")
    if med_flags["sedatives"]:
        conditions.add("Medication-related sedation or respiratory depression.")

    if not tests:
        tests.add("Continue routine monitoring and repeat key labs per clinical judgment.")
    if not conditions:
        conditions.add("No single dominant condition inferred; integrate with full clinical exam and imaging.")

    return sorted(tests), sorted(conditions)


def build_overall_summary(data: Dict[str, Any], primary_dx: str) -> str:
    pid = data["patient_id"]
    p_adm = data["admissions"]

    if not p_adm.empty:
        adm = p_adm.sort_values("admittime").iloc[-1]
        adm_time = adm.get("admittime", "unknown time")
        adm_type = adm.get("admission_type", "unknown type")
        intro = f"Patient {pid} was admitted on {adm_time} ({adm_type}) with a primary diagnosis of {primary_dx}."
    else:
        intro = f"Patient {pid} is currently under review; no admission record was found."

    overall = (
        intro + "\n\n"
        "The assistant has reviewed recent laboratory values, vital signs, medication history, ICU stays, "
        "and coded diagnoses to provide a focused overview of the current clinical situation. "
        "Findings below highlight organ function, hemodynamic and respiratory stability, infection risk, "
        "electrolyte disturbances, and medication-related safety considerations. "
        "All outputs are meant to support, not replace, bedside assessment and clinical judgment."
    )
    return overall


def build_progress_note(
    primary_dx: str,
    lab_flags: Dict[str, bool],
    vital_flags: Dict[str, bool],
    med_flags: Dict[str, bool],
    tests: List[str],
    conditions: List[str],
) -> str:
    assessment_lines = [f"Primary diagnosis: {primary_dx}."]

    if lab_flags["renal_issue"]:
        assessment_lines.append("Renal function appears stressed or impaired based on recent laboratory values.")
    if lab_flags["electrolyte_issue"]:
        assessment_lines.append("There is evidence of clinically relevant electrolyte disturbance.")
    if lab_flags["metabolic_stress"]:
        assessment_lines.append("Markers suggest global metabolic stress or possible hypoperfusion.")
    if lab_flags["bleeding_risk"] or med_flags["anticoagulant_use"]:
        assessment_lines.append("Bleeding risk may be increased due to coagulopathy, anemia, or anticoagulant use.")
    if lab_flags["infection_risk"] or vital_flags["fever"] or med_flags["antibiotic_use"]:
        assessment_lines.append("The pattern of labs and medications suggests concern for infection or inflammation.")
    if vital_flags["hemodynamic_issue"]:
        assessment_lines.append("Hemodynamic parameters have shown instability.")
    if vital_flags["resp_issue"]:
        assessment_lines.append("Respiratory parameters raise concern for compromise.")
    if len(assessment_lines) == 1:
        assessment_lines.append("No dominant acute organ system threat identified in the structured data alone.")

    plan_lines = ["Plan:"]
    for t in tests:
        plan_lines.append(f"- {t}")
    plan_lines.append("Reassess clinically and integrate trends, imaging, and bedside findings.")

    plan = "\n".join(plan_lines)

    assessment = "Assessment:\n" + "\n".join(f"- {x}" for x in assessment_lines)
    problems = "Problems to consider:\n" + "\n".join(f"- {c}" for c in conditions)

    return assessment + "\n\n" + problems + "\n\n" + plan


def build_alerts(
    lab_flags: Dict[str, bool],
    vital_flags: Dict[str, bool],
    med_flags: Dict[str, bool],
    med_interactions: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    high: List[str] = []
    moderate: List[str] = []
    mild: List[str] = []

    if lab_flags["metabolic_stress"]:
        high.append("Possible significant metabolic stress or hypoperfusion (e.g., elevated lactate).")
    if vital_flags["hemodynamic_issue"]:
        high.append("Potential hemodynamic instability (blood pressure, MAP, or heart rate).")
    if vital_flags["resp_issue"]:
        high.append("Possible respiratory compromise (respiratory rate or oxygen saturation changes).")

    if lab_flags["bleeding_risk"] or med_flags["anticoagulant_use"]:
        moderate.append("Increased bleeding risk due to coagulation profile, anemia, or antithrombotic therapy.")
    if lab_flags["renal_issue"]:
        moderate.append("Renal function is stressed or reduced.")
    if lab_flags["electrolyte_issue"]:
        moderate.append("Electrolyte disturbance (sodium and/or potassium).")
    if lab_flags["infection_risk"] or vital_flags["fever"] or med_flags["antibiotic_use"]:
        moderate.append("Pattern consistent with possible infection or systemic inflammation.")
    if med_interactions:
        moderate.append("Potential medication interactions that may affect safety.")

    if not high and not moderate:
        mild.append("No strong red-flag patterns detected from structured data alone; continue routine monitoring.")

    return high, moderate, mild


def compute_severity_score(
    lab_flags: Dict[str, bool],
    vital_flags: Dict[str, bool],
    med_flags: Dict[str, bool],
    med_interactions: List[str],
) -> Tuple[int, str]:
    """
    Very simple severity score based on flags.
    0–1: Low, 2–3: Moderate, >=4: High
    """
    score = 0
    if lab_flags["metabolic_stress"]:
        score += 2
    if vital_flags["hemodynamic_issue"]:
        score += 2
    if vital_flags["resp_issue"]:
        score += 2
    if lab_flags["renal_issue"]:
        score += 1
    if lab_flags["electrolyte_issue"]:
        score += 1
    if lab_flags["bleeding_risk"] or med_flags["anticoagulant_use"]:
        score += 1
    if lab_flags["infection_risk"] or vital_flags["fever"]:
        score += 1
    if med_interactions:
        score += 1

    if score <= 1:
        level = "Low"
    elif score <= 3:
        level = "Moderate"
    else:
        level = "High"
    return score, level


# ============================================================
# TRENDS & DELTA (Today vs Yesterday)
# ============================================================

def labs_by_day(p_labs: pd.DataFrame) -> pd.DataFrame:
    if p_labs.empty or "charttime" not in p_labs.columns:
        return pd.DataFrame()
    df = p_labs.dropna(subset=["charttime"]).copy()
    df["date"] = df["charttime"].dt.date
    return df


def compare_today_yesterday(p_labs: pd.DataFrame) -> pd.DataFrame | None:
    df = labs_by_day(p_labs)
    if df.empty or "valuenum" not in df.columns:
        return None

    last_date = df["date"].max()
    today = df[df["date"] == last_date]
    yesterday = df[df["date"] == (last_date - timedelta(days=1))]
    if today.empty or yesterday.empty:
        return None

    t = today.groupby("itemid")["valuenum"].mean().rename("today")
    y = yesterday.groupby("itemid")["valuenum"].mean().rename("yesterday")
    comp = pd.concat([t, y], axis=1).dropna()
    comp["delta"] = comp["today"] - comp["yesterday"]
    return comp.sort_values("delta", ascending=False)


def delta_text_from_df(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Not enough separated days of lab data to compute a meaningful day-to-day comparison."

    lines_worse = []
    lines_better = []
    lines_same = []

    for itemid, row in df.iterrows():
        label = LAB_LOOKUP.get(itemid, f"item {itemid}")
        delta = row["delta"]
        today = row["today"]
        yesterday = row["yesterday"]

        if abs(delta) < 0.01:
            arrow = "→"
            lines_same.append(f"{arrow} {label}: unchanged (approx {today:.2f}).")
        elif delta > 0:
            arrow = "↑"
            lines_worse.append(f"{arrow} {label}: increased {yesterday:.2f} → {today:.2f} (Δ +{delta:.2f}).")
        else:
            arrow = "↓"
            lines_better.append(f"{arrow} {label}: decreased {yesterday:.2f} → {today:.2f} (Δ {delta:.2f}).")

    out = []
    if lines_worse:
        out.append("Worsened (higher concerning values):")
        out.extend("• " + x for x in lines_worse)
    if lines_better:
        out.append("\nImproved (values moved in a favorable direction):")
        out.extend("• " + x for x in lines_better)
    if lines_same:
        out.append("\nStable:")
        out.extend("• " + x for x in lines_same)

    return "\n".join(out) if out else "No clear changes detected between today and yesterday."


# ============================================================
# RAG / Q&A
# ============================================================

def build_corpus_for_rag(
    data: Dict[str, Any],
    lab_sentences: List[str],
    vital_sentences: List[str],
    med_summary: str,
    med_interactions: List[str],
) -> List[str]:
    snippets: List[str] = []

    p_adm = data["admissions"]
    p_labs = data["labs"]
    p_meds = data["meds"]
    p_vitals = data["vitals"]
    p_dx = data["dx"]

    if not p_adm.empty:
        for _, row in p_adm.iterrows():
            snippets.append(
                f"Admission on {row.get('admittime')} (type {row.get('admission_type','')}) "
                f"with diagnosis text {row.get('diagnosis','')}."
            )

    if not p_dx.empty and "long_title" in p_dx.columns:
        diag_desc = "; ".join(p_dx["long_title"].dropna().unique().tolist())
        snippets.append(f"Diagnoses: {diag_desc}.")

    for s in lab_sentences:
        snippets.append(f"Lab summary: {s}")
    for s in vital_sentences:
        snippets.append(f"Vital summary: {s}")

    snippets.append(f"Medication summary: {med_summary}")
    for mi in med_interactions:
        snippets.append(f"Medication interaction: {mi}")

    if not p_meds.empty:
        for _, row in p_meds.head(200).iterrows():
            snippets.append(
                f"Medication {row.get('drug','')} route {row.get('route','')} "
                f"from {row.get('starttime','')} to {row.get('stoptime','')} "
                f"dose {row.get('dose_val_rx','')} {row.get('dose_unit_rx','')}."
            )

    if not p_labs.empty:
        labs_small = p_labs.sort_values("charttime").tail(200)
        for _, row in labs_small.iterrows():
            label = LAB_LOOKUP.get(row.get("itemid"), f"item {row.get('itemid')}")
            snippets.append(
                f"Lab value: {label} = {row.get('valuenum')} at {row.get('charttime')}."
            )

    if not p_vitals.empty:
        vit_small = p_vitals.sort_values("charttime").tail(200)
        for _, row in vit_small.iterrows():
            snippets.append(
                f"Vital sign: item {row.get('itemid')} = {row.get('valuenum')} at {row.get('charttime')}."
            )

    return snippets


def rule_based_answer(
    question: str,
    data: Dict[str, Any],
    lab_sentences: List[str],
    vital_sentences: List[str],
    med_summary: str,
    med_interactions: List[str],
    lab_flags: Dict[str, bool],
    vital_flags: Dict[str, bool],
    med_flags: Dict[str, bool],
) -> str | None:
    q = question.lower()
    p_labs = data["labs"]
    p_meds = data["meds"]
    p_vitals = data["vitals"]

    # Antibiotics
    if "antibiotic" in q or "antibiotics" in q:
        if p_meds.empty:
            return "No medications are recorded for this patient."
        antibiotics_kw = ["cillin", "mycin", "cef", "metro", "vanco", "clavulanate"]
        mask = p_meds["drug"].astype(str).str.lower().str.contains("|".join(antibiotics_kw))
        abx = p_meds[mask]
        if abx.empty:
            return "No clear antibiotic exposures identified in the medication list."
        lines = []
        for _, row in abx.iterrows():
            lines.append(
                f"- {row.get('drug','')} "
                f"(route {row.get('route','')}, start {row.get('starttime','')}, stop {row.get('stoptime','')})"
            )
        return "Antibiotic courses identified:\n" + "\n".join(lines)

    # Infection
    if "infection" in q or "sepsis" in q or "septic" in q:
        pieces = lab_sentences + vital_sentences
        hits = [
            s for s in pieces
            if any(k in s.lower() for k in ["white blood cell", "fever", "temperature", "lactate"])
        ]
        if not hits:
            return (
                "From structured labs and vitals alone, there is no strong isolated signal for infection. "
                "Cultures, imaging, and bedside assessment remain essential."
            )
        txt = (
            "Findings that may be consistent with infection or systemic inflammation:\n"
            + "\n".join(f"- {h}" for h in hits)
        )
        if med_flags["antibiotic_use"]:
            txt += "\nAntibiotic therapy is currently documented."
        return txt

    # Kidney function
    if "kidney" in q or "renal" in q or "creatinine" in q:
        creat = p_labs[p_labs["itemid"] == 50912]
        if creat.empty:
            base = "No creatinine values are available for this patient."
        else:
            creat = creat.sort_values("charttime")
            first = creat.iloc[0]
            last = creat.iloc[-1]
            base = (
                f"Creatinine trend: first recorded {first['valuenum']} at {first['charttime']}, "
                f"most recent {last['valuenum']} at {last['charttime']}."
            )
        if lab_flags["renal_issue"]:
            return base + "\nCurrent pattern suggests renal stress or impaired clearance."
        else:
            return base + "\nNo strong renal stress signal is detected from creatinine alone."

    # Anemia / hemoglobin
    if "anemia" in q or "hemoglobin" in q or "hb" in q:
        hb_ids = [50811, 51222, 51221]
        hb = p_labs[p_labs["itemid"].isin(hb_ids)]
        if hb.empty:
            return "No hemoglobin values are available for this patient."
        hb = hb.sort_values("charttime")
        last = hb.iloc[-1]
        ans = f"Most recent hemoglobin is {last['valuenum']} at {last['charttime']}."
        if lab_flags["bleeding_risk"]:
            ans += "\nThis level is low enough to raise concern for anemia or bleeding risk."
        return ans

    # Electrolytes
    if "electrolyte" in q or "sodium" in q or "potassium" in q or "na " in q or "k " in q:
        out = []
        na = p_labs[p_labs["itemid"] == 50983].sort_values("charttime")
        if not na.empty:
            last = na.iloc[-1]
            out.append(f"Most recent sodium: {last['valuenum']} at {last['charttime']}.")
        k = p_labs[p_labs["itemid"].isin([50822, 50971])].sort_values("charttime")
        if not k.empty:
            last = k.iloc[-1]
            out.append(f"Most recent potassium: {last['valuenum']} at {last['charttime']}.")
        if not out:
            return "No recent sodium or potassium values are available."
        if lab_flags["electrolyte_issue"]:
            out.append("These values include abnormalities consistent with electrolyte disturbance.")
        return "\n".join(out)

    # Respiratory / oxygenation
    if "respiratory" in q or "breath" in q or "spo2" in q or "oxygen" in q:
        spo2 = p_vitals[p_vitals["itemid"] == 220277].sort_values("charttime")
        rr = p_vitals[p_vitals["itemid"].isin([220210, 224690])].sort_values("charttime")
        out = []
        if not spo2.empty:
            last = spo2.iloc[-1]
            out.append(f"Most recent SpO₂: {last['valuenum']}% at {last['charttime']}.")
        if not rr.empty:
            last = rr.iloc[-1]
            out.append(f"Most recent respiratory rate: {last['valuenum']} breaths/min at {last['charttime']}.")
        if not out:
            return "No structured respiratory data are available."
        if vital_flags["resp_issue"]:
            out.append("Pattern suggests possible respiratory compromise.")
        return "\n".join(out)

    # Lactate trend
    if "lactate" in q:
        df = p_labs[p_labs["itemid"] == 50813]
        if df.empty:
            return "No lactate values are available for this patient."
        df = df.sort_values("charttime")
        first = df.iloc[0]
        last = df.iloc[-1]
        return (
            f"Lactate trend: first recorded {first['valuenum']} at {first['charttime']}, "
            f"most recent {last['valuenum']} at {last['charttime']}."
        )

    # Medication list
    if "what meds" in q or "what medications" in q or "medication list" in q or "all meds" in q:
        if p_meds.empty:
            return "No medications are recorded for this patient."
        meds_uniq = sorted(set(p_meds["drug"].dropna().astype(str).tolist()))
        return "Medication list includes:\n" + "\n".join(f"- {m}" for m in meds_uniq)

    # "Summary" of state
    if "summary" in q or "overview" in q:
        return (
            "High-level medication summary:\n"
            + med_summary
            + ("\n\nPotential medication concerns:\n" + "\n".join(f"- {mi}" for mi in med_interactions)
               if med_interactions else "")
        )

    # If question mentions specific lab name in LAB_LOOKUP
    for itemid, label in LAB_LOOKUP.items():
        if not label:
            continue
        if str(label).lower() in q:
            sub = p_labs[p_labs["itemid"] == itemid]
            if sub.empty:
                return f"No values for {label} are available."
            sub = sub.sort_values("charttime")
            last = sub.iloc[-1]
            return f"Most recent {label}: {last['valuenum']} at {last['charttime']}."

    return None


def rag_answer(question: str, corpus: List[str]) -> str:
    if not SKLEARN_AVAILABLE:
        return (
            "Advanced retrieval-based answers are not available because scikit-learn is not installed.\n"
            "Install scikit-learn to enable richer Q&A.\n"
        )
    if not corpus:
        return "There is not enough structured information to answer this question."

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus + [question])
    sims = cosine_similarity(X[-1], X[:-1]).flatten()

    top_idx = sims.argsort()[::-1][:5]
    picked = [corpus[i] for i in top_idx if sims[i] > 0.05]

    if not picked:
        return (
            "I could not find a clearly relevant segment in the structured record for this question. "
            "Please correlate with the full chart."
        )

    answer = "Context from the structured record:\n"
    for p in picked:
        answer += f"- {p}\n"
    return answer


def answer_question(question: str, data: Dict[str, Any], analysis: Dict[str, Any]) -> str:
    if not question.strip():
        return "Please enter a non-empty question."

    lab_sentences = analysis["lab_sentences"]
    vital_sentences = analysis["vital_sentences"]
    med_summary = analysis["med_summary"]
    med_interactions = analysis["med_interactions"]
    lab_flags = analysis["lab_flags"]
    vital_flags = analysis["vital_flags"]
    med_flags = analysis["med_flags"]

    # 1) Rule-based first
    rb = rule_based_answer(
        question,
        data,
        lab_sentences,
        vital_sentences,
        med_summary,
        med_interactions,
        lab_flags,
        vital_flags,
        med_flags,
    )

    # 2) RAG retrieval
    corpus = build_corpus_for_rag(
        data,
        lab_sentences,
        vital_sentences,
        med_summary,
        med_interactions,
    )
    rag = rag_answer(question, corpus)

    if rb is not None:
        return "Rule-based interpretation:\n" + rb + "\n\n" + rag
    else:
        return rag


# ============================================================
# TKINTER DESKTOP AGENT
# ============================================================

class EMRDesktopAgent:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("EMR Assistant")
        self.root.attributes("-topmost", True)  # Always on top (desktop agent feel)

        # Position on right side of screen (approx)
        self.root.geometry("430x780+1400+50")
        self.root.resizable(False, True)

        self.current_data: Dict[str, Any] | None = None
        self.analysis: Dict[str, Any] | None = None

        # For embedded matplotlib canvas in Trends tab
        self.trend_canvas = None

        self.build_ui()
        self.root.mainloop()

    def build_ui(self):
        # Header
        header = tk.Label(self.root, text="EMR Assistant", font=("Arial", 14, "bold"))
        header.pack(pady=(5, 0))
        sub = tk.Label(
            self.root,
            text="On-demand summary and Q&A from structured EMR data.",
            font=("Arial", 9),
            fg="gray",
        )
        sub.pack(pady=(0, 5))

        # Patient entry bar
        bar = tk.Frame(self.root)
        bar.pack(fill="x", padx=6, pady=(0, 4))

        self.patient_var = tk.StringVar()
        entry = tk.Entry(bar, textvariable=self.patient_var)
        entry.pack(side="left", fill="x", expand=True)
        load_btn = tk.Button(bar, text="Load", command=self.load_patient)
        load_btn.pack(side="left", padx=(4, 0))

        # Status label
        self.status_label = tk.Label(self.root, text="Enter a subject_id and click Load.", fg="gray")
        self.status_label.pack(anchor="w", padx=6)

        # Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)

        # Summary tab
        self.tab_summary = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_summary, text="Summary")

        self.txt_summary = scrolledtext.ScrolledText(self.tab_summary, wrap="word", font=("Consolas", 10), height=10)
        self.txt_summary.pack(fill="both", expand=True)

        self.txt_progress = scrolledtext.ScrolledText(self.tab_summary, wrap="word", font=("Consolas", 10), height=10)
        self.txt_progress.pack(fill="both", expand=True, pady=(4, 0))

        self.txt_alerts = scrolledtext.ScrolledText(self.tab_summary, wrap="word", font=("Consolas", 10), height=10)
        self.txt_alerts.pack(fill="both", expand=True, pady=(4, 4))

        # Export buttons
        export_frame = tk.Frame(self.tab_summary)
        export_frame.pack(fill="x", pady=(2, 2))
        tk.Button(export_frame, text="Export Summary TXT", command=self.export_summary).pack(side="left", padx=2)
        tk.Button(export_frame, text="Export Labs CSV", command=self.export_labs).pack(side="left", padx=2)
        tk.Button(export_frame, text="Export Vitals CSV", command=self.export_vitals).pack(side="left", padx=2)
        tk.Button(export_frame, text="Export Meds CSV", command=self.export_meds).pack(side="left", padx=2)

        # Labs tab
        self.tab_labs = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_labs, text="Labs")

        labs_btn_frame = tk.Frame(self.tab_labs)
        labs_btn_frame.pack(fill="x", pady=(2, 2))
        tk.Button(labs_btn_frame, text="View Full Lab Table", command=self.open_labs_table).pack(side="left", padx=2)

        self.txt_labs = scrolledtext.ScrolledText(self.tab_labs, wrap="word", font=("Consolas", 10))
        self.txt_labs.pack(fill="both", expand=True)

        # Vitals tab
        self.tab_vitals = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_vitals, text="Vitals")

        vitals_btn_frame = tk.Frame(self.tab_vitals)
        vitals_btn_frame.pack(fill="x", pady=(2, 2))
        tk.Button(vitals_btn_frame, text="View Full Vitals Table", command=self.open_vitals_table).pack(side="left", padx=2)

        self.txt_vitals = scrolledtext.ScrolledText(self.tab_vitals, wrap="word", font=("Consolas", 10))
        self.txt_vitals.pack(fill="both", expand=True)

        # Medications tab
        self.tab_meds = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_meds, text="Medications")

        meds_btn_frame = tk.Frame(self.tab_meds)
        meds_btn_frame.pack(fill="x", pady=(2, 2))
        tk.Button(meds_btn_frame, text="View Full Meds Table", command=self.open_meds_table).pack(side="left", padx=2)

        self.txt_meds = scrolledtext.ScrolledText(self.tab_meds, wrap="word", font=("Consolas", 10))
        self.txt_meds.pack(fill="both", expand=True)

        # Trends tab
        self.tab_trends = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_trends, text="Trends")

        trends_top = tk.Frame(self.tab_trends)
        trends_top.pack(fill="x", pady=(4, 4))

        tk.Label(trends_top, text="Plot lab trend for itemid:").pack(side="left")
        self.trend_item_var = tk.StringVar()
        self.trend_item_menu = ttk.Combobox(trends_top, textvariable=self.trend_item_var, width=20)
        self.trend_item_menu.pack(side="left", padx=(4, 4))

        tk.Button(trends_top, text="Plot", command=self.plot_trend).pack(side="left")

        # Canvas frame for matplotlib
        self.trend_canvas_frame = tk.Frame(self.tab_trends)
        self.trend_canvas_frame.pack(fill="both", expand=True)

        self.txt_trends = scrolledtext.ScrolledText(self.tab_trends, wrap="word", font=("Consolas", 10), height=8)
        self.txt_trends.pack(fill="x", expand=False)

        # Delta tab
        self.tab_delta = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_delta, text="Delta (Day)")

        self.txt_delta = scrolledtext.ScrolledText(self.tab_delta, wrap="word", font=("Consolas", 10))
        self.txt_delta.pack(fill="both", expand=True)

        # Q&A tab
        self.tab_qa = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_qa, text="Q&A")

        qa_top = tk.Frame(self.tab_qa)
        qa_top.pack(fill="x", pady=(4, 2))
        tk.Label(qa_top, text="Ask about this patient:").pack(anchor="w")

        qa_entry_frame = tk.Frame(self.tab_qa)
        qa_entry_frame.pack(fill="x")

        self.qa_var = tk.StringVar()
        self.qa_entry = tk.Entry(qa_entry_frame, textvariable=self.qa_var)
        self.qa_entry.pack(side="left", fill="x", expand=True)
        tk.Button(qa_entry_frame, text="Ask", command=self.handle_qa).pack(side="left", padx=(4, 0))

        self.txt_qa = scrolledtext.ScrolledText(self.tab_qa, wrap="word", font=("Consolas", 10))
        self.txt_qa.pack(fill="both", expand=True, pady=(4, 0))

    # --------------------- Helpers ---------------------

    def set_text(self, widget: scrolledtext.ScrolledText, text: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def highlight_phrases(self, widget: scrolledtext.ScrolledText, phrases: List[str], color: str = "red"):
        """Highlight given phrases in red inside a Text widget."""
        widget.config(state="normal")
        try:
            widget.tag_delete("alert")
        except tk.TclError:
            pass
        widget.tag_configure("alert", foreground=color)

        for phrase in phrases:
            if not phrase:
                continue
            start = "1.0"
            while True:
                idx = widget.search(phrase, start, stopindex="end")
                if not idx:
                    break
                end = f"{idx}+{len(phrase)}c"
                widget.tag_add("alert", idx, end)
                start = end
        widget.config(state="disabled")

    def open_table_window(self, title: str, df: pd.DataFrame):
        if df is None or df.empty:
            messagebox.showinfo(title, f"No {title} data available.")
            return

        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("900x500")

        cols = list(df.columns)
        tree = ttk.Treeview(win, columns=cols, show="headings")
        vsb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.pack(side="top", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, anchor="center")

        # Limit to first 1000 rows to avoid UI freeze
        for _, row in df.head(1000).iterrows():
            vals = [str(row[c]) for c in cols]
            tree.insert("", "end", values=vals)

        tk.Label(win, text=f"Showing up to 1000 rows of {title.lower()} data.").pack(anchor="w")

    # --------------------- Export helpers ---------------------

    def export_summary(self):
        if not self.current_data or not self.analysis:
            messagebox.showinfo("Export Summary", "Load a patient first.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Summary As"
        )
        if not filename:
            return
        summary_text = self.txt_summary.get("1.0", "end-1c") + "\n\n" + self.txt_progress.get("1.0", "end-1c") \
            + "\n\n" + self.txt_alerts.get("1.0", "end-1c")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(summary_text)
        messagebox.showinfo("Export Summary", f"Summary saved to {filename}")

    def export_labs(self):
        if not self.current_data:
            messagebox.showinfo("Export Labs", "Load a patient first.")
            return
        df = self.current_data["labs"]
        if df.empty:
            messagebox.showinfo("Export Labs", "No lab data for this patient.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Labs As"
        )
        if not filename:
            return
        df.to_csv(filename, index=False)
        messagebox.showinfo("Export Labs", f"Labs saved to {filename}")

    def export_vitals(self):
        if not self.current_data:
            messagebox.showinfo("Export Vitals", "Load a patient first.")
            return
        df = self.current_data["vitals"]
        if df.empty:
            messagebox.showinfo("Export Vitals", "No vital data for this patient.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Vitals As"
        )
        if not filename:
            return
        df.to_csv(filename, index=False)
        messagebox.showinfo("Export Vitals", f"Vitals saved to {filename}")

    def export_meds(self):
        if not self.current_data:
            messagebox.showinfo("Export Medications", "Load a patient first.")
            return
        df = self.current_data["meds"]
        if df.empty:
            messagebox.showinfo("Export Medications", "No medication data for this patient.")
            return
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Medications As"
        )
        if not filename:
            return
        df.to_csv(filename, index=False)
        messagebox.showinfo("Export Medications", f"Medications saved to {filename}")

    # --------------------- Table view actions ---------------------

    def open_labs_table(self):
        if not self.current_data:
            messagebox.showinfo("Labs", "Load a patient first.")
            return
        self.open_table_window("Labs", self.current_data["labs"])

    def open_vitals_table(self):
        if not self.current_data:
            messagebox.showinfo("Vitals", "Load a patient first.")
            return
        self.open_table_window("Vitals", self.current_data["vitals"])

    def open_meds_table(self):
        if not self.current_data:
            messagebox.showinfo("Medications", "Load a patient first.")
            return
        self.open_table_window("Medications", self.current_data["meds"])

    # --------------------- Core logic ---------------------

    def load_patient(self):
        text = self.patient_var.get().strip()
        if not text:
            self.status_label.configure(text="Please enter a subject_id.", fg="red")
            return

        try:
            pid = int(text)
        except ValueError:
            self.status_label.configure(text="Patient ID must be an integer.", fg="red")
            return

        try:
            data = load_patient_data(pid)
        except FileNotFoundError as e:
            self.status_label.configure(text=str(e), fg="red")
            return

        if data is None:
            self.status_label.configure(text=f"No admissions found for subject_id {pid}.", fg="red")
            return

        self.current_data = data

        # ---------------- DEATH CHECK ----------------
        p_adm = data["admissions"]
        death_time = None
        if "deathtime" in p_adm.columns:
            dt_series = p_adm["deathtime"].dropna()
            if not dt_series.empty:
                # use the latest recorded deathtime
                death_time = dt_series.max()

        if death_time is not None:
            # Deceased patient mode: no clinical reasoning, only factual lookup
            death_msg = (
                "⚠️ PATIENT DECEASED ⚠️\n\n"
                f"Recorded death time: {death_time}\n\n"
                "Clinical summaries, alerts, trends, and risk scoring are disabled for deceased patients.\n\n"
                "You may still:\n"
                "- View raw Labs, Vitals, and Medications tables\n"
                "- Ask factual questions (e.g., 'show recent labs', 'what meds were given?')\n"
            )

            # Summary / Progress / Alerts
            self.set_text(self.txt_summary, death_msg)
            self.set_text(self.txt_progress, "")
            self.set_text(self.txt_alerts, "Clinical alerts are disabled for deceased patients.")

            # Labs / Vitals / Meds text
            self.set_text(self.txt_labs, "Use 'View Full Lab Table' to inspect raw labs.")
            self.set_text(self.txt_vitals, "Use 'View Full Vitals Table' to inspect raw vitals.")
            self.set_text(self.txt_meds, "Use 'View Full Meds Table' to inspect medications.")

            # Trends / Delta disabled text
            self.set_text(self.txt_trends, "Trend plotting is disabled for deceased patients.")
            self.set_text(self.txt_delta, "Delta (day-to-day) analysis is disabled for deceased patients.")

            # Q&A history reset with hint
            self.txt_qa.config(state="normal")
            self.txt_qa.delete("1.0", "end")
            self.txt_qa.insert(
                "1.0",
                "Q&A history (deceased patient mode):\n\n"
                "You can ask factual questions like:\n"
                "- show recent labs\n"
                "- list medications\n"
                "- what diagnoses are recorded?\n\n"
            )
            self.txt_qa.config(state="disabled")

            # Store analysis as "death mode" only with raw tables
            self.analysis = {
                "death_mode": True,
                "labs": data["labs"],
                "vitals": data["vitals"],
                "meds": data["meds"],
                "dx": data["dx"],
            }

            self.status_label.configure(text=f"Patient {pid} is deceased.", fg="red")
            return
        # -------------- END DEATH CHECK ----------------

        # Compute analysis for living patients
        p_labs = data["labs"]
        p_vitals = data["vitals"]
        p_meds = data["meds"]
        p_dx = data["dx"]

        lab_sentences, lab_flags = detect_lab_abnormalities(p_labs)
        vital_sentences, vital_flags = detect_vital_abnormalities(p_vitals)
        med_summary, meds_list, med_flags, med_interactions = summarize_medications(p_meds)
        tests, conditions = suggest_tests_and_conditions(lab_flags, vital_flags, med_flags)

        if not p_dx.empty and "long_title" in p_dx.columns:
            dx_list = p_dx["long_title"].dropna().unique().tolist()
            primary_dx = dx_list[0] if dx_list else "no clearly documented primary diagnosis"
        else:
            primary_dx = "no clearly documented primary diagnosis"

        overall = build_overall_summary(data, primary_dx)
        progress_note = build_progress_note(primary_dx, lab_flags, vital_flags, med_flags, tests, conditions)
        high_alerts, moderate_alerts, mild_alerts = build_alerts(lab_flags, vital_flags, med_flags, med_interactions)
        severity_score, severity_level = compute_severity_score(lab_flags, vital_flags, med_flags, med_interactions)

        # Save analysis for Q&A and trends
        self.analysis = {
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

        # Fill Summary tab
        self.set_text(self.txt_summary, overall)
        self.set_text(self.txt_progress, progress_note)

        # Alerts with severity info
        severity_bar = "[" + "#" * min(severity_score, 10) + "-" * max(0, 10 - severity_score) + "]"
        alerts_text = f"Overall severity score: {severity_score} ({severity_level} risk) {severity_bar}\n\n"

        if high_alerts:
            alerts_text += "High-risk alerts:\n" + "\n".join(f"- {x}" for x in high_alerts) + "\n\n"
        if moderate_alerts:
            alerts_text += "Moderate concerns:\n" + "\n".join(f"- {x}" for x in moderate_alerts) + "\n\n"
        if not high_alerts and not moderate_alerts:
            alerts_text += "No strong red-flag patterns detected from structured data alone.\n\n"
        if mild_alerts:
            alerts_text += "\n".join(mild_alerts)

        self.set_text(self.txt_alerts, alerts_text)

        # Labs summary text
        labs_text = "Key recent lab findings:\n\n" + "\n".join(f"- {s}" for s in lab_sentences)
        self.set_text(self.txt_labs, labs_text)

        # Vitals summary text
        vitals_text = "Key recent vital sign findings:\n\n" + "\n".join(f"- {s}" for s in vital_sentences)
        self.set_text(self.txt_vitals, vitals_text)

        # Medications text
        meds_text = "Medication overview:\n" + med_summary + "\n\n"
        if meds_list:
            meds_text += "Medication list (compressed):\n" + "\n".join(f"- {m}" for m in meds_list) + "\n\n"
        if med_interactions:
            meds_text += "Potential medication concerns:\n" + "\n".join(f"- {m}" for m in med_interactions)
        self.set_text(self.txt_meds, meds_text)

        # Highlight abnormal sentences in red
        self.highlight_phrases(self.txt_labs, lab_sentences)
        self.highlight_phrases(self.txt_vitals, vital_sentences)
        self.highlight_phrases(self.txt_alerts, high_alerts + moderate_alerts)

        # Trends tab text (instructions)
        trend_instr = (
            "Trend plotting:\n"
            "- Select a lab itemid from the dropdown above and click 'Plot' to view a time-series plot.\n\n"
        )
        self.set_text(self.txt_trends, trend_instr)

        # Populate drop-down with common lab itemids
        if not p_labs.empty:
            itemids = p_labs["itemid"].value_counts().index.tolist()
            labels = []
            for it in itemids[:50]:
                label = LAB_LOOKUP.get(it, f"item {it}")
                labels.append(f"{it} – {label}")
            self.trend_item_menu["values"] = labels
            if labels:
                self.trend_item_var.set(labels[0])
        else:
            self.trend_item_menu["values"] = []
            self.trend_item_var.set("")

        # Delta tab
        delta_df = compare_today_yesterday(p_labs)
        delta_text = delta_text_from_df(delta_df)
        self.set_text(self.txt_delta, delta_text)

        # Clear Q&A text, reset as empty chat
        self.txt_qa.config(state="normal")
        self.txt_qa.delete("1.0", "end")
        self.txt_qa.insert("1.0", "Q&A history:\n\n")
        self.txt_qa.config(state="disabled")

        self.status_label.configure(text=f"Loaded patient {pid}.", fg="green")

    def handle_qa(self):
        if not self.current_data or not self.analysis:
            self.set_text(self.txt_qa, "Please load a patient first.")
            return
        q = self.qa_var.get().strip()
        if not q:
            self.set_text(self.txt_qa, "Please enter a question.")
            return

        # If patient is deceased → factual lookup only
        if self.analysis.get("death_mode", False):
            ans = self.factual_lookup_only(q)
        else:
            ans = answer_question(q, self.current_data, self.analysis)

        # Append to Q&A history (chat-style)
        self.txt_qa.config(state="normal")
        self.txt_qa.insert("end", f"You: {q}\n\nAgent:\n{ans}\n\n")
        self.txt_qa.see("end")
        self.txt_qa.config(state="disabled")

        # Clear entry box
        self.qa_var.set("")

    def factual_lookup_only(self, question: str) -> str:
        """
        Q&A mode for deceased patients — no clinical reasoning,
        only raw factual lookup from labs / vitals / meds / diagnoses.
        """
        q = question.lower()
        data = self.current_data

        if data is None:
            return "No patient loaded."

        # Show recent labs
        if "lab" in q or "labs" in q or "value" in q:
            df = data["labs"]
            if df.empty:
                return "No lab data available."
            return "Recent labs (last 20 rows):\n" + df.tail(20).to_string()

        # Show vitals
        if "vital" in q or "bp" in q or "heart rate" in q or "spo2" in q or "oxygen" in q:
            df = data["vitals"]
            if df.empty:
                return "No vital sign data available."
            return "Recent vitals (last 20 rows):\n" + df.tail(20).to_string()

        # Show medications
        if "med" in q or "drug" in q or "medication" in q:
            df = data["meds"]
            if df.empty:
                return "No medication data available."
            meds = sorted(df["drug"].dropna().unique().tolist())
            if not meds:
                return "No medication names recorded."
            return "Medications recorded:\n" + "\n".join(f"- {m}" for m in meds)

        # Diagnoses
        if "diagnosis" in q or "dx" in q:
            df = data["dx"]
            if df.empty:
                return "No diagnoses recorded."
            if "long_title" in df.columns:
                dx_list = df["long_title"].dropna().unique().tolist()
            else:
                dx_list = df["icd_code"].dropna().unique().tolist()
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

    def plot_trend(self):
        # Disable trends for deceased patients
        if self.analysis and self.analysis.get("death_mode", False):
            messagebox.showinfo("Trend plot", "Trend plotting is disabled for deceased patients.")
            return

        if not MATPLOTLIB_AVAILABLE:
            messagebox.showinfo("Trend plot", "matplotlib is not installed; cannot create plots.")
            return
        if not self.current_data:
            messagebox.showinfo("Trend plot", "Load a patient first.")
            return

        p_labs = self.current_data["labs"]
        if p_labs.empty:
            messagebox.showinfo("Trend plot", "No lab data available for this patient.")
            return

        choice = self.trend_item_var.get()
        if not choice:
            messagebox.showinfo("Trend plot", "Select a lab itemid first.")
            return

        # choice looks like "50912 – Creatinine"
        try:
            itemid = int(choice.split("–")[0].strip())
        except Exception:
            messagebox.showinfo("Trend plot", "Could not parse chosen itemid.")
            return

        df = p_labs[p_labs["itemid"] == itemid].dropna(subset=["charttime", "valuenum"]).sort_values("charttime")
        if df.empty:
            messagebox.showinfo("Trend plot", "No data points for that lab item.")
            return

        label = LAB_LOOKUP.get(itemid, f"item {itemid}")

        # Clear previous canvas
        for child in self.trend_canvas_frame.winfo_children():
            child.destroy()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(df["charttime"], df["valuenum"], marker="o", linestyle="-")
        ax.set_title(f"{label} (item {itemid}) over time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        fig.autofmt_xdate()

        canvas = FigureCanvasTkAgg(fig, master=self.trend_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.trend_canvas = canvas


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    EMRDesktopAgent()
