from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import timedelta

import pandas as pd

# Optional: scikit-learn for RAG Q&A
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# DATA REPO + LOADING
# ============================================================

@dataclass
class Tables:
    admissions: pd.DataFrame
    labs: pd.DataFrame
    meds: pd.DataFrame
    vitals: pd.DataFrame
    diag: pd.DataFrame
    icd: pd.DataFrame
    lab_lookup: Dict[int, str]


def load_all_tables(base: Path) -> Tables:
    """
    Load MIMIC-IV demo tables from:
      base/hosp/*.csv.gz
      base/icu/*.csv.gz

    base should be the folder: mimic-iv-clinical-database-demo-2.2
    """
    base = Path(base)
    if not base.exists():
        raise FileNotFoundError(f"Base folder not found: {base}")

    hosp = base / "hosp"
    icu = base / "icu"
    if not hosp.exists() or not icu.exists():
        raise FileNotFoundError(f"Expected 'hosp' and 'icu' under: {base}")

    admissions = pd.read_csv(
        hosp / "admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    labs = pd.read_csv(
        hosp / "labevents.csv.gz",
        parse_dates=["charttime"],
    )
    meds = pd.read_csv(
        hosp / "prescriptions.csv.gz",
        parse_dates=["starttime", "stoptime"],
    )
    vitals = pd.read_csv(
        icu / "chartevents.csv.gz",
        parse_dates=["charttime"],
    )
    diag = pd.read_csv(hosp / "diagnoses_icd.csv.gz")
    icd = pd.read_csv(hosp / "d_icd_diagnoses.csv.gz")
    labitems = pd.read_csv(hosp / "d_labitems.csv.gz")

    lab_lookup = dict(zip(labitems["itemid"].astype(int), labitems["label"].astype(str)))

    return Tables(
        admissions=admissions,
        labs=labs,
        meds=meds,
        vitals=vitals,
        diag=diag,
        icd=icd,
        lab_lookup=lab_lookup,
    )


def load_patient_data(t: Tables, subject_id: int) -> Optional[Dict[str, Any]]:
    p_adm = t.admissions[t.admissions["subject_id"] == subject_id].copy()
    if p_adm.empty:
        return None

    p_labs = t.labs[t.labs["subject_id"] == subject_id].copy()
    p_meds = t.meds[t.meds["subject_id"] == subject_id].copy()
    p_vitals = t.vitals[t.vitals["subject_id"] == subject_id].copy()

    p_dx = t.diag[t.diag["subject_id"] == subject_id].copy()
    if not p_dx.empty:
        # Merge to long_title if possible
        # diagnoses_icd has: subject_id, hadm_id, seq_num, icd_code, icd_version
        # d_icd_diagnoses has: icd_code, icd_version, long_title
        join_cols = ["icd_code"]
        if "icd_version" in p_dx.columns and "icd_version" in t.icd.columns:
            join_cols = ["icd_code", "icd_version"]
        p_dx = p_dx.merge(t.icd, on=join_cols, how="left")

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
        return float(sub.iloc[-1]["valuenum"])

    creat = get_latest(creat_ids)
    if creat is not None and creat > 1.5:
        flags["renal_issue"] = True
        sentences.append(f"Creatinine is elevated at {creat:.2f}, suggesting renal stress or impaired clearance.")

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
            sentences.append(f"Lactate is critically elevated at {lact:.1f}, suggesting significant hypoperfusion or severe stress.")
        elif lact > 2:
            flags["metabolic_stress"] = True
            sentences.append(f"Lactate is elevated at {lact:.1f}, suggesting increased metabolic demand or impaired clearance.")

    inr = get_latest(inr_ids)
    if inr is not None and inr > 1.5:
        flags["bleeding_risk"] = True
        sentences.append(f"INR is elevated at {inr:.2f}, indicating increased bleeding risk.")

    wbc = get_latest(wbc_ids)
    if wbc is not None:
        if wbc > 12:
            flags["infection_risk"] = True
            sentences.append(f"White blood cell count is elevated at {wbc:.1f}, which may indicate infection or inflammation.")
        elif wbc < 4:
            flags["infection_risk"] = True
            sentences.append(f"White blood cell count is low at {wbc:.1f}, which may reflect marrow suppression or severe infection.")

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
        ids = [iids] if isinstance(iids, int) else list(iids)
        sub = df[df["itemid"].isin(ids)]
        if sub.empty:
            return None
        return float(sub.iloc[-1]["valuenum"])

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
    if flags["anticoagulant_use"] and any(("aspirin" in m) or ("clopidogrel" in m) for m in meds_lower):
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


def compare_today_yesterday(p_labs: pd.DataFrame) -> Optional[pd.DataFrame]:
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


def delta_text_from_df(comp: Optional[pd.DataFrame], lab_lookup: Dict[int, str]) -> str:
    if comp is None or comp.empty:
        return "Not enough separated days of lab data to compute a meaningful day-to-day comparison."

    lines_worse, lines_better, lines_same = [], [], []

    for itemid, row in comp.iterrows():
        label = lab_lookup.get(int(itemid), f"item {itemid}")
        delta = float(row["delta"])
        today = float(row["today"])
        yesterday = float(row["yesterday"])

        if abs(delta) < 0.01:
            lines_same.append(f"→ {label}: unchanged (approx {today:.2f}).")
        elif delta > 0:
            lines_worse.append(f"↑ {label}: increased {yesterday:.2f} → {today:.2f} (Δ +{delta:.2f}).")
        else:
            lines_better.append(f"↓ {label}: decreased {yesterday:.2f} → {today:.2f} (Δ {delta:.2f}).")

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
    lab_lookup: Dict[int, str],
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
        if diag_desc:
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
