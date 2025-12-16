import streamlit as st
import emr_core as core

st.set_page_config(
    page_title="EMR Assistant",
    layout="wide",
)

st.title("EMR Assistant")
st.caption("On-demand summary and Q&A from structured EMR data")

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "data_initialized" not in st.session_state:
    st.session_state.data_initialized = False
if "current_data" not in st.session_state:
    st.session_state.current_data = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None


# ------------------------------------------------------------
# Sidebar – Data initialization
# ------------------------------------------------------------
st.sidebar.header("Data Setup")

base_path = st.sidebar.text_input(
    "Path to MIMIC-IV demo folder",
    placeholder="e.g. /mount/src/mimic-iv-clinical-database-demo-2.2",
)

if st.sidebar.button("Initialize data"):
    if not base_path.strip():
        st.sidebar.error("Please enter a valid base path.")
    else:
        try:
            core.initialize_data(base_path)
            st.session_state.data_initialized = True
            st.sidebar.success("Data loaded successfully.")
        except Exception as e:
            st.session_state.data_initialized = False
            st.sidebar.error(str(e))


# ------------------------------------------------------------
# Patient loader
# ------------------------------------------------------------
st.sidebar.header("Patient")

pid = st.sidebar.text_input("Subject ID")

load_patient = st.sidebar.button("Load patient")

if load_patient:
    if not st.session_state.data_initialized:
        st.sidebar.error("Please initialize data first.")
    elif not pid.strip():
        st.sidebar.error("Please enter a subject ID.")
    else:
        try:
            pid_int = int(pid)
            data = core.load_patient_data(pid_int)
            if data is None:
                st.error(f"No admissions found for subject_id {pid_int}.")
            else:
                st.session_state.current_data = data
                st.success(f"Loaded patient {pid_int}.")
        except Exception as e:
            st.error(str(e))


# ------------------------------------------------------------
# Main content
# ------------------------------------------------------------
if st.session_state.current_data is None:
    st.info("Load a patient to view summaries and analysis.")
    st.stop()

data = st.session_state.current_data

tabs = st.tabs(
    [
        "Summary",
        "Labs",
        "Vitals",
        "Medications",
        "Delta (Day)",
        "Q&A",
    ]
)

# ------------------------------------------------------------
# Summary tab
# ------------------------------------------------------------
with tabs[0]:
    st.subheader("Patient Overview")

    try:
        analysis = core.run_patient_analysis(data)
        st.session_state.analysis = analysis

        st.text_area(
            "Overall summary",
            analysis["overall_summary"],
            height=220,
        )

        st.text_area(
            "Assessment & Plan",
            analysis["progress_note"],
            height=260,
        )

        st.text_area(
            "Alerts",
            analysis["alerts_text"],
            height=220,
        )

    except Exception as e:
        st.error(str(e))


# ------------------------------------------------------------
# Labs tab
# ------------------------------------------------------------
with tabs[1]:
    st.subheader("Labs")
    st.dataframe(data["labs"].head(500), use_container_width=True)


# ------------------------------------------------------------
# Vitals tab
# ------------------------------------------------------------
with tabs[2]:
    st.subheader("Vitals")
    st.dataframe(data["vitals"].head(500), use_container_width=True)


# ------------------------------------------------------------
# Medications tab
# ------------------------------------------------------------
with tabs[3]:
    st.subheader("Medications")
    st.dataframe(data["meds"].head(500), use_container_width=True)


# ------------------------------------------------------------
# Delta tab
# ------------------------------------------------------------
with tabs[4]:
    st.subheader("Day-to-day lab changes")
    try:
        delta_df = core.compare_today_yesterday(data["labs"])
        if delta_df is None or delta_df.empty:
            st.info("Not enough data for day-to-day comparison.")
        else:
            st.dataframe(delta_df, use_container_width=True)
    except Exception as e:
        st.error(str(e))


# ------------------------------------------------------------
# Q&A tab
# ------------------------------------------------------------
with tabs[5]:
    st.subheader("Ask about this patient")

    question = st.text_input("Question")

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                answer = core.answer_question(
                    question,
                    data,
                    st.session_state.analysis,
                )
                st.text_area("Answer", answer, height=260)
            except Exception as e:
                st.error(str(e))
