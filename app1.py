import streamlit as st

st.set_page_config(layout="wide")
st.title("EMR AI Agent – Cloud Demo")

st.info(
    "This is a SAFE DEMO version for reviewers.\n\n"
    "• No real patient data\n"
    "• No MIMIC dataset\n"
    "• No clinical decision making\n"
)

# Demo patient selector
pid = st.text_input("Enter Demo Patient ID")

if pid:
    st.success(f"Demo patient {pid} loaded")

    st.subheader("Patient Summary (Demo)")
    st.markdown("""
    - **Hemodynamics:** Stable  
    - **Renal function:** Mild creatinine elevation  
    - **Infection markers:** No strong signal  
    - **Respiratory:** Stable on room air  
    """)

    st.subheader("Alerts")
    st.warning("No critical alerts detected")

    st.subheader("Ask the EMR Agent")
    q = st.text_input("Ask a question about the patient")

    if q:
        st.write("**Agent response (demo):**")
        st.info(
            "Based on structured EMR-style data, "
            "no high-risk abnormalities are detected at this time. "
            "Clinical correlation is required."
        )

st.divider()
st.caption(
    "⚠️ Research prototype. Not for clinical use. "
    "Demonstration only."
)
