import streamlit as st
import json
import os
from dotenv import load_dotenv

# Import your created sequential agent
# Ensure sequential_agent.py is in the same folder
from sequential_agent import healthflow_sequential

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="MediFlow AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MediFlow: Triage & Doctor Finder")
st.markdown("---")

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# We must persist the agent session ID or state if the ADK requires it.
# Assuming the healthflow_sequential object handles internal state automatically 
# or acts as a singleton for this local demo.

# --- Helper Functions for Formatting ---

def format_triage_card(data):
    """Formats the Patient Intake JSON into a nice UI card."""
    patient = data.get("patient_details", {})
    conditions = data.get("possible_conditions", [])
    rec = data.get("final_recommendation", {})
    
    with st.expander("📋 Triage Report Generated", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Patient:** {patient.get('name')} ({patient.get('age')}/{patient.get('gender')})")
            st.markdown(f"**Location:** {patient.get('location')}")
        with col2:
            st.markdown(f"**Urgency:** `{rec.get('urgency')}`")
            st.markdown(f"**Action:** {rec.get('action')}")
        
        st.divider()
        st.subheader("Top Possible Conditions")
        for cond in conditions:
            st.markdown(f"- **{cond.get('condition')}** ({cond.get('confidence_percentage')}): {cond.get('rationale')}")
        
        st.divider()
        st.markdown(f"**Next Steps:** {rec.get('next_steps')}")
        st.warning("⚠️ Medical Disclaimer: This is not a diagnosis. Consult a professional.")

def format_doctor_card(data):
    """Formats the Doctor Finder JSON into nice UI cards."""
    doctors = data.get("top_doctors", [])
    
    st.success(f"Found {data.get('total_doctors_found')} doctors for {data.get('mapped_specialty')} in {data.get('search_location')}")
    
    for doc in doctors:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.subheader(f"{doc.get('rank')}. {doc.get('doctor_name')}")
                st.markdown(f"**{doc.get('clinic_name')}**")
                st.markdown(f"📍 {doc.get('address')} ({doc.get('distance_km')} km)")
            with c2:
                st.markdown(f"⭐ **{doc.get('rating')}** ({doc.get('review_count')})")
                if doc.get('google_maps_link'):
                    st.link_button("View on Maps", doc.get('google_maps_link'))

def attempt_json_parse(text_content):
    """
    Tries to detect if the response contains specific JSON schemas 
    (TriageOutput or DoctorSearchOutput) and renders them visually.
    """
    try:
        # Find start and end of JSON block if mixed with text
        json_start = text_content.find('{')
        json_end = text_content.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = text_content[json_start:json_end]
            data = json.loads(json_str)
            
            # Check signatures to decide how to render
            if "patient_details" in data and "possible_conditions" in data:
                format_triage_card(data)
                return True
            elif "top_doctors" in data and "mapped_specialty" in data:
                format_doctor_card(data)
                return True
    except:
        pass # If parsing fails, just return False to print raw text
    return False

# --- Chat Interface ---

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # If the history contains specific JSON, we assume it was already rendered nicely, 
        # but for history, we usually show the raw text or a summary. 
        # For simplicity, we render the markdown content.
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Type your response here..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Agent Response
    with st.chat_message("assistant"):
        with st.spinner("MediFlow is thinking..."):
            try:
                # --- AGENT INTERACTION ---
                # Depending on your ADK version, the method might be .invoke(), .process(), or .run()
                # sequential_agent.py exposes 'healthflow_sequential'
                
                response_object = healthflow_sequential.invoke(prompt)
                
                # Extract text output. ADK agents often return an object with a .text attribute 
                # or a dictionary. Adjust this line based on strict ADK return types.
                if hasattr(response_object, 'text'):
                    response_text = response_object.text
                elif isinstance(response_object, dict) and 'output' in response_object:
                    response_text = response_object['output']
                else:
                    response_text = str(response_object)

                # 3. Render Logic
                # Check if it's a special JSON output first
                is_json = attempt_json_parse(response_text)
                
                # If it wasn't rendered as a special card, print as text
                if not is_json:
                    st.markdown(response_text)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                st.error(f"Error communicating with agent: {e}")