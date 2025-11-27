# gradio_app.py
import gradio as gr
from patient_intake_agent import patient_intake_agent
from doctor_finder_agent import doctor_finder_agent

def chat(message, history):
    # Step 1: Run Agent 1 (Tara)
    triage_result = patient_intake_agent.run(message)
    
    # Step 2: Run Agent 2 (Drishti) with Agent 1's output
    doctor_result = doctor_finder_agent.run(str(triage_result))
    
    return str(doctor_result)

demo = gr.ChatInterface(
    fn=chat, 
    title="🏥 HealthFlow AI",
    type="messages"
)

demo.launch()
