from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent, BaseAgent
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv
import os

try:
    GOOGLE_API_KEY = load_dotenv("GOOGLE_API_KEY")
    print("✅ Gemini API key setup complete.")
except Exception as e:
    print(
        f"🔑 Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )

# agent.py
  # or appropriate ADK entrypoint

### Helpers / Constants ###

LLM_MODEL = "gemini-2.0-flash"  # or whichever model you have access to

### 1. Agents ###

# 1.1 Symptom Collector — ask user for symptoms iteratively
symptom_collector = LlmAgent(
    name="SymptomCollectorAgent",
    model=LLM_MODEL,
    description="Collects symptoms from user; asks until user says done.",
    # For interactive mode: we will handle user I/O via session + external UI or CLI
    instruction=(
        "You are a health-assistant. Ask the user to list symptoms they are experiencing. "
        "Prompt: 'Please describe one symptom (or type \"done\" if finished)'. "
        "Return the symptom described or 'done'."
    ),
    output_key="last_symptom"
)

# 1.2 Condition Analyzer — analyze full symptom list & (optionally) context → possible conditions
condition_analyzer = LlmAgent(
    name="ConditionAnalyzerAgent",
    model=LLM_MODEL,
    description="Given list of symptoms, suggest possible conditions with risk/urgency levels and ask whether the user recognizes any of them.",
    # We'll format a prompt that reads from session.state['symptoms_list']
    instruction=(
        "You are a medical-assistant (non-clinician). Based only on the following symptoms: {symptoms_list}, "
        "provide 3–5 possible conditions or causes (non-definitive), each with a brief explanation, and a risk/urgency level (low / medium / high). "
        "Then ask the user: 'Do you think any of these seems like what you have? If yes, which one (by number)? If none, say \"none\".'"
    ),
    output_key="possible_conditions"
)

# 1.3 Guidance Agent — provide advice / disclaimers / home-remedy suggestions or recommend seeing doctor
guidance_agent = LlmAgent(
    name="GuidanceAgent",
    model=LLM_MODEL,
    description="Given selected condition (or none) plus symptoms, advise user on self-care, home remedies or suggest doctor consult; always include disclaimer.",
    instruction=(
        "You are a friendly health-assistant (non-doctor). Based on symptoms {symptoms_list} and selected condition {selected_condition}, "
        "provide a plain-language guidance: first a clear disclaimer that you are not a medical professional; then self-care advice or next-step recommendation. "
        "If condition seems serious, recommend doctor consult. Otherwise, you may suggest safe home-care steps."
    ),
    output_key="guidance_text"
)

# 1.4 Doctor Finder — search for doctors/clinics nearby if user wants to consult
doctor_finder = LlmAgent(
    name="DoctorFinderAgent",
    model=LLM_MODEL,
    description="If user wants doctor consult, search (via google_search) for relevant nearby doctors/clinics according to user location and condition.",
    instruction=(
        "You are an assistant that finds doctors or clinics nearby. Based on user's location {location} and probable condition {selected_condition}, "
        "formulate a set of Google-search queries to find at least 3 suitable doctors/clinics. Then parse the results and return a short list with name, specialty, address or contact (if available)."
    ),
    tools=[google_search],
    output_key="doctor_list"
)

# 1.5 Report Generator — compile final summary (symptoms, possible conditions, guidance, doctor list if any)
report_generator = LlmAgent(
    name="ReportGeneratorAgent",
    model=LLM_MODEL,
    description="Generate a final health summary report for user: symptoms, selected condition (if any), guidance, doctor options (if any).",
    instruction=(
        "You are a health-assistant summarizer. Create a clean, human-readable report containing:\n"
        "1. Symptoms list\n"
        "2. Possible conditions considered (with risk)\n"
        "3. Which condition user selected (if any)\n"
        "4. Guidance / advice (including disclaimers)\n"
        "5. Doctor list if any (name, specialty, address/contact)\n"
        "Output in Markdown format."
    ),
    output_key="final_report"
)

# 1.6 (Optional) Input Decision Agent — parse yes/no or selection answer from user after condition analysis
decision_agent = LlmAgent(
    name="UserDecisionAgent",
    model=LLM_MODEL,
    description="Ask user which condition they believe matches, or if none — parse their answer.",
    instruction=(
        "You asked the user: 'Do you think any of the suggested conditions matches you? Reply with the number of condition or \"none\".' "
        "Now parse their reply and return a JSON object: {\"selected_condition_index\": <int or null>} "
    ),
    # Optionally you could define output_schema to enforce JSON output — but for prototype you can parse from text
    output_key="selected_condition_index"
)

### 2. Controllers / Workflow Agents ###

# 2.1 Symptom Collection Loop Controller — repeatedly collect symptoms until user says done
symptom_loop = LoopAgent(
    name="SymptomCollectionLoop",
    sub_agents=[symptom_collector,  # ask symptom
                # maybe a Custom Agent to process and store symptom into symptoms_list
               ],
    max_iterations=20  # arbitrary limit to avoid infinite loop
)

# 2.2 Main Orchestrator — full pipeline
root_agent = SequentialAgent(
    name="HealthAssistantPipeline",
    sub_agents=[
        symptom_loop,
        condition_analyzer,
        decision_agent,
        # Later: depending on user decision we may loop back or continue
        guidance_agent,
        doctor_finder,      # optional — run only if user wants doctor consult
        report_generator
    ],
    description="Orchestrates full health-assistant workflow: symptom collection → analysis → guidance → optional doctor lookup → report."
)

### 3. Entry Point ###

