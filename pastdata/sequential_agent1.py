import os
import json
from datetime import datetime
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import google_search
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = load_dotenv("GOOGLE_API_KEY")


from google.adk.agents import SequentialAgent
from google.adk.sessions import InMemorySessionService
from patient_intake_agent import root_agent as patient_intake_agent
from doctor_finder_agent import root_agent as doctor_finder_agent

# ============================================================
# CREATE SEQUENTIAL AGENT
# ============================================================

healthflow_sequential = SequentialAgent(
    name="healthflow_triage_to_doctor",
    sub_agents=[patient_intake_agent, doctor_finder_agent],
    description="Complete HealthFlow system: Patient triage with Tara → Doctor discovery with Drishti"
)


root_agent = healthflow_sequential
