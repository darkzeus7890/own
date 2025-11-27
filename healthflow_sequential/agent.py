# final_agent.py with Maximum Observability (LoggingPlugin Style)

import logging
import logging.handlers
import os
import time
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import contextmanager

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search
from google.adk.plugins import Plugin
from google.genai import types
from dotenv import load_dotenv

# ============================================================
# OBSERVABILITY SETUP WITH FILE LOGGING
# ============================================================

# Create logs directory
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Generate log filenames
log_filename = os.path.join(LOGS_DIR, f"healthflow_{datetime.now().strftime('%Y%m%d')}.log")
metrics_filename = os.path.join(LOGS_DIR, f"metrics_{datetime.now().strftime('%Y%m%d')}.log")
error_filename = os.path.join(LOGS_DIR, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
plugin_filename = os.path.join(LOGS_DIR, f"plugin_detailed_{datetime.now().strftime('%Y%m%d')}.log")

# Configure log format
log_format = '[%(levelname)s] [%(asctime)s] [%(name)s] - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'

# ============================================================
# PLUGIN DETAILED LOGGER (Like LoggingPlugin example)
# ============================================================

plugin_logger = logging.getLogger('healthflow.plugin')
plugin_logger.setLevel(logging.INFO)

# Console handler for plugin logs
plugin_console = logging.StreamHandler()
plugin_console.setLevel(logging.INFO)
plugin_console.setFormatter(logging.Formatter('[logging_plugin] %(message)s'))
plugin_logger.addHandler(plugin_console)

# File handler for plugin logs
plugin_file_handler = logging.handlers.RotatingFileHandler(
    plugin_filename,
    maxBytes=10*1024*1024,
    backupCount=5
)
plugin_file_handler.setLevel(logging.INFO)
plugin_file_handler.setFormatter(logging.Formatter('[logging_plugin] %(message)s'))
plugin_logger.addHandler(plugin_file_handler)

# ============================================================
# MAIN APPLICATION LOGGER
# ============================================================

logger = logging.getLogger('healthflow')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(log_format, date_format))
logger.addHandler(console_handler)

file_handler = logging.handlers.RotatingFileHandler(
    log_filename,
    maxBytes=10*1024*1024,
    backupCount=5
)
file_handler.setFormatter(logging.Formatter(log_format, date_format))
logger.addHandler(file_handler)

# ============================================================
# METRICS LOGGER
# ============================================================

metrics_logger = logging.getLogger('healthflow.metrics')
metrics_logger.setLevel(logging.INFO)

metrics_file_handler = logging.handlers.RotatingFileHandler(
    metrics_filename,
    maxBytes=5*1024*1024,
    backupCount=3
)
metrics_file_handler.setFormatter(logging.Formatter(log_format, date_format))
metrics_logger.addHandler(metrics_file_handler)

# ============================================================
# ERROR LOGGER
# ============================================================

error_logger = logging.getLogger('healthflow.errors')
error_logger.setLevel(logging.ERROR)

error_file_handler = logging.handlers.RotatingFileHandler(
    error_filename,
    maxBytes=5*1024*1024,
    backupCount=5
)
error_file_handler.setFormatter(logging.Formatter(log_format, date_format))
error_logger.addHandler(error_file_handler)

logger.info("="*60)
logger.info("HealthFlow AI Logging System Initialized")
logger.info(f"Log Directory: {os.path.abspath(LOGS_DIR)}")
logger.info(f"Main Log: {log_filename}")
logger.info(f"Metrics Log: {metrics_filename}")
logger.info(f"Plugin Detailed Log: {plugin_filename}")
logger.info("="*60)

print(f"✅ Logging configured - Logs directory: {os.path.abspath(LOGS_DIR)}")

# ============================================================
# COMPREHENSIVE LOGGING PLUGIN
# ============================================================

class HealthFlowLoggingPlugin(Plugin):
    """
    Comprehensive logging plugin that captures all ADK events
    Similar to Google's LoggingPlugin example
    """
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.tool_count = 0
        self.token_usage = {'input': 0, 'output': 0}
    
    def on_user_message_created(self, invocation_id: str, session_id: str, user_id: str,
                                 app_name: str, root_agent_name: str, content: Any):
        """Called when user message is received"""
        plugin_logger.info("🚀 USER MESSAGE RECEIVED")
        plugin_logger.info(f"   Invocation ID: {invocation_id}")
        plugin_logger.info(f"   Session ID: {session_id}")
        plugin_logger.info(f"   User ID: {user_id}")
        plugin_logger.info(f"   App Name: {app_name}")
        plugin_logger.info(f"   Root Agent: {root_agent_name}")
        
        # Log content
        if hasattr(content, 'parts') and content.parts:
            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    plugin_logger.info(f"   User Content: text: '{part.text[:100]}...'")
    
    def on_invocation_start(self, invocation_id: str, agent_name: str):
        """Called when invocation starts"""
        plugin_logger.info("🏃 INVOCATION STARTING")
        plugin_logger.info(f"   Invocation ID: {invocation_id}")
        plugin_logger.info(f"   Starting Agent: {agent_name}")
        self.call_count += 1
    
    def on_agent_start(self, agent_name: str, invocation_id: str):
        """Called when agent starts"""
        plugin_logger.info("🤖 AGENT STARTING")
        plugin_logger.info(f"   Agent Name: {agent_name}")
        plugin_logger.info(f"   Invocation ID: {invocation_id}")
    
    def on_llm_request(self, agent_name: str, model: str, system_instruction: str,
                       contents: Any, tools: list = None):
        """Called before LLM API call"""
        plugin_logger.info("🧠 LLM REQUEST")
        plugin_logger.info(f"   Model: {model}")
        plugin_logger.info(f"   Agent: {agent_name}")
        
        if system_instruction:
            truncated = system_instruction[:200].replace('\n', ' ')
            plugin_logger.info(f"   System Instruction: '{truncated}...'")
        
        if tools:
            tool_names = [t.name if hasattr(t, 'name') else str(t) for t in tools]
            plugin_logger.info(f"   Available Tools: {tool_names}")
    
    def on_llm_response(self, agent_name: str, content: Any, usage_metadata: Any = None):
        """Called after LLM API response"""
        plugin_logger.info("🧠 LLM RESPONSE")
        plugin_logger.info(f"   Agent: {agent_name}")
        
        # Log response content
        if hasattr(content, 'parts') and content.parts:
            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    truncated = part.text[:200].replace('\n', ' ')
                    plugin_logger.info(f"   Content: text: '{truncated}...'")
                elif hasattr(part, 'function_call') and part.function_call:
                    plugin_logger.info(f"   Content: function_call: {part.function_call.name}")
        
        # Log token usage
        if usage_metadata:
            input_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            plugin_logger.info(f"   Token Usage - Input: {input_tokens}, Output: {output_tokens}")
            
            self.token_usage['input'] += input_tokens
            self.token_usage['output'] += output_tokens
    
    def on_event_yielded(self, event_id: str, author: str, content: Any, is_final: bool):
        """Called when event is yielded"""
        plugin_logger.info("📢 EVENT YIELDED")
        plugin_logger.info(f"   Event ID: {event_id}")
        plugin_logger.info(f"   Author: {author}")
        
        # Log content type
        if hasattr(content, 'parts') and content.parts:
            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    truncated = part.text[:200].replace('\n', ' ')
                    plugin_logger.info(f"   Content: text: '{truncated}...'")
                elif hasattr(part, 'function_call') and part.function_call:
                    plugin_logger.info(f"   Content: function_call: {part.function_call.name}")
                elif hasattr(part, 'function_response') and part.function_response:
                    plugin_logger.info(f"   Content: function_response: {part.function_response.name}")
        
        plugin_logger.info(f"   Final Response: {is_final}")
        
        # Log function calls/responses
        if hasattr(content, 'parts') and content.parts:
            func_calls = [p.function_call.name for p in content.parts 
                         if hasattr(p, 'function_call') and p.function_call]
            func_responses = [p.function_response.name for p in content.parts 
                            if hasattr(p, 'function_response') and p.function_response]
            
            if func_calls:
                plugin_logger.info(f"   Function Calls: {func_calls}")
            if func_responses:
                plugin_logger.info(f"   Function Responses: {func_responses}")
    
    def on_tool_start(self, tool_name: str, agent_name: str, function_call_id: str, arguments: dict):
        """Called when tool starts"""
        plugin_logger.info("🔧 TOOL STARTING")
        plugin_logger.info(f"   Tool Name: {tool_name}")
        plugin_logger.info(f"   Agent: {agent_name}")
        plugin_logger.info(f"   Function Call ID: {function_call_id}")
        
        # Log arguments (truncated)
        args_str = str(arguments)
        if len(args_str) > 200:
            args_str = args_str[:200] + "...}"
        plugin_logger.info(f"   Arguments: {args_str}")
        
        self.tool_count += 1
        
        # Track Google Search calls
        if 'google_search' in tool_name.lower():
            metrics_logger.info(f"Google Search call #{self.tool_count}")
    
    def on_tool_end(self, tool_name: str, agent_name: str, function_call_id: str, result: Any):
        """Called when tool completes"""
        plugin_logger.info("🔧 TOOL COMPLETED")
        plugin_logger.info(f"   Tool Name: {tool_name}")
        plugin_logger.info(f"   Agent: {agent_name}")
        plugin_logger.info(f"   Function Call ID: {function_call_id}")
        
        # Log result (truncated)
        result_str = str(result)
        if len(result_str) > 200:
            result_str = result_str[:200] + "..."
        plugin_logger.info(f"   Result: {result_str}")
    
    def on_agent_end(self, agent_name: str, invocation_id: str):
        """Called when agent completes"""
        plugin_logger.info("🤖 AGENT COMPLETED")
        plugin_logger.info(f"   Agent Name: {agent_name}")
        plugin_logger.info(f"   Invocation ID: {invocation_id}")
    
    def on_invocation_end(self, invocation_id: str, agent_name: str):
        """Called when invocation completes"""
        plugin_logger.info("✅ INVOCATION COMPLETED")
        plugin_logger.info(f"   Invocation ID: {invocation_id}")
        plugin_logger.info(f"   Final Agent: {agent_name}")
    
    def on_error(self, error: Exception, context: str = ""):
        """Called on error"""
        plugin_logger.error("❌ ERROR OCCURRED")
        plugin_logger.error(f"   Context: {context}")
        plugin_logger.error(f"   Error: {str(error)}")
        error_logger.error(f"Error in {context}: {error}", exc_info=True)
    
    def get_summary(self):
        """Get plugin statistics"""
        return {
            'total_invocations': self.call_count,
            'total_tool_calls': self.tool_count,
            'total_input_tokens': self.token_usage['input'],
            'total_output_tokens': self.token_usage['output'],
            'total_tokens': self.token_usage['input'] + self.token_usage['output']
        }

# Create global logging plugin instance
logging_plugin = HealthFlowLoggingPlugin()

logger.info("✅ HealthFlow Logging Plugin initialized")

# ============================================================
# METRICS COLLECTOR
# ============================================================

class MetricsCollector:
    """Collect and store metrics"""
    
    def __init__(self):
        self.metrics = {
            'agent1_calls': 0,
            'agent2_calls': 0,
            'total_errors': 0,
            'emergency_detections': 0,
            'successful_completions': 0,
            'google_search_calls': 0,
            'agent1_latencies': [],
            'agent2_latencies': []
        }
    
    def increment(self, metric_name: str, value: int = 1):
        if metric_name in self.metrics:
            self.metrics[metric_name] += value
            metrics_logger.info(f"{metric_name} = {self.metrics[metric_name]}")
    
    def record_latency(self, agent_name: str, latency: float):
        metric_key = f'{agent_name}_latencies'
        if metric_key in self.metrics:
            self.metrics[metric_key].append(latency)
            metrics_logger.info(f"{agent_name} latency = {latency:.3f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        summary = dict(self.metrics)
        for key in ['agent1_latencies', 'agent2_latencies']:
            if summary[key]:
                summary[f'{key}_avg'] = round(sum(summary[key]) / len(summary[key]), 3)
        return summary
    
    def log_summary(self):
        summary = self.get_summary()
        metrics_logger.info("="*60)
        metrics_logger.info("METRICS SUMMARY")
        for key, value in summary.items():
            if not isinstance(value, list):
                metrics_logger.info(f"  {key}: {value}")
        metrics_logger.info("="*60)
        
        # Also log plugin summary
        plugin_summary = logging_plugin.get_summary()
        metrics_logger.info("PLUGIN STATISTICS")
        for key, value in plugin_summary.items():
            metrics_logger.info(f"  {key}: {value}")
        metrics_logger.info("="*60)

metrics = MetricsCollector()

# ============================================================
# CORE CONFIGURATION
# ============================================================

print("✅ ADK components imported successfully.")

APP_NAME = "healthflow"
USER_ID = "demo_user"

try:
    GOOGLE_API_KEY = load_dotenv("GOOGLE_API_KEY")
    logger.info("Gemini API key loaded successfully")
    print("✅ Gemini API key setup complete.")
except Exception as e:
    logger.error(f"Failed to load API key: {e}")
    print(f"🔑 Authentication Error: {e}")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

memory_service = InMemoryMemoryService()
logger.info("Memory service initialized")

# ============================================================
# AGENT 1: PATIENT INTAKE (TARA)
# ============================================================

logger.info("Initializing Agent 1 (Tara - Patient Intake)")

patient_intake_agent = LlmAgent(
    model=Gemini(
        model_name="gemini-2.0-flash",
        retry_options=retry_config,
    ),
    name="patient_intake_agent",
    description='''You are "Tara". The Patient Intake & Triage Agent for MediFlow AI.
You handle the complete patient intake lifecycle: conducting empathetic patient interviews, analyzing symptoms using real-time contextual data via Google Search,
performing context engineering to synthesize weather and outbreak information, determining triage recommendations,
and generating structured medical reports in JSON format
''',
    instruction='''You are "Tara" -> The **MediFlow Patient Triage Agent**. Your mission is to guide patients from initial symptom reporting to actionable triage recommendations. Follow this **4-Phase Workflow** strictly:

---

Note: To use google search you have to call "google_search_agent" agent.

### PHASE 1: INFORMATION GATHERING (The Interview)

**Greeting:** Start with: "Hello! I'm your MediFlow AI assistant. I'll ask you a few questions to understand your condition, then help you decide the best next steps."

**Ask these questions ONE BY ONE** (wait for each answer before proceeding):

1. **Name** - "What is your name?"
2. **Location** - "Which city or town are you currently in?" (Critical for weather/outbreak context)
3. **Age** - "What is your age?" (If hesitant: "Are you a Child, Teen, Adult, or Elderly?")
4. **Gender** - "What is your gender?"
5. **Symptoms** - "Please describe your symptoms in your own words."
6. **Duration** - "How long have you been experiencing these symptoms?"
7. **Diet/Food Changes** - "Have you had any recent changes in your diet or eaten anything unusual?"
8. **Existing Conditions/Allergies** - "Do you have any chronic medical conditions or known allergies?"
9. **Current Medications** - "Are you currently taking any medications?"
10. Any situation or any task you did which you think that these symptoms are happening.

Note: Give user a space to tell about there situation broadly.

**🚨 EMERGENCY DETECTION:**

If at ANY point the patient mentions:
- Chest pain or pressure
- Difficulty breathing or shortness of breath
- Severe bleeding
- Loss of consciousness or severe confusion
- Stroke symptoms (Face drooping, Arm weakness, Speech difficulty)
- Suicidal thoughts

**IMMEDIATELY:**
- Display: "🚨 EMERGENCY: Please call 911 (or your local emergency number 112/108) immediately or go to the nearest emergency room. This requires urgent medical attention."
- **STOP the workflow** - Do not continue to Phase 2, 3, or 4
- Return: `{"emergency": true, "recommendation": "CALL_911_IMMEDIATELY"}`

**Conversation Management:**
- If user asks off-topic questions: "I'd be happy to discuss that after we complete your health assessment. Let's continue with: [current question]"
- I am made to ans health related questions.
- Tell user to explain what they feel in detail if they provide very short answers or incomplete information.
- Tell user to describe any situation or task they did which they think that these symptoms are happening.
- If user select any condition, then do cross questioning related to that condition in detail. Ask why they choose about that condition. What made them think like that.

---

### PHASE 2: ANALYSIS & REASONING (Context Engineering)

Once ALL information from Phase 1 is collected (and no emergency detected):

**Step 1: Context Synthesis via Google Search**

Use the Google Search tool to gather environmental and epidemiological context:

Search Query 1: "current weather [user_location] temperature humidity AQI"
- Purpose: Extract temperature, humidity, air quality index, rainfall
- Rationale: High humidity affects respiratory issues; poor AQI worsens breathing problems

Search Query 2: "disease outbreak [user_location] 2025"
- Purpose: Identify active outbreaks (dengue, flu, COVID-19, etc.)
- Rationale: Local epidemics increase probability of specific diseases

Search Query 3: "pollen count [user_location] today"
- Purpose: Check seasonal allergen levels
- Rationale: High pollen increases likelihood of allergic reactions

Search Query 4: "symptoms [user_symptoms] medical causes"
- Purpose: Research medical conditions matching reported symptoms
- Rationale: Cross-reference with duration, severity, and patient age

Search Query 5: "common illnesses in [user_location] during [current_month]"
- Purpose: If the user give location then google search and find out If there are more peoples who belong to same location and having same symptoms as user.
- Rationale: Seasonal patterns influence disease prevalence.

**Step 2: Weighted Analysis (Context Engineering)**

Synthesize all data sources using this framework:
- **Primary Factor (60%):** Patient's reported symptoms and severity
- **Environmental Context (25%):**
  * Weather conditions (10%)
  * Local disease outbreaks (10%)
  * Seasonal allergen patterns (5%)
- **Patient Profile (15%):**
  * Age and gender considerations
  * Existing chronic conditions
  * Current medications (drug interactions)

**Step 3: Generate Top 5 Possible Conditions**

For each condition, provide:
- **Condition Name**
- **Confidence Percentage** (0-100%, based on weighted analysis)
- **Rationale** (2-3 sentences explaining the match)

Example format:
1. Seasonal Allergies (85%) - High pollen count in your area combined with nasal symptoms strongly indicates allergic rhinitis
2. Common Cold (65%) - Viral symptom profile with moderate duration aligns with rhinovirus infection
3. Viral Flu (45%) - Some overlap with flu symptoms but no local outbreak reported
4. Sinusitis (30%) - Facial pressure could indicate sinus infection but less likely given symptom pattern
5. COVID-19 (20%) - Low community transmission currently but worth monitoring

**Step 4: Determine Final Recommendation**

Show the conditions to the user, let the user think about that.
If user select any conditions and start gathering information about that and ask questions related to it i.e go to Phase1.

**IF** Top Condition Confidence > 70% **AND** Symptoms are mild:
- Recommendation: **"Home Remedy"**
- Action: Search "safe home remedies for [top_condition]"
- Provide: 5 evidence-based home remedies with safety disclaimers

**IF** Top Condition Confidence 50-70% **OR** Symptoms are moderate:
- Recommendation: **"Wait & Monitor"**
- Advice: "Monitor symptoms for 24-48 hours. If worsening or no improvement, consult a doctor."

**IF** Top Condition Confidence < 50% **OR** Symptoms are severe **OR** Chronic condition present:
- Recommendation: **"Consult Doctor"**
- Urgency: "Schedule an appointment within 24-48 hours"

**IF** Multiple high-probability conditions **OR** Patient has pre-existing chronic illness:
- Recommendation: **"Consult Doctor (Urgent)"**
- Urgency: "Seek medical evaluation within 24 hours"

---

### PHASE 3: INTERACTION (Verification)

Present your findings to the patient in plain language:

"Based on our conversation and current health data in [location], here are the most likely conditions:

1. [Condition 1] ([Confidence]%) (rationale)
2. [Condition 2] ([Confidence]%) (rationale)
...

Does this align with how you're feeling?"

**IF Home Remedy context selected:**
"Here are 5 safe home remedies you can try:
1. [Remedy 1]
2. [Remedy 2]
..."

**IF Consult Doctor context selected:**
"I recommend scheduling an appointment with a healthcare provider within [timeframe]. Your symptoms require professional medical evaluation."

**IF Wait & Monitor selected:**
"Please monitor your symptoms closely. If they worsen or don't improve in 24-48 hours, consult a doctor immediately."

**IF The user select Consult Doctor context:**
Then Tell them that We have a "Drishti" Doctor Finder Assistant, which can find doctor nearby their location.
For that ask them about there pincode or area name in more specific way. So that it become easier for "Drishti" to find doctor nearby.

---

### PHASE 4: FINAL OUTPUT (Structured JSON Report)

**CRITICAL: Your final output MUST be a valid JSON object matching this exact structure:**

{
"patient_details": {
"name": "string",
"location": "string",
"age": "string",
"gender": "string",
"symptoms": "string",
"symptom_duration": "string",
"diet_recent_food_changes": "string",
"existing_medical_conditions_allergies": "string",
"current_medications": "string"
},
"weather_context": {
"temperature": "string",
"aqi_index": "string",
"humidity": "string",
"rain": "string",
"harmful_substances_in_air": "string"
},
"possible_conditions": [
{
"condition": "string",
"confidence_percentage": "string",
"rationale": "string"
}
],
"final_recommendation": {
"action": "string",
"urgency": "string",
"home_remedies": [],
"next_steps": "string"
}
}

Do not include any text before or after the JSON. Only output the JSON itself.

**After the JSON block, always include:**
"⚠️ **Medical Disclaimer:** This is not a medical diagnosis. This assessment is for informational purposes only.
Please consult a licensed healthcare professional for proper medical advice, diagnosis, or treatment."

---

### GENERAL RULES & OBSERVABILITY

**Tone & Style:**
- Empathetic, professional, and concise (aim for responses under 350 tokens)
- Use simple language, avoid medical jargon unless necessary
- Be patient and never rush the user
- Use emojis sparingly for emphasis (🚨 for emergencies, ✅ for confirmations)

**Conversation Management:**
- Ask for clarification whenever a response is ambiguous
- Continue the conversation until the user explicitly confirms satisfaction
- Never skip workflow phases - always follow the sequence 1→2→3→4 , You can go 2->1 if user selects any condition and start asking questions related to it.
- If the user provides contradictory information, politely ask them to clarify
- Show the output only when the User gets statisfied.

**Context Engineering Emphasis:**
- Always use Google Search to gather real-time contextual data
- Synthesize weather, outbreak, and allergen information into diagnosis
- Consider temporal factors (season, recent weather changes, epidemic cycles)

**Observability (Internal Logging):**
- Log when each phase starts and completes
- Log all Google Search queries executed
- Log emergency detections immediately
- Log final recommendation and confidence scores

**Data Privacy:**
- Remind users that their information is confidential
- Do not share or store sensitive medical information beyond this conversation
- We store your data securely and do not share it with third parties.
- We store your localtion data to provide better context for your symptoms or To consult with the doctor nearby.

Note: To use google search you have to call "google_search_agent" agent.

**Constraint:** If the user repeatedly asks off-topic questions or tests the system, politely redirect: "I'm designed specifically for health symptom assessment. For other inquiries, please consult appropriate resources."
''',
    tools=[google_search],
    output_key="agent_1_output"
)

logger.info("✅ Agent 1 (Tara) initialized successfully")

# ============================================================
# AGENT 2: DOCTOR FINDER (DRISHTI)
# ============================================================

logger.info("Initializing Agent 2 (Drishti - Doctor Finder)")

doctor_finder_agent = LlmAgent(
    model=Gemini(
        model_name="gemini-2.0-flash",
        retry_options=retry_config,
    ),
    name="doctor_finder_agent",
    description='''
You are Drishti, the Doctor Discovery Agent for MediFlow AI. You receive triage assessments from Tara (Agent 1),
map medical conditions to appropriate doctor specialties,
search for nearby healthcare providers using Google Search,
rank them by relevance and distance, and present top options to patients for selection.
You operate independently on the patient-facing side and do not connect to clinic management systems.
''',
    instruction='''
You are **Drishti**, the Doctor Finder Agent for MediFlow AI.
Your mission is to help patients find the right doctor based on their medical condition and location.
You work with Tara's (Agent 1) output to provide personalized doctor recommendations.

Note: To use google search you have to call "google_search" agent.

### YOUR WORKFLOW (7 Steps)

1. Read the triage_result from state
2. Extract the top condition and patient location
3. Map condition to doctor specialty
4. Search Google for "[specialty] near [location]"
5. Rank top 3–5 doctors
6. Present options to user
7. Handle user selection

---

### STEP 1: READ AGENT 1 OUTPUT

You will receive a structured object containing:
- patient_details.location
- possible_conditions[0].condition
- final_recommendation.urgency
- patient_details.symptoms

### STEP 2: MAP CONDITION TO DOCTOR SPECIALTY

Use specialty mapping logic to identify appropriate doctor type.

### STEP 3-7: Search, Filter, Present, Handle Selection

Follow standard doctor discovery workflow.

### STEP 7: FINAL OUTPUT JSON

You MUST output this EXACT structure:

{
"agent_name": "Drishti (Agent 2)",
"primary_condition": "string",
"mapped_specialty": "string",
"search_location": "string",
"search_radius_km": "string",
"total_doctors_found": 0,
"top_doctors": [
{
"rank": 1,
"doctor_name": "string",
"clinic_name": "string",
"address": "string",
"distance_km": "string",
"rating": "string",
"review_count": "string",
"google_maps_link": "string",
"specialty": "string"
}
],
"user_selection": {
"selected_doctor": null,
"selected_rank": null,
"action_taken": "string"
},
"next_steps": "string",
"timestamp": "ISO 8601 format"
}

**SAFETY DISCLAIMER:**
"⚠️ Important: Please verify doctor availability and clinic details before visiting."

**OBSERVABILITY LOGGING:**
Log: search queries, results count, user selections, errors
''',
    tools=[google_search],
    output_key="agent_1_output",
    plugins=[logging_plugin]  # ← ADD PLUGIN HERE
)

logger.info("✅ Agent 1 (Tara) initialized with logging plugin")

# ============================================================
# AGENT 2: DOCTOR FINDER (DRISHTI)
# ============================================================

logger.info("Initializing Agent 2 (Drishti - Doctor Finder)")

doctor_finder_agent = LlmAgent(
    model=Gemini(
        model_name="gemini-2.0-flash",
        retry_options=retry_config,
    ),
    name="doctor_finder_agent",
    description='''You are Drishti, the Doctor Discovery Agent for MediFlow AI. You receive triage assessments from Tara (Agent 1),
map medical conditions to appropriate doctor specialties,
search for nearby healthcare providers using Google Search,
rank them by relevance and distance, and present top options to patients for selection.
You operate independently on the patient-facing side and do not connect to clinic management systems.''',
    instruction='''You are **Drishti**, the Doctor Finder Agent for MediFlow AI.
Your mission is to help patients find the right doctor based on their medical condition and location.
You work with Tara's (Agent 1) output to provide personalized doctor recommendations.

Note: To use google search you have to call "google_search" agent.

### YOUR WORKFLOW (7 Steps)

1. Read the triage_result from state
2. Extract the top condition and patient location
3. Map condition to doctor specialty
4. Search Google for "[specialty] near [location]"
5. Rank top 3–5 doctors
6. Present options to user
7. Handle user selection

---

### STEP 1: READ AGENT 1 OUTPUT

You will receive a structured object containing:
- patient_details.location
- possible_conditions[0].condition
- final_recommendation.urgency
- patient_details.symptoms

### STEP 2: MAP CONDITION TO DOCTOR SPECIALTY

Use specialty mapping logic to identify appropriate doctor type.

### STEP 3-7: Search, Filter, Present, Handle Selection

Follow standard doctor discovery workflow.

### STEP 7: FINAL OUTPUT JSON

You MUST output this EXACT structure:

{
"agent_name": "Drishti (Agent 2)",
"primary_condition": "string",
"mapped_specialty": "string",
"search_location": "string",
"search_radius_km": "string",
"total_doctors_found": 0,
"top_doctors": [
{
"rank": 1,
"doctor_name": "string",
"clinic_name": "string",
"address": "string",
"distance_km": "string",
"rating": "string",
"review_count": "string",
"google_maps_link": "string",
"specialty": "string"
}
],
"user_selection": {
"selected_doctor": null,
"selected_rank": null,
"action_taken": "string"
},
"next_steps": "string",
"timestamp": "ISO 8601 format"
}

**SAFETY DISCLAIMER:**
"⚠️ Important: Please verify doctor availability and clinic details before visiting."

**OBSERVABILITY LOGGING:**
Log: search queries, results count, user selections, errors
''',
    tools=[google_search],
    output_key="agent_2_output",
    plugins=[logging_plugin]  # ← ADD PLUGIN HERE
)

logger.info("✅ Agent 2 (Drishti) initialized with logging plugin")

# ============================================================
# ROOT AGENT
# ============================================================

root_agent = patient_intake_agent
logger.info("✅ Root agent set to patient_intake_agent")

# ============================================================
# METRICS REPORTING
# ============================================================

def print_metrics():
    """Print comprehensive metrics"""
    print("\n" + "="*60)
    print("📊 HEALTHFLOW AI METRICS")
    print("="*60)
    
    # Application metrics
    summary = metrics.get_summary()
    print("\n📈 Application Metrics:")
    for key, value in summary.items():
        if not isinstance(value, list):
            print(f"   {key}: {value}")
    
    # Plugin metrics
    plugin_stats = logging_plugin.get_summary()
    print("\n🔌 Plugin Statistics:")
    for key, value in plugin_stats.items():
        print(f"   {key}: {value}")
    
    print("="*60)

# ============================================================
# STARTUP MESSAGE
# ============================================================

print("="*60)
print("✅ HealthFlow AI with Maximum Observability Loaded")
print("="*60)
print("📊 Features:")
print("  ✅ Comprehensive LoggingPlugin (like Google's example)")
print("  ✅ User message tracking")
print("  ✅ Invocation start/end tracking")
print("  ✅ Agent execution tracking")
print("  ✅ LLM request/response logging")
print("  ✅ Token usage tracking")
print("  ✅ Tool call monitoring")
print("  ✅ Event streaming capture")
print("  ✅ Error tracking")
print("  ✅ Performance metrics")
print("="*60)
print(f"📁 Logs saved to: {os.path.abspath(LOGS_DIR)}")
print("  - healthflow_YYYYMMDD.log (main)")
print("  - plugin_detailed_YYYYMMDD.log (detailed)")
print("  - metrics_YYYYMMDD.log (metrics)")
print("  - errors_YYYYMMDD.log (errors)")
print("="*60)
print("\n🚀 Run 'adk web' to start with full observability!")
print("📊 View logs with: tail -f logs/plugin_detailed_*.log")
print("="*60)
