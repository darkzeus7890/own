import os
import json
from datetime import datetime
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import google_search
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents import SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field
from typing import List, Optional

from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = load_dotenv("GOOGLE_API_KEY")

from pydantic import BaseModel, Field

class UserOutput(BaseModel):
    usr_otp: str = Field(description="User output will store here.")


google_search_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="google_search_agent",
    description="An agent that performs Google searches to find relevant information.",
    instruction="""
You are an agent that performs Google searches to find relevant information based on user queries.
When given a query, you will use the Google Search tool to retrieve the most relevant results.
Make sure to format your output as a JSON object with a single key 'search_results' containing the search results.
    """,
    tools=[google_search],
    output_key="search_results"
    )


# Example: Defining the basic identity
teacher_agent = SequentialAgent(
    name="teacher_agent",
    description="User will ask you about the latest cricket match scores.You will provide them with the most recent scores and updates from ongoing or recently concluded cricket matches.",  # Enforce JSON output
    sub_agents=[google_search_agent]
)

root_agent = teacher_agent