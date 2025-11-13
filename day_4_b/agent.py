# command to run tests

# cd /home/vinit/Webdev/google-ai-5-days
# adk eval day_4_b \
#     day_4_b/integration.evalset.json \
#     --config_file_path=day_4_b/test_config.json \
#     --print_detailed_results

import asyncio
import os
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini

from google.adk.runners import InMemoryRunner
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


def set_device_status(location: str, device_id: str, status: str) -> dict:
    """Sets the status of a smart home device.

    Args:
        location: The room where the device is located.
        device_id: The unique identifier for the device.
        status: The desired status, either 'ON' or 'OFF'.

    Returns:
        A dictionary confirming the action.
    """
    print(f"Tool Call: Setting {device_id} in {location} to {status}")
    return {
        "success": True,
        "message": f"Successfully set the {device_id} in {location} to {status.lower()}.",
    }


root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="home_automation_agent",
    description="An agent to control smart devices in a home.",
    instruction="""You are a home automation assistant. You control ALL smart devices in the house.
    
    You have access to lights, security systems, ovens, fireplaces, and any other device the user mentions.
    Always try to be helpful and control whatever device the user asks for.
    
    When users ask about device capabilities, tell them about all the amazing features you can control.""",
    tools=[set_device_status],
)

# runner = InMemoryRunner(agent=root_agent, app_name="home_automation_agent")


# async def main():
#     response = await runner.run_debug("Switch on the main light in the kitchen.")
#     print(response)
#
#
# asyncio.run(main())


class agent:
    root_agent = root_agent
