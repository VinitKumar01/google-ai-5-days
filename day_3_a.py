import os
import asyncio
from google.adk.agents import Agent, LlmAgent
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.models.google_llm import Gemini
from google.adk.sessions import DatabaseSessionService, InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

APP_NAME = "default"
USER_ID = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


# UNIVERSAL RUNNER HELPER
async def run_session(runner: Runner, user_queries, session_name: str):
    print(f"\n### SESSION: {session_name}")

    session_svc = runner.session_service

    try:
        session = await session_svc.create_session(
            app_name=runner.app_name,
            user_id=USER_ID,
            session_id=session_name,
        )
    except:
        session = await session_svc.get_session(
            app_name=runner.app_name,
            user_id=USER_ID,
            session_id=session_name,
        )

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query in user_queries:
        print(f"\nUser > {query}")
        query_obj = types.Content(role="user", parts=[types.Part(text=query)])
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session.id, new_message=query_obj
        ):
            if event.content and event.content.parts:
                text = event.content.parts[0].text
                if text and text != "None":
                    print(f"{MODEL_NAME} > {text}")


# BASE STATEFUL AGENT (IN-MEMORY)
root_agent = Agent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="text_chat_bot",
    description="Simple stateful chatbot",
)

session_service = InMemorySessionService()
memory_app = App(name=APP_NAME, root_agent=root_agent)
runner = Runner(app=memory_app, session_service=session_service)

# PERSISTENT AGENT (postgres)
chatbot_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="text_chat_bot",
    description="Chatbot with persistent sessions",
)

db_url = "postgresql://postgres:mypassword@localhost:5432/adk-test"
persistent_session_service = DatabaseSessionService(db_url=db_url)
chatbot_app = App(name=APP_NAME, root_agent=chatbot_agent)
persistent_runner = Runner(app=chatbot_app, session_service=persistent_session_service)

# EVENT COMPACTION AGENT
research_app_compacting = App(
    name=APP_NAME,
    root_agent=chatbot_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,
        overlap_size=1,
    ),
)

compact_runner = Runner(
    app=research_app_compacting, session_service=persistent_session_service
)


# SESSION STATE TOOLS
def save_userinfo(tool_context: ToolContext, user_name: str, country: str):
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country
    return {"status": "saved"}


def retrieve_userinfo(tool_context: ToolContext):
    return {
        "status": "success",
        "user_name": tool_context.state.get("user:name", "Unknown"),
        "country": tool_context.state.get("user:country", "Unknown"),
    }


state_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="text_chat_bot",
    description="Agent with session-state tools",
    tools=[save_userinfo, retrieve_userinfo],
)

state_service = InMemorySessionService()
state_app = App(name=APP_NAME, root_agent=state_agent)
state_runner = Runner(app=state_app, session_service=state_service)


# MAIN EXECUTION
async def main():
    print("\n===== DEMO: In-memory stateful agent =====")
    await run_session(
        runner,
        [
            "Hi, I am Sam! What is the capital of the United States?",
            "What is my name?",
        ],
        "memory-demo",
    )

    print("\n===== DEMO: Persistent agent (postgres) =====")
    await run_session(
        persistent_runner,
        [
            "Hi, I am Sam! What is the capital of the United States?",
            "What is my name?",
        ],
        "db-session-01",
    )

    print("\n===== DEMO: Context compaction =====")
    await run_session(
        compact_runner, "What is the latest news about AI?", "compact-demo"
    )
    await run_session(
        compact_runner, "Tell me more about drug discovery.", "compact-demo"
    )
    await run_session(
        compact_runner, "Explain the second part in detail.", "compact-demo"
    )
    await run_session(
        compact_runner, "Who are the main companies involved?", "compact-demo"
    )

    print("\nChecking compaction summary…")
    final_sess = await persistent_session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id="compact-demo",
    )
    for e in final_sess.events:
        if e.actions and e.actions.compaction:
            print("Compaction summary found.")
            break

    print("\n===== DEMO: Session state tools =====")
    await run_session(
        state_runner,
        [
            "Hi, what is my name?",
            "My name is Sam. I'm from Poland.",
            "What is my name? And where am I from?",
        ],
        "state-tools-demo",
    )


if __name__ == "__main__":
    asyncio.run(main())

