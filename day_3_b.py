import os
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory, preload_memory
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

# Setup

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

APP_NAME = "MemoryDemoApp"
USER_ID = "demo_user"
MODEL_NAME = "gemini-2.5-flash-lite"

session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Helper function for running a session


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


# Agent with load_memory tool (Reactive retrieval)

reactive_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="MemoryDemoAgent",
    instruction="Answer simply and use load_memory when needed.",
    tools=[load_memory],
)

reactive_runner = Runner(
    agent=reactive_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)


# Ingest a session manually
async def demo_manual_ingest():
    await run_session(
        reactive_runner,
        "My favorite color is blue-green.",
        "color_session_01",
    )

    session_svc = reactive_runner.session_service
    session = await session_svc.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="color_session_01"
    )
    if session is not None:
        await memory_service.add_session_to_memory(session)


# Automatic Memory Saving via callback


async def auto_save_to_memory(callback_context):
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )


auto_agent = LlmAgent(
    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
    name="AutoMemoryAgent",
    instruction="Answer questions.",
    tools=[preload_memory],
    after_agent_callback=auto_save_to_memory,
)

auto_runner = Runner(
    agent=auto_agent,
    app_name=APP_NAME,
    session_service=session_service,
    memory_service=memory_service,
)


# Demonstration of automated memory usage
async def demo_auto_memory():
    await run_session(
        auto_runner,
        "I gifted my nephew a new toy for his 1st birthday.",
        "auto_session_1",
    )

    await run_session(
        auto_runner,
        "What did I gift my nephew?",
        "auto_session_2",
    )


# Memory search example
async def demo_search():
    result = await memory_service.search_memory(
        app_name=APP_NAME, user_id=USER_ID, query="favorite color"
    )
    for m in result.memories:
        if m.content and m.content.parts:
            print(m.content.parts[0].text)


# Entrypoint for testing all examples
async def main():
    await demo_manual_ingest()
    await run_session(reactive_runner, "What is my favorite color?", "test_color_query")
    await demo_auto_memory()
    await demo_search()


asyncio.run(main())
