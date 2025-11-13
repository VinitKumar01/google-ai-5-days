import os
import asyncio
import base64
import uuid

from google.adk.apps.app import App, ResumabilityConfig
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner, Runner
from google.genai import types
from mcp import StdioServerParameters
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

mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-everything"],
        ),
        timeout=60,
    )
)

BULK_IMAGE_THRESHOLD = 1


def generate_images(prompt: str, num_images: int, tool_context: ToolContext) -> dict:
    """
    Requests image generation. Requires human approval if more than BULK_IMAGE_THRESHOLD images.

    Args:
        prompt: Description of images to generate.
        num_images: Number of images requested.

    Returns:
        Dict with approval status and parameters.
    """

    if num_images <= BULK_IMAGE_THRESHOLD:
        return {
            "status": "approved",
            "prompt": prompt,
            "num_images": num_images,
            "message": f"Auto-approved generation: {num_images} image(s) for '{prompt}'",
        }

    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Bulk image generation requested: {num_images} images for '{prompt}'. Approve?",
            payload={"prompt": prompt, "num_images": num_images},
        )
        return {
            "status": "pending",
            "message": "Bulk image generation requires approval",
        }

    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "prompt": prompt,
            "num_images": num_images,
            "message": f"Human-approved: {num_images} images for '{prompt}'",
        }
    else:
        return {
            "status": "rejected",
            "message": f"Request for {num_images} images of '{prompt}' was rejected.",
        }


image_agent = LlmAgent(
    name="bulk_image_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are an assistant that generates images using the MCP Image Server.
    
    Workflow:
     1. Use the generate_images tool to determine if approval is required.
     2. If approved, call the MCP server to actually generate the images.
     3. Decode and save images locally with clear filenames.
     4. Keep responses short but informative.
    """,
    tools=[FunctionTool(func=generate_images), mcp_image_server],
)

image_app = App(
    name="bulk_image_generator",
    root_agent=image_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

session_service = InMemorySessionService()
runner = Runner(app=image_app, session_service=session_service)


def check_for_approval(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None


def print_agent_response(events):
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")


def create_approval_response(approval_info, approved):
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(
        role="user", parts=[types.Part(function_response=confirmation_response)]
    )


async def generate_and_save_images(prompt: str, num_images: int):
    """Actually runs image generation through the MCP Tool and saves files."""
    runner = InMemoryRunner(agent=image_agent)

    # Generate the prompt for the MCP Image Server
    query = f"Generate {num_images} image(s) for: {prompt}"
    response = await runner.run_debug(query, verbose=False)

    output_dir = os.path.join(os.getcwd(), "generated_images")
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for event in response:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    for item in part.function_response.response.get("content", []):
                        if item.get("type") == "image":
                            count += 1
                            img_data = base64.b64decode(item["data"])
                            filename = os.path.join(
                                output_dir,
                                f"{prompt[:30].replace(' ', '_')}_{count}.png",
                            )
                            with open(filename, "wb") as f:
                                f.write(img_data)
                            print(f"✅ Saved image: {filename}")
    if count == 0:
        print("⚠️ No images returned from MCP server.")


async def run_image_workflow(prompt: str, num_images: int, auto_approve: bool = True):
    print(f"\n{'=' * 60}")
    print(f"User > Generate {num_images} image(s) for: '{prompt}'\n")

    session_id = f"img_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name="bulk_image_generator", user_id="test_user", session_id=session_id
    )

    query_content = types.Content(
        role="user",
        parts=[types.Part(text=f"Generate {num_images} image(s) for '{prompt}'")],
    )
    events = []

    async for event in runner.run_async(
        user_id="test_user", session_id=session_id, new_message=query_content
    ):
        events.append(event)

    approval_info = check_for_approval(events)

    if approval_info:
        print("⏸️  Pausing for approval...")
        print(f"🤔 Human Decision: {'APPROVE ✅' if auto_approve else 'REJECT ❌'}\n")

        async for event in runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(approval_info, auto_approve),
            invocation_id=approval_info["invocation_id"],
        ):
            print_agent_response([event])

        if auto_approve:
            print("\n🧠 Approval granted — generating images...")
            await generate_and_save_images(prompt, num_images)
    else:
        print_agent_response(events)
        print("\n🚀 Auto-approved — generating image(s)...")
        await generate_and_save_images(prompt, num_images)

    print(f"{'=' * 60}\n")


async def main():
    print("\n--- Running Bulk Image Generation Workflow ---\n")

    await run_image_workflow(
        "a futuristic city skyline at sunset", num_images=1, auto_approve=True
    )
    await run_image_workflow(
        "a medieval fantasy castle with dragons", num_images=3, auto_approve=True
    )
    await run_image_workflow(
        "a cyberpunk neon street market", num_images=6, auto_approve=True
    )

    await mcp_image_server.close()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
