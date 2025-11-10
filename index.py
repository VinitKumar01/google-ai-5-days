import os
import asyncio
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

# MULTI-AGENT

research_agent = Agent(
    name="ResearchAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Use the google_search tool to find 2–3 useful facts on the topic.
    Present findings with citations.""",
    tools=[google_search],
    output_key="research_findings",
)

summarizer_agent = Agent(
    name="SummarizerAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Summarize the research findings into 3–5 bullet points.
    Research: {research_findings}""",
    output_key="final_summary",
)

research_root = Agent(
    name="ResearchCoordinator",
    model="gemini-2.5-flash-lite",
    instruction="""
    1. Call ResearchAgent first.
    2. Then call SummarizerAgent.
    3. Respond with the final_summary.
    """,
    tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
)

# SEQUENTIAL AGENTS

outline_agent = Agent(
    name="OutlineAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Create a blog outline:
    - Headline
    - Intro hook
    - 3–5 sections with bullets
    - Conclusion""",
    output_key="blog_outline",
)

writer_agent = Agent(
    name="WriterAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Using this outline: {blog_outline}
    Write a 200–300 word blog post.""",
    output_key="blog_draft",
)

editor_agent = Agent(
    name="EditorAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Edit this draft for clarity: {blog_draft}""",
    output_key="final_blog",
)

blog_pipeline = SequentialAgent(
    name="BlogPipeline",
    sub_agents=[outline_agent, writer_agent, editor_agent],
)

# PARALLEL RESEARCH + AGGREGATOR

tech_researcher = Agent(
    name="TechResearcher",
    model="gemini-2.5-flash-lite",
    instruction="""Research latest AI/ML trends. 100 words.""",
    tools=[google_search],
    output_key="tech_research",
)

health_researcher = Agent(
    name="HealthResearcher",
    model="gemini-2.5-flash-lite",
    instruction="""Research recent medical breakthroughs. 100 words.""",
    tools=[google_search],
    output_key="health_research",
)

finance_researcher = Agent(
    name="FinanceResearcher",
    model="gemini-2.5-flash-lite",
    instruction="""Research fintech trends. 100 words.""",
    tools=[google_search],
    output_key="finance_research",
)

aggregator_agent = Agent(
    name="AggregatorAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Combine all research:

    Tech:
    {tech_research}

    Health:
    {health_research}

    Finance:
    {finance_research}

    Produce a 200-word executive summary.
    """,
    output_key="executive_summary",
)

parallel_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_researcher, health_researcher, finance_researcher],
)

parallel_pipeline = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_team, aggregator_agent],
)

# LOOP AGENT

initial_writer_agent = Agent(
    name="InitialWriterAgent",
    model="gemini-2.5-flash-lite",
    instruction="""Write the first 100–150 word draft of a short story based on the prompt.
    Output only the story.""",
    output_key="current_story",
)

critic_agent = Agent(
    name="CriticAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    Review the story:
    {current_story}

    If good, reply exactly: APPROVED
    Else give 2–3 improvement suggestions.
    """,
    output_key="critique",
)


def exit_loop():
    return {"status": "approved", "message": "Story approved. Exiting loop."}


refiner_agent = Agent(
    name="RefinerAgent",
    model="gemini-2.5-flash-lite",
    instruction="""
    Story: {current_story}
    Critique: {critique}

    If critique == APPROVED, call exit_loop().
    Otherwise rewrite the story using the critique.
    """,
    output_key="current_story",
    tools=[FunctionTool(exit_loop)],
)

story_loop = LoopAgent(
    name="StoryLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2,
)

story_pipeline = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_writer_agent, story_loop],
)


async def main():
    print("\n--- Running Multi-Agent Research Example ---\n")
    r = InMemoryRunner(agent=research_root)
    print(await r.run_debug("What are new advancements in robotics?"))

    print("\n--- Running Sequential Blog Pipeline ---\n")
    r = InMemoryRunner(agent=blog_pipeline)
    print(await r.run_debug("Benefits of multi-agent systems for developers"))

    print("\n--- Running Parallel Research System ---\n")
    r = InMemoryRunner(agent=parallel_pipeline)
    print(await r.run_debug("Daily executive briefing"))

    print("\n--- Running Story Refinement Loop ---\n")
    r = InMemoryRunner(agent=story_pipeline)
    print(
        await r.run_debug(
            "Write a story about a lighthouse keeper who finds a glowing map"
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
