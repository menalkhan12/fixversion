import os

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession

load_dotenv()


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession()

    instructions = (
        "You are IST Admissions Voice Agent. "
        "Answer using only official IST context provided by the backend."
    )

    await session.start(room=ctx.room, agent=Agent(instructions=instructions))


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
