"""The default agent team and one example workflow per orchestration pattern."""

from __future__ import annotations

from ..domain.models import AgentSpec
from .knowledge import SEARCH_TOOL_NAME
from .patterns import Concurrent, GroupChat, Magentic, Sequential, SingleAgent, Workflow
from .ports import AgentRuntime, ChatModel

DEFAULT_WORKFLOW = "support"

RESEARCHER = AgentSpec(
    name="researcher",
    description="Answers questions from the internal knowledge base, with citations.",
    instructions=(
        "Search the knowledge base before answering. Answer only from the passages you found "
        "and cite each fact as [source: <id>]. If nothing relevant is found, say so."
    ),
    tools=(SEARCH_TOOL_NAME,),
)

CRITIC = AgentSpec(
    name="critic",
    description="Finds risks, exceptions and open questions in the knowledge base for a request.",
    instructions=(
        "Search the knowledge base for rules, limits, exceptions and approvals that affect the "
        "request. List risks and open questions only, each with a [source: <id>] citation."
    ),
    tools=(SEARCH_TOOL_NAME,),
)

WRITER = AgentSpec(
    name="writer",
    description="Turns given material into clear, concise prose. Does not research.",
    instructions=(
        "Rewrite the material you are given into clear, concise text for the user. Do not add "
        "facts. Keep [source: ...] citations exactly as written. If you were given no researched "
        "material, say that research is needed instead of writing anything."
    ),
)

REVIEWER = AgentSpec(
    name="reviewer",
    description="Checks a draft against its sources and returns the corrected final version.",
    instructions=(
        "Check the latest draft against the research in the conversation. Remove any claim "
        "without a [source: ...] citation, fix errors, and return only the final text."
    ),
)

TRIAGE = AgentSpec(
    name="triage",
    description="Supervisor that calls specialists as tools and composes their answers.",
    instructions=(
        "You coordinate specialists and have no knowledge of your own. For every factual "
        "question, call ask_researcher with the user's question word for word. Call ask_writer "
        "only when the user asks for text to be rewritten or drafted. Answer greetings yourself. "
        "Return the specialist's answer without adding facts, keeping every [source: ...] citation."
    ),
    delegates=("researcher", "writer"),
)

CONCIERGE = AgentSpec(
    name="concierge",
    description="Front desk that hands the conversation over to the right specialist.",
    instructions=(
        "You route conversations. For factual questions, transfer to the researcher. For "
        "requests to draft or rewrite text, transfer to the writer. Answer greetings yourself."
    ),
    handoffs=("researcher", "writer"),
)

AGENTS: tuple[AgentSpec, ...] = (TRIAGE, CONCIERGE, RESEARCHER, CRITIC, WRITER, REVIEWER)


def build_workflows(runtime: AgentRuntime, manager: ChatModel) -> dict[str, Workflow]:
    workflows: dict[str, Workflow] = {a.name: SingleAgent(runtime, a.name) for a in AGENTS}
    workflows.update(
        {
            "support": SingleAgent(runtime, TRIAGE.name, pattern="agents_as_tools"),
            "handoff": SingleAgent(runtime, CONCIERGE.name, pattern="handoff"),
            "research-report": Sequential(runtime, (RESEARCHER.name, WRITER.name, REVIEWER.name)),
            "perspectives": Concurrent(runtime, (RESEARCHER.name, CRITIC.name), aggregator=WRITER.name),
            "editorial": GroupChat(runtime, (RESEARCHER, WRITER, REVIEWER), manager=manager, max_rounds=4),
            "magentic": Magentic(runtime, manager, (RESEARCHER, CRITIC, WRITER, REVIEWER)),
        }
    )
    return workflows
