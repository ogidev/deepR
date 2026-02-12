"""Graph state definitions and data structures for the Deep Research agent.

Architecture: Forum-based multi-agent coordination.

The system is optimised for a *forum* topology in which every agent node
(research supervisor, specialists, researcher, human supervisor) can
communicate with every other node via conditional, interruptible edges.

State is maintained as a **ledger** — an append-only log of timestamped
``LedgerEntry`` records that preserves the full network history — rather
than a rolling summary.  The ``forum_transcript`` field accumulates a
play-script–style dialogue so that every participant's contributions are
visible at all stages.
"""

import operator
from datetime import datetime, timezone
from typing import Annotated, Dict, List, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


class QuerySpecialist(BaseModel):
    """Query a specific specialist directly for expert opinion."""
    specialist: Literal["geneticist", "systems_theorist", "predictive_cognition"] = Field(
        description="Which specialist to query"
    )
    question: str = Field(
        description="The specific question to ask this specialist"
    )


class ConductNegotiationRound(BaseModel):
    """Run one round of specialist negotiation and return results."""
    round_instructions: str = Field(
        description="High-level instructions for what this round should focus on"
    )


class RecallFromNegotiation(BaseModel):
    """Ask the orchestrator to recall specific information from the negotiation history."""
    query: str = Field(
        description="What to recall, e.g. 'What did the geneticist propose about epigenetic markers?'"
    )
    specialist_filter: Optional[Literal["geneticist", "systems_theorist", "predictive_cognition"]] = Field(
        default=None,
        description="Optionally filter recall to a specific specialist's contributions"
    )


class SynthesizeNegotiation(BaseModel):
    """Synthesize all negotiation rounds into a final HypothesesBundle."""
    synthesis_instructions: str = Field(
        default="Synthesize all proposals and critiques into a comprehensive hypotheses bundle",
        description="Optional instructions for the synthesis process"
    )


class SupervisorSpecialistQuery(BaseModel):
    """Query from supervisor to a specific specialist."""
    specialist_role: Literal["geneticist", "systems_theorist", "predictive_cognition"]
    question: str = Field(description="Specific question for the specialist")
    context: str = Field(default="", description="Additional context for the query")


class HumanDirective(BaseModel):
    """Structured directive from the human supervisor.

    The human user acts as the supervisor and can address both the
    orchestrator (for research tasks and negotiation rounds) and the
    specialists directly.
    """

    action: Literal[
        "conduct_research",
        "query_specialist",
        "conduct_negotiation_round",
        "recall_from_negotiation",
        "synthesize_negotiation",
        "generate_report",
        "provide_feedback",
    ] = Field(
        description=(
            "The action to take. 'conduct_research' delegates a research task, "
            "'query_specialist' asks a specialist directly, "
            "'conduct_negotiation_round' runs a negotiation round, "
            "'recall_from_negotiation' queries negotiation history, "
            "'synthesize_negotiation' creates the final hypotheses bundle, "
            "'generate_report' produces the final report, "
            "'provide_feedback' sends guidance to the orchestrator."
        ),
    )
    content: str = Field(
        description=(
            "The content of the directive — a research topic, question for a "
            "specialist, negotiation round instructions, recall query, "
            "synthesis instructions, or feedback text."
        ),
    )
    specialist: Literal["geneticist", "systems_theorist", "predictive_cognition"] | None = Field(
        default=None,
        description="Which specialist to query (required for 'query_specialist' and optional for 'recall_from_negotiation').",
    )


###################
# Scientific Negotiation Models
###################
class Hypothesis(BaseModel):
    """A hypothesis generated during scientific negotiation."""
    
    id: str = Field(description="Unique identifier for the hypothesis")
    statement: str = Field(description="The hypothesis statement")
    rationale: str = Field(description="Reasoning behind the hypothesis")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions underlying the hypothesis")
    key_variables: List[str] = Field(default_factory=list, description="Key variables involved in the hypothesis")
    supporting_evidence: List[str] = Field(default_factory=list, description="Evidence supporting the hypothesis")
    counter_evidence: List[str] = Field(default_factory=list, description="Evidence against the hypothesis")
    confidence: str = Field(default="medium", description="Qualitative confidence level (low/medium/high)")
    proposing_role: str = Field(default="", description="The specialist role that proposed this hypothesis")


class Prediction(BaseModel):
    """A testable prediction derived from hypotheses."""
    
    hypothesis_ids: List[str] = Field(description="IDs of hypotheses this prediction relates to")
    prediction: str = Field(description="The specific testable prediction")
    test_method: str = Field(description="Method to test this prediction")
    required_data: List[str] = Field(default_factory=list, description="Data required to test the prediction")
    expected_if_true: str = Field(default="", description="Expected outcome if prediction is true")
    expected_if_false: str = Field(default="", description="Expected outcome if prediction is false")


class Disagreement(BaseModel):
    """A documented disagreement between specialists."""
    
    topic: str = Field(description="The topic of disagreement")
    positions_by_role: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of role name to their position on the topic"
    )
    what_data_would_resolve: str = Field(
        default="",
        description="What data or evidence would help resolve this disagreement"
    )


class HypothesesBundle(BaseModel):
    """Complete output from scientific negotiation process."""
    
    hypotheses: List[Hypothesis] = Field(default_factory=list, description="List of generated hypotheses")
    predictions: List[Prediction] = Field(default_factory=list, description="Testable predictions")
    open_questions: List[str] = Field(default_factory=list, description="Remaining open questions")
    disagreements: List[Disagreement] = Field(default_factory=list, description="Documented disagreements")

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""


###################
# Forum Ledger Models
###################

class LedgerEntry(BaseModel):
    """A single entry in the forum ledger.

    The ledger is the central state artefact of the forum architecture.
    Every meaningful action — research delegation, specialist response,
    human directive, negotiation round, etc. — is recorded as an
    immutable ``LedgerEntry`` so that the full network history is
    preserved with timestamps.
    """

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of when the entry was created",
    )
    agent: str = Field(
        description="The agent or participant that produced this entry "
        "(e.g. 'human_supervisor', 'research_supervisor', 'geneticist', "
        "'systems_theorist', 'predictive_cognition', 'researcher')",
    )
    action: str = Field(
        description="Short label for the action taken (e.g. 'directive', "
        "'research_result', 'proposal', 'critique', 'feedback')",
    )
    content: str = Field(
        description="The substantive content of this ledger entry",
    )
    target: Optional[str] = Field(
        default=None,
        description="The agent or node this entry is addressed to, if any",
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional key-value metadata (e.g. round number, topic)",
    )


def format_ledger_entry_as_script_line(entry: LedgerEntry) -> str:
    """Format a single ledger entry as a play-script line.

    The forum transcript is styled like the script of a play so that all
    participants' contributions are easy to follow at every stage.

    Example output::

        [2025-07-12T09:14:22+00:00] RESEARCH_SUPERVISOR → RESEARCHER:
            (research_result) Compressed findings on epigenetic markers …
    """
    target_str = f" → {entry.target.upper()}" if entry.target else ""
    return (
        f"[{entry.timestamp}] {entry.agent.upper()}{target_str}:\n"
        f"    ({entry.action}) {entry.content}"
    )


class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data.

    The state follows a **forum** architecture:
    - ``ledger``: append-only log of timestamped ``LedgerEntry`` records
      capturing the full network history of all agent interactions.
    - ``forum_transcript``: play-script–style dialogue string built from
      ledger entries so that every participant's contributions are
      visible at all stages.
    """
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    # Forum ledger: append-only timestamped log of all agent interactions
    ledger: Annotated[list[dict], operator.add] = []
    # Play-script transcript built from ledger entries
    forum_transcript: Annotated[list[str], operator.add] = []
    # Scientific negotiation state promoted from negotiation subgraph
    negotiation_round: int = 0
    negotiation_max_rounds: int = 2
    negotiation_messages: Annotated[list[MessageLikeRepresentation], override_reducer] = []
    geneticist_proposals: List[Hypothesis] = []
    systems_theorist_proposals: List[Hypothesis] = []
    predictive_cognition_proposals: List[Hypothesis] = []
    critiques: Annotated[list[str], override_reducer] = []
    hypotheses_bundle: Optional[HypothesesBundle] = None
    # Human supervisor loop tracking
    human_supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer] = []

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    research_iterations: int
    raw_notes: Annotated[list[str], override_reducer]
    specialist_queries: Annotated[list[dict], operator.add]
    specialist_responses: Annotated[list[dict], operator.add]
    # Forum ledger: append-only timestamped log of all agent interactions
    ledger: Annotated[list[dict], operator.add]
    forum_transcript: Annotated[list[str], operator.add]
    # Negotiation state visible to supervisor
    negotiation_round: int
    negotiation_max_rounds: int
    negotiation_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    geneticist_proposals: List[Hypothesis]
    systems_theorist_proposals: List[Hypothesis]
    predictive_cognition_proposals: List[Hypothesis]
    critiques: Annotated[list[str], operator.add]
    hypotheses_bundle: Optional[HypothesesBundle]


class NegotiationState(TypedDict):
    """State for the scientific negotiation subgraph."""
    
    research_brief: str
    notes: List[str]
    raw_notes: List[str]
    # Current round in the negotiation process (1-indexed)
    current_round: int
    max_rounds: int
    # Per-specialist proposal buffers
    geneticist_proposals: List[Hypothesis]
    systems_theorist_proposals: List[Hypothesis]
    predictive_cognition_proposals: List[Hypothesis]
    # Critique and convergence data
    critiques: Annotated[list[str], operator.add]
    # Final outputs
    hypotheses_bundle: Optional[HypothesesBundle]
    negotiation_messages: Annotated[list[MessageLikeRepresentation], operator.add]

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []