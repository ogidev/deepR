"""Graph state definitions and data structures for the Deep Research agent."""

import operator
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


class SupervisorSpecialistQuery(BaseModel):
    """Query from supervisor to a specific specialist."""
    specialist_role: Literal["geneticist", "systems_theorist", "predictive_cognition"]
    question: str = Field(description="Specific question for the specialist")
    context: str = Field(default="", description="Additional context for the query")


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
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    # Scientific negotiation outputs promoted from negotiation subgraph
    negotiation_round: int
    negotiation_max_rounds: int
    negotiation_messages: Annotated[list[MessageLikeRepresentation], override_reducer] = []
    geneticist_proposals: List[Hypothesis]
    systems_theorist_proposals: List[Hypothesis]
    predictive_cognition_proposals: List[Hypothesis]
    critiques: Annotated[list[str], override_reducer] = []
    hypotheses_bundle: Optional[HypothesesBundle] = None

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    research_iterations: int
    raw_notes: Annotated[list[str], override_reducer]
    specialist_queries: Annotated[list[dict], operator.add]
    specialist_responses: Annotated[list[dict], operator.add]
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