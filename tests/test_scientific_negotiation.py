"""Tests for scientific negotiation feature."""

import pytest
from open_deep_research.state import (
    Hypothesis,
    Prediction,
    Disagreement,
    HypothesesBundle,
    NegotiationState,
    AgentState,
    QuerySpecialist,
    SupervisorSpecialistQuery,
    ConductNegotiationRound,
    RecallFromNegotiation,
    SynthesizeNegotiation,
)
from open_deep_research.configuration import Configuration


class TestHypothesisModel:
    """Tests for the Hypothesis Pydantic model."""
    
    def test_hypothesis_creation_minimal(self):
        """Test creating a hypothesis with minimal fields."""
        h = Hypothesis(
            id="H1",
            statement="Test hypothesis statement",
            rationale="Test rationale"
        )
        assert h.id == "H1"
        assert h.statement == "Test hypothesis statement"
        assert h.rationale == "Test rationale"
        assert h.confidence == "medium"  # default
        assert h.assumptions == []
        assert h.key_variables == []
    
    def test_hypothesis_creation_full(self):
        """Test creating a hypothesis with all fields."""
        h = Hypothesis(
            id="G1",
            statement="Genetic variants in XYZ gene affect outcome",
            rationale="Based on GWAS findings",
            assumptions=["Gene is expressed in relevant tissue", "Effect is measurable"],
            key_variables=["XYZ gene expression", "Outcome measure"],
            supporting_evidence=["Study A found association"],
            counter_evidence=["Study B found no effect"],
            confidence="high",
            proposing_role="geneticist"
        )
        assert h.id == "G1"
        assert len(h.assumptions) == 2
        assert h.confidence == "high"
        assert h.proposing_role == "geneticist"


class TestPredictionModel:
    """Tests for the Prediction Pydantic model."""
    
    def test_prediction_creation(self):
        """Test creating a prediction."""
        p = Prediction(
            hypothesis_ids=["H1", "H2"],
            prediction="If H1 and H2 are true, we expect X",
            test_method="Randomized controlled trial",
            required_data=["Patient outcomes", "Gene expression data"],
            expected_if_true="Significant effect size",
            expected_if_false="No difference from control"
        )
        assert len(p.hypothesis_ids) == 2
        assert "H1" in p.hypothesis_ids
        assert p.test_method == "Randomized controlled trial"


class TestDisagreementModel:
    """Tests for the Disagreement Pydantic model."""
    
    def test_disagreement_creation(self):
        """Test creating a disagreement record."""
        d = Disagreement(
            topic="Role of feedback loops",
            positions_by_role={
                "geneticist": "Genetic factors dominate",
                "systems_theorist": "Feedback loops are primary drivers"
            },
            what_data_would_resolve="Longitudinal study with genetic and dynamic measures"
        )
        assert d.topic == "Role of feedback loops"
        assert len(d.positions_by_role) == 2
        assert "geneticist" in d.positions_by_role


class TestHypothesesBundleModel:
    """Tests for the HypothesesBundle Pydantic model."""
    
    def test_empty_bundle(self):
        """Test creating an empty bundle."""
        bundle = HypothesesBundle()
        assert bundle.hypotheses == []
        assert bundle.predictions == []
        assert bundle.open_questions == []
        assert bundle.disagreements == []
    
    def test_full_bundle(self):
        """Test creating a complete bundle."""
        h1 = Hypothesis(id="H1", statement="Hypothesis 1", rationale="Reason 1")
        h2 = Hypothesis(id="H2", statement="Hypothesis 2", rationale="Reason 2")
        
        p1 = Prediction(
            hypothesis_ids=["H1"],
            prediction="Prediction 1",
            test_method="Method 1"
        )
        
        d1 = Disagreement(
            topic="Topic 1",
            positions_by_role={"role1": "position1"}
        )
        
        bundle = HypothesesBundle(
            hypotheses=[h1, h2],
            predictions=[p1],
            open_questions=["Question 1", "Question 2"],
            disagreements=[d1]
        )
        
        assert len(bundle.hypotheses) == 2
        assert len(bundle.predictions) == 1
        assert len(bundle.open_questions) == 2
        assert len(bundle.disagreements) == 1


class TestConfiguration:
    """Tests for scientific negotiation configuration options."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = Configuration()
        assert config.enable_scientific_negotiation is False
        assert config.negotiation_rounds == 2
        assert config.negotiation_model is None
        assert config.negotiation_max_tokens == 8192
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = Configuration(
            enable_scientific_negotiation=True,
            negotiation_rounds=3,
            negotiation_model="openai:gpt-4",
            negotiation_max_tokens=16000
        )
        assert config.enable_scientific_negotiation is True
        assert config.negotiation_rounds == 3
        assert config.negotiation_model == "openai:gpt-4"
        assert config.negotiation_max_tokens == 16000


class TestNegotiationState:
    """Tests for NegotiationState TypedDict structure."""
    
    def test_negotiation_state_structure(self):
        """Test that NegotiationState has expected keys."""
        # NegotiationState is a TypedDict, so we test its annotations
        from typing import get_type_hints
        hints = get_type_hints(NegotiationState)
        
        expected_keys = [
            'research_brief',
            'notes',
            'raw_notes',
            'current_round',
            'max_rounds',
            'geneticist_proposals',
            'systems_theorist_proposals',
            'predictive_cognition_proposals',
            'critiques',
            'hypotheses_bundle',
            'negotiation_messages'
        ]
        
        for key in expected_keys:
            assert key in hints, f"NegotiationState missing key: {key}"


class TestQuerySpecialistModel:
    """Tests for the QuerySpecialist Pydantic model."""
    
    def test_query_specialist_creation_geneticist(self):
        """Test creating a query for geneticist specialist."""
        query = QuerySpecialist(
            specialist="geneticist",
            question="What genetic mechanisms might explain X?"
        )
        assert query.specialist == "geneticist"
        assert query.question == "What genetic mechanisms might explain X?"
    
    def test_query_specialist_creation_systems_theorist(self):
        """Test creating a query for systems theorist specialist."""
        query = QuerySpecialist(
            specialist="systems_theorist",
            question="What feedback loops could produce this behavior?"
        )
        assert query.specialist == "systems_theorist"
        assert query.question == "What feedback loops could produce this behavior?"
    
    def test_query_specialist_creation_predictive_cognition(self):
        """Test creating a query for predictive cognition specialist."""
        query = QuerySpecialist(
            specialist="predictive_cognition",
            question="What Bayesian priors are relevant here?"
        )
        assert query.specialist == "predictive_cognition"
        assert query.question == "What Bayesian priors are relevant here?"
    
    def test_query_specialist_invalid_specialist(self):
        """Test that invalid specialist role raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            QuerySpecialist(
                specialist="invalid_role",
                question="Test question"
            )


class TestSupervisorSpecialistQueryModel:
    """Tests for the SupervisorSpecialistQuery Pydantic model."""
    
    def test_supervisor_query_minimal(self):
        """Test creating a supervisor query with minimal fields."""
        query = SupervisorSpecialistQuery(
            specialist_role="geneticist",
            question="What is the role of gene X?"
        )
        assert query.specialist_role == "geneticist"
        assert query.question == "What is the role of gene X?"
        assert query.context == ""  # default
    
    def test_supervisor_query_with_context(self):
        """Test creating a supervisor query with context."""
        query = SupervisorSpecialistQuery(
            specialist_role="systems_theorist",
            question="How do these systems interact?",
            context="Based on the research findings about feedback loops..."
        )
        assert query.specialist_role == "systems_theorist"
        assert query.question == "How do these systems interact?"
        assert query.context == "Based on the research findings about feedback loops..."
    
    def test_supervisor_query_all_specialists(self):
        """Test that all three specialist roles are valid."""
        specialists = ["geneticist", "systems_theorist", "predictive_cognition"]
        
        for specialist in specialists:
            query = SupervisorSpecialistQuery(
                specialist_role=specialist,
                question=f"Question for {specialist}"
            )
            assert query.specialist_role == specialist


class TestConductNegotiationRoundModel:
    """Tests for the ConductNegotiationRound Pydantic model."""
    
    def test_conduct_negotiation_round_creation(self):
        """Test creating a ConductNegotiationRound tool call."""
        round_call = ConductNegotiationRound(
            round_instructions="Focus on generating testable hypotheses"
        )
        assert round_call.round_instructions == "Focus on generating testable hypotheses"
    
    def test_conduct_negotiation_round_with_detailed_instructions(self):
        """Test creating a ConductNegotiationRound with detailed instructions."""
        detailed_instructions = """
        This round should focus on:
        1. Proposing 3-6 hypotheses each
        2. Ensuring hypotheses are testable
        3. Identifying key variables
        """
        round_call = ConductNegotiationRound(round_instructions=detailed_instructions)
        assert "3-6 hypotheses" in round_call.round_instructions
        assert "testable" in round_call.round_instructions


class TestRecallFromNegotiationModel:
    """Tests for the RecallFromNegotiation Pydantic model."""
    
    def test_recall_without_filter(self):
        """Test creating a recall query without specialist filter."""
        recall = RecallFromNegotiation(
            query="What variables were identified by the specialists?"
        )
        assert recall.query == "What variables were identified by the specialists?"
        assert recall.specialist_filter is None
    
    def test_recall_with_geneticist_filter(self):
        """Test creating a recall query filtered to geneticist."""
        recall = RecallFromNegotiation(
            query="What did the geneticist say about epigenetic markers?",
            specialist_filter="geneticist"
        )
        assert recall.query == "What did the geneticist say about epigenetic markers?"
        assert recall.specialist_filter == "geneticist"
    
    def test_recall_with_systems_theorist_filter(self):
        """Test creating a recall query filtered to systems theorist."""
        recall = RecallFromNegotiation(
            query="What feedback loops were proposed?",
            specialist_filter="systems_theorist"
        )
        assert recall.specialist_filter == "systems_theorist"
    
    def test_recall_with_predictive_cognition_filter(self):
        """Test creating a recall query filtered to predictive cognition scientist."""
        recall = RecallFromNegotiation(
            query="What Bayesian priors were discussed?",
            specialist_filter="predictive_cognition"
        )
        assert recall.specialist_filter == "predictive_cognition"
    
    def test_recall_invalid_specialist_filter(self):
        """Test that invalid specialist filter raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            RecallFromNegotiation(
                query="Test query",
                specialist_filter="invalid_specialist"
            )


class TestSynthesizeNegotiationModel:
    """Tests for the SynthesizeNegotiation Pydantic model."""
    
    def test_synthesize_with_default_instructions(self):
        """Test creating a synthesis call with default instructions."""
        synthesis = SynthesizeNegotiation()
        assert synthesis.synthesis_instructions == "Synthesize all proposals and critiques into a comprehensive hypotheses bundle"
    
    def test_synthesize_with_custom_instructions(self):
        """Test creating a synthesis call with custom instructions."""
        custom_instructions = "Focus on the most testable hypotheses and generate specific predictions"
        synthesis = SynthesizeNegotiation(synthesis_instructions=custom_instructions)
        assert synthesis.synthesis_instructions == custom_instructions
        assert "testable hypotheses" in synthesis.synthesis_instructions


class TestHumanDirectiveModel:
    """Tests for the HumanDirective Pydantic model."""

    def test_conduct_research_directive(self):
        """Test creating a conduct_research directive."""
        from open_deep_research.state import HumanDirective
        directive = HumanDirective(
            action="conduct_research",
            content="Research the role of epigenetic markers in stress response",
        )
        assert directive.action == "conduct_research"
        assert "epigenetic" in directive.content
        assert directive.specialist is None

    def test_query_specialist_directive(self):
        """Test creating a query_specialist directive with specialist."""
        from open_deep_research.state import HumanDirective
        directive = HumanDirective(
            action="query_specialist",
            content="What genes are involved in circadian rhythm regulation?",
            specialist="geneticist",
        )
        assert directive.action == "query_specialist"
        assert directive.specialist == "geneticist"

    def test_generate_report_directive(self):
        """Test creating a generate_report directive."""
        from open_deep_research.state import HumanDirective
        directive = HumanDirective(
            action="generate_report",
            content="Generate the final report",
        )
        assert directive.action == "generate_report"

    def test_provide_feedback_directive(self):
        """Test creating a provide_feedback directive."""
        from open_deep_research.state import HumanDirective
        directive = HumanDirective(
            action="provide_feedback",
            content="Focus more on the genetic aspects of the research",
        )
        assert directive.action == "provide_feedback"

    def test_all_valid_actions(self):
        """Test that all valid actions are accepted."""
        from open_deep_research.state import HumanDirective
        valid_actions = [
            "conduct_research",
            "query_specialist",
            "conduct_negotiation_round",
            "recall_from_negotiation",
            "synthesize_negotiation",
            "generate_report",
            "provide_feedback",
        ]
        for action in valid_actions:
            directive = HumanDirective(action=action, content=f"Test {action}")
            assert directive.action == action

    def test_invalid_action(self):
        """Test that invalid action raises validation error."""
        from open_deep_research.state import HumanDirective
        with pytest.raises(Exception):
            HumanDirective(action="invalid_action", content="Test")

    def test_all_valid_specialists(self):
        """Test that all valid specialist values are accepted."""
        from open_deep_research.state import HumanDirective
        for specialist in ["geneticist", "systems_theorist", "predictive_cognition"]:
            directive = HumanDirective(
                action="query_specialist",
                content="Test question",
                specialist=specialist,
            )
            assert directive.specialist == specialist

    def test_invalid_specialist(self):
        """Test that invalid specialist raises validation error."""
        from open_deep_research.state import HumanDirective
        with pytest.raises(Exception):
            HumanDirective(
                action="query_specialist",
                content="Test",
                specialist="invalid_specialist",
            )


class TestBuildStatusMessage:
    """Tests for the _build_status_message helper function."""

    def test_status_message_with_empty_state(self):
        """Test status message with minimal state."""
        from open_deep_research.deep_researcher import _build_status_message
        state = {
            "messages": [],
            "research_brief": "Test brief",
            "notes": [],
            "raw_notes": [],
            "negotiation_round": 0,
            "negotiation_max_rounds": 2,
            "geneticist_proposals": [],
            "systems_theorist_proposals": [],
            "predictive_cognition_proposals": [],
            "critiques": [],
            "hypotheses_bundle": None,
        }
        msg = _build_status_message(state)
        assert "Test brief" in msg
        assert "No research notes collected yet" in msg
        assert "human supervisor" in msg.lower()

    def test_status_message_with_notes(self):
        """Test status message with research notes."""
        from open_deep_research.deep_researcher import _build_status_message
        state = {
            "messages": [],
            "research_brief": "Research about genes",
            "notes": ["Finding 1: Gene X is important", "Finding 2: Gene Y is related"],
            "raw_notes": [],
            "negotiation_round": 1,
            "negotiation_max_rounds": 3,
            "geneticist_proposals": [],
            "systems_theorist_proposals": [],
            "predictive_cognition_proposals": [],
            "critiques": [],
            "hypotheses_bundle": None,
        }
        msg = _build_status_message(state)
        assert "Research about genes" in msg
        assert "2 notes" in msg
        assert "Gene X" in msg


class TestGraphStructure:
    """Tests for the human supervisor graph structure."""

    def test_graph_has_human_supervisor_node(self):
        """Test that the compiled graph includes the human_supervisor node."""
        from open_deep_research.deep_researcher import deep_researcher
        assert "human_supervisor" in deep_researcher.nodes

    def test_graph_has_process_human_directive_node(self):
        """Test that the compiled graph includes the process_human_directive node."""
        from open_deep_research.deep_researcher import deep_researcher
        assert "process_human_directive" in deep_researcher.nodes

    def test_graph_preserves_existing_nodes(self):
        """Test that all original nodes are still present."""
        from open_deep_research.deep_researcher import deep_researcher
        expected_nodes = [
            "clarify_with_user",
            "write_research_brief",
            "research_supervisor",
            "final_report_generation",
        ]
        for node in expected_nodes:
            assert node in deep_researcher.nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
