"""Tests for scientific negotiation feature."""

import pytest
from open_deep_research.state import (
    Hypothesis,
    Prediction,
    Disagreement,
    HypothesesBundle,
    NegotiationState,
    AgentState,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
