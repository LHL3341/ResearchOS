"""Core data models for the ResearchOS agentic research system.

Defines Pydantic models for research skills, skill graph edges,
and multi-level research design outputs (DesignCard, ResearchProposal, ExperimentPlan).
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class SkillLevel(str, Enum):
    """The level at which a research skill operates."""
    TRAINING_TIME = "training-time"
    INFERENCE_TIME = "inference-time"
    DATA = "data"
    ARCHITECTURE = "architecture"
    PROMPTING = "prompting"


class ResearchSkill(BaseModel):
    """A discrete, reusable research technique extracted from a paper.

    Each skill captures a self-contained methodological unit with enough
    detail that it could be re-implemented and composed with other skills.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Short descriptive name of the skill")
    goal: str = Field(..., description="What this skill aims to achieve")
    mechanism: str = Field(..., description="How the skill works, at an implementable level of detail")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions required for the skill to work")
    inputs: list[str] = Field(default_factory=list, description="Required inputs (data, models, etc.)")
    outputs: list[str] = Field(default_factory=list, description="What the skill produces")
    level: SkillLevel = Field(..., description="Operational level of the skill")
    required_resources: str = Field(default="", description="Compute, data, or other resource requirements")
    empirical_gains: list[str] = Field(default_factory=list, description="Reported improvements from the source paper")
    failure_modes: list[str] = Field(default_factory=list, description="Known failure modes or limitations")
    compatibility: list[str] = Field(default_factory=list, description="IDs of skills known to be compatible")
    conflict: list[str] = Field(default_factory=list, description="IDs of skills known to conflict")
    source_papers: list[str] = Field(default_factory=list, description="Paper IDs or references")
    tags: list[str] = Field(default_factory=list, description="Free-form tags for categorization")


class EdgeType(str, Enum):
    """Types of relationships between research skills."""
    PREREQUISITE = "PREREQUISITE"   # A must be done before B
    ENHANCES = "ENHANCES"           # A improves the effectiveness of B
    SUBSTITUTES = "SUBSTITUTES"     # A can replace B
    CONFLICTS = "CONFLICTS"         # A and B cannot be used together
    COMPOSES = "COMPOSES"           # A and B can be combined into a pipeline
    REFINES = "REFINES"             # A is a more specific version of B


class SkillEdge(BaseModel):
    """A directed edge between two skills in the skill graph."""
    source_id: str = Field(..., description="ID of the source skill")
    target_id: str = Field(..., description="ID of the target skill")
    edge_type: EdgeType = Field(..., description="Type of relationship")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this edge (0-1)")
    evidence: str = Field(default="", description="Textual evidence supporting this edge")


class DesignCard(BaseModel):
    """Level A output: a high-level research design composed from skills.

    Captures the core idea, selected skills, and composition rationale
    without full experimental detail.
    """
    design_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    research_goal: str = Field(..., description="The research question or objective")
    selected_skills: list[str] = Field(default_factory=list, description="IDs of skills used in this design")
    composition_rationale: str = Field(default="", description="Why these skills were composed in this way")
    novel_bridging: str = Field(default="", description="Novel mechanisms introduced to bridge gaps between skills")
    expected_gain: str = Field(default="", description="Expected improvement or contribution")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions underlying the design")
    risks: list[str] = Field(default_factory=list, description="Identified risks and potential issues")


class ResearchProposal(DesignCard):
    """Level B output: extends DesignCard with experimental methodology.

    Adds hypothesis, variables, controls, and positioning within
    the related work landscape.
    """
    hypothesis: str = Field(default="", description="Formal hypothesis to be tested")
    independent_variables: list[str] = Field(default_factory=list, description="Variables being manipulated")
    dependent_variables: list[str] = Field(default_factory=list, description="Variables being measured")
    controls: list[str] = Field(default_factory=list, description="Control conditions")
    metrics: list[str] = Field(default_factory=list, description="Evaluation metrics")
    related_work_positioning: str = Field(default="", description="How this work relates to existing literature")
    implementation_notes: str = Field(default="", description="Notes on implementation approach")


class ExperimentPlan(ResearchProposal):
    """Level C output: extends ResearchProposal with full implementation detail.

    Contains everything needed to execute the experiment, including
    code modifications, training setup, and expected outcomes.
    """
    code_modifications: list[str] = Field(default_factory=list, description="Specific code changes required")
    training_setup: str = Field(default="", description="Training configuration and procedure")
    inference_setup: str = Field(default="", description="Inference configuration and procedure")
    datasets: list[str] = Field(default_factory=list, description="Datasets to use")
    ablations: list[str] = Field(default_factory=list, description="Ablation studies to run")
    sanity_checks: list[str] = Field(default_factory=list, description="Sanity checks before full experiments")
    expected_outcomes: list[str] = Field(default_factory=list, description="Expected experimental outcomes")
