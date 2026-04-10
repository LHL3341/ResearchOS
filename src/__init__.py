"""ResearchOS: Composing Research Skills for Automated Method Design."""

from .schema import (
    ResearchSkill,
    SkillLevel,
    EdgeType,
    SkillEdge,
    DesignCard,
    ResearchProposal,
    ExperimentPlan,
)
from .skill_compiler import SkillCompiler
from .skill_graph import SkillGraph
from .composer import SkillComposer
from .validator import DesignValidator
from .evaluator import DesignEvaluator
from .pipeline import Paper2SkillPipeline, TaxonomyNode
from .baselines import (
    direct_prompting,
    rag_over_papers,
    flat_skill_list,
    graph_retrieval_only,
    random_composition,
)

__all__ = [
    "ResearchSkill",
    "SkillLevel",
    "EdgeType",
    "SkillEdge",
    "DesignCard",
    "ResearchProposal",
    "ExperimentPlan",
    "SkillCompiler",
    "SkillGraph",
    "SkillComposer",
    "DesignValidator",
    "DesignEvaluator",
    "direct_prompting",
    "rag_over_papers",
    "flat_skill_list",
    "graph_retrieval_only",
    "random_composition",
]
