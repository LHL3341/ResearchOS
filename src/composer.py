"""Skill Composer: composes research designs from skill graphs.

Decomposes research goals into sub-goals, retrieves candidate skills,
builds composition DAGs, identifies and bridges gaps, and synthesizes
multi-level design outputs (DesignCard, ResearchProposal, ExperimentPlan).
"""

import json
import random
import re
from typing import Optional, Union

from .schema import (
    ResearchSkill, DesignCard, ResearchProposal, ExperimentPlan, EdgeType,
)
from .skill_graph import SkillGraph
from .utils import call_llm, get_embedding, cosine_similarity


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DECOMPOSE_GOAL_PROMPT = """\
You are a research planning expert. Decompose the following research goal \
into 3-6 concrete sub-goals that, when addressed together, would achieve the \
overall goal. Each sub-goal should correspond to a distinct functional \
component or methodological requirement.

Research Goal: {goal}
{constraints_section}

Return a JSON array of sub-goal strings. Output ONLY valid JSON.
"""

GAP_ANALYSIS_PROMPT = """\
You are a research design analyst. Given the following composition DAG of \
research skills, identify functional gaps:
1. Output/input mismatches between connected skills
2. Sub-goals not covered by any skill
3. Missing preprocessing or postprocessing steps
4. Integration challenges between skills

DAG:
{dag_json}

Sub-goals:
{sub_goals}

Return a JSON array of gap objects, each with fields:
- type: "input_output_mismatch" | "uncovered_subgoal" | "missing_step" | "integration_challenge"
- description: what the gap is
- location: which skills/sub-goals are involved
- severity: "low" | "medium" | "high"

Output ONLY valid JSON.
"""

BRIDGE_GAPS_PROMPT = """\
You are a creative research methodologist. For each gap identified in a \
research design, propose a novel bridging mechanism that resolves the gap. \
The bridge should be technically feasible and clearly described.

DAG:
{dag_json}

GAPS:
{gaps_json}

For each gap, provide:
- gap_index: index of the gap being addressed
- bridge_name: short name for the bridging mechanism
- mechanism: how it works
- rationale: why this bridge is appropriate

Return a JSON array. Output ONLY valid JSON.
"""

SYNTHESIZE_DESIGN_PROMPT = """\
You are a research design architect. Given the following components, synthesize \
a coherent research design.

Research Goal: {goal}

Skills used:
{skills_text}

Composition DAG:
{dag_json}

Bridges (novel mechanisms to fill gaps):
{bridges_text}

Provide a JSON object with:
- composition_rationale: why these skills work together
- novel_bridging: summary of novel bridging mechanisms introduced
- expected_gain: expected improvement or contribution
- assumptions: list of assumptions
- risks: list of risks

Output ONLY valid JSON.
"""

EXPAND_PROPOSAL_PROMPT = """\
You are a research methodology expert. Expand the following design card into \
a full research proposal with experimental methodology.

Design:
Goal: {goal}
Skills: {skills}
Rationale: {rationale}
Novel bridging: {bridging}
Expected gain: {expected_gain}

Provide a JSON object with:
- hypothesis: formal hypothesis to test
- independent_variables: list of variables to manipulate
- dependent_variables: list of variables to measure
- controls: list of control conditions
- metrics: list of evaluation metrics
- related_work_positioning: how this relates to existing work
- implementation_notes: implementation approach notes

Output ONLY valid JSON.
"""

EXPAND_PLAN_PROMPT = """\
You are a research engineer. Expand the following research proposal into \
a concrete experiment plan with implementation details.

Proposal:
Goal: {goal}
Hypothesis: {hypothesis}
Independent vars: {ind_vars}
Dependent vars: {dep_vars}
Metrics: {metrics}

Provide a JSON object with:
- code_modifications: list of specific code changes needed
- training_setup: training configuration and procedure
- inference_setup: inference configuration and procedure
- datasets: list of datasets to use
- ablations: list of ablation studies to run
- sanity_checks: list of sanity checks before full experiments
- expected_outcomes: list of expected experimental outcomes

Output ONLY valid JSON.
"""

SELF_CRITIQUE_PROMPT = """\
You are a critical research reviewer. Evaluate the following research design \
and suggest improvements.

Design:
Goal: {goal}
Skills: {skills}
Rationale: {rationale}
Novel bridging: {bridging}
Expected gain: {expected_gain}
Assumptions: {assumptions}
Risks: {risks}

Critique this design on:
1. Logical coherence of the skill composition
2. Strength of the expected gains argument
3. Completeness of assumptions and risks
4. Novelty and significance

Then provide an improved version as a JSON object with the same fields \
as the design (composition_rationale, novel_bridging, expected_gain, \
assumptions, risks). Output ONLY valid JSON.
"""


class SkillComposer:
    """Composes research designs by assembling skills from a skill graph.

    Implements the full pipeline from goal decomposition through multi-level
    design output (DesignCard -> ResearchProposal -> ExperimentPlan), including
    gap analysis, bridging, and self-critique.
    """

    def __init__(self, graph: SkillGraph, model: str = "claude-sonnet-4-20250514"):
        """Initialize the composer.

        Args:
            graph: The SkillGraph to retrieve skills from.
            model: LLM model identifier.
        """
        self.graph = graph
        self.model = model

    def decompose_goal(self, goal: str,
                       constraints: Optional[list[str]] = None) -> list[str]:
        """Decompose a research goal into sub-goals using LLM.

        Args:
            goal: High-level research goal.
            constraints: Optional constraints to respect.

        Returns:
            List of sub-goal strings.
        """
        constraints_section = ""
        if constraints:
            constraints_section = f"Constraints:\n" + "\n".join(f"- {c}" for c in constraints)

        prompt = DECOMPOSE_GOAL_PROMPT.format(
            goal=goal,
            constraints_section=constraints_section,
        )

        response = call_llm(prompt=prompt, model=self.model,
                            system="You are a research planner. Return only valid JSON.")

        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                sub_goals = json.loads(match.group())
            else:
                sub_goals = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: treat the goal as a single sub-goal
            sub_goals = [goal]

        return sub_goals

    def retrieve_candidates(self, sub_goals: list[str],
                            top_k: int = 5) -> dict[str, list[ResearchSkill]]:
        """Retrieve candidate skills for each sub-goal.

        Args:
            sub_goals: List of sub-goal strings.
            top_k: Number of skills to retrieve per sub-goal.

        Returns:
            Dict mapping sub-goal string to list of candidate skills.
        """
        candidates: dict[str, list[ResearchSkill]] = {}
        for sg in sub_goals:
            skills = self.graph.retrieve_skills(sg, top_k=top_k)
            candidates[sg] = skills
        return candidates

    def build_dag(self, candidates: dict[str, list[ResearchSkill]],
                  constraints: list[str]) -> list[dict]:
        """Construct candidate DAGs from skill candidates via beam search.

        Each DAG maps sub-goals to selected skills and includes edges between
        them. Returns top-B DAGs ranked by a compatibility/coverage score.

        Args:
            candidates: Output of retrieve_candidates().
            constraints: Constraints to satisfy.

        Returns:
            List of DAG dicts, each with 'sub_goals', 'skills', 'edges',
            and 'score' keys. Sorted by score descending.
        """
        sub_goals = list(candidates.keys())
        beam_width = 3

        # Initialize beam with empty DAGs
        beam: list[dict] = [{"sub_goals": {}, "skills": [], "edges": [], "score": 0.0}]

        for sg in sub_goals:
            sg_candidates = candidates[sg]
            if not sg_candidates:
                continue

            new_beam: list[dict] = []
            for dag in beam:
                for skill in sg_candidates[:beam_width]:
                    new_dag = {
                        "sub_goals": {**dag["sub_goals"], sg: skill.id},
                        "skills": dag["skills"] + [skill.id],
                        "edges": list(dag["edges"]),
                        "score": dag["score"],
                    }

                    # Check compatibility with existing skills
                    is_compatible, conflicts = self.graph.check_compatibility(new_dag["skills"])
                    if is_compatible:
                        # Bonus for compatible composition
                        new_dag["score"] += 1.0
                    else:
                        new_dag["score"] -= len(conflicts) * 0.5

                    new_beam.append(new_dag)

            # Keep top beam_width DAGs
            new_beam.sort(key=lambda d: d["score"], reverse=True)
            beam = new_beam[:beam_width]

        return beam

    def analyze_gaps(self, dag: dict) -> list[dict]:
        """Identify functional gaps in a composition DAG.

        Args:
            dag: A DAG dict from build_dag().

        Returns:
            List of gap dicts with type, description, location, severity.
        """
        prompt = GAP_ANALYSIS_PROMPT.format(
            dag_json=json.dumps(dag, indent=2),
            sub_goals=json.dumps(list(dag.get("sub_goals", {}).keys()), indent=2),
        )

        response = call_llm(prompt=prompt, model=self.model,
                            system="You are a design analyst. Return only valid JSON.")

        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                gaps = json.loads(match.group())
            else:
                gaps = json.loads(response)
        except json.JSONDecodeError:
            gaps = []

        return gaps

    def bridge_gaps(self, dag: dict, gaps: list[dict]) -> dict:
        """Generate novel bridging mechanisms for gaps in the DAG.

        Args:
            dag: The composition DAG.
            gaps: List of identified gaps.

        Returns:
            Updated DAG dict with a 'bridges' key containing the bridging mechanisms.
        """
        if not gaps:
            dag["bridges"] = []
            return dag

        prompt = BRIDGE_GAPS_PROMPT.format(
            dag_json=json.dumps(dag, indent=2),
            gaps_json=json.dumps(gaps, indent=2),
        )

        response = call_llm(prompt=prompt, model=self.model,
                            system="You are a creative methodologist. Return only valid JSON.")

        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                bridges = json.loads(match.group())
            else:
                bridges = json.loads(response)
        except json.JSONDecodeError:
            bridges = []

        dag["bridges"] = bridges
        return dag

    def synthesize_design(self, dag: dict, goal: str) -> DesignCard:
        """Produce a Level A DesignCard from a composition DAG.

        Args:
            dag: The composition DAG (with bridges if applicable).
            goal: The original research goal.

        Returns:
            A DesignCard instance.
        """
        # Collect skill descriptions
        skill_ids = dag.get("skills", [])
        skills_text_parts = []
        for sid in skill_ids:
            skill = self.graph.nodes.get(sid)
            if skill:
                skills_text_parts.append(f"- {skill.name}: {skill.goal} ({skill.mechanism[:200]})")

        bridges_text = json.dumps(dag.get("bridges", []), indent=2)

        prompt = SYNTHESIZE_DESIGN_PROMPT.format(
            goal=goal,
            skills_text="\n".join(skills_text_parts) or "(no skills selected)",
            dag_json=json.dumps(dag, indent=2),
            bridges_text=bridges_text,
        )

        response = call_llm(prompt=prompt, model=self.model,
                            system="You are a design architect. Return only valid JSON.")

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            data = {}

        return DesignCard(
            research_goal=goal,
            selected_skills=skill_ids,
            composition_rationale=data.get("composition_rationale", ""),
            novel_bridging=data.get("novel_bridging", ""),
            expected_gain=data.get("expected_gain", ""),
            assumptions=data.get("assumptions", []),
            risks=data.get("risks", []),
        )

    def expand_to_proposal(self, design: DesignCard) -> ResearchProposal:
        """Expand a Level A DesignCard to a Level B ResearchProposal.

        Args:
            design: The DesignCard to expand.

        Returns:
            A ResearchProposal instance.
        """
        prompt = EXPAND_PROPOSAL_PROMPT.format(
            goal=design.research_goal,
            skills=", ".join(design.selected_skills),
            rationale=design.composition_rationale,
            bridging=design.novel_bridging,
            expected_gain=design.expected_gain,
        )

        response = call_llm(prompt=prompt, model=self.model,
                            system="You are a methodology expert. Return only valid JSON.")

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            data = {}

        return ResearchProposal(
            **design.model_dump(),
            hypothesis=data.get("hypothesis", ""),
            independent_variables=data.get("independent_variables", []),
            dependent_variables=data.get("dependent_variables", []),
            controls=data.get("controls", []),
            metrics=data.get("metrics", []),
            related_work_positioning=data.get("related_work_positioning", ""),
            implementation_notes=data.get("implementation_notes", ""),
        )

    def expand_to_plan(self, proposal: ResearchProposal) -> ExperimentPlan:
        """Expand a Level B ResearchProposal to a Level C ExperimentPlan.

        Args:
            proposal: The ResearchProposal to expand.

        Returns:
            An ExperimentPlan instance.
        """
        prompt = EXPAND_PLAN_PROMPT.format(
            goal=proposal.research_goal,
            hypothesis=proposal.hypothesis,
            ind_vars=", ".join(proposal.independent_variables),
            dep_vars=", ".join(proposal.dependent_variables),
            metrics=", ".join(proposal.metrics),
        )

        response = call_llm(prompt=prompt, model=self.model,
                            system="You are a research engineer. Return only valid JSON.")

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            data = {}

        return ExperimentPlan(
            **proposal.model_dump(),
            code_modifications=data.get("code_modifications", []),
            training_setup=data.get("training_setup", ""),
            inference_setup=data.get("inference_setup", ""),
            datasets=data.get("datasets", []),
            ablations=data.get("ablations", []),
            sanity_checks=data.get("sanity_checks", []),
            expected_outcomes=data.get("expected_outcomes", []),
        )

    def self_critique(self, design: DesignCard, rounds: int = 2) -> DesignCard:
        """Iteratively self-critique and refine a design.

        Args:
            design: The DesignCard to refine.
            rounds: Number of critique-refine iterations.

        Returns:
            Refined DesignCard.
        """
        current = design
        for i in range(rounds):
            prompt = SELF_CRITIQUE_PROMPT.format(
                goal=current.research_goal,
                skills=", ".join(current.selected_skills),
                rationale=current.composition_rationale,
                bridging=current.novel_bridging,
                expected_gain=current.expected_gain,
                assumptions=", ".join(current.assumptions),
                risks=", ".join(current.risks),
            )

            response = call_llm(
                prompt=prompt, model=self.model,
                system=f"You are a critical reviewer (round {i+1}/{rounds}). Return only valid JSON.",
            )

            try:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    data = json.loads(response)

                current = DesignCard(
                    design_id=current.design_id,
                    research_goal=current.research_goal,
                    selected_skills=current.selected_skills,
                    composition_rationale=data.get("composition_rationale", current.composition_rationale),
                    novel_bridging=data.get("novel_bridging", current.novel_bridging),
                    expected_gain=data.get("expected_gain", current.expected_gain),
                    assumptions=data.get("assumptions", current.assumptions),
                    risks=data.get("risks", current.risks),
                )
            except json.JSONDecodeError:
                print(f"[Composer] Self-critique round {i+1} failed to parse, keeping current design.")

        return current

    def compose(self, goal: str, constraints: Optional[list[str]] = None,
                level: str = "A") -> Union[DesignCard, ResearchProposal, ExperimentPlan]:
        """Full composition pipeline: decompose, retrieve, compose, refine.

        Args:
            goal: Research goal to design for.
            constraints: Optional constraints.
            level: Output level - "A" (DesignCard), "B" (ResearchProposal),
                   or "C" (ExperimentPlan).

        Returns:
            Design output at the requested level.
        """
        print(f"[Composer] Starting composition for goal: {goal}")
        print(f"[Composer] Output level: {level}")

        # Step 1: Decompose goal
        sub_goals = self.decompose_goal(goal, constraints)
        print(f"[Composer] Decomposed into {len(sub_goals)} sub-goals")

        # Step 2: Retrieve candidates
        candidates = self.retrieve_candidates(sub_goals)
        total_candidates = sum(len(v) for v in candidates.values())
        print(f"[Composer] Retrieved {total_candidates} candidate skills")

        # Step 3: Build DAGs
        dags = self.build_dag(candidates, constraints or [])
        if not dags:
            print("[Composer] No valid DAGs found, creating empty design")
            return DesignCard(research_goal=goal)

        best_dag = dags[0]
        print(f"[Composer] Best DAG score: {best_dag['score']:.2f}")

        # Step 4: Gap analysis and bridging
        gaps = self.analyze_gaps(best_dag)
        print(f"[Composer] Found {len(gaps)} gaps")
        best_dag = self.bridge_gaps(best_dag, gaps)

        # Step 5: Synthesize design
        design = self.synthesize_design(best_dag, goal)

        # Step 6: Self-critique
        design = self.self_critique(design, rounds=2)

        # Step 7: Expand to requested level
        if level == "A":
            return design
        elif level == "B":
            return self.expand_to_proposal(design)
        elif level == "C":
            proposal = self.expand_to_proposal(design)
            return self.expand_to_plan(proposal)
        else:
            raise ValueError(f"Unknown level '{level}'. Use 'A', 'B', or 'C'.")
