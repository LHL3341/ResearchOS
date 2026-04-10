"""Design Validator: validates research designs for novelty, feasibility, impact, and risk.

Provides automated checks against existing papers, skill prerequisites,
resource budgets, and failure mode aggregation.
"""

import json
import re
from typing import Optional

from .schema import DesignCard, EdgeType
from .skill_graph import SkillGraph
from .utils import call_llm, get_embedding, cosine_similarity


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

IMPACT_ASSESSMENT_PROMPT = """\
You are a research impact assessor. Evaluate the potential significance \
of the following research design.

Research Goal: {goal}
Composition Rationale: {rationale}
Novel Bridging: {bridging}
Expected Gain: {expected_gain}
Assumptions: {assumptions}
Risks: {risks}

Rate the impact as one of:
- incremental: minor improvement over existing methods
- moderate: meaningful advance with clear practical value
- significant: substantial advance that opens new directions
- breakthrough: paradigm-shifting contribution

Return a JSON object with:
- impact_level: one of the above
- justification: brief explanation
- key_contributions: list of main contributions
- limitations: list of factors limiting impact

Output ONLY valid JSON.
"""


class DesignValidator:
    """Validates research designs across multiple dimensions.

    Checks novelty (vs existing papers), feasibility (prerequisites and
    resources), impact (LLM-assessed significance), and risk (aggregated
    failure modes).
    """

    def __init__(self, graph: SkillGraph,
                 paper_corpus: Optional[list[dict]] = None):
        """Initialize the validator.

        Args:
            graph: The SkillGraph containing skill definitions.
            paper_corpus: Optional list of paper dicts with 'id', 'title',
                         'abstract' fields for novelty checking.
        """
        self.graph = graph
        self.paper_corpus = paper_corpus or []

    def check_novelty(self, design: DesignCard,
                      existing_papers: Optional[list[dict]] = None) -> dict:
        """Check the novelty of a design against existing papers.

        Computes embedding similarity between the design and existing
        paper abstracts.

        Args:
            design: The DesignCard to evaluate.
            existing_papers: Papers to compare against (overrides corpus).

        Returns:
            Dict with novelty_score (0-1, higher = more novel),
            most_similar_papers (list), and assessment (str).
        """
        papers = existing_papers or self.paper_corpus
        if not papers:
            return {
                "novelty_score": 1.0,
                "most_similar_papers": [],
                "assessment": "No existing papers to compare against.",
            }

        design_text = (
            f"{design.research_goal} {design.composition_rationale} "
            f"{design.novel_bridging} {design.expected_gain}"
        )
        design_emb = get_embedding(design_text)

        similarities: list[tuple[float, dict]] = []
        for paper in papers:
            paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            paper_emb = get_embedding(paper_text)
            sim = cosine_similarity(design_emb, paper_emb)
            similarities.append((sim, paper))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_similar = similarities[:5]

        max_sim = top_similar[0][0] if top_similar else 0.0
        novelty_score = 1.0 - max_sim

        most_similar = [
            {"paper_id": p.get("id", "unknown"), "title": p.get("title", ""),
             "similarity": round(s, 4)}
            for s, p in top_similar
        ]

        if novelty_score > 0.7:
            assessment = "Highly novel - no closely related work found."
        elif novelty_score > 0.4:
            assessment = "Moderately novel - some related work exists but approach is distinct."
        else:
            assessment = "Low novelty - very similar work already exists."

        return {
            "novelty_score": round(novelty_score, 4),
            "most_similar_papers": most_similar,
            "assessment": assessment,
        }

    def check_feasibility(self, design: DesignCard) -> dict:
        """Check the feasibility of a design.

        Verifies that prerequisites are satisfied and resources are
        reasonable.

        Args:
            design: The DesignCard to evaluate.

        Returns:
            Dict with feasibility_score (0-1), issues (list of str),
            prerequisites_met (bool), and resource_assessment (str).
        """
        issues: list[str] = []
        skill_ids = design.selected_skills

        # Check that all selected skills exist in the graph
        missing_skills = [sid for sid in skill_ids if sid not in self.graph.nodes]
        if missing_skills:
            issues.append(f"Skills not found in graph: {missing_skills}")

        # Check prerequisites
        for sid in skill_ids:
            prereqs = self.graph.get_prerequisites(sid)
            for prereq in prereqs:
                if prereq.id not in skill_ids:
                    skill_name = self.graph.nodes.get(sid)
                    label = skill_name.name if skill_name else sid
                    issues.append(
                        f"Skill '{label}' requires prerequisite '{prereq.name}' "
                        f"which is not in the selected skills."
                    )

        # Check compatibility
        valid_ids = [sid for sid in skill_ids if sid in self.graph.nodes]
        is_compatible, conflicts = self.graph.check_compatibility(valid_ids)
        if not is_compatible:
            issues.extend(conflicts)

        # Assess resources
        resource_descriptions = []
        for sid in valid_ids:
            skill = self.graph.nodes[sid]
            if skill.required_resources:
                resource_descriptions.append(f"{skill.name}: {skill.required_resources}")

        resource_assessment = (
            "Resources needed:\n" + "\n".join(f"- {r}" for r in resource_descriptions)
            if resource_descriptions else "No specific resource requirements identified."
        )

        # Compute score
        penalty = len(issues) * 0.15
        feasibility_score = max(0.0, 1.0 - penalty)

        return {
            "feasibility_score": round(feasibility_score, 4),
            "issues": issues,
            "prerequisites_met": not any("prerequisite" in i.lower() for i in issues),
            "resource_assessment": resource_assessment,
        }

    def check_impact(self, design: DesignCard,
                     model: str = "claude-sonnet-4-20250514") -> dict:
        """Estimate the potential impact of a design using LLM.

        Args:
            design: The DesignCard to evaluate.
            model: LLM model to use.

        Returns:
            Dict with impact_level (str), justification (str),
            key_contributions (list), limitations (list).
        """
        prompt = IMPACT_ASSESSMENT_PROMPT.format(
            goal=design.research_goal,
            rationale=design.composition_rationale,
            bridging=design.novel_bridging,
            expected_gain=design.expected_gain,
            assumptions=", ".join(design.assumptions),
            risks=", ".join(design.risks),
        )

        response = call_llm(prompt=prompt, model=model,
                            system="You are an impact assessor. Return only valid JSON.")

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "impact_level": "moderate",
                "justification": "Could not assess impact (LLM parse error).",
                "key_contributions": [],
                "limitations": [],
            }

        return result

    def check_risk(self, design: DesignCard) -> dict:
        """Aggregate and assess risks from selected skills.

        Collects failure modes from all selected skills and checks for
        risk amplification when skills are combined.

        Args:
            design: The DesignCard to evaluate.

        Returns:
            Dict with risk_list (list of dicts), overall_risk_level (str),
            and amplification_warnings (list of str).
        """
        risk_list: list[dict] = []
        all_failure_modes: list[str] = []

        for sid in design.selected_skills:
            skill = self.graph.nodes.get(sid)
            if not skill:
                continue
            for fm in skill.failure_modes:
                risk_list.append({
                    "source_skill": skill.name,
                    "failure_mode": fm,
                })
                all_failure_modes.append(fm)

        # Check for amplification: if multiple skills share similar failure modes
        amplification_warnings: list[str] = []
        seen_modes: dict[str, list[str]] = {}
        for item in risk_list:
            mode_lower = item["failure_mode"].lower()
            for keyword in ["overfit", "catastrophic", "degenerat", "unstable", "diverge",
                            "bias", "memoriz", "hallucin", "collapse"]:
                if keyword in mode_lower:
                    if keyword not in seen_modes:
                        seen_modes[keyword] = []
                    seen_modes[keyword].append(item["source_skill"])

        for keyword, skills in seen_modes.items():
            if len(skills) > 1:
                amplification_warnings.append(
                    f"Risk amplification: '{keyword}' failure mode appears in "
                    f"multiple skills ({', '.join(skills)}). Combined risk may be higher."
                )

        # Overall risk level
        if amplification_warnings:
            overall = "high"
        elif len(risk_list) > 5:
            overall = "medium"
        elif risk_list:
            overall = "low"
        else:
            overall = "minimal"

        return {
            "risk_list": risk_list,
            "overall_risk_level": overall,
            "amplification_warnings": amplification_warnings,
        }

    def validate(self, design: DesignCard,
                 existing_papers: Optional[list[dict]] = None) -> dict:
        """Run all validation checks and return a combined report.

        Args:
            design: The DesignCard to validate.
            existing_papers: Optional papers for novelty check.

        Returns:
            Dict with novelty, feasibility, impact, risk sub-reports
            and an overall_assessment string.
        """
        novelty = self.check_novelty(design, existing_papers)
        feasibility = self.check_feasibility(design)
        impact = self.check_impact(design)
        risk = self.check_risk(design)

        # Overall assessment
        scores = {
            "novelty": novelty["novelty_score"],
            "feasibility": feasibility["feasibility_score"],
        }
        avg_score = sum(scores.values()) / len(scores)

        if avg_score > 0.7 and risk["overall_risk_level"] in ("low", "minimal"):
            overall = "STRONG: Design is novel, feasible, and low-risk."
        elif avg_score > 0.5:
            overall = "MODERATE: Design has merit but some concerns should be addressed."
        else:
            overall = "WEAK: Design has significant issues that need resolution."

        return {
            "novelty": novelty,
            "feasibility": feasibility,
            "impact": impact,
            "risk": risk,
            "overall_assessment": overall,
        }
