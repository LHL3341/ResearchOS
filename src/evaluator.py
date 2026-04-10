"""Design Evaluator: scores and compares research designs using structured rubrics.

Provides absolute scoring (1-5 scale), pairwise comparison with position-bias
detection, Elo rating computation, batch evaluation, and future-grounded
validation against subsequently published papers.
"""

import json
import re
from typing import Optional

import pandas as pd

from .schema import DesignCard
from .utils import call_llm, get_embedding, cosine_similarity


# ---------------------------------------------------------------------------
# Scoring rubrics
# ---------------------------------------------------------------------------

NOVELTY_RUBRIC = """\
1 - Trivial: Directly replicates existing work with no new contribution.
2 - Incremental: Minor variation on known methods; obvious next step.
3 - Moderate: Combines existing ideas in a non-obvious way or applies to a new domain.
4 - Significant: Introduces a clearly new mechanism or composition with strong justification.
5 - Breakthrough: Proposes a fundamentally new paradigm or resolves a long-standing open problem.
"""

FEASIBILITY_RUBRIC = """\
1 - Infeasible: Requires unavailable resources, violates known constraints, or is logically flawed.
2 - Difficult: Requires very significant resources or relies on unproven assumptions.
3 - Challenging: Feasible but requires substantial effort and some risky assumptions.
4 - Practical: Clearly feasible with reasonable resources and well-supported assumptions.
5 - Straightforward: Could be implemented quickly with existing tools and infrastructure.
"""

IMPACT_RUBRIC = """\
1 - Negligible: Results would not change practice or understanding in the field.
2 - Minor: Small improvement relevant to a narrow sub-community.
3 - Moderate: Meaningful advance with clear applications and moderate audience.
4 - High: Significant advance that would influence multiple research directions.
5 - Transformative: Would fundamentally reshape the field or enable entirely new capabilities.
"""

CLARITY_RUBRIC = """\
1 - Incomprehensible: Cannot understand what is being proposed.
2 - Vague: General direction is clear but specifics are missing or contradictory.
3 - Adequate: Main idea is clear but some details are ambiguous.
4 - Clear: Well-articulated proposal with clear methodology and expectations.
5 - Exemplary: Crystal-clear presentation; could be directly implemented from the description.
"""

SOUNDNESS_RUBRIC = """\
1 - Flawed: Contains logical errors, unsupported claims, or contradictory elements.
2 - Weak: Some reasoning gaps; assumptions are not well-justified.
3 - Acceptable: Reasoning is generally sound but has minor gaps.
4 - Strong: Well-reasoned with clearly stated and justified assumptions.
5 - Rigorous: Logically airtight; all claims are well-supported and assumptions are minimal.
"""

RUBRICS = {
    "novelty": NOVELTY_RUBRIC,
    "feasibility": FEASIBILITY_RUBRIC,
    "impact": IMPACT_RUBRIC,
    "clarity": CLARITY_RUBRIC,
    "soundness": SOUNDNESS_RUBRIC,
}

ABSOLUTE_SCORING_PROMPT = """\
You are a research design evaluator. Score the following design on the \
"{metric}" dimension using the rubric below.

RUBRIC:
{rubric}

DESIGN:
Goal: {goal}
Skills: {skills}
Rationale: {rationale}
Novel Bridging: {bridging}
Expected Gain: {expected_gain}
Assumptions: {assumptions}
Risks: {risks}

Return a JSON object with:
- score: integer 1-5
- justification: brief explanation for the score

Output ONLY valid JSON.
"""

PAIRWISE_COMPARISON_PROMPT = """\
You are a research design evaluator. Compare the following two designs on \
the "{metric}" dimension and determine which is better.

RUBRIC:
{rubric}

DESIGN A:
Goal: {a_goal}
Rationale: {a_rationale}
Novel Bridging: {a_bridging}
Expected Gain: {a_expected_gain}

DESIGN B:
Goal: {b_goal}
Rationale: {b_rationale}
Novel Bridging: {b_bridging}
Expected Gain: {b_expected_gain}

Return a JSON object with:
- winner: "A" or "B" or "tie"
- justification: brief explanation

Output ONLY valid JSON.
"""

FUTURE_GROUNDED_PROMPT = """\
You are evaluating whether a research design anticipated ideas that later \
appeared in published papers. Check at three levels:

1. Direction: Did the design identify the same general research direction?
2. Method: Did the design propose similar methods or techniques?
3. Composition: Did the design propose the same composition of techniques?

DESIGN:
Goal: {goal}
Rationale: {rationale}
Novel Bridging: {bridging}
Expected Gain: {expected_gain}

FUTURE PAPER:
Title: {paper_title}
Abstract: {paper_abstract}

Return a JSON object with:
- direction_hit: boolean
- method_hit: boolean
- composition_hit: boolean
- explanation: brief explanation of matches

Output ONLY valid JSON.
"""


class DesignEvaluator:
    """Evaluates research designs using structured rubrics and LLM judges.

    Supports absolute scoring, pairwise comparison with position-bias
    detection, Elo rating computation, and future-grounded validation.
    """

    def __init__(self, judge_model: str = "claude-sonnet-4-20250514"):
        """Initialize the evaluator.

        Args:
            judge_model: LLM model to use as the judge.
        """
        self.judge_model = judge_model

    def score_absolute(self, design: DesignCard, metric: str) -> dict:
        """Score a design on a 1-5 scale using a structured rubric.

        Args:
            design: The DesignCard to score.
            metric: Metric name (novelty, feasibility, impact, clarity, soundness).

        Returns:
            Dict with score (int 1-5) and justification (str).
        """
        rubric = RUBRICS.get(metric)
        if not rubric:
            return {"score": 3, "justification": f"Unknown metric '{metric}'."}

        prompt = ABSOLUTE_SCORING_PROMPT.format(
            metric=metric,
            rubric=rubric,
            goal=design.research_goal,
            skills=", ".join(design.selected_skills),
            rationale=design.composition_rationale,
            bridging=design.novel_bridging,
            expected_gain=design.expected_gain,
            assumptions=", ".join(design.assumptions),
            risks=", ".join(design.risks),
        )

        response = call_llm(prompt=prompt, model=self.judge_model,
                            system="You are a fair evaluator. Return only valid JSON.")

        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = json.loads(response)
        except json.JSONDecodeError:
            result = {"score": 3, "justification": "Failed to parse judge response."}

        # Clamp score to 1-5
        score = result.get("score", 3)
        result["score"] = max(1, min(5, int(score)))
        return result

    def score_pairwise(self, design_a: DesignCard, design_b: DesignCard,
                       metric: str) -> dict:
        """Pairwise comparison of two designs with position-bias detection.

        Runs the comparison in both orderings (A vs B and B vs A) to detect
        position bias.

        Args:
            design_a: First design.
            design_b: Second design.
            metric: Metric to compare on.

        Returns:
            Dict with winner (design_id), justification, position_bias_detected (bool),
            and raw results from both orderings.
        """
        rubric = RUBRICS.get(metric, "")

        def _compare(first: DesignCard, second: DesignCard, label_first: str, label_second: str) -> dict:
            prompt = PAIRWISE_COMPARISON_PROMPT.format(
                metric=metric, rubric=rubric,
                a_goal=first.research_goal, a_rationale=first.composition_rationale,
                a_bridging=first.novel_bridging, a_expected_gain=first.expected_gain,
                b_goal=second.research_goal, b_rationale=second.composition_rationale,
                b_bridging=second.novel_bridging, b_expected_gain=second.expected_gain,
            )
            response = call_llm(prompt=prompt, model=self.judge_model,
                                system="You are a fair evaluator. Return only valid JSON.")
            try:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    return json.loads(match.group())
                return json.loads(response)
            except json.JSONDecodeError:
                return {"winner": "tie", "justification": "Parse error."}

        # Forward ordering: A=first, B=second
        forward = _compare(design_a, design_b, "A", "B")
        # Reverse ordering: B=first, A=second
        reverse = _compare(design_b, design_a, "B", "A")

        # Map reverse result back to original labels
        reverse_winner = reverse.get("winner", "tie")
        if reverse_winner == "A":
            reverse_mapped = "B"  # "A" in reverse = design_b
        elif reverse_winner == "B":
            reverse_mapped = "A"  # "B" in reverse = design_a
        else:
            reverse_mapped = "tie"

        forward_winner = forward.get("winner", "tie")

        # Detect position bias
        position_bias = (forward_winner != reverse_mapped and
                         forward_winner != "tie" and reverse_mapped != "tie")

        # Determine final winner
        if forward_winner == reverse_mapped:
            final_winner = forward_winner
        elif forward_winner == "tie" or reverse_mapped == "tie":
            final_winner = forward_winner if forward_winner != "tie" else reverse_mapped
        else:
            final_winner = "tie"  # Disagreement = tie

        winner_id = ""
        if final_winner == "A":
            winner_id = design_a.design_id
        elif final_winner == "B":
            winner_id = design_b.design_id

        return {
            "winner": winner_id,
            "winner_label": final_winner,
            "justification": forward.get("justification", ""),
            "position_bias_detected": position_bias,
            "forward_result": forward,
            "reverse_result": reverse,
        }

    def compute_elo(self, pairwise_results: list[dict],
                    initial_elo: int = 1000, k: int = 32) -> dict[str, float]:
        """Compute Elo ratings from pairwise comparison results.

        Args:
            pairwise_results: List of dicts with 'winner' and participant IDs.
                Each dict should have 'design_a_id', 'design_b_id', 'winner_label'.
            initial_elo: Starting Elo rating.
            k: K-factor for Elo updates.

        Returns:
            Dict mapping design_id to Elo rating.
        """
        ratings: dict[str, float] = {}

        for result in pairwise_results:
            a_id = result.get("design_a_id", "")
            b_id = result.get("design_b_id", "")
            winner = result.get("winner_label", "tie")

            if a_id not in ratings:
                ratings[a_id] = float(initial_elo)
            if b_id not in ratings:
                ratings[b_id] = float(initial_elo)

            ra, rb = ratings[a_id], ratings[b_id]
            ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
            eb = 1.0 - ea

            if winner == "A":
                sa, sb = 1.0, 0.0
            elif winner == "B":
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5

            ratings[a_id] = ra + k * (sa - ea)
            ratings[b_id] = rb + k * (sb - eb)

        return ratings

    def evaluate_batch(self, designs: list[DesignCard],
                       metrics: list[str]) -> pd.DataFrame:
        """Evaluate all designs on all metrics, returning a DataFrame.

        Args:
            designs: List of DesignCard objects.
            metrics: List of metric names to score.

        Returns:
            DataFrame with design_id as index and metrics as columns.
            Each cell contains the integer score (1-5).
        """
        rows = []
        for design in designs:
            row = {"design_id": design.design_id, "goal": design.research_goal}
            for metric in metrics:
                result = self.score_absolute(design, metric)
                row[metric] = result["score"]
                row[f"{metric}_justification"] = result.get("justification", "")
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("design_id")
        return df

    def future_grounded_check(self, design: DesignCard,
                              future_papers: list[dict],
                              levels: Optional[list[str]] = None) -> dict:
        """Check if a design anticipated ideas in subsequently published papers.

        Args:
            design: The DesignCard to check.
            future_papers: List of paper dicts with 'title' and 'abstract'.
            levels: Which levels to check (default: direction, method, composition).

        Returns:
            Dict with per-paper results and aggregate hit rates at each level.
        """
        if levels is None:
            levels = ["direction", "method", "composition"]

        paper_results = []
        hits = {level: 0 for level in levels}
        total = len(future_papers)

        for paper in future_papers:
            prompt = FUTURE_GROUNDED_PROMPT.format(
                goal=design.research_goal,
                rationale=design.composition_rationale,
                bridging=design.novel_bridging,
                expected_gain=design.expected_gain,
                paper_title=paper.get("title", ""),
                paper_abstract=paper.get("abstract", ""),
            )

            response = call_llm(prompt=prompt, model=self.judge_model,
                                system="You are a grounding evaluator. Return only valid JSON.")

            try:
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                else:
                    result = json.loads(response)
            except json.JSONDecodeError:
                result = {
                    "direction_hit": False,
                    "method_hit": False,
                    "composition_hit": False,
                    "explanation": "Parse error.",
                }

            paper_results.append({
                "paper": paper.get("title", "unknown"),
                **result,
            })

            if result.get("direction_hit") and "direction" in levels:
                hits["direction"] += 1
            if result.get("method_hit") and "method" in levels:
                hits["method"] += 1
            if result.get("composition_hit") and "composition" in levels:
                hits["composition"] += 1

        hit_rates = {
            level: hits[level] / total if total > 0 else 0.0
            for level in levels
        }

        return {
            "paper_results": paper_results,
            "hit_rates": hit_rates,
            "total_papers": total,
        }
