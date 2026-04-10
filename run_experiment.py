#!/usr/bin/env python3
"""Main experiment runner for ResearchOS.

Orchestrates five experiment types:
  1. Design quality benchmark (compare baselines vs full pipeline)
  2. Proposal benchmark (expand top designs to proposals)
  3. Future-grounded evaluation (check against future papers)
  4. (reserved)
  5. Judge reliability (compare LLM judge to human scores)
  + Ablation studies

Usage:
    python run_experiment.py --experiment 1 --goals-file goals.json
    python run_experiment.py --experiment ablation --goals-file goals.json
    python run_experiment.py --experiment all
"""

import argparse
import json
import os
import sys
from typing import Optional

import pandas as pd

from src.schema import DesignCard, ResearchSkill, SkillLevel
from src.skill_graph import SkillGraph
from src.composer import SkillComposer
from src.evaluator import DesignEvaluator
from src.validator import DesignValidator
from src.baselines import (
    direct_prompting,
    rag_over_papers,
    flat_skill_list,
    graph_retrieval_only,
    random_composition,
)
from src.utils import call_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_goals(path: Optional[str]) -> list[str]:
    """Load research goals from a JSON file or return defaults.

    Args:
        path: Path to a JSON file containing a list of goal strings.

    Returns:
        List of research goal strings.
    """
    if path and os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]

    # Default demo goals
    return [
        "Improve zero-shot reasoning in large language models without additional training data.",
        "Reduce hallucination in retrieval-augmented generation systems.",
        "Enable efficient fine-tuning of 70B parameter models on a single GPU.",
    ]


def build_demo_graph() -> SkillGraph:
    """Build a small demo skill graph for testing.

    Returns:
        A SkillGraph with a handful of example skills and edges.
    """
    from src.schema import SkillEdge, EdgeType

    graph = SkillGraph()

    skills = [
        ResearchSkill(
            id="skill-cot", name="Chain-of-Thought Prompting",
            goal="Improve multi-step reasoning by eliciting intermediate steps.",
            mechanism="Append 'Let's think step by step' or provide few-shot examples with reasoning traces.",
            assumptions=["Model has sufficient capacity for multi-step reasoning."],
            inputs=["LLM", "task prompt"], outputs=["reasoning trace", "final answer"],
            level=SkillLevel.PROMPTING,
            required_resources="Minimal (prompting only)",
            empirical_gains=["GSM8K: +15% accuracy"],
            failure_modes=["May produce plausible but incorrect reasoning chains."],
            tags=["reasoning", "prompting"],
        ),
        ResearchSkill(
            id="skill-sc", name="Self-Consistency Decoding",
            goal="Improve answer reliability by sampling multiple reasoning paths and voting.",
            mechanism="Sample N reasoning chains via temperature sampling, then take majority vote on final answer.",
            assumptions=["Correct answers are more likely than any single incorrect answer."],
            inputs=["LLM", "task prompt", "temperature"], outputs=["voted answer", "confidence"],
            level=SkillLevel.INFERENCE_TIME,
            required_resources="N times single-inference cost",
            empirical_gains=["GSM8K: +5% over CoT"],
            failure_modes=["High compute cost.", "Fails when all paths converge on wrong answer."],
            tags=["reasoning", "decoding", "ensemble"],
        ),
        ResearchSkill(
            id="skill-lora", name="LoRA (Low-Rank Adaptation)",
            goal="Enable parameter-efficient fine-tuning by adding low-rank matrices.",
            mechanism="Inject trainable low-rank decomposition matrices into attention layers while freezing base weights.",
            assumptions=["Task-specific adaptations lie in a low-rank subspace."],
            inputs=["pretrained model", "task data"], outputs=["adapted model"],
            level=SkillLevel.TRAINING_TIME,
            required_resources="1 GPU, task dataset",
            empirical_gains=["Matches full fine-tuning with <1% trainable parameters."],
            failure_modes=["May underperform full fine-tuning on complex tasks."],
            tags=["fine-tuning", "efficiency", "adaptation"],
        ),
        ResearchSkill(
            id="skill-rag", name="Retrieval-Augmented Generation",
            goal="Ground generation in retrieved documents to improve factuality.",
            mechanism="Retrieve top-k documents from a corpus, prepend to context, then generate.",
            assumptions=["Relevant documents exist in the corpus.", "Model can faithfully use retrieved context."],
            inputs=["query", "document corpus", "LLM"], outputs=["grounded response"],
            level=SkillLevel.INFERENCE_TIME,
            required_resources="Vector store, embedding model",
            empirical_gains=["Reduces hallucination by 30-50%"],
            failure_modes=["Retrieval may return irrelevant documents.", "Context window overflow."],
            tags=["retrieval", "grounding", "factuality"],
        ),
        ResearchSkill(
            id="skill-dpo", name="Direct Preference Optimization",
            goal="Align model outputs with human preferences without RL.",
            mechanism="Directly optimize policy using preference pairs via a classification loss, bypassing reward model.",
            assumptions=["Preference data is available.", "Bradley-Terry model of preferences holds."],
            inputs=["base model", "preference pairs"], outputs=["aligned model"],
            level=SkillLevel.TRAINING_TIME,
            required_resources="GPUs for training, preference dataset",
            empirical_gains=["Matches RLHF quality with simpler training."],
            failure_modes=["Sensitive to preference data quality.", "May overfit to annotation artifacts."],
            tags=["alignment", "preferences", "training"],
        ),
    ]

    for skill in skills:
        graph.add_skill(skill)

    edges = [
        SkillEdge(source_id="skill-cot", target_id="skill-sc",
                  edge_type=EdgeType.ENHANCES, confidence=0.95,
                  evidence="Self-consistency builds on chain-of-thought by sampling multiple chains."),
        SkillEdge(source_id="skill-cot", target_id="skill-rag",
                  edge_type=EdgeType.COMPOSES, confidence=0.8,
                  evidence="CoT reasoning can be applied over retrieved documents."),
        SkillEdge(source_id="skill-lora", target_id="skill-dpo",
                  edge_type=EdgeType.COMPOSES, confidence=0.85,
                  evidence="LoRA can be used as the adaptation method within DPO training."),
        SkillEdge(source_id="skill-rag", target_id="skill-sc",
                  edge_type=EdgeType.COMPOSES, confidence=0.7,
                  evidence="Self-consistency can be applied over RAG outputs."),
    ]

    for edge in edges:
        graph.add_edge(edge)

    return graph


# ---------------------------------------------------------------------------
# Experiment functions
# ---------------------------------------------------------------------------

def run_design_quality_benchmark(goals: list[str], graph: SkillGraph) -> pd.DataFrame:
    """Experiment 1: Compare design quality across baselines and full pipeline.

    For each goal, generates designs using all baselines and the full
    ResearchOS pipeline, then evaluates them on multiple metrics.

    Args:
        goals: List of research goal strings.
        graph: The SkillGraph to use.

    Returns:
        DataFrame with scores for each method x metric combination.
    """
    print("=" * 70)
    print("EXPERIMENT 1: Design Quality Benchmark")
    print("=" * 70)

    evaluator = DesignEvaluator()
    composer = SkillComposer(graph)
    all_skills = list(graph.nodes.values())
    metrics = ["novelty", "feasibility", "impact", "clarity", "soundness"]

    all_results = []

    for goal in goals:
        print(f"\n--- Goal: {goal[:80]}... ---")

        # Generate designs from each method
        methods = {}
        methods["direct_prompting"] = direct_prompting(goal, ["Sample abstract."])
        methods["rag_over_papers"] = rag_over_papers(goal, ["Sample chunk."])
        methods["flat_skill_list"] = flat_skill_list(goal, all_skills)
        methods["graph_retrieval"] = graph_retrieval_only(goal, graph)
        methods["random_composition"] = random_composition(goal, graph)
        methods["researchos_full"] = composer.compose(goal, level="A")

        # Evaluate each
        for method_name, design in methods.items():
            row = {"goal": goal[:60], "method": method_name}
            for metric in metrics:
                result = evaluator.score_absolute(design, metric)
                row[metric] = result["score"]
            all_results.append(row)
            print(f"  {method_name}: {', '.join(f'{m}={row[m]}' for m in metrics)}")

    df = pd.DataFrame(all_results)
    print("\n--- Aggregate Results ---")
    if not df.empty:
        agg = df.groupby("method")[metrics].mean()
        print(agg.to_string())
    return df


def run_proposal_benchmark(goals: list[str], graph: SkillGraph) -> list[dict]:
    """Experiment 2: Expand top designs to Level B proposals and evaluate.

    Args:
        goals: List of research goal strings.
        graph: The SkillGraph to use.

    Returns:
        List of proposal evaluation dicts.
    """
    print("=" * 70)
    print("EXPERIMENT 2: Proposal Benchmark")
    print("=" * 70)

    composer = SkillComposer(graph)
    evaluator = DesignEvaluator()
    validator = DesignValidator(graph)
    results = []

    for goal in goals:
        print(f"\n--- Goal: {goal[:80]}... ---")
        proposal = composer.compose(goal, level="B")
        validation = validator.validate(proposal)
        score = evaluator.score_absolute(proposal, "soundness")

        result = {
            "goal": goal,
            "hypothesis": proposal.hypothesis,
            "metrics": proposal.metrics,
            "validation": validation["overall_assessment"],
            "soundness_score": score["score"],
        }
        results.append(result)
        print(f"  Hypothesis: {proposal.hypothesis[:100]}")
        print(f"  Validation: {validation['overall_assessment']}")
        print(f"  Soundness: {score['score']}/5")

    return results


def run_future_grounded(goals: list[str], graph: SkillGraph,
                        future_papers: Optional[list[dict]] = None) -> dict:
    """Experiment 3: Future-grounded evaluation.

    Checks whether generated designs anticipate ideas that appear in
    subsequent papers.

    Args:
        goals: List of research goal strings.
        graph: The SkillGraph to use.
        future_papers: List of paper dicts with 'title' and 'abstract'.

    Returns:
        Dict with aggregate hit rates.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Future-Grounded Evaluation")
    print("=" * 70)

    if not future_papers:
        future_papers = [
            {"title": "Example Future Paper", "abstract": "This paper proposes..."},
        ]

    composer = SkillComposer(graph)
    evaluator = DesignEvaluator()

    all_hit_rates = {"direction": [], "method": [], "composition": []}

    for goal in goals:
        print(f"\n--- Goal: {goal[:80]}... ---")
        design = composer.compose(goal, level="A")
        result = evaluator.future_grounded_check(design, future_papers)

        for level in ["direction", "method", "composition"]:
            rate = result["hit_rates"].get(level, 0.0)
            all_hit_rates[level].append(rate)
            print(f"  {level}: {rate:.2%}")

    avg_rates = {
        level: sum(rates) / len(rates) if rates else 0.0
        for level, rates in all_hit_rates.items()
    }
    print(f"\n--- Average Hit Rates ---")
    for level, rate in avg_rates.items():
        print(f"  {level}: {rate:.2%}")

    return {"average_hit_rates": avg_rates, "per_goal": all_hit_rates}


def run_judge_reliability(goals: list[str], graph: SkillGraph,
                          human_scores: Optional[dict] = None) -> dict:
    """Experiment 5: Judge reliability — compare LLM judge to human scores.

    Args:
        goals: List of research goal strings.
        graph: The SkillGraph to use.
        human_scores: Optional dict mapping design_id -> {metric: score}.

    Returns:
        Dict with agreement statistics.
    """
    print("=" * 70)
    print("EXPERIMENT 5: Judge Reliability")
    print("=" * 70)

    composer = SkillComposer(graph)
    evaluator = DesignEvaluator()
    metrics = ["novelty", "feasibility", "soundness"]

    designs = []
    for goal in goals:
        design = composer.compose(goal, level="A")
        designs.append(design)

    # Pairwise comparisons for position bias detection
    bias_count = 0
    total_pairs = 0
    for i in range(len(designs)):
        for j in range(i + 1, len(designs)):
            for metric in metrics:
                result = evaluator.score_pairwise(designs[i], designs[j], metric)
                total_pairs += 1
                if result["position_bias_detected"]:
                    bias_count += 1

    bias_rate = bias_count / total_pairs if total_pairs else 0.0
    print(f"\nPosition bias rate: {bias_rate:.2%} ({bias_count}/{total_pairs})")

    # If human scores are provided, compute agreement
    agreement = {}
    if human_scores:
        matches = 0
        total = 0
        for design in designs:
            if design.design_id in human_scores:
                human = human_scores[design.design_id]
                for metric in metrics:
                    if metric in human:
                        llm_result = evaluator.score_absolute(design, metric)
                        if abs(llm_result["score"] - human[metric]) <= 1:
                            matches += 1
                        total += 1
        agreement = {
            "agreement_rate": matches / total if total else 0.0,
            "total_comparisons": total,
        }
        print(f"Human agreement (within 1): {agreement['agreement_rate']:.2%}")
    else:
        print("No human scores provided. Skipping agreement analysis.")

    return {
        "position_bias_rate": bias_rate,
        "total_pairwise": total_pairs,
        "human_agreement": agreement,
    }


def run_ablations(goals: list[str], graph: SkillGraph) -> dict:
    """Ablation studies: measure the contribution of each pipeline component.

    Tests:
    - No self-critique
    - No gap analysis/bridging
    - No graph neighborhood expansion
    - Single sub-goal (no decomposition)

    Args:
        goals: List of research goal strings.
        graph: The SkillGraph to use.

    Returns:
        Dict with ablation results.
    """
    print("=" * 70)
    print("ABLATION STUDIES")
    print("=" * 70)

    evaluator = DesignEvaluator()
    metrics = ["novelty", "feasibility", "soundness"]
    results: dict[str, list[dict]] = {}

    for goal in goals[:2]:  # Limit for speed
        print(f"\n--- Goal: {goal[:80]}... ---")

        # Full pipeline
        composer = SkillComposer(graph)
        full_design = composer.compose(goal, level="A")

        # Ablation: no self-critique
        composer_no_critique = SkillComposer(graph)
        sub_goals = composer_no_critique.decompose_goal(goal)
        candidates = composer_no_critique.retrieve_candidates(sub_goals)
        dags = composer_no_critique.build_dag(candidates, [])
        if dags:
            gaps = composer_no_critique.analyze_gaps(dags[0])
            dag_bridged = composer_no_critique.bridge_gaps(dags[0], gaps)
            no_critique_design = composer_no_critique.synthesize_design(dag_bridged, goal)
        else:
            no_critique_design = DesignCard(research_goal=goal)

        # Ablation: no gap analysis
        composer_no_gaps = SkillComposer(graph)
        sub_goals2 = composer_no_gaps.decompose_goal(goal)
        candidates2 = composer_no_gaps.retrieve_candidates(sub_goals2)
        dags2 = composer_no_gaps.build_dag(candidates2, [])
        if dags2:
            no_gaps_design = composer_no_gaps.synthesize_design(dags2[0], goal)
        else:
            no_gaps_design = DesignCard(research_goal=goal)

        # Evaluate all variants
        variants = {
            "full": full_design,
            "no_critique": no_critique_design,
            "no_gaps": no_gaps_design,
        }

        for variant_name, design in variants.items():
            scores = {}
            for metric in metrics:
                result = evaluator.score_absolute(design, metric)
                scores[metric] = result["score"]

            if variant_name not in results:
                results[variant_name] = []
            results[variant_name].append(scores)
            print(f"  {variant_name}: {scores}")

    # Aggregate
    print("\n--- Aggregate Ablation Results ---")
    for variant, score_list in results.items():
        if score_list:
            avg = {m: sum(s[m] for s in score_list) / len(score_list) for m in metrics}
            print(f"  {variant}: {avg}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="ResearchOS Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Experiments:
  1        Design quality benchmark (baselines vs full pipeline)
  2        Proposal benchmark (expand to Level B)
  3        Future-grounded evaluation
  5        Judge reliability
  ablation Ablation studies
  all      Run all experiments
""",
    )
    parser.add_argument(
        "--experiment", type=str, default="all",
        help="Which experiment to run: 1, 2, 3, 5, ablation, all",
    )
    parser.add_argument(
        "--goals-file", type=str, default=None,
        help="Path to JSON file with research goals",
    )
    parser.add_argument(
        "--graph-file", type=str, default=None,
        help="Path to JSON file with serialized SkillGraph",
    )
    parser.add_argument(
        "--future-papers-file", type=str, default=None,
        help="Path to JSON file with future papers (for experiment 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Load or build graph
    if args.graph_file and os.path.isfile(args.graph_file):
        with open(args.graph_file) as f:
            graph = SkillGraph.from_dict(json.load(f))
        print(f"Loaded graph from {args.graph_file}: {graph.stats()}")
    else:
        graph = build_demo_graph()
        print(f"Using demo graph: {graph.stats()}")

    # Load goals
    goals = load_goals(args.goals_file)
    print(f"Loaded {len(goals)} goals")

    # Load future papers if provided
    future_papers = None
    if args.future_papers_file and os.path.isfile(args.future_papers_file):
        with open(args.future_papers_file) as f:
            future_papers = json.load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run selected experiments
    experiment = args.experiment.lower()

    if experiment in ("1", "all"):
        df = run_design_quality_benchmark(goals, graph)
        df.to_csv(os.path.join(args.output_dir, "exp1_design_quality.csv"), index=False)

    if experiment in ("2", "all"):
        results = run_proposal_benchmark(goals, graph)
        with open(os.path.join(args.output_dir, "exp2_proposals.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    if experiment in ("3", "all"):
        results = run_future_grounded(goals, graph, future_papers)
        with open(os.path.join(args.output_dir, "exp3_future_grounded.json"), "w") as f:
            json.dump(results, f, indent=2)

    if experiment in ("5", "all"):
        results = run_judge_reliability(goals, graph)
        with open(os.path.join(args.output_dir, "exp5_judge_reliability.json"), "w") as f:
            json.dump(results, f, indent=2)

    if experiment in ("ablation", "all"):
        results = run_ablations(goals, graph)
        with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
