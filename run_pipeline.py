#!/usr/bin/env python3
"""Run the Paper-to-Skill-to-KnowledgeGraph pipeline.

Usage:
    # Process papers from a directory
    python run_pipeline.py --papers data/papers/ --output data/skillgraph/

    # Process a single paper
    python run_pipeline.py --paper data/papers/cot.txt --output data/skillgraph/

    # Use demo data (no papers needed)
    python run_pipeline.py --demo --output data/skillgraph/

    # Only build edges on existing graph
    python run_pipeline.py --edges-only --output data/skillgraph/

    # Only export existing graph
    python run_pipeline.py --export-only --output data/skillgraph/
"""

import argparse
import json
import logging
from pathlib import Path

from src.pipeline import Paper2SkillPipeline, TaxonomyNode
from src.schema import ResearchSkill, SkillLevel, SkillEdge, EdgeType
from src.skill_graph import SkillGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def build_demo_graph() -> Paper2SkillPipeline:
    """Build a demo pipeline with hand-crafted skills for illustration.

    Creates ~15 skills from the LLM reasoning domain with known
    relationships, without requiring LLM API calls.
    """
    pipeline = Paper2SkillPipeline(output_dir="data/skillgraph")

    # -- Skills --
    skills = [
        ResearchSkill(
            id="cot", name="Chain-of-Thought Prompting",
            goal="Improve LLM multi-step reasoning by eliciting intermediate steps",
            mechanism="Insert step-by-step reasoning demonstrations in the prompt before the query, guiding the model to produce intermediate derivation steps rather than jumping to the answer.",
            assumptions=["Model >= 60B parameters", "Task is decomposable into steps"],
            inputs=["Prompt template", "Few-shot exemplars with reasoning chains", "Query"],
            outputs=["Model output with explicit reasoning chain"],
            level=SkillLevel.PROMPTING,
            required_resources="No training; inference tokens increase 3-5x",
            empirical_gains=["GSM8K: 58% -> 75% (PaLM 540B)", "SVAMP: +15% (GPT-3.5)"],
            failure_modes=["Unfaithful chains", "Small models show no gain", "Increased latency"],
            compatibility=["sc", "verification", "retrieval"],
            conflict=["direct_answer"],
            source_papers=["wei2022chain", "kojima2022zeroshotcot"],
            tags=["reasoning", "prompting", "few-shot"],
        ),
        ResearchSkill(
            id="zerocot", name="Zero-shot CoT",
            goal="Enable chain-of-thought reasoning without demonstrations",
            mechanism="Append 'Let's think step by step' to the prompt to trigger reasoning without few-shot exemplars.",
            assumptions=["Model >= 100B parameters"],
            inputs=["Query with trigger phrase"],
            outputs=["Model output with reasoning chain"],
            level=SkillLevel.PROMPTING,
            empirical_gains=["MultiArith: 78.7% (zero-shot CoT vs 17.7% standard)"],
            failure_modes=["Lower quality chains than few-shot CoT", "Inconsistent activation"],
            source_papers=["kojima2022zeroshotcot"],
            tags=["reasoning", "prompting", "zero-shot"],
        ),
        ResearchSkill(
            id="sc", name="Self-Consistency",
            goal="Improve reasoning reliability via majority voting over multiple chains",
            mechanism="Sample multiple reasoning chains from the model with temperature > 0, extract the final answer from each, and return the most frequent answer (majority vote).",
            assumptions=["CoT is effective for the task", "Multiple samples are affordable"],
            inputs=["CoT prompt", "Number of samples K", "Temperature"],
            outputs=["Majority-voted answer", "Consistency score"],
            level=SkillLevel.INFERENCE_TIME,
            required_resources="K times the compute of single CoT (typically K=5-40)",
            empirical_gains=["GSM8K: +6-8% over single CoT", "ARC-c: +3%"],
            failure_modes=["Fails when all chains share systematic errors", "High compute cost"],
            compatibility=["cot", "tot"],
            source_papers=["wang2023selfconsistency"],
            tags=["reasoning", "ensemble", "voting"],
        ),
        ResearchSkill(
            id="tot", name="Tree of Thoughts",
            goal="Enable deliberate planning via tree-structured reasoning exploration",
            mechanism="Decompose problem into thought steps, generate multiple candidates per step, evaluate each with a value function, and search the tree (BFS/DFS) to find the best reasoning path.",
            assumptions=["Task has evaluable intermediate states", "Search budget available"],
            inputs=["Problem", "Thought decomposition strategy", "Value function", "Search algorithm"],
            outputs=["Best reasoning path with justification"],
            level=SkillLevel.INFERENCE_TIME,
            required_resources="10-100x compute of single CoT depending on branching/depth",
            empirical_gains=["Game of 24: 4% -> 74%", "Creative writing: improved coherence"],
            failure_modes=["Exponential cost", "Value function accuracy bottleneck"],
            compatibility=["sc", "verification"],
            conflict=["direct_answer"],
            source_papers=["yao2023tree"],
            tags=["reasoning", "search", "planning"],
        ),
        ResearchSkill(
            id="got", name="Graph of Thoughts",
            goal="Enable non-linear reasoning via graph-structured thought exploration",
            mechanism="Extend Tree of Thoughts by allowing merging and refining of thoughts, creating a DAG/graph structure instead of a tree. Supports aggregation of partial solutions.",
            assumptions=["Problem benefits from combining partial solutions"],
            inputs=["Problem", "Graph operations (generate, aggregate, refine)", "Scoring function"],
            outputs=["Optimized reasoning graph with solution"],
            level=SkillLevel.INFERENCE_TIME,
            empirical_gains=["Sorting: improved over ToT", "Set operations: +15% vs ToT"],
            failure_modes=["Complex implementation", "Graph explosion"],
            source_papers=["besta2024got"],
            tags=["reasoning", "search", "graph"],
        ),
        ResearchSkill(
            id="verification", name="Step-level Verification",
            goal="Detect and correct errors in intermediate reasoning steps",
            mechanism="Train or prompt a verifier model to score each intermediate step of a reasoning chain for correctness. Use process reward model (PRM) to assign step-level rewards.",
            assumptions=["Step-level annotations available", "Verifier is more accurate than generator"],
            inputs=["Reasoning chain", "Step boundaries", "Verifier model"],
            outputs=["Per-step correctness scores", "Identified error locations"],
            level=SkillLevel.INFERENCE_TIME,
            required_resources="Separate verifier model (can be smaller); step-level training data",
            empirical_gains=["MATH: +5-10% with process supervision vs outcome supervision"],
            failure_modes=["Verifier errors cascade", "Annotation cost for training"],
            compatibility=["cot", "tot", "sc"],
            source_papers=["lightman2023lets"],
            tags=["reasoning", "verification", "reward-model"],
        ),
        ResearchSkill(
            id="reflexion", name="Reflexion",
            goal="Enable LLM self-improvement through verbal self-reflection",
            mechanism="After task failure, generate a verbal reflection analyzing what went wrong, store it in memory, and use it to inform the next attempt. Iterates until success or budget exhausted.",
            assumptions=["Environment provides success/failure signal", "Model can self-diagnose"],
            inputs=["Task", "Environment feedback", "Reflection memory"],
            outputs=["Improved action trajectory", "Accumulated reflections"],
            level=SkillLevel.INFERENCE_TIME,
            empirical_gains=["AlfWorld: 134% improvement", "HotpotQA: +14%"],
            failure_modes=["Reflection quality degrades with weak models", "Infinite retry loops"],
            source_papers=["shinn2023reflexion"],
            tags=["reasoning", "self-improvement", "reflection"],
        ),
        ResearchSkill(
            id="selfrefine", name="Self-Refine",
            goal="Iteratively improve LLM outputs via self-feedback",
            mechanism="Generate initial output, then prompt the same model to critique it, then refine based on the critique. Repeat for N rounds.",
            assumptions=["Model can generate useful self-feedback"],
            inputs=["Initial output", "Critique prompt", "Refinement prompt"],
            outputs=["Refined output after N iterations"],
            level=SkillLevel.INFERENCE_TIME,
            empirical_gains=["Code: +13% pass rate", "Math reasoning: +5%"],
            failure_modes=["Diminishing returns after 2-3 rounds", "Self-feedback can be incorrect"],
            compatibility=["cot", "verification"],
            source_papers=["madaan2023selfrefine"],
            tags=["reasoning", "self-improvement", "refinement"],
        ),
        ResearchSkill(
            id="pal", name="Program-Aided Language Models",
            goal="Offload computation to code execution for faithful reasoning",
            mechanism="Prompt LLM to generate Python code that solves the reasoning problem, then execute the code to obtain the answer. Separates reasoning (LLM) from computation (interpreter).",
            assumptions=["Problem is expressible as code", "Code execution environment available"],
            inputs=["Problem description", "Code generation prompt", "Python interpreter"],
            outputs=["Generated code", "Execution result"],
            level=SkillLevel.PROMPTING,
            empirical_gains=["GSM8K: +12% over CoT", "Colored Objects: +30%"],
            failure_modes=["Code generation errors", "Not applicable to non-computational tasks"],
            conflict=["cot"],
            source_papers=["gao2023pal"],
            tags=["reasoning", "tool-use", "code-generation"],
        ),
        ResearchSkill(
            id="pot", name="Program of Thoughts",
            goal="Disentangle computation from reasoning via code",
            mechanism="Prompt LLM to express reasoning as interleaved natural language thoughts and executable code statements. Execute code parts, feed results back as context.",
            assumptions=["Problem has computational sub-tasks", "Code interpreter available"],
            inputs=["Problem", "Interleaved prompt template"],
            outputs=["Thought-code trace", "Computed answer"],
            level=SkillLevel.PROMPTING,
            empirical_gains=["GSM8K: +8% over CoT", "FinQA: +10%"],
            failure_modes=["Complex reasoning not easily expressible as code"],
            compatibility=["verification"],
            source_papers=["chen2023program"],
            tags=["reasoning", "tool-use", "code-generation"],
        ),
        ResearchSkill(
            id="retrieval", name="Evidence-Grounded Retrieval for Reasoning",
            goal="Ground reasoning steps in retrieved external evidence",
            mechanism="At each reasoning step, retrieve relevant evidence from a knowledge corpus. The model conditions its next reasoning step on both the chain so far and the retrieved evidence.",
            assumptions=["Relevant knowledge corpus exists", "Retriever is accurate enough"],
            inputs=["Reasoning chain so far", "Knowledge corpus", "Retriever model"],
            outputs=["Retrieved evidence", "Evidence-grounded reasoning step"],
            level=SkillLevel.INFERENCE_TIME,
            empirical_gains=["Multi-hop QA: +8-15% factual accuracy"],
            failure_modes=["Retriever noise", "Corpus coverage gaps", "Increased latency"],
            compatibility=["cot", "verification"],
            source_papers=["asai2024selfrag"],
            tags=["reasoning", "retrieval", "grounding"],
        ),
        ResearchSkill(
            id="tool_use", name="Tool-Augmented Reasoning",
            goal="Extend LLM capabilities by calling external tools during reasoning",
            mechanism="Teach or prompt the model to emit tool calls (calculator, search, code interpreter) at appropriate points in the reasoning chain, then inject tool outputs back into the context.",
            assumptions=["Tools are available and accessible", "Model can learn tool call syntax"],
            inputs=["Problem", "Available tool descriptions", "Tool call format specification"],
            outputs=["Reasoning trace with tool calls and results"],
            level=SkillLevel.PROMPTING,
            empirical_gains=["Math: near-perfect with calculator", "QA: +20% with search"],
            failure_modes=["Wrong tool selection", "Tool API changes", "Hallucinated tool calls"],
            compatibility=["cot", "pal"],
            source_papers=["schick2023toolformer"],
            tags=["reasoning", "tool-use", "augmentation"],
        ),
        ResearchSkill(
            id="direct_answer", name="Direct Answer Prompting",
            goal="Obtain answer directly without intermediate reasoning",
            mechanism="Standard prompting: ask the question and expect a direct answer without chain-of-thought or other scaffolding.",
            assumptions=["Task is simple enough for direct recall"],
            inputs=["Query"],
            outputs=["Direct answer"],
            level=SkillLevel.PROMPTING,
            failure_modes=["Fails on multi-step reasoning", "No interpretability"],
            conflict=["cot", "tot"],
            source_papers=[],
            tags=["baseline", "prompting", "direct"],
        ),
        ResearchSkill(
            id="debate", name="Multi-Agent Debate",
            goal="Improve reasoning quality through adversarial multi-agent discussion",
            mechanism="Multiple LLM instances independently solve a problem, then iteratively debate their answers, challenging each other's reasoning. After N rounds, converge on a consensus answer.",
            assumptions=["Multiple LLM calls affordable", "Diversity in initial solutions"],
            inputs=["Problem", "Number of agents", "Number of debate rounds"],
            outputs=["Consensus answer", "Debate transcript"],
            level=SkillLevel.INFERENCE_TIME,
            empirical_gains=["MMLU: +5% over single model", "Math: +8%"],
            failure_modes=["Echo chamber with identical models", "High compute cost"],
            compatibility=["cot", "sc"],
            source_papers=[],
            tags=["reasoning", "multi-agent", "debate"],
        ),
        ResearchSkill(
            id="prm", name="Process Reward Model",
            goal="Provide dense step-level feedback for reasoning chain generation",
            mechanism="Train a reward model on step-level human annotations to score each intermediate step. Use as a value function for best-of-N selection or tree search guidance.",
            assumptions=["Step-level annotations available", "Reward model generalizes to test distribution"],
            inputs=["Reasoning chain with step boundaries", "Trained PRM model"],
            outputs=["Per-step reward scores", "Aggregate chain quality score"],
            level=SkillLevel.TRAINING_TIME,
            required_resources="Step-level annotation dataset (800K+ labels); GPU training for reward model",
            empirical_gains=["MATH: process supervision > outcome supervision (+8%)"],
            failure_modes=["Annotation cost", "Reward hacking", "Distribution shift"],
            compatibility=["verification", "tot"],
            source_papers=["lightman2023lets"],
            tags=["reasoning", "reward-model", "process-supervision"],
        ),
    ]

    for skill in skills:
        pipeline.graph.add_skill(skill)

    # -- Edges --
    edge_defs = [
        # REFINES
        ("zerocot", "cot", EdgeType.REFINES, 0.9, "Zero-shot CoT is a simplified variant of few-shot CoT"),
        ("got", "tot", EdgeType.REFINES, 0.85, "GoT extends ToT with graph structure"),
        ("pot", "pal", EdgeType.REFINES, 0.8, "PoT refines PAL with interleaved reasoning"),
        # ENHANCES
        ("sc", "cot", EdgeType.ENHANCES, 0.95, "Self-consistency improves CoT via majority voting"),
        ("verification", "cot", EdgeType.ENHANCES, 0.9, "Step verification catches CoT errors"),
        ("retrieval", "cot", EdgeType.ENHANCES, 0.8, "Evidence grounding reduces hallucination in CoT"),
        ("prm", "tot", EdgeType.ENHANCES, 0.85, "PRM provides value function for tree search"),
        ("prm", "verification", EdgeType.ENHANCES, 0.9, "PRM is a trained step-level verifier"),
        # SUBSTITUTES
        ("cot", "direct_answer", EdgeType.SUBSTITUTES, 0.7, "CoT replaces direct answering for complex tasks"),
        ("pal", "cot", EdgeType.SUBSTITUTES, 0.6, "PAL substitutes natural language chains with code"),
        ("tot", "cot", EdgeType.SUBSTITUTES, 0.5, "ToT is a more expensive alternative to CoT for hard problems"),
        ("selfrefine", "reflexion", EdgeType.SUBSTITUTES, 0.7, "Both do iterative self-improvement"),
        ("debate", "sc", EdgeType.SUBSTITUTES, 0.5, "Both use multiple solutions; debate via adversarial discussion"),
        # CONFLICTS
        ("pal", "direct_answer", EdgeType.CONFLICTS, 0.9, "PAL requires structured code generation"),
        ("cot", "direct_answer", EdgeType.CONFLICTS, 0.95, "CoT and direct answering are mutually exclusive strategies"),
        # COMPOSES
        ("cot", "sc", EdgeType.COMPOSES, 0.95, "CoT + Self-Consistency is a standard pipeline"),
        ("cot", "verification", EdgeType.COMPOSES, 0.9, "Generate chain then verify steps"),
        ("cot", "retrieval", EdgeType.COMPOSES, 0.8, "Retrieve evidence to ground each CoT step"),
        ("cot", "tool_use", EdgeType.COMPOSES, 0.8, "Use tools within a CoT reasoning chain"),
        ("verification", "reflexion", EdgeType.COMPOSES, 0.7, "Verify then reflect on failures"),
        ("tot", "verification", EdgeType.COMPOSES, 0.8, "Use verifier as value function in tree search"),
        # PREREQUISITE
        ("cot", "sc", EdgeType.PREREQUISITE, 0.95, "SC requires CoT to generate multiple chains"),
        ("cot", "verification", EdgeType.PREREQUISITE, 0.8, "Verification requires a reasoning chain to verify"),
    ]

    for src, tgt, etype, conf, evidence in edge_defs:
        pipeline.graph.add_edge(SkillEdge(
            source_id=src, target_id=tgt,
            edge_type=etype, confidence=conf, evidence=evidence,
        ))

    # Register dummy papers
    pipeline.paper_registry = {
        "wei2022chain": {"paper_id": "wei2022chain", "title": "Chain-of-Thought Prompting", "year": 2022},
        "kojima2022zeroshotcot": {"paper_id": "kojima2022zeroshotcot", "title": "Zero-shot CoT", "year": 2022},
        "wang2023selfconsistency": {"paper_id": "wang2023selfconsistency", "title": "Self-Consistency", "year": 2023},
        "yao2023tree": {"paper_id": "yao2023tree", "title": "Tree of Thoughts", "year": 2023},
        "besta2024got": {"paper_id": "besta2024got", "title": "Graph of Thoughts", "year": 2024},
        "lightman2023lets": {"paper_id": "lightman2023lets", "title": "Let's Verify Step by Step", "year": 2023},
        "shinn2023reflexion": {"paper_id": "shinn2023reflexion", "title": "Reflexion", "year": 2023},
        "madaan2023selfrefine": {"paper_id": "madaan2023selfrefine", "title": "Self-Refine", "year": 2023},
        "gao2023pal": {"paper_id": "gao2023pal", "title": "PAL", "year": 2023},
        "chen2023program": {"paper_id": "chen2023program", "title": "Program of Thoughts", "year": 2023},
        "asai2024selfrag": {"paper_id": "asai2024selfrag", "title": "Self-RAG", "year": 2024},
        "schick2023toolformer": {"paper_id": "schick2023toolformer", "title": "Toolformer", "year": 2023},
    }

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Paper-to-Skill-to-KnowledgeGraph Pipeline")
    parser.add_argument("--papers", type=str, help="Directory of paper text files to process")
    parser.add_argument("--paper", type=str, help="Single paper file to process")
    parser.add_argument("--output", type=str, default="data/skillgraph", help="Output directory")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="LLM model")
    parser.add_argument("--demo", action="store_true", help="Run with demo data (no API needed)")
    parser.add_argument("--edges-only", action="store_true", help="Only infer edges on existing graph")
    parser.add_argument("--export-only", action="store_true", help="Only export existing graph")
    args = parser.parse_args()

    if args.demo:
        logger.info("Building demo skill graph...")
        pipeline = build_demo_graph()
        pipeline.build_taxonomy()
        pipeline.save_state()
        outputs = pipeline.export()
        stats = pipeline.graph.stats()
        logger.info(f"Demo graph: {stats}")
        logger.info(f"Outputs: {outputs}")
        return

    pipeline = Paper2SkillPipeline(output_dir=args.output, model=args.model)

    if args.export_only:
        outputs = pipeline.export()
        logger.info(f"Exported: {outputs}")
        return

    if args.edges_only:
        edge_count = pipeline.build_edges()
        pipeline.build_taxonomy()
        pipeline.save_state()
        outputs = pipeline.export()
        logger.info(f"Added {edge_count} edges. Exported: {outputs}")
        return

    if args.paper:
        pipeline.ingest_paper(args.paper)
        pipeline.build_edges()
        pipeline.build_taxonomy()
        pipeline.save_state()
        outputs = pipeline.export()
        logger.info(f"Outputs: {outputs}")
    elif args.papers:
        summary = pipeline.run(args.papers)
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")
    else:
        parser.print_help()
        print("\nTip: use --demo for a quick demo without API keys.")


if __name__ == "__main__":
    main()
