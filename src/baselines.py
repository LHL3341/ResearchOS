"""Baseline methods for comparison against the full ResearchOS pipeline.

Implements five baselines of increasing sophistication:
1. Direct prompting (no structure)
2. RAG over paper chunks
3. Flat skill list (skills without graph)
4. Graph retrieval only (graph but no composition planning)
5. Random composition (random skill selection)
"""

import json
import random
import re
from typing import Optional

from .schema import ResearchSkill, DesignCard
from .skill_graph import SkillGraph
from .utils import call_llm, get_embedding, cosine_similarity


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DIRECT_PROMPTING_TEMPLATE = """\
You are a research design expert. Given the following research goal and \
paper abstracts, propose a novel research design.

Research Goal: {goal}

Related Paper Abstracts:
{abstracts}

Propose a research design as a JSON object with:
- research_goal: the goal
- composition_rationale: why this approach works
- novel_bridging: what new mechanisms are introduced
- expected_gain: expected improvement
- assumptions: list of assumptions
- risks: list of risks

Output ONLY valid JSON.
"""

RAG_PROMPTING_TEMPLATE = """\
You are a research design expert. Given the following research goal and \
the most relevant passages from related papers, propose a novel research design.

Research Goal: {goal}

Most Relevant Passages:
{chunks}

Propose a research design as a JSON object with:
- research_goal: the goal
- composition_rationale: why this approach works
- novel_bridging: what new mechanisms are introduced
- expected_gain: expected improvement
- assumptions: list of assumptions
- risks: list of risks

Output ONLY valid JSON.
"""

FLAT_SKILL_TEMPLATE = """\
You are a research design expert. Given the following research goal and \
a list of available research skills, propose a novel research design that \
composes some of these skills.

Research Goal: {goal}

Available Skills:
{skills_text}

Propose a research design as a JSON object with:
- research_goal: the goal
- selected_skills: list of skill names you chose
- composition_rationale: why these skills work together
- novel_bridging: what new mechanisms bridge gaps between skills
- expected_gain: expected improvement
- assumptions: list of assumptions
- risks: list of risks

Output ONLY valid JSON.
"""

GRAPH_RETRIEVAL_TEMPLATE = """\
You are a research design expert. Given the following research goal and \
skills retrieved from a knowledge graph (with their relationships), \
propose a novel research design.

Research Goal: {goal}

Retrieved Skills and Relationships:
{skills_with_edges}

Propose a research design as a JSON object with:
- research_goal: the goal
- selected_skills: list of skill names you chose
- composition_rationale: why these skills work together
- novel_bridging: what new mechanisms bridge gaps between skills
- expected_gain: expected improvement
- assumptions: list of assumptions
- risks: list of risks

Output ONLY valid JSON.
"""


def _parse_design_response(response: str, goal: str) -> DesignCard:
    """Parse an LLM response into a DesignCard.

    Args:
        response: Raw LLM response (expected JSON).
        goal: The research goal.

    Returns:
        A DesignCard instance.
    """
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(response)
    except json.JSONDecodeError:
        data = {}

    return DesignCard(
        research_goal=data.get("research_goal", goal),
        selected_skills=data.get("selected_skills", []),
        composition_rationale=data.get("composition_rationale", ""),
        novel_bridging=data.get("novel_bridging", ""),
        expected_gain=data.get("expected_gain", ""),
        assumptions=data.get("assumptions", []),
        risks=data.get("risks", []),
    )


def direct_prompting(goal: str, paper_abstracts: list[str],
                     model: str = "claude-sonnet-4-20250514") -> DesignCard:
    """Baseline 1: Directly prompt LLM with goal and paper abstracts.

    No skill extraction, no graph structure. The LLM sees raw abstracts
    and generates a design in one shot.

    Args:
        goal: Research goal.
        paper_abstracts: List of paper abstract strings.
        model: LLM model to use.

    Returns:
        A DesignCard.
    """
    abstracts_text = "\n\n".join(
        f"[Paper {i+1}]: {a}" for i, a in enumerate(paper_abstracts[:20])
    )

    prompt = DIRECT_PROMPTING_TEMPLATE.format(goal=goal, abstracts=abstracts_text)
    response = call_llm(prompt=prompt, model=model,
                        system="You are a research design expert. Return only valid JSON.")
    return _parse_design_response(response, goal)


def rag_over_papers(goal: str, paper_chunks: list[str],
                    model: str = "claude-sonnet-4-20250514") -> DesignCard:
    """Baseline 2: Retrieve relevant paper chunks then generate design.

    Uses embedding similarity to select the most relevant chunks before
    prompting. No skill extraction.

    Args:
        goal: Research goal.
        paper_chunks: List of paper text chunks.
        model: LLM model to use.

    Returns:
        A DesignCard.
    """
    if not paper_chunks:
        return DesignCard(research_goal=goal)

    # Retrieve top-k chunks by embedding similarity
    goal_emb = get_embedding(goal)
    scored = []
    for chunk in paper_chunks:
        chunk_emb = get_embedding(chunk)
        sim = cosine_similarity(goal_emb, chunk_emb)
        scored.append((sim, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:10]]

    chunks_text = "\n\n".join(
        f"[Chunk {i+1}]: {c}" for i, c in enumerate(top_chunks)
    )

    prompt = RAG_PROMPTING_TEMPLATE.format(goal=goal, chunks=chunks_text)
    response = call_llm(prompt=prompt, model=model,
                        system="You are a research design expert. Return only valid JSON.")
    return _parse_design_response(response, goal)


def flat_skill_list(goal: str, skills: list[ResearchSkill],
                    model: str = "claude-sonnet-4-20250514") -> DesignCard:
    """Baseline 3: Give skill list without graph structure.

    Skills are presented as a flat list with no relationship information.
    The LLM must figure out composition on its own.

    Args:
        goal: Research goal.
        skills: List of ResearchSkill objects.
        model: LLM model to use.

    Returns:
        A DesignCard.
    """
    skills_text = "\n\n".join(
        f"Skill: {s.name}\n  Goal: {s.goal}\n  Mechanism: {s.mechanism[:300]}\n"
        f"  Level: {s.level.value}\n  Inputs: {', '.join(s.inputs)}\n"
        f"  Outputs: {', '.join(s.outputs)}"
        for s in skills[:30]
    )

    prompt = FLAT_SKILL_TEMPLATE.format(goal=goal, skills_text=skills_text)
    response = call_llm(prompt=prompt, model=model,
                        system="You are a research design expert. Return only valid JSON.")
    return _parse_design_response(response, goal)


def graph_retrieval_only(goal: str, graph: SkillGraph,
                         model: str = "claude-sonnet-4-20250514") -> DesignCard:
    """Baseline 4: Retrieve from graph but no composition planning.

    Uses the graph for retrieval (embedding + neighborhood expansion) and
    presents skills with their edges, but does not perform DAG construction,
    gap analysis, or bridging.

    Args:
        goal: Research goal.
        graph: The SkillGraph to retrieve from.
        model: LLM model to use.

    Returns:
        A DesignCard.
    """
    skills = graph.retrieve_skills(goal, top_k=10)

    parts = []
    for skill in skills:
        neighbors = graph.get_neighbors(skill.id)
        neighbor_info = "; ".join(
            f"{edge.edge_type.value} -> {n.name}" for n, edge in neighbors[:5]
        )
        parts.append(
            f"Skill: {skill.name}\n  Goal: {skill.goal}\n"
            f"  Mechanism: {skill.mechanism[:300]}\n"
            f"  Relationships: {neighbor_info or 'none'}"
        )

    skills_with_edges = "\n\n".join(parts)
    prompt = GRAPH_RETRIEVAL_TEMPLATE.format(goal=goal, skills_with_edges=skills_with_edges)
    response = call_llm(prompt=prompt, model=model,
                        system="You are a research design expert. Return only valid JSON.")
    return _parse_design_response(response, goal)


def random_composition(goal: str, graph: SkillGraph, k: int = 5) -> DesignCard:
    """Baseline 5: Randomly select k non-conflicting skills.

    No LLM involvement -- just random selection with conflict avoidance.

    Args:
        goal: Research goal.
        graph: The SkillGraph to select from.
        k: Number of skills to select.

    Returns:
        A DesignCard with randomly selected skills.
    """
    all_skills = list(graph.nodes.values())
    if not all_skills:
        return DesignCard(research_goal=goal)

    random.shuffle(all_skills)

    selected: list[ResearchSkill] = []
    selected_ids: list[str] = []

    for skill in all_skills:
        candidate_ids = selected_ids + [skill.id]
        is_compatible, _ = graph.check_compatibility(candidate_ids)
        if is_compatible:
            selected.append(skill)
            selected_ids.append(skill.id)
        if len(selected) >= k:
            break

    return DesignCard(
        research_goal=goal,
        selected_skills=selected_ids,
        composition_rationale=f"Randomly selected {len(selected)} non-conflicting skills.",
        novel_bridging="None (random baseline).",
        expected_gain="Unknown (random baseline).",
        assumptions=["Random selection may not be coherent."],
        risks=["Skills may not address the goal.", "No composition planning was done."],
    )
