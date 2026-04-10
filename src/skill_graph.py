"""Skill Graph: stores and queries the network of research skills.

Provides graph operations including adding nodes/edges, inferring
relationships via LLM, neighborhood queries, compatibility checking,
embedding-based retrieval, and serialization.
"""

import json
from typing import Optional

from .schema import ResearchSkill, SkillEdge, EdgeType
from .utils import call_llm, get_embedding, cosine_similarity


# ---------------------------------------------------------------------------
# Prompt template for edge inference
# ---------------------------------------------------------------------------

EDGE_INFERENCE_PROMPT = """\
You are a research methodology expert. Given two research skills, determine \
the relationship(s) between them.

SKILL A:
Name: {a_name}
Goal: {a_goal}
Mechanism: {a_mechanism}
Level: {a_level}
Inputs: {a_inputs}
Outputs: {a_outputs}

SKILL B:
Name: {b_name}
Goal: {b_goal}
Mechanism: {b_mechanism}
Level: {b_level}
Inputs: {b_inputs}
Outputs: {b_outputs}

Possible relationship types (from A -> B):
- PREREQUISITE: A must be done before B can be applied
- ENHANCES: A improves the effectiveness of B
- SUBSTITUTES: A can replace B (they serve the same function)
- CONFLICTS: A and B cannot be used together
- COMPOSES: A and B can be combined into a pipeline
- REFINES: A is a more specific version of B

For each relationship that applies, provide:
- edge_type: one of the types above
- confidence: float between 0 and 1
- evidence: brief explanation

Return a JSON array of edge objects. If no relationship exists, return an empty array [].
Output ONLY valid JSON.
"""


class SkillGraph:
    """A directed graph of research skills connected by typed edges.

    Nodes are ResearchSkill objects indexed by ID. Edges are typed
    relationships (prerequisite, enhances, substitutes, etc.) with
    confidence scores.
    """

    def __init__(self):
        """Initialize an empty skill graph."""
        self.nodes: dict[str, ResearchSkill] = {}
        self.edges: list[SkillEdge] = []

    def add_skill(self, skill: ResearchSkill) -> None:
        """Add a skill node to the graph.

        Args:
            skill: The ResearchSkill to add.
        """
        self.nodes[skill.id] = skill

    def add_edge(self, edge: SkillEdge) -> None:
        """Add an edge to the graph with validation.

        Args:
            edge: The SkillEdge to add.

        Raises:
            ValueError: If source or target skill is not in the graph.
        """
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source skill '{edge.source_id}' not found in graph.")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target skill '{edge.target_id}' not found in graph.")
        self.edges.append(edge)

    def infer_edges(self, skill_a: ResearchSkill, skill_b: ResearchSkill,
                    model: str = "claude-sonnet-4-20250514") -> list[SkillEdge]:
        """Use LLM to infer relationship edges between two skills.

        Args:
            skill_a: First skill.
            skill_b: Second skill.
            model: LLM model to use.

        Returns:
            List of inferred SkillEdge objects.
        """
        prompt = EDGE_INFERENCE_PROMPT.format(
            a_name=skill_a.name, a_goal=skill_a.goal,
            a_mechanism=skill_a.mechanism, a_level=skill_a.level.value,
            a_inputs=", ".join(skill_a.inputs), a_outputs=", ".join(skill_a.outputs),
            b_name=skill_b.name, b_goal=skill_b.goal,
            b_mechanism=skill_b.mechanism, b_level=skill_b.level.value,
            b_inputs=", ".join(skill_b.inputs), b_outputs=", ".join(skill_b.outputs),
        )

        response = call_llm(
            prompt=prompt,
            model=model,
            system="You are a research methodology expert. Return only valid JSON.",
        )

        edges = []
        try:
            import re
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)

            if not isinstance(data, list):
                data = [data]

            for item in data:
                try:
                    edge = SkillEdge(
                        source_id=skill_a.id,
                        target_id=skill_b.id,
                        edge_type=EdgeType(item["edge_type"]),
                        confidence=float(item.get("confidence", 0.5)),
                        evidence=item.get("evidence", ""),
                    )
                    edges.append(edge)
                except (ValueError, KeyError) as e:
                    print(f"[SkillGraph] Could not parse edge: {e}")
        except json.JSONDecodeError:
            print("[SkillGraph] Failed to parse edge inference response as JSON.")

        return edges

    def get_neighbors(self, skill_id: str,
                      edge_types: Optional[list[EdgeType]] = None
                      ) -> list[tuple[ResearchSkill, SkillEdge]]:
        """Get all neighboring skills connected by edges of given types.

        Searches both outgoing and incoming edges.

        Args:
            skill_id: ID of the skill to query.
            edge_types: Filter to these edge types (None = all types).

        Returns:
            List of (neighbor_skill, edge) tuples.
        """
        neighbors = []
        for edge in self.edges:
            if edge_types and edge.edge_type not in edge_types:
                continue

            if edge.source_id == skill_id and edge.target_id in self.nodes:
                neighbors.append((self.nodes[edge.target_id], edge))
            elif edge.target_id == skill_id and edge.source_id in self.nodes:
                neighbors.append((self.nodes[edge.source_id], edge))

        return neighbors

    def get_substitutes(self, skill_id: str) -> list[ResearchSkill]:
        """Get all skills that can substitute for the given skill.

        Args:
            skill_id: ID of the skill.

        Returns:
            List of substitute ResearchSkill objects.
        """
        pairs = self.get_neighbors(skill_id, [EdgeType.SUBSTITUTES])
        return [skill for skill, _ in pairs]

    def get_prerequisites(self, skill_id: str) -> list[ResearchSkill]:
        """Get all prerequisite skills for the given skill.

        Returns skills that are connected via PREREQUISITE edges pointing
        toward the given skill (i.e. the prerequisite is the source).

        Args:
            skill_id: ID of the skill.

        Returns:
            List of prerequisite ResearchSkill objects.
        """
        prereqs = []
        for edge in self.edges:
            if (edge.target_id == skill_id
                    and edge.edge_type == EdgeType.PREREQUISITE
                    and edge.source_id in self.nodes):
                prereqs.append(self.nodes[edge.source_id])
        return prereqs

    def check_compatibility(self, skill_ids: list[str]) -> tuple[bool, list[str]]:
        """Check if a set of skills has no CONFLICTS edges among them.

        Args:
            skill_ids: List of skill IDs to check.

        Returns:
            Tuple of (is_compatible, list_of_conflict_descriptions).
        """
        conflicts = []
        id_set = set(skill_ids)

        for edge in self.edges:
            if edge.edge_type == EdgeType.CONFLICTS:
                if edge.source_id in id_set and edge.target_id in id_set:
                    src_name = self.nodes.get(edge.source_id, None)
                    tgt_name = self.nodes.get(edge.target_id, None)
                    src_label = src_name.name if src_name else edge.source_id
                    tgt_label = tgt_name.name if tgt_name else edge.target_id
                    desc = f"CONFLICT: '{src_label}' <-> '{tgt_label}': {edge.evidence}"
                    conflicts.append(desc)

        return (len(conflicts) == 0, conflicts)

    def retrieve_skills(self, goal: str, top_k: int = 10) -> list[ResearchSkill]:
        """Retrieve skills relevant to a research goal.

        Uses embedding similarity to rank skills, then expands via graph
        neighborhood to include related skills.

        Args:
            goal: Research goal description.
            top_k: Number of skills to return.

        Returns:
            List of relevant ResearchSkill objects, ranked by relevance.
        """
        if not self.nodes:
            return []

        goal_emb = get_embedding(goal)

        # Score each skill by embedding similarity
        scored: list[tuple[float, ResearchSkill]] = []
        for skill in self.nodes.values():
            skill_text = f"{skill.name} {skill.goal} {skill.mechanism}"
            skill_emb = get_embedding(skill_text)
            sim = cosine_similarity(goal_emb, skill_emb)
            scored.append((sim, skill))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top-k/2 by embedding, then expand via graph neighbors
        initial_count = max(top_k // 2, 1)
        result_ids: set[str] = set()
        result: list[ResearchSkill] = []

        for _, skill in scored[:initial_count]:
            if skill.id not in result_ids:
                result_ids.add(skill.id)
                result.append(skill)

        # Expand via neighbors of top skills
        for skill in list(result):
            neighbors = self.get_neighbors(
                skill.id,
                [EdgeType.ENHANCES, EdgeType.COMPOSES, EdgeType.PREREQUISITE],
            )
            for neighbor_skill, _ in neighbors:
                if neighbor_skill.id not in result_ids and len(result) < top_k:
                    result_ids.add(neighbor_skill.id)
                    result.append(neighbor_skill)

        # Fill remaining slots from embedding ranking
        for _, skill in scored:
            if skill.id not in result_ids and len(result) < top_k:
                result_ids.add(skill.id)
                result.append(skill)

        return result[:top_k]

    def to_dict(self) -> dict:
        """Serialize the graph to a dictionary.

        Returns:
            Dict with 'nodes' and 'edges' keys.
        """
        return {
            "nodes": {sid: skill.model_dump() for sid, skill in self.nodes.items()},
            "edges": [edge.model_dump() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillGraph":
        """Deserialize a graph from a dictionary.

        Args:
            data: Dict with 'nodes' and 'edges' keys.

        Returns:
            Reconstructed SkillGraph instance.
        """
        graph = cls()
        for sid, node_data in data.get("nodes", {}).items():
            skill = ResearchSkill(**node_data)
            graph.nodes[sid] = skill
        for edge_data in data.get("edges", []):
            edge = SkillEdge(**edge_data)
            graph.edges.append(edge)
        return graph

    def stats(self) -> dict:
        """Return summary statistics about the graph.

        Returns:
            Dict with node_count, edge_count, edges_by_type, skill_levels,
            and avg_edges_per_node.
        """
        edges_by_type: dict[str, int] = {}
        for edge in self.edges:
            key = edge.edge_type.value
            edges_by_type[key] = edges_by_type.get(key, 0) + 1

        levels: dict[str, int] = {}
        for skill in self.nodes.values():
            key = skill.level.value
            levels[key] = levels.get(key, 0) + 1

        node_count = len(self.nodes)
        return {
            "node_count": node_count,
            "edge_count": len(self.edges),
            "edges_by_type": edges_by_type,
            "skill_levels": levels,
            "avg_edges_per_node": len(self.edges) / node_count if node_count else 0.0,
        }
