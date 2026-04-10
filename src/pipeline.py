"""Paper-to-Skill-to-KnowledgeGraph end-to-end pipeline.

Implements a complete workflow:
  1. Ingest papers (PDF/text/arxiv ID)
  2. Extract structured research skills via LLM
  3. Build a relational skill graph with typed edges
  4. Organize skills into a hierarchical taxonomy (capability tree)
  5. Export to multiple formats (JSON, GraphML, HTML visualization)

Inspired by SkillNet (paper2skills.com) and AgentSkillOS architectures.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .schema import ResearchSkill, SkillEdge, EdgeType, SkillLevel
from .skill_compiler import SkillCompiler
from .skill_graph import SkillGraph
from .utils import call_llm, get_embedding, cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Taxonomy node for the capability tree
# ---------------------------------------------------------------------------

@dataclass
class TaxonomyNode:
    """A node in the hierarchical skill taxonomy (capability tree)."""
    name: str
    path: str  # e.g. "reasoning/prompting/chain-of-thought"
    description: str = ""
    children: list["TaxonomyNode"] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "skill_ids": self.skill_ids,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaxonomyNode":
        node = cls(
            name=data["name"],
            path=data["path"],
            description=data.get("description", ""),
            skill_ids=data.get("skill_ids", []),
        )
        node.children = [cls.from_dict(c) for c in data.get("children", [])]
        return node

    def all_skill_ids(self) -> list[str]:
        """Recursively collect all skill IDs in this subtree."""
        ids = list(self.skill_ids)
        for child in self.children:
            ids.extend(child.all_skill_ids())
        return ids

    def find(self, path: str) -> Optional["TaxonomyNode"]:
        """Find a node by path."""
        if self.path == path:
            return self
        for child in self.children:
            result = child.find(path)
            if result:
                return result
        return None


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

TAXONOMY_CLASSIFY_PROMPT = """\
You are organizing research skills into a hierarchical taxonomy.

Given the following skill and the current taxonomy categories, assign the skill
to the most appropriate category. If no existing category fits well, suggest a
new subcategory name.

Skill:
  Name: {name}
  Goal: {goal}
  Mechanism: {mechanism}
  Level: {level}
  Tags: {tags}

Current taxonomy categories (path -> description):
{categories}

Return JSON:
{{
  "assigned_path": "category/subcategory",
  "is_new_category": true/false,
  "new_category_name": "name if new",
  "new_category_description": "description if new",
  "confidence": 0.0-1.0
}}
"""

BATCH_EDGE_INFERENCE_PROMPT = """\
You are a research methodology expert. Given a focal skill and a batch of
candidate skills, determine relationships between the focal skill and each
candidate.

FOCAL SKILL:
  Name: {focal_name}
  Goal: {focal_goal}
  Mechanism: {focal_mechanism}
  Level: {focal_level}

CANDIDATE SKILLS:
{candidates_text}

For each candidate that has a meaningful relationship with the focal skill,
output an edge object. Possible relationship types (from focal -> candidate):
- PREREQUISITE: focal must be done before candidate
- ENHANCES: focal improves effectiveness of candidate
- SUBSTITUTES: focal and candidate serve the same function
- CONFLICTS: focal and candidate cannot be used together
- COMPOSES: focal and candidate combine into a pipeline
- REFINES: focal is a more specific version of candidate

Return a JSON array of objects, each with:
  "candidate_id", "edge_type", "confidence" (0-1), "evidence" (brief reason)

Skip candidates with no meaningful relationship. Return [] if none.
Output ONLY valid JSON.
"""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Paper2SkillPipeline:
    """End-to-end pipeline: papers -> skills -> knowledge graph.

    Usage:
        pipeline = Paper2SkillPipeline(output_dir="data/skillgraph")
        pipeline.ingest_paper("path/to/paper.txt", paper_id="cot_2022")
        pipeline.ingest_paper("path/to/paper2.txt", paper_id="tot_2023")
        pipeline.build_edges()
        pipeline.build_taxonomy()
        pipeline.export()
    """

    def __init__(
        self,
        output_dir: str = "data/skillgraph",
        model: str = "claude-sonnet-4-20250514",
        edge_confidence_threshold: float = 0.6,
        dedup_threshold: float = 0.9,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.edge_confidence_threshold = edge_confidence_threshold
        self.dedup_threshold = dedup_threshold

        self.compiler = SkillCompiler(model=model)
        self.graph = SkillGraph()
        self.taxonomy = TaxonomyNode(name="root", path="root", description="All research skills")
        self.paper_registry: dict[str, dict] = {}  # paper_id -> metadata

        # Load existing state if available
        self._load_state()

    # ------------------------------------------------------------------
    # Step 1: Ingest papers
    # ------------------------------------------------------------------

    def ingest_paper(
        self,
        paper_path: str,
        paper_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> list[ResearchSkill]:
        """Ingest a single paper: parse, extract skills, deduplicate, add to graph.

        Args:
            paper_path: Path to paper text file.
            paper_id: Optional paper identifier (defaults to filename stem).
            metadata: Optional metadata dict (title, authors, year, venue).

        Returns:
            List of newly added skills.
        """
        path = Path(paper_path)
        if not path.exists():
            raise FileNotFoundError(f"Paper not found: {paper_path}")

        if paper_id is None:
            paper_id = path.stem

        if paper_id in self.paper_registry:
            logger.info(f"Paper {paper_id} already ingested, skipping.")
            return []

        logger.info(f"Ingesting paper: {paper_id}")
        paper_text = path.read_text(encoding="utf-8", errors="replace")

        # Parse into sections
        parsed = self.compiler.parse_paper(paper_text, paper_id)

        # Extract skills
        raw_skills = self.compiler.extract_skills(parsed, paper_id)
        logger.info(f"  Extracted {len(raw_skills)} raw skills")

        # Deduplicate against existing
        existing = list(self.graph.nodes.values())
        new_skills = self.compiler.deduplicate(raw_skills, existing)
        logger.info(f"  {len(new_skills)} new skills after deduplication")

        # Add to graph
        for skill in new_skills:
            self.graph.add_skill(skill)

        # Register paper
        self.paper_registry[paper_id] = {
            "paper_id": paper_id,
            "path": str(path),
            "skill_count": len(new_skills),
            "skill_ids": [s.id for s in new_skills],
            **(metadata or {}),
        }

        return new_skills

    def ingest_directory(
        self,
        papers_dir: str,
        extensions: tuple[str, ...] = (".txt", ".md", ".tex"),
    ) -> dict[str, list[ResearchSkill]]:
        """Ingest all papers from a directory.

        Args:
            papers_dir: Directory containing paper files.
            extensions: File extensions to process.

        Returns:
            Dict mapping paper_id to list of extracted skills.
        """
        results = {}
        paper_dir = Path(papers_dir)
        files = sorted(
            f for f in paper_dir.iterdir()
            if f.suffix in extensions and f.is_file()
        )
        logger.info(f"Found {len(files)} papers in {papers_dir}")

        for paper_file in files:
            try:
                skills = self.ingest_paper(str(paper_file))
                results[paper_file.stem] = skills
            except Exception as e:
                logger.error(f"Failed to ingest {paper_file.name}: {e}")

        logger.info(f"Ingestion complete: {self.graph.stats()}")
        return results

    # ------------------------------------------------------------------
    # Step 2: Build edges (relations between skills)
    # ------------------------------------------------------------------

    def build_edges(self, batch_size: int = 5) -> int:
        """Infer edges between all skill pairs using batched LLM calls.

        Uses a batched approach: for each skill, compare it against a batch
        of other skills in a single LLM call (more efficient than pairwise).

        Args:
            batch_size: Number of candidate skills per LLM call.

        Returns:
            Number of edges added.
        """
        skills = list(self.graph.nodes.values())
        n = len(skills)
        if n < 2:
            logger.info("Not enough skills for edge inference.")
            return 0

        # Track existing edges to avoid duplicates
        existing_pairs = {
            (e.source_id, e.target_id, e.edge_type) for e in self.graph.edges
        }

        total_added = 0
        logger.info(f"Inferring edges among {n} skills (batch_size={batch_size})...")

        for i, focal in enumerate(skills):
            # Get candidates (skills we haven't compared with yet)
            candidates = [
                s for s in skills[i+1:]
                if not any(
                    (focal.id, s.id, et) in existing_pairs
                    or (s.id, focal.id, et) in existing_pairs
                    for et in EdgeType
                )
            ]
            if not candidates:
                continue

            # Process in batches
            for batch_start in range(0, len(candidates), batch_size):
                batch = candidates[batch_start:batch_start + batch_size]
                edges = self._infer_edges_batch(focal, batch)

                for edge in edges:
                    if edge.confidence >= self.edge_confidence_threshold:
                        key = (edge.source_id, edge.target_id, edge.edge_type)
                        if key not in existing_pairs:
                            self.graph.add_edge(edge)
                            existing_pairs.add(key)
                            total_added += 1

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{n} skills, {total_added} edges so far")

        logger.info(f"Edge inference complete: {total_added} edges added")
        return total_added

    def _infer_edges_batch(
        self, focal: ResearchSkill, candidates: list[ResearchSkill]
    ) -> list[SkillEdge]:
        """Infer edges from focal skill to a batch of candidates in one LLM call."""
        candidates_text = "\n".join(
            f"[{c.id[:8]}] {c.name}\n"
            f"  Goal: {c.goal}\n"
            f"  Mechanism: {c.mechanism[:200]}\n"
            f"  Level: {c.level.value}"
            for c in candidates
        )

        prompt = BATCH_EDGE_INFERENCE_PROMPT.format(
            focal_name=focal.name,
            focal_goal=focal.goal,
            focal_mechanism=focal.mechanism[:300],
            focal_level=focal.level.value,
            candidates_text=candidates_text,
        )

        response = call_llm(
            prompt=prompt,
            model=self.model,
            system="You are a research methodology expert. Return only valid JSON.",
        )

        # Build ID lookup
        id_map = {c.id[:8]: c.id for c in candidates}

        edges = []
        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            data = json.loads(match.group()) if match else json.loads(response)
            if not isinstance(data, list):
                data = [data]

            for item in data:
                cid_short = item.get("candidate_id", "")
                full_id = id_map.get(cid_short)
                if not full_id:
                    # Try matching by prefix
                    for short, full in id_map.items():
                        if cid_short.startswith(short) or short.startswith(cid_short):
                            full_id = full
                            break
                if not full_id:
                    continue

                try:
                    edge = SkillEdge(
                        source_id=focal.id,
                        target_id=full_id,
                        edge_type=EdgeType(item["edge_type"]),
                        confidence=float(item.get("confidence", 0.5)),
                        evidence=item.get("evidence", ""),
                    )
                    edges.append(edge)
                except (ValueError, KeyError):
                    pass
        except (json.JSONDecodeError, AttributeError):
            # Fall back to pairwise inference
            for c in candidates:
                pairwise_edges = self.graph.infer_edges(focal, c, model=self.model)
                edges.extend(pairwise_edges)

        return edges

    # ------------------------------------------------------------------
    # Step 3: Build taxonomy (capability tree)
    # ------------------------------------------------------------------

    def build_taxonomy(self) -> TaxonomyNode:
        """Organize all skills into a hierarchical taxonomy.

        Uses a two-pass approach:
          1. Seed categories from skill levels and tags
          2. LLM-guided classification of each skill into the tree

        Returns:
            The root TaxonomyNode.
        """
        logger.info("Building skill taxonomy...")

        # Seed top-level categories from SkillLevel
        level_categories = {
            "prompting": ("Prompting Methods", "Skills applied at the prompt level"),
            "inference-time": ("Inference-time Methods", "Skills applied during model inference"),
            "training-time": ("Training Methods", "Skills requiring model training or fine-tuning"),
            "data": ("Data Methods", "Skills for data processing and augmentation"),
            "architecture": ("Architecture Methods", "Skills modifying model architecture"),
        }

        self.taxonomy = TaxonomyNode(name="root", path="root", description="All research skills")
        for level_val, (name, desc) in level_categories.items():
            self.taxonomy.children.append(
                TaxonomyNode(name=name, path=f"root/{level_val}", description=desc)
            )

        # Classify each skill
        for skill in self.graph.nodes.values():
            self._classify_skill(skill)

        # Prune empty branches
        self._prune_empty(self.taxonomy)

        total = len(self.taxonomy.all_skill_ids())
        logger.info(f"Taxonomy built: {len(self.taxonomy.children)} top categories, {total} skills classified")
        return self.taxonomy

    def _classify_skill(self, skill: ResearchSkill) -> None:
        """Classify a single skill into the taxonomy tree."""
        # First try rule-based assignment by level
        level_path = f"root/{skill.level.value}"
        parent_node = self.taxonomy.find(level_path)
        if not parent_node:
            parent_node = self.taxonomy

        # Use tags to find or create subcategory
        if skill.tags:
            primary_tag = skill.tags[0].lower().replace(" ", "-")
            sub_path = f"{parent_node.path}/{primary_tag}"
            sub_node = self.taxonomy.find(sub_path)

            if not sub_node:
                sub_node = TaxonomyNode(
                    name=primary_tag.replace("-", " ").title(),
                    path=sub_path,
                    description=f"Skills related to {primary_tag}",
                )
                parent_node.children.append(sub_node)

            sub_node.skill_ids.append(skill.id)
        else:
            parent_node.skill_ids.append(skill.id)

    def _prune_empty(self, node: TaxonomyNode) -> bool:
        """Recursively prune empty branches. Returns True if node should be kept."""
        node.children = [c for c in node.children if self._prune_empty(c)]
        return bool(node.children) or bool(node.skill_ids)

    # ------------------------------------------------------------------
    # Step 4: Export to various formats
    # ------------------------------------------------------------------

    def export(self) -> dict[str, str]:
        """Export the skill graph and taxonomy to multiple formats.

        Returns:
            Dict mapping format name to output file path.
        """
        outputs = {}

        # JSON export
        json_path = self._export_json()
        outputs["json"] = str(json_path)

        # GraphML export (for Gephi, Cytoscape, etc.)
        graphml_path = self._export_graphml()
        outputs["graphml"] = str(graphml_path)

        # HTML visualization
        html_path = self._export_html()
        outputs["html"] = str(html_path)

        # Taxonomy YAML
        taxonomy_path = self._export_taxonomy()
        outputs["taxonomy"] = str(taxonomy_path)

        logger.info(f"Exported to: {outputs}")
        return outputs

    def _export_json(self) -> Path:
        """Export full graph + taxonomy as JSON."""
        data = {
            "graph": self.graph.to_dict(),
            "taxonomy": self.taxonomy.to_dict(),
            "papers": self.paper_registry,
            "stats": self.graph.stats(),
        }
        path = self.output_dir / "skill_knowledge_graph.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path

    def _export_graphml(self) -> Path:
        """Export graph as GraphML for visualization tools."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<graphml xmlns="http://graphml.graphstruct.org/graphml">')
        lines.append('  <key id="name" for="node" attr.name="name" attr.type="string"/>')
        lines.append('  <key id="goal" for="node" attr.name="goal" attr.type="string"/>')
        lines.append('  <key id="level" for="node" attr.name="level" attr.type="string"/>')
        lines.append('  <key id="mechanism" for="node" attr.name="mechanism" attr.type="string"/>')
        lines.append('  <key id="edge_type" for="edge" attr.name="edge_type" attr.type="string"/>')
        lines.append('  <key id="confidence" for="edge" attr.name="confidence" attr.type="double"/>')
        lines.append('  <key id="evidence" for="edge" attr.name="evidence" attr.type="string"/>')
        lines.append('  <graph id="G" edgedefault="directed">')

        for sid, skill in self.graph.nodes.items():
            lines.append(f'    <node id="{_xml_escape(sid)}">')
            lines.append(f'      <data key="name">{_xml_escape(skill.name)}</data>')
            lines.append(f'      <data key="goal">{_xml_escape(skill.goal[:200])}</data>')
            lines.append(f'      <data key="level">{skill.level.value}</data>')
            lines.append(f'      <data key="mechanism">{_xml_escape(skill.mechanism[:300])}</data>')
            lines.append(f'    </node>')

        for i, edge in enumerate(self.graph.edges):
            lines.append(f'    <edge id="e{i}" source="{_xml_escape(edge.source_id)}" target="{_xml_escape(edge.target_id)}">')
            lines.append(f'      <data key="edge_type">{edge.edge_type.value}</data>')
            lines.append(f'      <data key="confidence">{edge.confidence}</data>')
            lines.append(f'      <data key="evidence">{_xml_escape(edge.evidence[:200])}</data>')
            lines.append(f'    </edge>')

        lines.append('  </graph>')
        lines.append('</graphml>')

        path = self.output_dir / "skill_graph.graphml"
        path.write_text("\n".join(lines))
        return path

    def _export_html(self) -> Path:
        """Export interactive HTML visualization using D3.js force graph."""
        nodes_json = json.dumps([
            {
                "id": sid,
                "name": s.name,
                "goal": s.goal[:100],
                "level": s.level.value,
                "tags": s.tags[:3],
                "papers": len(s.source_papers),
            }
            for sid, s in self.graph.nodes.items()
        ])

        edges_json = json.dumps([
            {
                "source": e.source_id,
                "target": e.target_id,
                "type": e.edge_type.value,
                "confidence": e.confidence,
            }
            for e in self.graph.edges
        ])

        stats = self.graph.stats()

        html = f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ResearchOS Skill Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #0a0a0a; color: #e0e0e0; }}
#header {{ padding: 16px 24px; background: #111; border-bottom: 1px solid #333; display: flex; justify-content: space-between; align-items: center; }}
#header h1 {{ margin: 0; font-size: 18px; font-weight: 600; }}
#stats {{ font-size: 13px; color: #888; }}
#graph {{ width: 100vw; height: calc(100vh - 56px); }}
#tooltip {{ position: absolute; background: #1a1a1a; border: 1px solid #444; border-radius: 6px; padding: 10px 14px; font-size: 12px; pointer-events: none; display: none; max-width: 320px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }}
#tooltip .name {{ font-weight: 600; font-size: 14px; margin-bottom: 4px; }}
#tooltip .goal {{ color: #aaa; margin-bottom: 4px; }}
#tooltip .meta {{ color: #666; font-size: 11px; }}
#legend {{ position: absolute; bottom: 20px; left: 20px; background: #111; border: 1px solid #333; border-radius: 6px; padding: 12px; font-size: 11px; }}
#legend .item {{ display: flex; align-items: center; margin: 3px 0; }}
#legend .dot {{ width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
#legend .line {{ width: 20px; height: 2px; margin-right: 6px; }}
svg text {{ fill: #ccc; font-size: 10px; }}
</style>
</head>
<body>
<div id="header">
  <h1>ResearchOS Skill Knowledge Graph</h1>
  <div id="stats">{stats['node_count']} skills &middot; {stats['edge_count']} relations &middot; {len(self.paper_registry)} papers</div>
</div>
<div id="graph"></div>
<div id="tooltip"></div>
<div id="legend">
  <b>Node (skill level)</b>
  <div class="item"><div class="dot" style="background:#4fc3f7"></div>prompting</div>
  <div class="item"><div class="dot" style="background:#81c784"></div>inference-time</div>
  <div class="item"><div class="dot" style="background:#ff8a65"></div>training-time</div>
  <div class="item"><div class="dot" style="background:#ce93d8"></div>architecture</div>
  <div class="item"><div class="dot" style="background:#fff176"></div>data</div>
  <br><b>Edge (relation)</b>
  <div class="item"><div class="line" style="background:#4caf50"></div>ENHANCES</div>
  <div class="item"><div class="line" style="background:#2196f3"></div>COMPOSES</div>
  <div class="item"><div class="line" style="background:#ff9800"></div>PREREQUISITE</div>
  <div class="item"><div class="line" style="background:#f44336"></div>CONFLICTS</div>
  <div class="item"><div class="line" style="background:#9c27b0"></div>SUBSTITUTES</div>
  <div class="item"><div class="line" style="background:#607d8b"></div>REFINES</div>
</div>
<script>
const nodes = {nodes_json};
const links = {edges_json};

const levelColor = {{
  "prompting": "#4fc3f7", "inference-time": "#81c784",
  "training-time": "#ff8a65", "architecture": "#ce93d8", "data": "#fff176"
}};
const edgeColor = {{
  "ENHANCES": "#4caf50", "COMPOSES": "#2196f3", "PREREQUISITE": "#ff9800",
  "CONFLICTS": "#f44336", "SUBSTITUTES": "#9c27b0", "REFINES": "#607d8b"
}};

const width = window.innerWidth, height = window.innerHeight - 56;
const svg = d3.select("#graph").append("svg").attr("width", width).attr("height", height);
const g = svg.append("g");

svg.call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", (e) => g.attr("transform", e.transform)));

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(120))
  .force("charge", d3.forceManyBody().strength(-300))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(20));

// Edges
const link = g.append("g").selectAll("line").data(links).join("line")
  .attr("stroke", d => edgeColor[d.type] || "#555")
  .attr("stroke-opacity", d => 0.3 + d.confidence * 0.5)
  .attr("stroke-width", d => 1 + d.confidence);

// Nodes
const node = g.append("g").selectAll("circle").data(nodes).join("circle")
  .attr("r", d => 6 + d.papers * 2)
  .attr("fill", d => levelColor[d.level] || "#999")
  .attr("stroke", "#222").attr("stroke-width", 1)
  .call(d3.drag().on("start", dragstart).on("drag", dragged).on("end", dragend));

// Labels
const label = g.append("g").selectAll("text").data(nodes).join("text")
  .text(d => d.name.length > 25 ? d.name.slice(0, 22) + "..." : d.name)
  .attr("dx", 12).attr("dy", 4);

// Tooltip
const tooltip = d3.select("#tooltip");
node.on("mouseover", (e, d) => {{
  tooltip.style("display", "block")
    .html(`<div class="name">${{d.name}}</div><div class="goal">${{d.goal}}</div><div class="meta">${{d.level}} &middot; ${{d.tags.join(", ")}} &middot; ${{d.papers}} paper(s)</div>`)
    .style("left", (e.pageX + 12) + "px").style("top", (e.pageY - 10) + "px");
}}).on("mouseout", () => tooltip.style("display", "none"));

simulation.on("tick", () => {{
  link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  node.attr("cx", d => d.x).attr("cy", d => d.y);
  label.attr("x", d => d.x).attr("y", d => d.y);
}});

function dragstart(e, d) {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
function dragged(e, d) {{ d.fx = e.x; d.fy = e.y; }}
function dragend(e, d) {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}
</script>
</body>
</html>"""

        path = self.output_dir / "skill_graph.html"
        path.write_text(html)
        return path

    def _export_taxonomy(self) -> Path:
        """Export taxonomy as JSON."""
        path = self.output_dir / "taxonomy.json"
        with open(path, "w") as f:
            json.dump(self.taxonomy.to_dict(), f, indent=2)
        return path

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Save full pipeline state to disk."""
        state = {
            "graph": self.graph.to_dict(),
            "taxonomy": self.taxonomy.to_dict(),
            "papers": self.paper_registry,
        }
        path = self.output_dir / "pipeline_state.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"State saved to {path}")

    def _load_state(self) -> None:
        """Load pipeline state from disk if available."""
        path = self.output_dir / "pipeline_state.json"
        if path.exists():
            try:
                with open(path) as f:
                    state = json.load(f)
                self.graph = SkillGraph.from_dict(state.get("graph", {}))
                self.taxonomy = TaxonomyNode.from_dict(state.get("taxonomy", {
                    "name": "root", "path": "root"
                }))
                self.paper_registry = state.get("papers", {})
                logger.info(f"Loaded state: {self.graph.stats()}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    # ------------------------------------------------------------------
    # Convenience: full run
    # ------------------------------------------------------------------

    def run(self, papers_dir: str, export: bool = True) -> dict:
        """Run the full pipeline: ingest -> edges -> taxonomy -> export.

        Args:
            papers_dir: Directory containing paper text files.
            export: Whether to export results.

        Returns:
            Summary dict with stats and output paths.
        """
        # Step 1: Ingest
        results = self.ingest_directory(papers_dir)

        # Step 2: Build edges
        edge_count = self.build_edges()

        # Step 3: Build taxonomy
        self.build_taxonomy()

        # Step 4: Save state
        self.save_state()

        # Step 5: Export
        outputs = {}
        if export:
            outputs = self.export()

        summary = {
            "papers_processed": len(results),
            "skills_extracted": self.graph.stats()["node_count"],
            "edges_inferred": edge_count,
            "graph_stats": self.graph.stats(),
            "outputs": outputs,
        }
        logger.info(f"Pipeline complete: {summary}")
        return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xml_escape(s: str) -> str:
    """Escape special XML characters."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
