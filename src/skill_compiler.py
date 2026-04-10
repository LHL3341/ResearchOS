"""Skill Compiler: extracts ResearchSkill objects from academic papers.

Parses papers into sections, uses LLM to identify and structure
research skills, deduplicates against existing skills, and performs
quality checks on extracted skills.
"""

import re
from typing import Optional

from .schema import ResearchSkill, SkillLevel
from .utils import call_llm, get_embedding, cosine_similarity


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SKILL_EXTRACTION_PROMPT = """\
You are a research skill extractor. Given the METHOD section of a research paper, \
identify all distinct, reusable research techniques ("skills") described.

For each skill, provide a JSON object with the following fields:
- name: Short descriptive name
- goal: What this skill aims to achieve
- mechanism: How the skill works, with enough detail that someone could reimplement it
- assumptions: List of assumptions required for the skill to work
- inputs: Required inputs (data, models, etc.)
- outputs: What the skill produces
- level: One of "training-time", "inference-time", "data", "architecture", "prompting"
- required_resources: Compute, data, or other resource requirements
- empirical_gains: Reported improvements from the paper
- failure_modes: Known failure modes or limitations
- tags: Free-form tags for categorization

PAPER ID: {paper_id}

METHOD SECTION:
{method_text}

INTRODUCTION (for context):
{intro_text}

EXPERIMENTS (for empirical gains):
{experiments_text}

LIMITATIONS (for failure modes):
{limitations_text}

Return a JSON array of skill objects. Output ONLY valid JSON, no commentary.
"""

QUALITY_CHECK_PROMPT = """\
You are a research methodology reviewer. Evaluate whether the following \
research skill schema contains enough information for someone to reimplement \
the technique without access to the original paper.

SKILL:
Name: {name}
Goal: {goal}
Mechanism: {mechanism}
Assumptions: {assumptions}
Inputs: {inputs}
Outputs: {outputs}
Level: {level}
Required Resources: {required_resources}

Rate the following on a 1-5 scale:
1. Reconstructability: Could someone reimplement this from the schema alone?
2. Completeness: Are all necessary details present?
3. Clarity: Is the description unambiguous?

Return JSON with fields: reconstructability (int 1-5), completeness (int 1-5), \
clarity (int 1-5), overall_score (float), feedback (str with improvement suggestions).
"""


class SkillCompiler:
    """Extracts and structures research skills from academic papers.

    Uses LLM-based extraction to identify discrete techniques from paper
    text, structures them into ResearchSkill objects, deduplicates against
    existing skills, and quality-checks the results.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """Initialize the SkillCompiler.

        Args:
            model: LLM model identifier for extraction and quality checks.
        """
        self.model = model

    def parse_paper(self, paper_text: str, paper_id: str) -> dict:
        """Split a paper into sections using regex heuristics.

        Attempts to identify introduction, method, experiments, related work,
        and limitation sections. Falls back to treating the full text as the
        method section if no structure is detected.

        Args:
            paper_text: Full text of the paper.
            paper_id: Identifier for the paper.

        Returns:
            Dict with keys: paper_id, intro, method, experiments,
            related_work, limitation, full_text.
        """
        sections = {
            "paper_id": paper_id,
            "intro": "",
            "method": "",
            "experiments": "",
            "related_work": "",
            "limitation": "",
            "full_text": paper_text,
        }

        # Regex patterns for common section headings
        patterns = {
            "intro": r"(?i)(?:^|\n)#{0,3}\s*\d*\.?\s*introduction\s*\n",
            "method": r"(?i)(?:^|\n)#{0,3}\s*\d*\.?\s*(?:method(?:ology|s)?|approach|proposed\s+method|our\s+approach|framework)\s*\n",
            "experiments": r"(?i)(?:^|\n)#{0,3}\s*\d*\.?\s*(?:experiment(?:s|al)?(?:\s+(?:results|setup))?|results|evaluation)\s*\n",
            "related_work": r"(?i)(?:^|\n)#{0,3}\s*\d*\.?\s*(?:related\s+work|background|prior\s+work|literature\s+review)\s*\n",
            "limitation": r"(?i)(?:^|\n)#{0,3}\s*\d*\.?\s*(?:limitation(?:s)?|discussion|conclusion(?:s)?)\s*\n",
        }

        # Find section boundaries
        boundaries: list[tuple[str, int]] = []
        for section_name, pattern in patterns.items():
            match = re.search(pattern, paper_text)
            if match:
                boundaries.append((section_name, match.start()))

        if not boundaries:
            # No sections found — treat entire text as method
            sections["method"] = paper_text
            return sections

        # Sort by position
        boundaries.sort(key=lambda x: x[1])

        # Extract text between boundaries
        for i, (name, start) in enumerate(boundaries):
            end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(paper_text)
            sections[name] = paper_text[start:end].strip()

        # If method section is empty, use full text
        if not sections["method"]:
            sections["method"] = paper_text

        return sections

    def extract_skills(self, parsed_paper: dict, paper_id: str) -> list[ResearchSkill]:
        """Extract candidate research skills from a parsed paper.

        Uses LLM to identify techniques in the method section, cross-referencing
        other sections for empirical gains and failure modes.

        Args:
            parsed_paper: Output of parse_paper().
            paper_id: Identifier for the paper.

        Returns:
            List of extracted ResearchSkill objects.
        """
        prompt = SKILL_EXTRACTION_PROMPT.format(
            paper_id=paper_id,
            method_text=parsed_paper.get("method", "")[:6000],
            intro_text=parsed_paper.get("intro", "")[:2000],
            experiments_text=parsed_paper.get("experiments", "")[:3000],
            limitations_text=parsed_paper.get("limitation", "")[:2000],
        )

        response = call_llm(
            prompt=prompt,
            model=self.model,
            system="You are a precise research skill extractor. Return only valid JSON.",
        )

        skills = self._parse_skills_response(response, paper_id)
        return skills

    def _parse_skills_response(self, response: str, paper_id: str) -> list[ResearchSkill]:
        """Parse the LLM response into ResearchSkill objects.

        Args:
            response: Raw LLM response text (expected JSON array).
            paper_id: Paper ID to attach to each skill.

        Returns:
            List of ResearchSkill objects.
        """
        import json

        # Try to extract JSON from the response
        try:
            # Look for JSON array in the response
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                data = json.loads(response)
        except json.JSONDecodeError:
            print(f"[SkillCompiler] Failed to parse LLM response as JSON for paper {paper_id}")
            return []

        if not isinstance(data, list):
            data = [data]

        skills = []
        for item in data:
            try:
                # Map level string to enum
                level_str = item.get("level", "prompting")
                try:
                    level = SkillLevel(level_str)
                except ValueError:
                    level = SkillLevel.PROMPTING

                skill = ResearchSkill(
                    name=item.get("name", "Unknown Skill"),
                    goal=item.get("goal", ""),
                    mechanism=item.get("mechanism", ""),
                    assumptions=item.get("assumptions", []),
                    inputs=item.get("inputs", []),
                    outputs=item.get("outputs", []),
                    level=level,
                    required_resources=item.get("required_resources", ""),
                    empirical_gains=item.get("empirical_gains", []),
                    failure_modes=item.get("failure_modes", []),
                    source_papers=[paper_id],
                    tags=item.get("tags", []),
                )
                skills.append(skill)
            except Exception as e:
                print(f"[SkillCompiler] Failed to create ResearchSkill from item: {e}")

        return skills

    def deduplicate(self, new_skills: list[ResearchSkill],
                    existing_skills: list[ResearchSkill]) -> list[ResearchSkill]:
        """Deduplicate new skills against existing ones using embedding similarity.

        Skills with >0.9 cosine similarity to an existing skill are merged
        (the existing skill absorbs source papers and empirical gains).

        Args:
            new_skills: Newly extracted skills to check.
            existing_skills: Existing skill library to deduplicate against.

        Returns:
            List of genuinely new skills (after merging duplicates).
        """
        if not existing_skills:
            return new_skills

        # Precompute embeddings for existing skills
        existing_embeddings = [
            get_embedding(f"{s.name} {s.goal} {s.mechanism}")
            for s in existing_skills
        ]

        unique_skills = []
        for new_skill in new_skills:
            new_emb = get_embedding(f"{new_skill.name} {new_skill.goal} {new_skill.mechanism}")

            is_duplicate = False
            for i, existing_emb in enumerate(existing_embeddings):
                sim = cosine_similarity(new_emb, existing_emb)
                if sim > 0.9:
                    # Merge into existing skill
                    existing = existing_skills[i]
                    existing.source_papers = list(set(existing.source_papers + new_skill.source_papers))
                    existing.empirical_gains = list(set(existing.empirical_gains + new_skill.empirical_gains))
                    existing.failure_modes = list(set(existing.failure_modes + new_skill.failure_modes))
                    is_duplicate = True
                    print(f"[SkillCompiler] Merged duplicate: '{new_skill.name}' -> '{existing.name}' (sim={sim:.3f})")
                    break

            if not is_duplicate:
                unique_skills.append(new_skill)

        return unique_skills

    def quality_check(self, skill: ResearchSkill) -> dict:
        """Assess the quality and reconstructability of a skill schema.

        Uses LLM to evaluate whether the skill contains enough information
        for someone to reimplement the technique without the original paper.

        Args:
            skill: The ResearchSkill to evaluate.

        Returns:
            Dict with keys: reconstructability (1-5), completeness (1-5),
            clarity (1-5), overall_score (float), feedback (str).
        """
        prompt = QUALITY_CHECK_PROMPT.format(
            name=skill.name,
            goal=skill.goal,
            mechanism=skill.mechanism,
            assumptions=", ".join(skill.assumptions),
            inputs=", ".join(skill.inputs),
            outputs=", ".join(skill.outputs),
            level=skill.level.value,
            required_resources=skill.required_resources,
        )

        response = call_llm(
            prompt=prompt,
            model=self.model,
            system="You are a methodology reviewer. Return only valid JSON.",
        )

        import json
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "reconstructability": 3,
                "completeness": 3,
                "clarity": 3,
                "overall_score": 3.0,
                "feedback": "Could not parse quality check response.",
            }

        return result
