# ResearchOS

## Project Overview

ResearchOS is an agentic research system that extracts structured "research skills" from scientific papers, builds a relational skill graph, and composes novel research designs through constraint-aware skill composition.

**Target venue:** NeurIPS 2026

## Repository Structure

```
ResearchOS/
├── paper/                    # LaTeX paper (NeurIPS 2026 format)
│   ├── main.tex              # Main file
│   ├── sections/             # Modular sections (0_abstract ~ A_appendix)
│   ├── refs.bib              # Bibliography
│   ├── build.sh              # Compile: outputs ResearchOS.pdf
│   └── .claude/skills/       # Paper writing skills
├── src/                      # Core Python library
│   ├── schema.py             # Pydantic data models (ResearchSkill, DesignCard, etc.)
│   ├── skill_compiler.py     # Paper → Skill extraction (LLM-based)
│   ├── skill_graph.py        # Skill graph with 6 typed edge relations
│   ├── pipeline.py           # End-to-end paper→skill→KG workflow
│   ├── composer.py           # Constraint-aware skill composition engine
│   ├── validator.py          # Multi-dimensional design validation
│   ├── evaluator.py          # 9-metric evaluation + Elo + future-grounded check
│   └── baselines.py          # 5 baseline implementations
├── run_pipeline.py           # CLI: paper→skill→KG pipeline (--demo for no-API test)
├── run_experiment.py         # CLI: run paper experiments
├── data/skillgraph/          # Generated knowledge graph outputs
│   ├── skill_knowledge_graph.json
│   ├── skill_graph.graphml   # For Gephi/Cytoscape
│   ├── skill_graph.html      # Interactive D3.js visualization
│   └── taxonomy.json         # Hierarchical capability tree
├── conversation.jsonl        # Design conversation log
└── requirements.txt
```

## Key Concepts

- **Research Skill**: A structured method primitive with 15 fields (goal, mechanism, assumptions, inputs/outputs, empirical gains, failure modes, compatibility/conflict, etc.)
- **Skill Graph**: Directed graph with 6 edge types: PREREQUISITE, ENHANCES, SUBSTITUTES, CONFLICTS, COMPOSES, REFINES
- **Composition**: Goal decomposition → skill retrieval → DAG construction → gap-bridging → self-critique
- **Three-level output**: Design Card (Level A) → Research Proposal (Level B) → Executable Experiment Plan (Level C)

## Development Commands

```bash
# Run demo pipeline (no API key needed)
python run_pipeline.py --demo

# Process papers from directory
ANTHROPIC_API_KEY=... python run_pipeline.py --papers data/papers/

# Compile paper
cd paper && bash build.sh

# Run experiments
python run_experiment.py --experiment design_quality
```

## Conventions

- Paper compiles with `pdflatex + bibtex` (3-pass), outputs `paper/ResearchOS.pdf`
- LLM calls use `src/utils.call_llm()` — set `RESEARCHOS_MOCK=1` for mock mode
- All data models in `src/schema.py` use Pydantic v2
- Skill IDs are UUIDs by default; demo uses short readable IDs (e.g., "cot", "tot")
- Edge confidence threshold: 0.6 (configurable in pipeline)

## Current Status (2026-04-10)

- **Paper**: Complete draft (21 pages), all sections written with placeholder experimental results
- **Code**: Full pipeline implemented, demo runs end-to-end
- **Next steps**:
  1. Collect LLM reasoning papers into `data/papers/`
  2. Run real skill extraction with API key
  3. Replace placeholder results with actual experimental data
  4. Generate publication-quality figures (pipeline diagram, graph visualization)

## References

- SkillNet (paper2skills.com, arXiv:2603.04448): Three-layer skill infrastructure (taxonomy → relation graph → skill packages)
- AgentSkillOS (arXiv:2603.02176): Capability tree + DAG orchestration for 200K+ skills
