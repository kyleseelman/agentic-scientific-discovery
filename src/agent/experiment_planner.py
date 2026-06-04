from __future__ import annotations

import json
from typing import Any

from src.agent.schemas import ExperimentPlan, ExperimentStep, Hypothesis
from src.config import LLMBackend, get_config
from src.memory.retriever import MemoryRetriever
from src.tools.tool_retriever import ToolRetriever
from src.utils.json_extract import extract_json_object

_cfg = get_config()
_tool_retriever = ToolRetriever(
    use_gpu=_cfg.use_gpu_embeddings,
    device=_cfg.device,
    embedding_model=_cfg.embedding_model,
)


def plan_experiment(
    llm: LLMBackend,
    hypothesis: Hypothesis,
    dataset_summary: dict[str, Any],
    retriever: MemoryRetriever,
) -> ExperimentPlan:
    similar = retriever.similar_past_experiments(hypothesis.statement, k=2)
    sim_blob = "\n".join(f"- score={s.score:.2f} {s.text[:200]}" for s in similar)

    retrieval_query = f"{hypothesis.statement} {hypothesis.testable_prediction}"
    retrieved_tools_text = _tool_retriever.format_for_prompt(retrieval_query, top_k=10)

    known_tools = [t.name for t in _tool_retriever.retrieve(retrieval_query, top_k=12)]
    if "profile_dataset" not in known_tools:
        known_tools.insert(0, "profile_dataset")
    if "execute_code" not in known_tools:
        known_tools.append("execute_code")

    prompt = f"""
Design a computational experiment plan as ordered tool steps.

Hypothesis id: {hypothesis.id}
Statement: {hypothesis.statement}
Testable prediction: {hypothesis.testable_prediction}
Rationale: {hypothesis.rationale}

Dataset summary: {json.dumps(dataset_summary)[:2500]}

Similar past experiments:
{sim_blob}

Retrieved relevant tools (with descriptions):
{retrieved_tools_text}

Full tool list (use exact names):
{known_tools}

Note: You may also use "execute_code" to write custom Python analysis code
when the pre-defined tools don't cover the needed analysis. Code has access
to: expression (DataFrame), groups (Series), np, pd, plt, scipy.stats.
Variables persist across code executions within the same experiment.

Return JSON:
{{
  "hypothesis_id": "{hypothesis.id}",
  "steps": [
    {{"tool": "...", "params": {{}}, "description": "..."}}
  ],
  "expected_duration": "short|medium|long",
  "success_criteria": "...",
  "failure_criteria": "..."
}}

Rules:
- First step should usually profile the dataset unless already fully characterized.
- Include DE and pathway enrichment if the hypothesis concerns coordinated expression.
- Add at least one visualization when valid (volcano or PCA).
"""
    text = llm.generate(prompt, system="Return strictly valid JSON only.", temperature=0.35)
    raw = extract_json_object(text)
    steps_raw = raw.get("steps", [])
    steps: list[ExperimentStep] = []
    for s in steps_raw:
        steps.append(
            ExperimentStep(
                tool=str(s.get("tool", "")),
                params=dict(s.get("params", {})),
                description=str(s.get("description", "")),
            )
        )
    return ExperimentPlan(
        hypothesis_id=str(raw.get("hypothesis_id", hypothesis.id)),
        steps=steps,
        expected_duration=str(raw.get("expected_duration", "short")),
        success_criteria=str(raw.get("success_criteria", "")),
        failure_criteria=str(raw.get("failure_criteria", "")),
    )


def sanitize_plan(plan: ExperimentPlan, fallback_hypothesis_id: str) -> ExperimentPlan:
    allowed = {
        # Core analysis & statistics
        "profile_dataset",
        "differential_expression",
        "pathway_enrichment",
        "dimensionality_reduction",
        "feature_importance_variance",
        "group_comparison_summary",
        "correlation_analysis",
        "clustering_samples",
        # Visualization
        "plot_pca",
        "plot_volcano",
        "plot_heatmap",
        "plot_box_gene",
        # Databases
        "string_network",
        "uniprot_lookup",
        "gene_ontology_quickgo",
        # Literature
        "literature_pubmed",
        "literature_fetch_abstracts",
        "literature_search_biorxiv",
        # ML model building
        "train_classifier",
        "train_neural_network",
        "evaluate_model",
        "feature_selection",
        "train_gene_embeddings",
        "cross_validate_hypothesis",
        # LLM tools
        "finetune_text_classifier",
        "generate_with_llm",
        "embed_texts",
        "extract_entities_llm",
        "extract_architecture_from_paper",
        "list_recommended_models",
        "search_hf_models",
        "download_model",
        # Novel model building
        "build_architecture",
        "train_model_pipeline",
        "finetune_protein_lm",
        "finetune_genomic_lm",
        "build_graph_model",
        "design_from_paper",
        "benchmark_model",
        # Knowledge graph
        "query_knowledge_graph",
        "add_to_knowledge_graph",
        # Causal inference
        "causal_graph_discovery",
        "mediation_analysis",
        "instrumental_variable_analysis",
        "counterfactual_analysis",
        "interaction_network_analysis",
        # Code sandbox
        "execute_code",
    }
    cleaned: list[ExperimentStep] = []
    for st in plan.steps:
        if st.tool in allowed:
            cleaned.append(st)
    if not cleaned:
        cleaned = [
            ExperimentStep("profile_dataset", {}, "Profile dataset"),
            ExperimentStep("differential_expression", {"method": "welch_t", "correction": "fdr_bh"}, "DE"),
            ExperimentStep("pathway_enrichment", {"direction": "up_down", "correction": "fdr_bh"}, "Enrichment"),
            ExperimentStep("plot_volcano", {"fdr_threshold": 0.05}, "Volcano plot"),
        ]
    hid = plan.hypothesis_id or fallback_hypothesis_id
    return ExperimentPlan(
        hypothesis_id=hid,
        steps=cleaned,
        expected_duration=plan.expected_duration,
        success_criteria=plan.success_criteria,
        failure_criteria=plan.failure_criteria,
    )
