# Alzheimer's Neuroinflammation Investigation

Autonomous agentic investigation of **neuroinflammatory mechanisms in Alzheimer's
disease** using real brain transcriptomics data from NCBI GEO.

## Dataset

| Field | Value |
|-------|-------|
| Primary accession | [GSE5281](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE5281) |
| Organism | Homo sapiens |
| Tissue | Brain (multiple regions: entorhinal cortex, hippocampus, frontal cortex, etc.) |
| Comparison | Alzheimer's disease vs normal controls |
| Platform | Affymetrix HG-U133 Plus 2.0 |

GSE5281 is a well-characterized Alzheimer's disease brain transcriptomics dataset
with samples from multiple brain regions, enabling investigation of
neuroinflammatory signatures and immune dysregulation in AD pathology.

Fallback accessions (GSE44770, GSE33000) are attempted if the primary fails.

## Gene Sets

MSigDB Hallmark collection (50 curated pathway sets) filtered to genes present
in the expression matrix.

## Running

```bash
# Default (mock LLM — tests pipeline without API calls)
python examples/alzheimers_neuroinflammation/run.py

# With Gateway LLM (requires ML_GATEWAY_URL_CODEX)
LLM_PROVIDER=gateway python examples/alzheimers_neuroinflammation/run.py

# With Ollama (requires local Ollama server)
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2 python examples/alzheimers_neuroinflammation/run.py

# With OpenAI
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python examples/alzheimers_neuroinflammation/run.py
```

## Expected Outputs

After a successful run, the `session/` directory will contain:

```
session/
├── knowledge/          # Accumulated findings & hypotheses
├── papers/             # Literature scan results
├── outputs/            # Plots, DE tables, enrichment results
├── experiments.json    # Experiment execution log
├── orchestrator_state.json
└── run_summary.json    # Full cycle-by-cycle summary
```

The agent will autonomously:
1. Profile the dataset (sample counts, variance structure)
2. Run differential expression (AD vs control brain tissue)
3. Perform pathway enrichment (GSEA/ORA on Hallmark sets)
4. Generate and test hypotheses about neuroinflammatory mechanisms
5. Scan PubMed literature for supporting/contradicting evidence

Key pathways expected to emerge: complement system, TNF-alpha/NF-kB signaling,
interferon response, oxidative phosphorylation (mitochondrial dysfunction),
apoptosis, and IL6/JAK/STAT3 signaling.

## Scientific Context

Neuroinflammation is increasingly recognized as a central driver of Alzheimer's
disease pathology, not merely a bystander response. Microglial activation,
complement deposition, and pro-inflammatory cytokine cascades contribute to
synaptic loss and neurodegeneration. This investigation aims to identify the
specific transcriptomic signatures of these processes.

## Requirements

- `GEOparse` — for GEO data download
- Network access to NCBI GEO and MSigDB (Broad Institute)
- Optional: LLM API access for non-mock runs
