# Cellular Senescence Investigation

Autonomous agentic investigation of **replicative senescence in human fibroblasts**
using real transcriptomics data from NCBI GEO.

## Dataset

| Field | Value |
|-------|-------|
| Primary accession | [GSE130727](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE130727) |
| Organism | Homo sapiens |
| Tissue | Dermal fibroblasts |
| Comparison | Proliferating vs replicative senescent cells |
| Platform | Microarray (Affymetrix/Illumina) |

The dataset profiles transcriptomic changes in human fibroblasts undergoing
replicative senescence — a fundamental aging mechanism linked to tissue
dysfunction, inflammation (SASP), and age-related disease.

Fallback accessions (GSE63577, GSE53356) are attempted if the primary fails.

## Gene Sets

MSigDB Hallmark collection (50 curated pathway sets) filtered to genes present
in the expression matrix.

## Running

```bash
# Default (mock LLM — tests pipeline without API calls)
python examples/cellular_senescence/run.py

# With Gateway LLM (requires ML_GATEWAY_URL_CODEX)
LLM_PROVIDER=gateway python examples/cellular_senescence/run.py

# With Ollama (requires local Ollama server)
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2 python examples/cellular_senescence/run.py

# With OpenAI
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python examples/cellular_senescence/run.py
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
2. Run differential expression (senescent vs proliferating)
3. Perform pathway enrichment (GSEA/ORA on Hallmark sets)
4. Generate and test hypotheses about senescence mechanisms
5. Scan PubMed literature for supporting/contradicting evidence

Key pathways expected to emerge: p53 signaling, inflammatory response (SASP),
oxidative phosphorylation, cell cycle arrest, and TGF-beta signaling.

## Requirements

- `GEOparse` — for GEO data download
- Network access to NCBI GEO and MSigDB (Broad Institute)
- Optional: LLM API access for non-mock runs
