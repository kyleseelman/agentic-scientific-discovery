# Cancer Immunotherapy Response Investigation

Autonomous agentic investigation of **anti-PD1 immunotherapy response in melanoma**
using real transcriptomics data from NCBI GEO.

## Dataset

| Field | Value |
|-------|-------|
| Primary accession | [GSE91061](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE91061) |
| Publication | Riaz et al., Cell 2017 |
| Organism | Homo sapiens |
| Tissue | Melanoma tumor biopsies |
| Comparison | Responders vs Non-responders to anti-PD1 (nivolumab) |
| Platform | RNA-seq |

This landmark dataset profiled tumor transcriptomes from melanoma patients
treated with nivolumab (anti-PD1 checkpoint inhibitor). Identifying gene
expression signatures predictive of response is critical for precision
immunotherapy.

Fallback accessions (GSE78220, GSE93157) are attempted if the primary fails.

## Gene Sets

MSigDB Hallmark collection (50 curated pathway sets) filtered to genes present
in the expression matrix.

## Running

```bash
# Default (mock LLM — tests pipeline without API calls)
python examples/cancer_immunotherapy/run.py

# With Gateway LLM (requires ML_GATEWAY_URL_CODEX)
LLM_PROVIDER=gateway python examples/cancer_immunotherapy/run.py

# With Ollama (requires local Ollama server)
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2 python examples/cancer_immunotherapy/run.py

# With OpenAI
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python examples/cancer_immunotherapy/run.py
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
2. Run differential expression (responders vs non-responders)
3. Perform pathway enrichment (GSEA/ORA on Hallmark sets)
4. Generate and test hypotheses about response mechanisms
5. Scan PubMed literature for supporting/contradicting evidence

Key pathways expected to emerge: interferon gamma response, allograft rejection
(immune infiltration), inflammatory response, TNF-alpha signaling, and
complement activation.

## Requirements

- `GEOparse` — for GEO data download
- Network access to NCBI GEO and MSigDB (Broad Institute)
- Optional: LLM API access for non-mock runs
