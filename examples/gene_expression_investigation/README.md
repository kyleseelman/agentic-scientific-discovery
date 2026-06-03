# Gene expression investigation (synthetic cohort)

This example instantiates the autonomous discovery loop on a **synthetic** gene-expression matrix where a known gene set (`PLANTED_STRESS`) is coordinately up-regulated in the treatment arm.

## Why synthetic data?

- No API keys or downloads required.
- You can verify that pathway enrichment and differential expression recover the planted signal.
- The same code path applies to real GEO / TCGA tables once you swap in file loaders.

## Run

From the **repository root** `agentic-scientific-discovery/` (use **Python 3.10–3.12**; e.g. `python3.11`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
export MPLBACKEND=Agg
python examples/gene_expression_investigation/run.py
```

By default `LLM_PROVIDER=mock` uses deterministic structured outputs so the demo runs offline.

For a local model:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2
```

For OpenAI:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=...
pip install openai
```

Outputs land in `examples/gene_expression_investigation/session_demo/`:

- `knowledge/` — hypotheses, findings, relations
- `experiments.json` — append-only experiment log
- `outputs/` — DE tables, enrichment CSVs, figures
- `orchestrator_state.json` — serialized decision trace for pause/resume workflows
