# Agentic Scientific Discovery System

An autonomous research agent that performs computational biology investigations end-to-end. Given a research question and gene expression data, the system searches scientific literature, proposes testable hypotheses, designs and executes computational experiments using 39 integrated tools, applies adversarial falsification testing and multi-agent critique, and iterates with full checkpoint/resume support across multi-cycle investigations.

The system supports 5 LLM backends (including local open-source models via HuggingFace and Ollama), GPU-accelerated analysis, causal inference, neural network model building, and persistent knowledge accumulation across sessions via meta-learning and continual adaptation.

## Architecture

The 12-phase orchestrator pipeline runs in cycles until a configurable budget (max cycles, wall time, or consecutive pivots) is exhausted:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ResearchOrchestrator.run()                         │
└──────────────────────────────────────────────────────────────────────┘
    │
    ├── Phase 0: Literature Review (periodic)
    │     PubMed + bioRxiv search → fetch abstracts → LLM extracts
    │     structured insights → PaperStore persists papers + findings
    │
    ├── Phase 1: Review Knowledge State
    │     Summarize KnowledgeStore → MemoryRetriever fetches related context
    │
    ├── Phase 2: Hypothesis Generation (literature-aware)
    │     Data + memory + literature + suggested focus → ranked hypotheses
    │     Literature-grounded hypotheses cite source papers
    │
    ├── Phase 3: Rank & Select
    │     Score by information gain (45%), feasibility (35%), novelty (15%),
    │     impact (5%) → select top hypothesis
    │
    ├── Phase 4: Experiment Planning
    │     LLM designs ExperimentPlan with ordered tool steps
    │
    ├── Phase 5: Experiment Execution
    │     Execute plan steps → collect traces and aggregated results
    │     5b: Capture reproducibility bundle (data hash, config, tool sequence)
    │
    ├── Phase 6: Result Analysis
    │     LLM interprets results → verdict (supported/refuted/inconclusive)
    │     + posterior confidence + follow-up suggestions
    │
    ├── Phase 6b: Multi-Agent Critique (optional)
    │     Experimentalist → Critic → Synthesis agents adjust posterior
    │
    ├── Phase 6c: Adversarial Hypothesis Testing
    │     Devil's-advocate LLM attempts to falsify conclusions
    │     Generates falsification experiments, adjusts confidence
    │
    ├── Phase 7: Knowledge Update
    │     Persist findings, update hypothesis status, log open questions
    │
    ├── Phase 8: Strategy Assessment (periodic)
    │     LLM evaluates progress → continue, pivot, or suggest new focus
    │
    └── Auto-checkpoint after each cycle (optional)
```

## LLM Backends

| Provider | `LLM_PROVIDER` | Key Env Vars | Description |
|----------|---------------|--------------|-------------|
| Mock | `mock` | — | Deterministic template responses for demos without API keys |
| Ollama | `ollama` | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Local models via Ollama server (quantized, lower memory) |
| OpenAI | `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` | OpenAI API (GPT-4o-mini default) |
| HuggingFace | `huggingface` | `HF_MODEL`, `HF_MAX_NEW_TOKENS` | Local GPU-accelerated models via transformers |
| Gateway | `gateway` | `ML_GATEWAY_URL_CODEX`, `GATEWAY_MODEL` | Internal OpenAI-compatible gateway |

### Using Local Open-Source Models

The system runs entirely offline using open-source models. Two paths are available:

**Option A: HuggingFace Transformers (full precision, maximum quality)**

Downloads the model from HuggingFace Hub and runs inference locally on GPU with float16:

```bash
# Example: Use Qwen 2.5 7B for research
export LLM_PROVIDER=huggingface
export HF_MODEL=Qwen/Qwen2.5-7B-Instruct
export HF_MAX_NEW_TOKENS=2048
python examples/cellular_senescence/run.py

# Example: Use DeepSeek-R1 for reasoning-intensive research
export LLM_PROVIDER=huggingface
export HF_MODEL=deepseek-ai/DeepSeek-R1-0528
export HF_MAX_NEW_TOKENS=4096
python examples/cellular_senescence/run.py
```

**Option B: Ollama (quantized, lower memory, easier setup)**

Uses Ollama to serve quantized models with minimal configuration:

```bash
# Install and start Ollama, then pull a model
ollama pull qwen2.5:7b

export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:7b
python examples/cellular_senescence/run.py

# Use a biomedical model
ollama pull biomistral:7b
export OLLAMA_MODEL=biomistral:7b
python examples/cellular_senescence/run.py
```

**Recommended Local Models for Biological Research:**

| Model | HF ID | Size | GPU VRAM | Best For |
|-------|--------|------|----------|----------|
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | 16 GB+ | General research, good JSON output |
| Kimi K2 | `moonshotai/Kimi-K2-Instruct` | ~16 GB | 16 GB+ | Strong reasoning, long context |
| DeepSeek-R1 | `deepseek-ai/DeepSeek-R1-0528` | ~16 GB | 16 GB+ | Reasoning-heavy research, chain-of-thought |
| BioMistral 7B | `BioMistral/BioMistral-7B` | ~14 GB | 16 GB+ | Biomedical domain, PubMed-tuned |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | ~16 GB | 16 GB+ | Strong general reasoning |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | ~15 GB | 16 GB+ | Good instruction following, fast |
| TinyLlama 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2.2 GB | 4 GB+ | Quick prototyping, low resource |

**GPU Memory Requirements:**

| Model Size | FP16 VRAM | INT8 VRAM (Ollama) | INT4 VRAM (Ollama) |
|-----------|-----------|--------------------|--------------------|
| 1-3B | 4-8 GB | 2-4 GB | 1-2 GB |
| 7B | 14-16 GB | 8-10 GB | 4-6 GB |
| 13B | 26-30 GB | 14-16 GB | 8-10 GB |
| 70B | 140+ GB | 70-80 GB | 40-48 GB |

**Step-by-step (HuggingFace path):**

1. Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Set environment variables for your chosen model
3. First run downloads the model (~15 GB for 7B models); subsequent runs use cache
4. The backend auto-detects GPU and uses float16 for inference
5. Chat template is applied automatically if the model supports it

## Tools

39 tools organized by category, all following the signature `(ctx: ToolContext, params: dict) -> dict`:

### Data Analysis (8 tools)

| Tool | Description |
|------|-------------|
| `profile_dataset` | Dataset dimensions, group counts, missing values, top variable genes |
| `differential_expression` | Per-gene DE with Welch t-test or Mann-Whitney U, FDR correction, GPU-accelerated |
| `pathway_enrichment` | Hypergeometric test on DE genes against pathway gene sets |
| `correlation_analysis` | Pairwise Pearson/Spearman correlations between genes |
| `feature_importance_variance` | Top-variance genes + Spearman association with group labels |
| `dimensionality_reduction` | PCA with explained variance ratios |
| `clustering_samples` | K-means clustering on standardized expression |
| `group_comparison_summary` | Per-gene mean comparison with t-tests |

### Visualization (4 tools)

| Tool | Description |
|------|-------------|
| `plot_volcano` | Volcano plot (log2FC vs -log10 q-value) |
| `plot_heatmap` | Expression heatmap of top variable/DE genes |
| `plot_pca` | PCA scatter plot colored by group |
| `plot_box_gene` | Box plot of single gene expression by group |

### Biological Databases (3 tools)

| Tool | Description |
|------|-------------|
| `string_network` | STRING protein-protein interactions |
| `uniprot_lookup` | UniProt gene/protein annotation |
| `gene_ontology_quickgo` | QuickGO Gene Ontology annotations |

### Literature (3 tools)

| Tool | Description |
|------|-------------|
| `literature_pubmed` | PubMed keyword search |
| `literature_fetch_abstracts` | Fetch full abstracts by PMIDs |
| `literature_search_biorxiv` | bioRxiv preprint search |

### Machine Learning (6 tools)

| Tool | Description |
|------|-------------|
| `train_classifier` | Logistic regression, random forest, SVM, gradient boosting, MLP |
| `train_neural_network` | PyTorch autoencoder, VAE, or MLP (GPU-accelerated) |
| `evaluate_model` | Comprehensive model evaluation (accuracy, F1, AUC, MCC, CV) |
| `feature_selection` | LASSO, random forest, mutual information, Boruta, RFE |
| `train_gene_embeddings` | Autoencoder or contrastive learning for gene embeddings |
| `cross_validate_hypothesis` | Multi-method CV + permutation testing for gene sets |

### LLM Tools (7 tools)

| Tool | Description |
|------|-------------|
| `finetune_text_classifier` | Fine-tune PubMedBERT/BioBERT for text classification |
| `generate_with_llm` | Local LLM text generation (summarization, extraction) |
| `embed_texts` | Dense embeddings via sentence-transformers or HuggingFace |
| `extract_entities_llm` | NER for genes, diseases, drugs, pathways from text |
| `list_recommended_models` | Curated catalog of 16 biomedical models |
| `search_hf_models` | Search full HuggingFace Hub by keyword/task |
| `download_model` | Pre-download and cache a model for later use |

### Causal Inference (5 tools)

| Tool | Description |
|------|-------------|
| `causal_graph_discovery` | PC algorithm, GES, LiNGAM, or Granger causality |
| `mediation_analysis` | Baron & Kenny + Sobel test for mediation |
| `instrumental_variable_analysis` | 2SLS with first-stage F-test and Hausman test |
| `counterfactual_analysis` | Propensity score matching, IPTW, doubly robust estimation |
| `interaction_network_analysis` | Mutual information, partial correlation, ARACNe networks |

### Knowledge Graph (2 tools)

| Tool | Description |
|------|-------------|
| `query_knowledge_graph` | Hybrid graph+vector retrieval from bio-knowledge-graph-rag |
| `add_to_knowledge_graph` | Persist findings with provenance into the KG |

### Code Execution (1 tool)

| Tool | Description |
|------|-------------|
| `execute_code` | Sandboxed Python with numpy, pandas, scipy, matplotlib, PyTorch |

## Model Building

The ML tools enable the agent to autonomously train, evaluate, and iterate on predictive models:

**Classifiers** (`train_classifier`): Logistic regression, random forest, SVM, gradient boosting, or PyTorch MLP. Returns accuracy, F1, AUC-ROC, cross-validation scores, feature importances, and persists the trained model.

**Neural Networks** (`train_neural_network`): GPU-accelerated PyTorch architectures — MLP for classification, autoencoder for dimensionality reduction, VAE for generative modeling. Supports configurable hidden dimensions, epochs, batch size.

**Gene Embeddings** (`train_gene_embeddings`): Learn dense vector representations of genes from expression patterns using autoencoder, gene2vec-style, or contrastive learning. Outputs similar gene pairs and embedding matrices.

**Model Evaluation** (`evaluate_model`): Load a saved model (.pkl or .pt) and compute additional metrics or run cross-validation on the full dataset.

**Feature Selection** (`feature_selection`): Five methods (LASSO, random forest importance, mutual information, Boruta, recursive elimination) to identify discriminative gene sets.

**Hypothesis Validation** (`cross_validate_hypothesis`): Test whether a gene set distinguishes groups using 4 classifiers + permutation testing for statistical significance. Returns an evidence assessment (strong/moderate/weak/insufficient).

All trained models are tracked in `ModelStore` with metrics, hyperparameters, feature genes, training time, and model paths.

## Causal Inference

Five tools that go beyond correlation to estimate causal structure and effects:

- **Causal Graph Discovery**: Infer directed causal graphs using PC (conditional independence testing), GES (greedy BIC-based search), LiNGAM (ICA-based non-Gaussian), or Granger causality. Identifies potential driver genes (high out-degree) and target genes (high in-degree). GPU-accelerated partial correlation via precision matrix inversion.

- **Mediation Analysis**: Tests whether gene M mediates the effect of experimental condition on outcome gene Y. Baron & Kenny regression decomposes total effect into direct and indirect paths; Sobel test provides statistical significance for the indirect effect.

- **Instrumental Variable Analysis**: Two-stage least squares (2SLS) for causal effect estimation when confounders are present. First-stage F-test warns about weak instruments; Hausman test compares OLS vs IV estimates.

- **Counterfactual Analysis**: Propensity score methods estimate what a gene's expression would be under the counterfactual condition. Three estimators: nearest-neighbor matching, inverse probability weighting (IPTW), and doubly robust. Reports ATE, ATT, confidence intervals, and covariate balance diagnostics.

- **Interaction Network Analysis**: Infer gene regulatory networks using mutual information (with permutation thresholding), partial correlation (via precision matrix), or ARACNe (MI + Data Processing Inequality pruning). Identifies hub genes and regulatory relationships.

## Multi-Agent Coordination

Four specialist agents with role-specific system prompts, coordinated by `AgentCoordinator`:

| Agent | Role |
|-------|------|
| **CriticAgent** | Devil's advocate. Finds confounders, statistical issues, selection bias. Suggests falsification experiments. |
| **LiteratureAgent** | Identifies search terms, summarizes findings, flags contradictions with published work. |
| **ExperimentalistAgent** | Designs experiments with controls, replication strategies, power considerations. |
| **SynthesisAgent** | Integrates perspectives, identifies consensus/disagreement, adjusts confidence. |

**Protocols:**

- **Critique Protocol**: Experimentalist → Critic → Synthesis. Three-step review that produces objections, severity rating, falsification experiments, and a combined confidence adjustment.
- **Hypothesis Debate**: Each agent scores hypotheses; Critic raises objections; Synthesis produces final ranking with multi-agent scores.
- **Deep Review**: Literature searches → Experimentalist evaluates methodology → Critic identifies gaps → Synthesis produces structured review.
- **Replication Protocol**: Experimentalist designs alternative experiments → Critic compares methodology → Synthesis assesses reproducibility score.

## Adversarial Hypothesis Testing

After initial analysis, a separate adversarial review pass stress-tests every conclusion:

1. An LLM with a skeptical system prompt receives the hypothesis, analysis results, and experiment trace
2. It generates: specific objections, severity rating (minor/moderate/serious/fatal), alternative explanations, statistical concerns, and at least one falsification experiment
3. A confidence adjustment multiplier (0.1–1.0) is applied to the posterior
4. Recommendation: accept, revise, retest, or reject
5. If severity is "fatal" or recommendation is "reject" with a "supported" verdict, the verdict is downgraded to "inconclusive"
6. Falsification experiments are added to follow-up suggestions

## Meta-Learning & Continual Learning

### MetaLearner (Recursive Self-Improvement)

Tracks research strategy performance across sessions:
- Records which tools, hypothesis types, and analysis sequences lead to supported outcomes
- Recommends tools based on recency-weighted success rates
- Recommends overall strategies by finding similar past research questions (Jaccard similarity)
- Identifies failing hypothesis types (high refutation rate) and suggests prompt refinements
- Provides adaptation summaries showing what the agent has learned

### ContinualLearner (Cross-Session Adaptation)

Consolidates knowledge across multiple research sessions:
- Groups findings by topic using embedding similarity (sentence-transformers) or keyword overlap
- Detects knowledge drift: contradictions, refinements, extensions between sessions
- Maintains Bayesian priors per hypothesis type (Beta distribution updated with outcomes)
- Provides cross-session reports highlighting agreements and unresolved conflicts
- Resolution tracking for contradictions with explanations

## Memory Architecture

| Component | Purpose |
|-----------|---------|
| **KnowledgeStore** | Hypotheses, findings, relations, open questions — persistent JSON store |
| **ExperimentLog** | Full experiment records: plan, execution trace, results, interpretation |
| **PaperStore** | Papers read, LLM-extracted insights, hypothesis-paper links |
| **ModelStore** | Trained model registry: metrics, hyperparameters, feature genes, paths |
| **ReproducibilityLog** | Data hashes, config snapshots, tool sequences for each experiment |
| **MemoryRetriever** | Semantic retrieval over knowledge + experiments using GPU embeddings |

## Checkpoint/Resume

Long-running research sessions can be checkpointed and resumed:

```python
# Checkpoint saves full state
orch.checkpoint("./sessions/my_research")

# Resume from checkpoint (reconstructs orchestrator at exact state)
orch = ResearchOrchestrator.resume(
    "./sessions/my_research",
    llm=llm, knowledge=knowledge, experiments=experiments,
    tool_ctx_factory=factory, dataset_summary_provider=provider,
)
orch.run()  # Continues from where it left off
```

**Auto-checkpoint**: When `auto_checkpoint=True` in `OrchestratorConfig`, the system automatically saves state after every cycle. If a run crashes, resume picks up at the last completed cycle with timing information preserved.

Checkpoint includes: session ID, cycle count, pivot count, suggested focus, elapsed time, full orchestrator config, and decision log.

## Data Loaders

| Loader | Source | Output |
|--------|--------|--------|
| `geo_loader` | NCBI GEO (via GEOparse) | Expression DataFrame + group Series |
| `msigdb_loader` | MSigDB gene sets | Pathway dict `{name: [genes]}` |
| `hf_loader` | HuggingFace Datasets | Expression DataFrame + groups |

## Knowledge Graph Integration

The `query_knowledge_graph` and `add_to_knowledge_graph` tools connect to the sibling `bio-knowledge-graph-rag` project:

- **Query**: Free-text questions are routed to hybrid graph+vector retrieval. Entity names are resolved to CURIE IDs, ego subgraphs are expanded, and relational paths are serialized for the LLM context.
- **Write**: Research findings, hypotheses, and entity relations are persisted as typed `KGNode`/`KGEdge` entries with bronze-tier provenance for future retrieval.

The tools gracefully degrade if the KG project is not installed — they return informative fallback messages rather than crashing.

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `mock` | LLM backend: mock, ollama, openai, huggingface, gateway |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `OLLAMA_MODEL` | `llama3.2` | Model name for Ollama |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `HF_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID |
| `HF_MAX_NEW_TOKENS` | `1024` | Max tokens to generate (HuggingFace) |
| `ML_GATEWAY_URL_CODEX` | internal URL | Gateway endpoint |
| `GATEWAY_MODEL` | `gpt-5.3-codex` | Gateway model name |
| `RESEARCH_SESSION_DIR` | `./research_sessions` | Session output directory |
| `RANDOM_SEED` | `42` | Global random seed |
| `DEVICE` | auto-detected | `cuda` or `cpu` |
| `USE_GPU_EMBEDDINGS` | `true` | GPU sentence-transformers for retrieval |
| `USE_GPU_COMPUTE` | `true` | GPU for DE, neural networks, causal inference |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model for embeddings |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
# PyTorch (GPU): pip install torch --index-url https://download.pytorch.org/whl/cu121
# PyTorch (CPU): pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Autonomous Mode: Give a Topic, Agent Does Everything

The agent reads recent papers, identifies knowledge gaps, formulates novel research questions, finds relevant GEO datasets, and runs the full investigation — no human input beyond the topic:

```bash
# Just give a topic — the agent reads papers, finds gaps, generates questions, picks data
python run_research.py --topic "Alzheimer's disease" \
    --provider ollama --model qwen2.5:7b

# More specific topics work too
python run_research.py --topic "drug resistance in triple-negative breast cancer" \
    --provider openai --model gpt-4o-mini

# The agent will:
# 1. Search PubMed + bioRxiv for recent papers on your topic
# 2. Extract key findings, gaps, and future directions from each paper
# 3. Search GEO for relevant expression datasets
# 4. Use the LLM to synthesize novel research questions that address gaps
# 5. Select the most promising (question, dataset) pair
# 6. Run the full research pipeline (hypotheses → experiments → analysis → report)
```

### Directed Mode: Specify Question + Dataset

For more control, provide a specific question and dataset:

```bash
python run_research.py \
    --question "What are the shared mechanisms between diabetes and Alzheimer's?" \
    --geo GSE5281 --group-column "disease state" \
    --control normal --treatment "Alzheimer's Disease" \
    --provider openai --model gpt-4o-mini
```

### Demo Mode

```bash
python run_research.py --demo  # No API keys, synthetic data, mock LLM
```

### CLI Arguments

| Argument | Mode | Description |
|----------|------|-------------|
| `--topic` / `-t` | Autonomous | Broad topic — agent discovers questions from literature |
| `--question` / `-q` | Directed | Specific research question |
| `--geo` / `-g` | Directed | GEO accession (e.g. GSE5281) |
| `--provider` | Both | LLM backend: mock, ollama, openai, huggingface |
| `--model` | Both | Model name for chosen provider |
| `--group-column` | Both | Metadata field for grouping (auto-detected if omitted) |
| `--control` | Both | Control group label |
| `--treatment` | Both | Treatment group label |
| `--cycles` | Both | Max research cycles (default: 3) |
| `--output-dir` / `-o` | Both | Output directory (default: ./research_output) |
| `--organism` | Both | Organism (default: Homo sapiens) |
| `--tissue` | Both | Tissue type for literature queries |
| `--condition` | Both | Disease/condition for literature queries |
| `--demo` | — | Run demo with mock LLM + synthetic data |

### Run Pre-Built Examples

```bash
# Set your LLM backend first
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=qwen2.5:7b

# Run any example
python examples/novel_research/run.py              # T2D-Alzheimer's molecular link (has sample results)
python examples/cellular_senescence/run.py          # Cellular senescence mechanisms
python examples/alzheimers_neuroinflammation/run.py # Neuroinflammation in AD
python examples/cancer_immunotherapy/run.py         # Anti-PD1 response in melanoma

# Or with mock LLM (no API keys, for testing the pipeline)
export LLM_PROVIDER=mock
python examples/cellular_senescence/run.py
```

## Novel Research Example: T2D-Alzheimer's Molecular Link

The `examples/novel_research/` directory contains a completed research investigation — the system autonomously investigated shared molecular mechanisms between type 2 diabetes and Alzheimer's disease using real data.

**What it did across 3 research cycles:**
- Searched PubMed and found 5 relevant papers (including MMP9 as shared immune gene, lncRNA dynamics in AD)
- Generated 9 testable hypotheses (insulin signaling, MMP9 inflammation, OXPHOS suppression, sleep/glymphatic disruption)
- Ran differential expression on 21,655 genes × 161 samples (GSE5281), finding 6,378 significant genes (FDR<0.05)
- Identified 17 enriched pathways including IFN-gamma (q=7.36e-6), TNFa/NFkB (q=7.36e-6), PI3K-AKT-mTOR (q=3.3e-3)
- Applied adversarial review to each cycle, downgrading unsupported claims
- Produced a full scientific report with citations

**Key files included in the repo:**
- `examples/novel_research/research_report.md` — Full scientific report
- `examples/novel_research/run_summary.json` — Structured results (hypotheses, findings, confidence scores)
- `examples/novel_research/outputs/volcano.png` — Volcano plot of differential expression
- `examples/novel_research/outputs/box_MMP9.png` — MMP9 expression comparison
- `examples/novel_research/outputs/pathway_enrichment.csv` — Enriched pathways

**Available examples:**
- `examples/novel_research/` — T2D-Alzheimer's link (completed, with results)
- `examples/cellular_senescence/` — Cellular senescence gene expression analysis
- `examples/alzheimers_neuroinflammation/` — Alzheimer's neuroinflammation investigation
- `examples/cancer_immunotherapy/` — Cancer immunotherapy response markers
- `examples/gene_expression_investigation/` — General gene expression study with synthetic data

## Project Layout

```
agentic-scientific-discovery/
├── src/
│   ├── config.py                    # AppConfig, LLMProvider enum, backend factory
│   ├── llm_backends.py              # Mock, Ollama, OpenAI, HuggingFace, Gateway backends
│   ├── gpu_embeddings.py            # GPU-accelerated sentence-transformer embeddings
│   ├── agent/
│   │   ├── orchestrator.py          # 12-phase ResearchOrchestrator with checkpoint/resume
│   │   ├── multi_agent.py           # 4 specialist agents + AgentCoordinator
│   │   ├── adversarial.py           # Adversarial hypothesis testing & falsification
│   │   ├── schemas.py              # Hypothesis, ExperimentPlan, ExperimentStep
│   │   ├── hypothesis_generator.py  # Literature-aware hypothesis generation
│   │   ├── experiment_planner.py    # LLM-driven experiment planning
│   │   ├── experiment_executor.py   # Ordered tool execution engine
│   │   └── result_analyzer.py       # LLM-driven result interpretation
│   ├── planning/
│   │   ├── meta_learner.py          # Recursive self-improvement from past outcomes
│   │   ├── continual_learner.py     # Cross-session adaptation & drift detection
│   │   ├── strategy.py             # Progress assessment & pivot decisions
│   │   └── evaluation.py           # Hypothesis scoring (info gain, feasibility, novelty)
│   ├── memory/
│   │   ├── knowledge_store.py       # Hypotheses, findings, relations, open questions
│   │   ├── experiment_log.py        # Experiment records with full traces
│   │   ├── paper_store.py           # Literature papers + extracted insights
│   │   ├── model_store.py           # Trained model registry
│   │   ├── reproducibility.py       # Data hashes, config snapshots, tool sequences
│   │   └── retriever.py            # Semantic memory retrieval (GPU embeddings)
│   ├── tools/
│   │   ├── data_analysis.py         # 8 core analysis tools + TOOL_REGISTRY
│   │   ├── visualization.py         # 4 plot tools (volcano, heatmap, PCA, box)
│   │   ├── bio_databases.py         # STRING, UniProt, QuickGO
│   │   ├── literature.py            # PubMed, bioRxiv search + LLM insight extraction
│   │   ├── ml_models.py             # 6 ML tools (classifiers, NN, embeddings, CV)
│   │   ├── llm_tools.py             # 7 LLM tools + 16-model biomedical catalog
│   │   ├── causal_inference.py      # 5 causal tools (PC, mediation, IV, propensity, network)
│   │   ├── knowledge_graph.py       # KG query/write integration
│   │   ├── code_executor.py         # Sandboxed Python execution
│   │   └── tool_retriever.py        # Semantic tool selection (GPU or TF-IDF)
│   ├── data/
│   │   ├── geo_loader.py            # NCBI GEO dataset loading
│   │   ├── msigdb_loader.py         # MSigDB pathway gene sets
│   │   └── hf_loader.py             # HuggingFace Datasets loading
│   └── utils/
│       └── json_extract.py          # Robust JSON extraction from LLM output
├── run_research.py                  # CLI entry point for running any research question
├── examples/
│   ├── novel_research/              # Completed T2D-Alzheimer's investigation with results
│   ├── cellular_senescence/         # Cellular senescence example
│   ├── alzheimers_neuroinflammation/
│   ├── cancer_immunotherapy/
│   └── gene_expression_investigation/
├── requirements.txt
└── tests/
```
