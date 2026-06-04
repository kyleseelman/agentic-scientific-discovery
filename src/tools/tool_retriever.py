"""Dynamic tool retrieval via semantic similarity over tool descriptions.

Inspired by Biomni's ToolRetriever: instead of listing all tools in the prompt,
embed the user query and retrieve the top-k most relevant tools. This scales
to large tool registries without overwhelming the LLM context.

Uses GPU-accelerated sentence-transformer embeddings when available,
falling back to TF-IDF for environments without GPU or sentence-transformers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ToolDescription:
    name: str
    description: str
    category: str
    example_use: str


TOOL_CATALOG: list[ToolDescription] = [
    ToolDescription(
        "profile_dataset",
        "Summarize dataset dimensions, group counts, missing values, and top variable genes",
        "analysis",
        "Initial data profiling before any analysis",
    ),
    ToolDescription(
        "differential_expression",
        "Run per-gene differential expression between groups using Welch t-test or Mann-Whitney U with FDR correction",
        "statistics",
        "Test which genes differ between treatment and control",
    ),
    ToolDescription(
        "pathway_enrichment",
        "Hypergeometric test for pathway enrichment on differentially expressed genes with FDR correction",
        "statistics",
        "Test if DE genes are enriched in known biological pathways",
    ),
    ToolDescription(
        "correlation_analysis",
        "Compute pairwise Pearson or Spearman correlations between genes",
        "statistics",
        "Check if genes of interest are co-expressed",
    ),
    ToolDescription(
        "feature_importance_variance",
        "Rank genes by variance and test association with group labels via Spearman correlation",
        "analysis",
        "Identify high-variance genes and check if they associate with treatment",
    ),
    ToolDescription(
        "dimensionality_reduction",
        "PCA on gene expression matrix to find principal components separating samples",
        "analysis",
        "Explore overall structure in the data, check for batch effects or group separation",
    ),
    ToolDescription(
        "clustering_samples",
        "K-means clustering on standardized expression profiles to find sample subgroups",
        "analysis",
        "Discover unsupervised clusters in samples",
    ),
    ToolDescription(
        "group_comparison_summary",
        "Compare mean expression of specific genes between groups with t-tests",
        "statistics",
        "Focused comparison of candidate genes between conditions",
    ),
    ToolDescription(
        "plot_pca",
        "Generate PCA scatter plot colored by group labels",
        "visualization",
        "Visualize sample separation in reduced dimensions",
    ),
    ToolDescription(
        "plot_volcano",
        "Volcano plot of differential expression: log2 fold change vs -log10 p-value",
        "visualization",
        "Visualize effect size vs significance for all genes",
    ),
    ToolDescription(
        "plot_heatmap",
        "Heatmap of top variable or DE genes across samples",
        "visualization",
        "Visualize expression patterns across samples and genes",
    ),
    ToolDescription(
        "plot_box_gene",
        "Box plot of a specific gene's expression across groups",
        "visualization",
        "Compare expression of a single gene between conditions",
    ),
    ToolDescription(
        "string_network",
        "Query STRING database for protein-protein interactions among genes of interest",
        "database",
        "Find known physical or functional interactions between proteins",
    ),
    ToolDescription(
        "uniprot_lookup",
        "Look up gene/protein information in UniProt (function, accession, protein name)",
        "database",
        "Get detailed protein annotation for a gene",
    ),
    ToolDescription(
        "gene_ontology_quickgo",
        "Query QuickGO for Gene Ontology annotations of a gene",
        "database",
        "Find biological process, molecular function, and cellular component annotations",
    ),
    ToolDescription(
        "literature_pubmed",
        "Search PubMed for publications matching a query string",
        "literature",
        "Find relevant papers on a gene, pathway, or biological process",
    ),
    ToolDescription(
        "literature_fetch_abstracts",
        "Fetch full abstracts from PubMed given a list of PMIDs",
        "literature",
        "Read the details of specific papers found via search",
    ),
    ToolDescription(
        "literature_search_biorxiv",
        "Search recent bioRxiv preprints for a query",
        "literature",
        "Find cutting-edge preprints on a topic",
    ),
    ToolDescription(
        "execute_code",
        "Write and execute arbitrary Python analysis code with access to expression data, numpy, pandas, scipy, matplotlib, and PyTorch (GPU). Variables persist across executions.",
        "code",
        "Run custom GPU-accelerated analysis, novel statistical tests, complex data transformations, custom visualizations, or torch-based computation",
    ),
    ToolDescription(
        "train_classifier",
        "Train a machine learning classifier (logistic regression, random forest, SVM, gradient boosting, or neural network MLP) on gene expression data to predict sample groups. Returns accuracy, AUC, feature importances, and saves the trained model.",
        "machine_learning",
        "Build a classifier to predict treatment response from gene expression, identify discriminative gene signatures",
    ),
    ToolDescription(
        "train_neural_network",
        "Train a PyTorch neural network on GPU (MLP classifier, autoencoder, or VAE) on expression data. Supports classification, dimensionality reduction, and learning gene embeddings.",
        "machine_learning",
        "Train a deep learning model for complex pattern recognition, learn latent representations of gene expression",
    ),
    ToolDescription(
        "evaluate_model",
        "Evaluate a previously trained model with comprehensive metrics (accuracy, F1, AUC, MCC) and optional cross-validation",
        "machine_learning",
        "Assess model performance, compare models, validate on held-out data",
    ),
    ToolDescription(
        "feature_selection",
        "ML-based feature selection using LASSO, random forest importance, mutual information, or recursive elimination to identify the most discriminative genes",
        "machine_learning",
        "Find minimal gene sets that distinguish groups, biomarker discovery",
    ),
    ToolDescription(
        "train_gene_embeddings",
        "Learn dense vector representations of genes from expression patterns using autoencoders or contrastive learning on GPU",
        "machine_learning",
        "Create gene embeddings for downstream clustering, similarity search, or knowledge graph enrichment",
    ),
    ToolDescription(
        "cross_validate_hypothesis",
        "Rigorously validate whether a gene set distinguishes sample groups using multiple ML classifiers plus permutation testing for statistical significance",
        "machine_learning",
        "Formally test if a hypothesis about discriminative genes is statistically supported by predictive modeling",
    ),
    ToolDescription(
        "finetune_text_classifier",
        "Fine-tune a pre-trained biomedical language model (PubMedBERT, BioBERT) for text classification on GPU. Train custom classifiers for paper relevance, gene function categorization, or finding stance detection.",
        "llm",
        "Fine-tune PubMedBERT to classify abstracts as relevant/irrelevant to a research question",
    ),
    ToolDescription(
        "generate_with_llm",
        "Generate text using a local pre-trained language model for summarization, hypothesis generation, or entity extraction from biological text",
        "llm",
        "Summarize gene function descriptions or generate structured hypothesis text from findings",
    ),
    ToolDescription(
        "embed_texts",
        "Generate dense vector embeddings for biological texts (abstracts, gene descriptions, findings) using sentence-transformers or BiomedBERT for similarity search and clustering",
        "llm",
        "Embed paper abstracts for semantic similarity search or cluster gene descriptions",
    ),
    ToolDescription(
        "extract_entities_llm",
        "Extract biological entities (genes, diseases, drugs, pathways) from unstructured text using LLM-based named entity recognition",
        "llm",
        "Extract gene names, diseases, and drug mentions from a paper abstract",
    ),
    ToolDescription(
        "list_recommended_models",
        "List curated open-source biomedical models (PubMedBERT, BioBERT, BioGPT, ESM-2, etc.) filtered by task, domain, size, or tags. Helps the agent choose the right model before fine-tuning or inference.",
        "llm",
        "Find the best biomedical encoder under 500MB for NER fine-tuning",
    ),
    ToolDescription(
        "search_hf_models",
        "Search the full HuggingFace Hub for any open-source model by keyword, task, and popularity. Returns model IDs the agent can download and fine-tune for biological research questions.",
        "llm",
        "Search HuggingFace for 'drug interaction prediction' models sorted by downloads",
    ),
    ToolDescription(
        "download_model",
        "Pre-download and cache a HuggingFace model (tokenizer + weights) before a research loop so fine-tuning or inference runs without network delays.",
        "llm",
        "Cache BioBERT locally before starting a multi-cycle NER fine-tuning experiment",
    ),
    ToolDescription(
        "causal_graph_discovery",
        "Infer causal graph structure from gene expression data using PC algorithm, GES, LiNGAM, or Granger causality. Identifies directed causal edges, potential driver genes (high out-degree), and target genes (high in-degree).",
        "causal_inference",
        "Discover which genes causally regulate others rather than just correlate",
    ),
    ToolDescription(
        "mediation_analysis",
        "Test whether a mediator gene M mediates the effect of experimental condition on an outcome gene Y using Baron & Kenny regression and the Sobel test. Decomposes total effect into direct and indirect paths.",
        "causal_inference",
        "Test if a transcription factor mediates the treatment effect on a downstream target gene",
    ),
    ToolDescription(
        "instrumental_variable_analysis",
        "Two-stage least squares (2SLS) instrumental variable analysis to estimate causal effects between genes while accounting for hidden confounders. Includes first-stage F-test and Hausman specification test.",
        "causal_inference",
        "Estimate the causal effect of one gene on another using upstream regulators as instruments",
    ),
    ToolDescription(
        "counterfactual_analysis",
        "Estimate causal treatment effects using propensity score methods: nearest-neighbor matching, inverse probability weighting (IPTW), or doubly robust estimation. Reports ATE, ATT, confidence intervals, and covariate balance diagnostics.",
        "causal_inference",
        "Estimate what a gene's expression would be under the counterfactual condition",
    ),
    ToolDescription(
        "interaction_network_analysis",
        "Infer gene regulatory networks using mutual information, partial correlation, or ARACNe (with DPI pruning). Identifies hub genes and regulatory relationships beyond simple pairwise correlation.",
        "causal_inference",
        "Build a regulatory network to find master regulators and key interaction hubs",
    ),
    ToolDescription(
        "query_knowledge_graph",
        "Query the biomedical knowledge graph for structured context about genes, drugs, diseases, pathways, and their relationships using hybrid graph+vector retrieval",
        "knowledge_graph",
        "Look up what drugs target a gene, find pathway relationships, or retrieve similar entities by embedding",
    ),
    ToolDescription(
        "add_to_knowledge_graph",
        "Persist research findings, hypotheses, and discovered entity relations into the knowledge graph with provenance tracking",
        "knowledge_graph",
        "Save newly discovered gene-disease associations or experimental findings for future retrieval",
    ),
    # Novel model building
    ToolDescription(
        "build_architecture",
        "Build and instantiate a custom PyTorch nn.Module from an architecture catalog: gene transformers, GAT, multi-modal encoders, residual MLPs, contrastive encoders, VAEs, GCNs, or custom LLM-generated code. Returns a saved model checkpoint ready for training.",
        "model_building",
        "Design a self-attention gene network architecture for gene regulation prediction",
    ),
    ToolDescription(
        "train_model_pipeline",
        "Full GPU training pipeline for any PyTorch model: data prep, train/val/test split, early stopping, cosine/plateau LR scheduling, gradient clipping, mixed precision (AMP), metric logging, and checkpoint saving.",
        "model_building",
        "Train the custom architecture on expression data with early stopping and mixed precision",
    ),
    ToolDescription(
        "finetune_protein_lm",
        "Fine-tune ESM-2 protein language models (8M/35M/650M) with LoRA for protein function prediction, protein-protein interaction, or property regression. Parameter-efficient via PEFT adapters.",
        "model_building",
        "Fine-tune ESM-2 with LoRA to predict protein binding affinity from sequence",
    ),
    ToolDescription(
        "finetune_genomic_lm",
        "Fine-tune any HuggingFace language model with LoRA/PEFT for genomic or biomedical text tasks: classification, NER, sequence-to-sequence. Supports 4-bit quantized loading for large models.",
        "model_building",
        "Fine-tune BiomedBERT with LoRA for classifying gene function descriptions",
    ),
    ToolDescription(
        "build_graph_model",
        "Build and train graph neural networks (GCN, GAT, GraphSAGE) for biological networks: protein interactions, gene regulatory networks, metabolic pathways. Manual message-passing with optional PyTorch Geometric backend.",
        "model_building",
        "Build a GAT model on the STRING protein interaction network for node classification",
    ),
    ToolDescription(
        "design_from_paper",
        "The 'novel' tool: given a paper abstract or method description, uses LLM to extract the architecture design, maps it to the architecture catalog, and instantiates a buildable PyTorch model. Optionally auto-trains the designed model.",
        "model_building",
        "Read a transformer-based gene regulation paper and build the described architecture",
    ),
    ToolDescription(
        "benchmark_model",
        "Evaluate a trained PyTorch model with comprehensive metrics (accuracy, F1, AUC, precision, recall), cross-validation, and optional comparison against baseline models. Records results with lineage tracking.",
        "model_building",
        "Benchmark the novel architecture against the baseline MLP on the same dataset",
    ),
    ToolDescription(
        "extract_architecture_from_paper",
        "Extract structured model architecture details from a paper: architecture components, training procedure, key innovation, hyperparameters, and reproducibility info. Maps extracted concepts to the buildable architecture catalog.",
        "llm",
        "Extract the transformer architecture and training details from an Enformer paper abstract",
    ),
]


class ToolRetriever:
    """Retrieve the most relevant tools for a given research query.

    Uses GPU sentence-transformer embeddings when config.use_gpu_embeddings is
    True and sentence-transformers is available; falls back to TF-IDF otherwise.
    """

    def __init__(
        self,
        catalog: list[ToolDescription] | None = None,
        use_gpu: bool = True,
        device: str = "cuda",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.catalog = catalog or TOOL_CATALOG
        self._corpus = [
            f"{t.name} {t.description} {t.category} {t.example_use}"
            for t in self.catalog
        ]
        self._use_gpu = use_gpu
        self._device = device
        self._embedding_model = embedding_model
        self._doc_embeddings: np.ndarray | None = None

        if use_gpu:
            try:
                self._precompute_gpu_embeddings()
                logger.info("ToolRetriever using GPU embeddings on %s", device)
            except Exception as e:
                logger.warning("GPU embeddings failed (%s), falling back to TF-IDF", e)
                self._use_gpu = False

        if not self._use_gpu:
            self._vectorizer = TfidfVectorizer(stop_words="english", max_features=512)
            self._doc_matrix = self._vectorizer.fit_transform(self._corpus)

    def _precompute_gpu_embeddings(self) -> None:
        from src.gpu_embeddings import _get_embedding_model
        model = _get_embedding_model(self._embedding_model, self._device)
        if model is None:
            raise RuntimeError("Could not load embedding model")
        self._doc_embeddings = model.encode(self._corpus, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(self._doc_embeddings, axis=1, keepdims=True).clip(min=1e-10)
        self._doc_embeddings = self._doc_embeddings / norms

    def retrieve(self, query: str, top_k: int = 8) -> list[ToolDescription]:
        if self._use_gpu and self._doc_embeddings is not None:
            from src.gpu_embeddings import _get_embedding_model
            model = _get_embedding_model(self._embedding_model, self._device)
            q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
            q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True).clip(min=1e-10)
            sims = (q_emb @ self._doc_embeddings.T).flatten()
        else:
            q_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self._doc_matrix).flatten()

        top_indices = np.argsort(-sims)[:top_k]
        return [self.catalog[int(i)] for i in top_indices if sims[int(i)] > 0.0]

    def format_for_prompt(self, query: str, top_k: int = 8) -> str:
        tools = self.retrieve(query, top_k)
        lines = [f"- {t.name}: {t.description}" for t in tools]
        return "\n".join(lines)
