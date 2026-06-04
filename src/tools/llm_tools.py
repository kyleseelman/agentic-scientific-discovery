"""LLM-based tools for biological text analysis.

Fine-tune biomedical language models, generate text, compute embeddings,
and extract entities from biological literature.  All tools follow the
TOOL_REGISTRY signature: ``(ctx: ToolContext, params: dict) -> dict``.

Heavy imports (transformers, sentence-transformers, huggingface_hub) are
deferred to call-time so the module loads quickly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.tools.data_analysis import ToolContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Biomedical model catalog
# ---------------------------------------------------------------------------

@dataclass
class ModelCard:
    """Metadata for a recommended open-source model."""
    model_id: str
    task: str
    domain: str
    size_mb: int
    description: str
    tags: list[str] = field(default_factory=list)


BIOMEDICAL_MODEL_CATALOG: list[ModelCard] = [
    # -- Text classification / NER fine-tuning bases -----------------------
    ModelCard("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
              "text-classification", "biomedical", 440,
              "Pre-trained on PubMed abstracts. Best general biomedical encoder.",
              ["encoder", "pubmed", "fine-tune", "NER", "classification"]),
    ModelCard("dmis-lab/biobert-base-cased-v1.2",
              "text-classification", "biomedical", 440,
              "BioBERT v1.2 pre-trained on PubMed + PMC full-text.",
              ["encoder", "pubmed", "pmc", "fine-tune", "NER"]),
    ModelCard("allenai/scibert_scivocab_uncased",
              "text-classification", "scientific", 440,
              "SciBERT trained on Semantic Scholar papers across all sciences.",
              ["encoder", "multidomain", "fine-tune"]),
    ModelCard("michiyasunaga/BioLinkBERT-base",
              "text-classification", "biomedical", 440,
              "BioLinkBERT uses citation-link pre-training for better relation understanding.",
              ["encoder", "relations", "fine-tune", "link-prediction"]),
    ModelCard("distilbert-base-uncased",
              "text-classification", "general", 260,
              "Lightweight and fast. Good for quick prototyping or small datasets.",
              ["encoder", "fast", "fine-tune", "small"]),

    # -- Named entity recognition ------------------------------------------
    ModelCard("d4data/biomedical-ner-all",
              "token-classification", "biomedical", 440,
              "Fine-tuned for biomedical NER: genes, diseases, chemicals, species, cell types.",
              ["NER", "ready-to-use", "genes", "diseases", "chemicals"]),
    ModelCard("blaze999/Medical-NER",
              "token-classification", "clinical", 440,
              "Medical NER for clinical entity extraction.",
              ["NER", "ready-to-use", "clinical"]),

    # -- Text generation / summarization -----------------------------------
    ModelCard("BioMistral/BioMistral-7B",
              "text-generation", "biomedical", 14000,
              "7B biomedical LLM fine-tuned from Mistral. Strong for biomedical QA and summarization.",
              ["generation", "QA", "summarization", "large"]),
    ModelCard("TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              "text-generation", "general", 2200,
              "Compact 1.1B chat model. Good for fast inference when GPU memory is limited.",
              ["generation", "fast", "small", "chat"]),
    ModelCard("microsoft/BioGPT",
              "text-generation", "biomedical", 1500,
              "Generative pre-trained model for biomedical text. Good for literature-style generation.",
              ["generation", "pubmed", "biomedical"]),

    # -- Embeddings / similarity -------------------------------------------
    ModelCard("all-MiniLM-L6-v2",
              "sentence-similarity", "general", 90,
              "Fast, compact sentence embeddings. Good default for similarity search.",
              ["embeddings", "fast", "small", "similarity"]),
    ModelCard("pritamdeka/S-PubMedBert-MS-MARCO",
              "sentence-similarity", "biomedical", 440,
              "PubMedBERT fine-tuned for sentence similarity on biomedical text.",
              ["embeddings", "pubmed", "similarity", "retrieval"]),
    ModelCard("cambridgeltl/SapBERT-from-PubMedBERT",
              "sentence-similarity", "biomedical", 440,
              "SapBERT for biomedical entity linking and concept matching.",
              ["embeddings", "entity-linking", "UMLS", "concepts"]),

    # -- Relation extraction -----------------------------------------------
    ModelCard("alvaroalon2/biobert_genetic_ner",
              "token-classification", "genetics", 440,
              "BioBERT fine-tuned for genetic entity recognition.",
              ["NER", "genetics", "ready-to-use"]),

    # -- Protein / sequence models -----------------------------------------
    ModelCard("facebook/esm2_t6_8M_UR50D",
              "feature-extraction", "protein", 33,
              "ESM-2 8M: tiny protein language model for sequence embeddings. Fast prototyping.",
              ["protein", "embeddings", "small", "sequence"]),
    ModelCard("facebook/esm2_t33_650M_UR50D",
              "feature-extraction", "protein", 2600,
              "ESM-2 650M: strong protein embeddings for structure/function prediction.",
              ["protein", "embeddings", "medium", "sequence"]),
]

# Index by task for fast lookup
_CATALOG_BY_TASK: dict[str, list[ModelCard]] = {}
for _mc in BIOMEDICAL_MODEL_CATALOG:
    _CATALOG_BY_TASK.setdefault(_mc.task, []).append(_mc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_device():
    """Return the best available torch device."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_output(ctx: ToolContext) -> Path:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    return ctx.output_dir


def _resolve_hf_model(model_id: str | None, default: str) -> str:
    """Return *model_id* if provided, otherwise *default*."""
    if model_id and model_id.strip():
        return model_id.strip()
    return default


# ---------------------------------------------------------------------------
# Tool: finetune_text_classifier
# ---------------------------------------------------------------------------

def finetune_text_classifier(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune a pre-trained language model for biological text classification.

    Use cases the agent might invoke this for:
    - Classify paper abstracts by relevance to a research question
    - Classify gene descriptions by functional category
    - Sentiment/stance classification on study findings

    params:
        base_model: str — HuggingFace model ID
            (default "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        texts: list[str] — training texts
        labels: list[str] — corresponding labels
        task_description: str — what the classifier does
        epochs: int — (default 3)
        batch_size: int — (default 16)
        learning_rate: float — (default 2e-5)
        max_length: int — (default 256)
        test_fraction: float — (default 0.2)
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        return {"error": "transformers is not installed. Run: pip install transformers"}

    base_model = _resolve_hf_model(
        params.get("base_model"),
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    )
    texts: list[str] = list(params.get("texts", []))
    labels: list[str] = list(params.get("labels", []))
    task_description = str(params.get("task_description", "biological text classification"))
    epochs = int(params.get("epochs", 3))
    batch_size = int(params.get("batch_size", 16))
    lr = float(params.get("learning_rate", 2e-5))
    max_length = int(params.get("max_length", 256))
    test_fraction = float(params.get("test_fraction", 0.2))

    if not texts or not labels:
        return {"error": "texts and labels are required (non-empty lists)"}
    if len(texts) != len(labels):
        return {"error": f"texts ({len(texts)}) and labels ({len(labels)}) must have equal length"}

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    label_names = list(le.classes_)
    n_classes = len(label_names)

    # Train/test split (stratify when possible)
    try:
        tr_texts, te_texts, tr_labels, te_labels = train_test_split(
            texts, y_encoded, test_size=test_fraction, random_state=42, stratify=y_encoded,
        )
    except ValueError:
        tr_texts, te_texts, tr_labels, te_labels = train_test_split(
            texts, y_encoded, test_size=test_fraction, random_state=42,
        )

    device = _get_device()
    print(f"Fine-tuning {base_model} for {task_description}")
    print(f"  {len(tr_texts)} train / {len(te_texts)} test, {n_classes} classes, device={device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=n_classes,
        ).to(device)
    except Exception as e:
        return {"error": f"Failed to load model {base_model}: {type(e).__name__}: {e}"}

    # ---- Dataset ---------------------------------------------------------
    class TextDataset(Dataset):
        def __init__(self, txts: list[str], lbls: np.ndarray) -> None:
            self.encodings = tokenizer(
                txts, truncation=True, padding=True, max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(lbls, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = self.labels[idx]
            return item

    train_ds = TextDataset(tr_texts, tr_labels)
    test_ds = TextDataset(te_texts, te_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ---- Optimizer + scheduler -------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    # ---- Training loop ---------------------------------------------------
    t0 = time.time()
    loss_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    training_time = time.time() - t0

    # ---- Evaluation ------------------------------------------------------
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cr = classification_report(all_labels, all_preds, target_names=label_names, output_dict=True)

    # ---- Save model + tokenizer ------------------------------------------
    out_dir = _ensure_output(ctx)
    model_dir = out_dir / "finetuned_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | saved to {model_dir}")

    result = {
        "accuracy": float(acc),
        "f1": float(f1),
        "per_class_metrics": {k: v for k, v in cr.items() if k not in ("accuracy", "macro avg", "weighted avg")},
        "model_path": str(model_dir),
        "base_model": base_model,
        "task_description": task_description,
        "label_names": label_names,
        "n_classes": n_classes,
        "n_train": len(tr_texts),
        "n_test": len(te_texts),
        "training_loss_history": loss_history,
        "training_time_s": training_time,
        "device": str(device),
    }
    try:
        from src.memory.model_store import ModelRecord, ModelStore
        ms = ModelStore(Path("./outputs/model_registry.json"))
        ms.add(ModelRecord(
            id=f"llm_{int(time.time())}",
            model_type=f"finetune_{base_model.split('/')[-1]}",
            task="text_classification",
            hypothesis_id=params.get("hypothesis_id"),
            experiment_id=None,
            metrics={"accuracy": float(acc), "f1": float(f1)},
            hyperparameters={"base_model": base_model, "epochs": epochs, "lr": lr},
            feature_genes=[],
            model_path=str(model_dir),
            training_time_s=training_time,
        ))
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Tool: generate_with_llm
# ---------------------------------------------------------------------------

def generate_with_llm(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Generate text using a local pre-trained language model.

    Use cases:
    - Summarize gene function descriptions
    - Generate hypothesis text from structured data
    - Extract entities from biological text

    params:
        model: str — HuggingFace model ID or local path
            (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
        prompt: str — input prompt
        max_new_tokens: int — (default 256)
        temperature: float — (default 0.7)
        task: str — "generation", "summarization", "extraction"
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return {"error": "transformers is not installed. Run: pip install transformers"}

    from src.config import get_config
    cfg = get_config()

    model_id = _resolve_hf_model(params.get("model"), cfg.hf_model)
    prompt = str(params.get("prompt", ""))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    temperature = float(params.get("temperature", 0.7))
    task = str(params.get("task", "generation"))

    if not prompt:
        return {"error": "prompt is required"}

    device = _get_device()
    print(f"Generating with {model_id} on {device} (task={task})")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto",
        )
    except Exception as e:
        return {"error": f"Failed to load model {model_id}: {type(e).__name__}: {e}"}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    elapsed = time.time() - t0
    tokens_generated = len(generated_ids)

    print(f"Generated {tokens_generated} tokens in {elapsed:.1f}s")

    return {
        "generated_text": generated_text,
        "model_used": model_id,
        "tokens_generated": tokens_generated,
        "task": task,
        "elapsed_s": elapsed,
        "device": str(device),
    }


# ---------------------------------------------------------------------------
# Tool: embed_texts
# ---------------------------------------------------------------------------

def embed_texts(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Generate dense embeddings for biological texts using a language model.

    Use cases:
    - Embed paper abstracts for similarity search
    - Embed gene descriptions for clustering
    - Embed hypothesis statements for retrieval

    params:
        texts: list[str] — texts to embed
        model: str — sentence-transformers or HuggingFace model ID
            (default "all-MiniLM-L6-v2")
        output_name: str — name for saved embeddings file
    """
    texts: list[str] = list(params.get("texts", []))
    if not texts:
        return {"error": "texts is required (non-empty list of strings)"}

    model_id = _resolve_hf_model(params.get("model"), "all-MiniLM-L6-v2")
    output_name = str(params.get("output_name", "text_embeddings"))

    # Try sentence-transformers first (fast, purpose-built)
    embeddings: np.ndarray | None = None
    model_used = model_id

    try:
        from sentence_transformers import SentenceTransformer
        device = str(_get_device())
        print(f"Embedding {len(texts)} texts with sentence-transformers ({model_id}) on {device}")
        st_model = SentenceTransformer(model_id, device=device)
        t0 = time.time()
        embeddings = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        elapsed = time.time() - t0
    except ImportError:
        logger.info("sentence-transformers not available, falling back to transformers")
    except Exception as e:
        logger.warning("sentence-transformers encode failed (%s), trying transformers", e)

    # Fallback: use transformers directly (mean-pooling over last hidden state)
    if embeddings is None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            return {"error": "Neither sentence-transformers nor transformers is installed"}

        fallback_model = (
            model_id if "/" in model_id
            else "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        )
        model_used = fallback_model
        device = _get_device()
        print(f"Embedding {len(texts)} texts with transformers ({fallback_model}) on {device}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            lm = AutoModel.from_pretrained(fallback_model).to(device)
        except Exception as e:
            return {"error": f"Failed to load model {fallback_model}: {type(e).__name__}: {e}"}

        t0 = time.time()
        lm.eval()
        all_embs: list[np.ndarray] = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tokenizer(
                    batch, truncation=True, padding=True, max_length=512,
                    return_tensors="pt",
                ).to(device)
                out = lm(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                all_embs.append(pooled.cpu().numpy())
        embeddings = np.vstack(all_embs)
        elapsed = time.time() - t0

    # Compute sample pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    n_show = min(5, len(texts))
    sample_sims = cosine_similarity(embeddings[:n_show]).tolist()

    # Save embeddings
    out_dir = _ensure_output(ctx)
    output_path = out_dir / f"{output_name}.npy"
    np.save(str(output_path), embeddings)

    print(f"Embedded {len(texts)} texts → shape {embeddings.shape} in {elapsed:.1f}s")

    return {
        "embedding_shape": list(embeddings.shape),
        "model_used": model_used,
        "output_path": str(output_path),
        "sample_similarities": sample_sims,
        "n_texts": len(texts),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Tool: extract_entities_llm
# ---------------------------------------------------------------------------

def extract_entities_llm(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Extract biological entities (genes, diseases, drugs, pathways) from text using an LLM.

    params:
        text: str — input text (abstract, finding statement, etc.)
        entity_types: list[str] — types to extract
            (default ["gene", "disease", "drug", "pathway"])
        model: str — model to use (default: configured LLM backend)
    """
    text = str(params.get("text", ""))
    if not text:
        return {"error": "text is required"}

    entity_types = list(params.get("entity_types", ["gene", "disease", "drug", "pathway"]))
    model_id = params.get("model")

    types_str = ", ".join(entity_types)
    extraction_prompt = (
        f"Extract all biological entities from the following text.\n"
        f"Entity types to extract: {types_str}\n\n"
        f"Return a JSON object with keys for each entity type, "
        f"where each value is a list of extracted entity names.\n"
        f"Only include entities that are explicitly mentioned.\n\n"
        f"Text: {text}\n\n"
        f"JSON:"
    )

    # Try using the configured LLM backend first
    generated_text: str | None = None

    try:
        from src.config import create_llm_backend
        backend = create_llm_backend()
        generated_text = backend.generate(
            extraction_prompt,
            system="You are a biomedical named entity recognition system. "
                   "Extract entities and return valid JSON only.",
            temperature=0.1,
        )
    except Exception as e:
        logger.info("LLM backend failed (%s), trying local model", e)

    # Fallback: use a local transformers model
    if not generated_text:
        result = generate_with_llm(ctx, {
            "model": model_id,
            "prompt": extraction_prompt,
            "max_new_tokens": 512,
            "temperature": 0.1,
            "task": "extraction",
        })
        if "error" in result:
            return result
        generated_text = result.get("generated_text", "")

    # Parse the JSON response
    entities: dict[str, list[str]] = {t: [] for t in entity_types}
    try:
        import json
        clean = generated_text.strip()
        # Try to find JSON in the response
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(clean[start:end])
            for t in entity_types:
                if t in parsed and isinstance(parsed[t], list):
                    entities[t] = [str(e) for e in parsed[t]]
    except (json.JSONDecodeError, ValueError):
        logger.warning("Could not parse entity extraction JSON, returning raw text")

    total = sum(len(v) for v in entities.values())
    print(f"Extracted {total} entities from {len(text)} chars of text")

    return {
        "entities": entities,
        "entity_types": entity_types,
        "total_extracted": total,
        "raw_extraction": generated_text[:2000] if generated_text else "",
        "text_length": len(text),
    }


# ---------------------------------------------------------------------------
# Tool: extract_architecture_from_paper
# ---------------------------------------------------------------------------

def extract_architecture_from_paper(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Extract model architecture details from a paper abstract or method section.

    Extracts structured information that feeds into ``design_from_paper``
    and can update the architecture catalog with new patterns.

    params:
        text: str — paper abstract or method section
        paper_id: str | None — identifier for provenance
        paper_title: str | None — title for reference
    """
    text = str(params.get("text", ""))
    if not text:
        return {"error": "text is required (paper abstract or method section)"}

    paper_id = params.get("paper_id", "unknown")
    paper_title = params.get("paper_title", "")

    from src.tools.architecture_catalog import get_catalog_summary, COMPONENT_VOCABULARY

    extraction_prompt = f"""Extract the model architecture details from this paper.

Paper: {paper_title}
Text:
{text[:4000]}

Known architecture components: {COMPONENT_VOCABULARY}

Known architecture patterns:
{get_catalog_summary()}

Return a JSON object with:
{{
  "architecture_description": "<1-2 sentence description of the model>",
  "architecture_type": "<closest match from known patterns, or 'novel'>",
  "components": ["<list of architecture components used>"],
  "layers": {{
    "encoder": "<description>",
    "decoder": "<description if any>",
    "attention": "<type: self-attention, cross-attention, graph-attention, none>",
    "normalization": "<type: layer_norm, batch_norm, none>",
    "activation": "<type: relu, gelu, swish, etc.>"
  }},
  "training_procedure": {{
    "optimizer": "<adam, adamw, sgd, etc.>",
    "lr_schedule": "<cosine, linear_warmup, step, etc.>",
    "loss_function": "<cross_entropy, mse, contrastive, etc.>",
    "regularization": ["<dropout, weight_decay, etc.>"]
  }},
  "key_innovation": "<what is novel compared to standard approaches>",
  "input_output": {{
    "input_type": "<gene_expression, protein_sequence, graph, text, multi_modal>",
    "output_type": "<classification, regression, embedding, generation>"
  }},
  "hyperparameters": {{
    "hidden_dim": <int or null>,
    "n_layers": <int or null>,
    "n_heads": <int or null>,
    "dropout": <float or null>
  }},
  "reproducibility": {{
    "dataset": "<dataset used>",
    "compute": "<GPU type and count if mentioned>",
    "code_available": <true/false/null>
  }}
}}

Return strictly valid JSON only."""

    extracted: dict[str, Any] | None = None

    try:
        from src.config import create_llm_backend
        backend = create_llm_backend()
        response = backend.generate(
            extraction_prompt,
            system="You are an expert ML architecture analyst. Extract architecture details "
                   "from papers and return structured JSON.",
            temperature=0.1,
        )
        import json as json_mod
        clean = response.strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            extracted = json_mod.loads(clean[start:end])
    except Exception as e:
        logger.warning("LLM extraction failed (%s), using keyword analysis", e)

    if extracted is None:
        text_lower = text.lower()
        components: list[str] = []
        comp_keywords = {
            "multi_head_attention": ["attention", "self-attention", "multi-head"],
            "graph_convolution": ["graph convolution", "gcn", "message passing"],
            "graph_attention": ["graph attention", "gat"],
            "feed_forward": ["feed-forward", "ffn", "mlp"],
            "skip_connection": ["residual", "skip connection"],
            "layer_norm": ["layer norm", "layernorm"],
            "batch_norm": ["batch norm", "batchnorm"],
            "dropout": ["dropout"],
            "positional_encoding": ["positional encoding", "position embedding"],
            "encoder": ["encoder"],
            "decoder": ["decoder"],
            "cross_attention": ["cross-attention", "cross attention"],
        }
        for comp, kws in comp_keywords.items():
            if any(kw in text_lower for kw in kws):
                components.append(comp)

        if any(kw in text_lower for kw in ["transformer", "attention"]):
            arch_type = "gene_transformer"
        elif any(kw in text_lower for kw in ["graph neural", "gcn", "gnn", "gat"]):
            arch_type = "gcn_message_passing"
        elif any(kw in text_lower for kw in ["contrastive", "simclr"]):
            arch_type = "contrastive_encoder"
        elif any(kw in text_lower for kw in ["variational", "vae"]):
            arch_type = "expression_vae"
        elif any(kw in text_lower for kw in ["multi-modal", "multimodal"]):
            arch_type = "multi_modal_encoder"
        else:
            arch_type = "residual_mlp"

        extracted = {
            "architecture_description": f"Keyword-extracted architecture from paper",
            "architecture_type": arch_type,
            "components": components or ["feed_forward"],
            "layers": {"encoder": "unknown", "attention": "unknown"},
            "training_procedure": {"optimizer": "unknown", "loss_function": "unknown"},
            "key_innovation": "Could not extract (LLM unavailable)",
            "input_output": {"input_type": "unknown", "output_type": "unknown"},
            "hyperparameters": {},
            "reproducibility": {},
        }

    extracted["paper_id"] = paper_id
    extracted["paper_title"] = paper_title

    from src.tools.architecture_catalog import match_paper_concepts
    components = extracted.get("components", [])
    task = extracted.get("input_output", {}).get("output_type")
    matches = match_paper_concepts(components, task)
    extracted["catalog_matches"] = [
        {"name": m.name, "paper": m.paper, "description": m.description[:100]}
        for m in matches[:3]
    ]

    total_fields = sum(1 for v in extracted.values() if v and v != "unknown")
    print(f"Extracted {total_fields} architecture fields from paper "
          f"({extracted.get('architecture_type', 'unknown')})")

    return extracted


# ---------------------------------------------------------------------------
# Tool: list_recommended_models
# ---------------------------------------------------------------------------

def list_recommended_models(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """List curated open-source models for a biological task.

    The agent calls this to decide which model to download and fine-tune.

    params:
        task: str — filter by task type. Options:
            "text-classification", "token-classification", "text-generation",
            "sentence-similarity", "feature-extraction", or "all"
        domain: str — filter by domain: "biomedical", "clinical", "protein",
            "general", "scientific", or "all" (default "all")
        max_size_mb: int — exclude models larger than this (default: no limit)
        tags: list[str] — only include models matching ANY of these tags
    """
    task = str(params.get("task", "all")).strip().lower()
    domain = str(params.get("domain", "all")).strip().lower()
    max_size = params.get("max_size_mb")
    required_tags = [t.lower() for t in params.get("tags", [])]

    results: list[dict[str, Any]] = []
    for mc in BIOMEDICAL_MODEL_CATALOG:
        if task != "all" and mc.task != task:
            continue
        if domain != "all" and mc.domain != domain:
            continue
        if max_size is not None and mc.size_mb > int(max_size):
            continue
        if required_tags and not any(t in [tag.lower() for tag in mc.tags] for t in required_tags):
            continue
        results.append({
            "model_id": mc.model_id,
            "task": mc.task,
            "domain": mc.domain,
            "size_mb": mc.size_mb,
            "description": mc.description,
            "tags": mc.tags,
        })

    return {
        "models": results,
        "total": len(results),
        "filters": {"task": task, "domain": domain, "max_size_mb": max_size, "tags": required_tags},
    }


# ---------------------------------------------------------------------------
# Tool: search_hf_models
# ---------------------------------------------------------------------------

def search_hf_models(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Search HuggingFace Hub for open-source models the agent can download.

    Use this when the curated catalog doesn't have the right model -- the agent
    can search the full HuggingFace Hub and pick any public model to fine-tune
    or use for inference.

    params:
        query: str — free-text search (e.g. "biomedical NER", "protein folding")
        task: str — HF pipeline tag filter (e.g. "text-classification",
            "token-classification", "text-generation", "feature-extraction")
        sort: str — "downloads", "likes", "trending" (default "downloads")
        limit: int — max results (default 10, max 25)
        language: str — filter by language (default "en")
        min_downloads: int — only models with at least this many downloads
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {"error": "huggingface_hub is not installed. Run: pip install huggingface_hub"}

    query = str(params.get("query", ""))
    task_filter = params.get("task")
    sort = str(params.get("sort", "downloads"))
    limit = min(int(params.get("limit", 10)), 25)
    language = params.get("language", "en")
    min_downloads = int(params.get("min_downloads", 0))

    if not query:
        return {"error": "query is required (e.g. 'biomedical text classification')"}

    api = HfApi()
    try:
        models = api.list_models(
            search=query,
            pipeline_tag=task_filter if task_filter else None,
            sort=sort,
            limit=limit,
        )

        results: list[dict[str, Any]] = []
        for m in models:
            downloads = getattr(m, "downloads", 0) or 0
            if downloads < min_downloads:
                continue
            results.append({
                "model_id": m.id,
                "task": getattr(m, "pipeline_tag", None),
                "downloads": downloads,
                "likes": getattr(m, "likes", 0),
                "tags": getattr(m, "tags", [])[:10],
                "last_modified": str(getattr(m, "lastModified", "")),
            })

        return {
            "models": results,
            "total": len(results),
            "query": query,
            "task_filter": task_filter,
            "note": "Use any model_id with finetune_text_classifier, generate_with_llm, or embed_texts",
        }
    except Exception as e:
        return {"error": f"HuggingFace Hub search failed: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Tool: download_model
# ---------------------------------------------------------------------------

def download_model(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
    """Pre-download a HuggingFace model for later use (fine-tuning or inference).

    Useful when the agent wants to cache a model before a long experiment loop.

    params:
        model_id: str — HuggingFace model ID (e.g. "dmis-lab/biobert-base-cased-v1.2")
        model_type: str — "encoder" (AutoModel), "classifier" (AutoModelForSequenceClassification),
            "causal" (AutoModelForCausalLM), "tokenizer_only" (default "encoder")
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return {"error": "transformers is not installed. Run: pip install transformers"}

    model_id = str(params.get("model_id", "")).strip()
    model_type = str(params.get("model_type", "encoder"))

    if not model_id:
        return {"error": "model_id is required"}

    print(f"Downloading {model_id} ({model_type})...")
    t0 = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tok_size = sum(
            f.stat().st_size for f in Path(tokenizer.name_or_path).rglob("*")
            if f.is_file()
        ) if Path(tokenizer.name_or_path).exists() else 0

        if model_type == "tokenizer_only":
            elapsed = time.time() - t0
            return {
                "model_id": model_id,
                "status": "downloaded",
                "components": ["tokenizer"],
                "elapsed_s": elapsed,
            }

        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

        loaders = {
            "encoder": AutoModel,
            "classifier": AutoModelForSequenceClassification,
            "causal": AutoModelForCausalLM,
        }
        loader = loaders.get(model_type, AutoModel)
        model = loader.from_pretrained(model_id)
        n_params = sum(p.numel() for p in model.parameters())
        elapsed = time.time() - t0

        print(f"Downloaded {model_id}: {n_params/1e6:.1f}M parameters in {elapsed:.1f}s")

        return {
            "model_id": model_id,
            "status": "downloaded",
            "components": ["tokenizer", "model"],
            "parameters_millions": round(n_params / 1e6, 1),
            "model_type": model_type,
            "elapsed_s": elapsed,
        }
    except Exception as e:
        return {"error": f"Failed to download {model_id}: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_llm_tools(registry: dict[str, Callable]) -> None:
    """Register all LLM tools into the TOOL_REGISTRY."""

    tools: dict[str, Callable] = {
        "finetune_text_classifier": finetune_text_classifier,
        "generate_with_llm": generate_with_llm,
        "embed_texts": embed_texts,
        "extract_entities_llm": extract_entities_llm,
        "extract_architecture_from_paper": extract_architecture_from_paper,
        "list_recommended_models": list_recommended_models,
        "search_hf_models": search_hf_models,
        "download_model": download_model,
    }

    for name, fn in tools.items():
        def _make_safe(tool_fn: Callable, tool_name: str) -> Callable:
            def safe_wrapper(ctx: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
                try:
                    return tool_fn(ctx, params)
                except Exception as e:
                    import traceback
                    return {
                        "error": f"{tool_name} failed: {type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }
            safe_wrapper.__name__ = tool_name
            safe_wrapper.__doc__ = tool_fn.__doc__
            return safe_wrapper

        registry[name] = _make_safe(fn, name)

    print(f"Registered {len(tools)} LLM tools: {', '.join(tools)}")
