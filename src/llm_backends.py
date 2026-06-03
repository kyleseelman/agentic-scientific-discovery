from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from src.config import AppConfig


@dataclass
class MockLLMBackend:
    """Deterministic template-driven responses so demos run without API keys."""

    config: AppConfig

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        p = prompt.lower()
        if "extract structured scientific insights" in p:
            return json.dumps(
                {
                    "key_methods": [
                        "Differential gene expression analysis with FDR correction",
                        "Pathway enrichment via hypergeometric test",
                        "PCA-based dimensionality reduction on transcriptomic profiles",
                    ],
                    "key_findings": [
                        "Stress-response pathways show coordinated upregulation under treatment",
                        "Cell-cycle genes exhibit inverse correlation with stress markers",
                        "Top principal components separate treatment from control with >80% variance explained",
                    ],
                    "limitations": [
                        "Small sample size limits statistical power for rare pathway detection",
                        "Batch effects not fully controlled across experimental runs",
                    ],
                    "future_work": [
                        "Validate findings with independent cohort or orthogonal assay",
                        "Investigate upstream regulators driving the observed expression changes",
                        "Apply single-cell resolution to identify cell-type-specific responses",
                    ],
                    "relevance_summary": (
                        "This paper's methods and findings directly overlap with our investigation "
                        "of gene expression differences between treatment groups. Their pathway enrichment "
                        "approach validates our analytical strategy."
                    ),
                    "suggested_hypotheses": [
                        "The stress-response pathway activation reported in this paper should be "
                        "reproducible in our dataset if the treatment mechanism is conserved",
                        "Upstream transcription factors identified in this study (e.g., NF-kB, AP-1) "
                        "may show differential binding site enrichment in our DE gene promoters",
                    ],
                }
            )
        if "interpret computational results" in p:
            return json.dumps(
                {
                    "verdict": "supported",
                    "confidence": 0.72,
                    "summary": (
                        "Observed statistically significant separation consistent with "
                        "pathway-level differential expression; caveats on confounding remain."
                    ),
                    "confounders": ["Potential batch effects if not randomized", "Sample size"],
                    "follow_ups": [
                        "Validate top hits with orthogonal database evidence",
                        "Check correlation of DE with technical covariates",
                    ],
                    "evidence_strength": 0.65,
                }
            )
        if "coordinate autonomous scientific discovery" in p:
            return json.dumps(
                {
                    "continue_thread": True,
                    "reason": "Information gain remains high; next cycle should refine mechanism.",
                    "suggested_focus": "Upstream regulators and protein–protein context",
                }
            )
        if "propose testable biological hypotheses" in p:
            return json.dumps(
                {
                    "hypotheses": [
                        {
                            "statement": (
                                "Coordinated differential expression in stress-response "
                                "and cell-cycle gene sets distinguishes treatment from control."
                            ),
                            "rationale": (
                                "Data summary shows separation on PC1 with enrichment of "
                                "annotated gene sets related to these processes."
                            ),
                            "testable_prediction": (
                                "Gene set enrichment on ranked differential expression "
                                "yields q<0.05 for at least one a priori pathway."
                            ),
                            "required_data": ["expression_matrix", "sample_groups", "pathway_gene_sets"],
                            "confidence_prior": 0.45,
                            "novelty_score": 0.55,
                        },
                        {
                            "statement": (
                                "A subset of genes with high variance drives batch-like "
                                "structure confounding group comparisons."
                            ),
                            "rationale": (
                                "High fraction of variance on early PCs may reflect technical "
                                "or biological heterogeneity worth testing before causal claims."
                            ),
                            "testable_prediction": (
                                "Association of top variable genes with batch covariates "
                                "is stronger than with treatment label."
                            ),
                            "required_data": ["expression_matrix", "sample_groups"],
                            "confidence_prior": 0.35,
                            "novelty_score": 0.4,
                        },
                    ]
                }
            )
        if "design a computational experiment plan" in p:
            hid_match = re.search(r"hypothesis[_ ]?id[\"']?\s*[:=]\s*([a-zA-Z0-9_\-]+)", prompt)
            hid = hid_match.group(1) if hid_match else f"h_mock"
            return json.dumps(
                {
                    "hypothesis_id": hid,
                    "steps": [
                        {
                            "tool": "profile_dataset",
                            "params": {},
                            "description": "Summarize dimensions, groups, and missingness.",
                        },
                        {
                            "tool": "differential_expression",
                            "params": {"method": "welch_t", "correction": "fdr_bh"},
                            "description": "Test per-gene differences between groups.",
                        },
                        {
                            "tool": "pathway_enrichment",
                            "params": {"direction": "up_down", "correction": "fdr_bh"},
                            "description": "Enrichment on ranked genes or DE hits.",
                        },
                        {
                            "tool": "plot_volcano",
                            "params": {"fdr_threshold": 0.05},
                            "description": "Visualize effect size vs significance.",
                        },
                    ],
                    "expected_duration": "short",
                    "success_criteria": (
                        "Significant pathway enrichment or coherent DE pattern matching "
                        "the biological process named in the hypothesis."
                    ),
                    "failure_criteria": "No significant enrichment after multiple-testing correction.",
                }
            )
        return json.dumps(
            {
                "note": "mock_llm_default",
                "echo_prefix": prompt[:200],
            }
        )


@dataclass
class OllamaLLMBackend:
    config: AppConfig

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/generate"
        payload: dict[str, Any] = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        r = requests.post(url, json=payload, timeout=self.config.request_timeout_s)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


@dataclass
class OpenAILLMBackend:
    config: AppConfig

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install openai package for OpenAI backend: pip install openai"
            ) from e
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=config.openai_api_key)

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class GatewayLLMBackend:
    """Internal LLM gateway via OpenAI-compatible API with Roblox auth."""

    def __init__(self, config: AppConfig) -> None:
        import httpx
        from openai import OpenAI
        from roblox_ml import cluster

        def _strip_sdk_auth(request: httpx.Request):
            request.headers.pop("authorization", None)

        api_key = cluster.get_secret("llm-gateway-api-key")
        self._client = OpenAI(
            base_url=config.gateway_base_url,
            api_key="unused",
            http_client=httpx.Client(event_hooks={"request": [_strip_sdk_auth]}),
            default_headers={"Roblox-Api-Key": api_key},
        )
        self._model = config.gateway_model

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class HuggingFaceLLMBackend:
    """Local GPU-accelerated LLM via HuggingFace transformers pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Install torch and transformers for HuggingFace backend"
            ) from e

        device = config.device
        print(f"[HuggingFaceLLMBackend] Loading {config.hf_model} on {device}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.hf_model, trust_remote_code=True
        )
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            config.hf_model,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._device = device

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print(f"[HuggingFaceLLMBackend] Model loaded. Device: {device}, "
              f"dtype: {dtype}, params: {sum(p.numel() for p in self._model.parameters()) / 1e6:.0f}M")

    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        import torch

        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            try:
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = f"{system}\n\n{prompt}" if system else prompt
        else:
            text = f"{system}\n\n{prompt}" if system else prompt

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.config.hf_max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
