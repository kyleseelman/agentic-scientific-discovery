from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(str, Enum):
    MOCK = "mock"
    OLLAMA = "ollama"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    GATEWAY = "gateway"


@runtime_checkable
class LLMBackend(Protocol):
    def generate(self, prompt: str, system: str = "", temperature: float = 0.7) -> str:
        """Return model text completion for the given prompt."""
        ...


def _detect_gpu_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class AppConfig:
    llm_provider: LLMProvider = LLMProvider.MOCK
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    openai_model: str = "gpt-4o-mini"
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    gateway_base_url: str = "https://apis.sitetest3.simulpong.com/llm-gateway/v1"
    gateway_model: str = "gpt-5.3-codex"
    hf_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    hf_max_new_tokens: int = 1024
    research_session_dir: str = "./research_sessions"
    random_seed: int = 42
    request_timeout_s: int = 60
    embedding_max_features: int = 256
    device: str = field(default_factory=_detect_gpu_device)
    use_gpu_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    use_gpu_compute: bool = True


def get_config() -> AppConfig:
    provider = os.environ.get("LLM_PROVIDER", "mock").lower()
    try:
        llm = LLMProvider(provider)
    except ValueError:
        llm = LLMProvider.MOCK
    device = os.environ.get("DEVICE", _detect_gpu_device())
    use_gpu_embeddings = os.environ.get("USE_GPU_EMBEDDINGS", "true").lower() == "true"
    use_gpu_compute = os.environ.get("USE_GPU_COMPUTE", "true").lower() == "true"
    return AppConfig(
        llm_provider=llm,
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        gateway_base_url=os.environ.get(
            "ML_GATEWAY_URL_CODEX",
            "https://apis.sitetest3.simulpong.com/llm-gateway/v1",
        ),
        gateway_model=os.environ.get("GATEWAY_MODEL", "gpt-5.3-codex"),
        hf_model=os.environ.get("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        hf_max_new_tokens=int(os.environ.get("HF_MAX_NEW_TOKENS", "1024")),
        research_session_dir=os.environ.get("RESEARCH_SESSION_DIR", "./research_sessions"),
        random_seed=int(os.environ.get("RANDOM_SEED", "42")),
        device=device,
        use_gpu_embeddings=use_gpu_embeddings,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        use_gpu_compute=use_gpu_compute,
    )


def create_llm_backend(config: AppConfig | None = None) -> LLMBackend:
    from src.llm_backends import (
        GatewayLLMBackend,
        HuggingFaceLLMBackend,
        MockLLMBackend,
        OllamaLLMBackend,
        OpenAILLMBackend,
    )

    cfg = config or get_config()
    if cfg.llm_provider == LLMProvider.OLLAMA:
        return OllamaLLMBackend(cfg)
    if cfg.llm_provider == LLMProvider.OPENAI:
        return OpenAILLMBackend(cfg)
    if cfg.llm_provider == LLMProvider.HUGGINGFACE:
        return HuggingFaceLLMBackend(cfg)
    if cfg.llm_provider == LLMProvider.GATEWAY:
        return GatewayLLMBackend(cfg)
    return MockLLMBackend(cfg)
