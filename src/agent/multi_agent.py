"""Multi-agent coordination system for adversarial scientific discovery.

Specialist agents (Critic, Literature, Experimentalist, Synthesis) each use
the same LLM backend but with role-specific system prompts.  The
AgentCoordinator manages turn-taking, message routing, and structured
protocols such as critique, hypothesis debate, deep review, and adversarial
replication.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.config import LLMBackend
from src.memory.knowledge_store import KnowledgeStore
from src.utils.json_extract import extract_json_object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    content: str
    message_type: str  # "request", "response", "critique", "suggestion"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class CritiqueResult:
    objections: list[str]
    severity: str  # "minor", "moderate", "major", "fatal"
    falsification_experiments: list[str]
    confidence_adjustment: float
    summary: str
    agent_messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class ReviewResult:
    structured_review: str
    papers_found: int
    methodology_assessment: str
    gaps: list[str]
    novel_angles: list[str]
    agent_messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class DebateOutcome:
    ranked_hypotheses: list[dict[str, Any]]
    objections: dict[str, list[str]]
    consensus_notes: str
    agent_messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class ReplicationResult:
    reproducibility_score: float
    alternative_experiments: list[dict[str, Any]]
    discrepancies: list[str]
    assessment: str
    agent_messages: list[AgentMessage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Specialist Agent protocol & implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class SpecialistAgent(Protocol):
    name: str

    def process(self, message: AgentMessage, context: dict[str, Any]) -> AgentMessage:
        ...


class _BaseAgent:
    """Shared LLM-calling logic for all specialist agents."""

    def __init__(self, name: str, system_prompt: str, llm: LLMBackend) -> None:
        self.name = name
        self._system_prompt = system_prompt
        self._llm = llm

    def _call_llm(
        self,
        user_prompt: str,
        *,
        temperature: float = 0.4,
        json_mode: bool = False,
    ) -> str:
        system = self._system_prompt
        if json_mode:
            system += "\n\nRespond ONLY with valid JSON."
        return self._llm.generate(user_prompt, system=system, temperature=temperature)

    def _call_llm_json(
        self, user_prompt: str, *, temperature: float = 0.3
    ) -> dict[str, Any]:
        raw = self._call_llm(user_prompt, temperature=temperature, json_mode=True)
        return extract_json_object(raw)

    def _make_reply(
        self,
        original: AgentMessage,
        content: str,
        message_type: str = "response",
        **meta: Any,
    ) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            recipient=original.sender,
            content=content,
            message_type=message_type,
            metadata=meta,
        )


# ---- CriticAgent -----------------------------------------------------------

_CRITIC_SYSTEM = """\
You are a rigorous scientific critic.  Your job is to find flaws in
hypotheses and experimental conclusions.  Act as a devil's advocate.

Check for:
- Confounders and alternative explanations
- Insufficient sample sizes or statistical power
- Multiple testing / p-hacking risk
- Overfitting risk in ML analyses
- Contradictions with published literature
- Selection bias, survivorship bias, batch effects

Always suggest at least one falsification experiment that could disprove
the claim.  Rate the overall severity of your objections as one of:
minor, moderate, major, fatal.

Output a confidence_adjustment between -0.3 (severe problems) and 0.0
(no issues found).
"""


class CriticAgent(_BaseAgent):
    def __init__(self, llm: LLMBackend) -> None:
        super().__init__("critic", _CRITIC_SYSTEM, llm)

    def process(self, message: AgentMessage, context: dict[str, Any]) -> AgentMessage:
        prompt = (
            f"Review the following hypothesis and evidence.\n\n"
            f"Hypothesis: {context.get('hypothesis', 'N/A')}\n\n"
            f"Evidence / Results:\n{message.content}\n\n"
            f"Additional context:\n{json.dumps(context.get('extra', {}), default=str)[:2000]}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "objections": ["..."],\n'
            '  "severity": "minor|moderate|major|fatal",\n'
            '  "falsification_experiments": ["..."],\n'
            '  "confidence_adjustment": -0.1,\n'
            '  "summary": "..."\n'
            "}\n"
        )
        obj = self._call_llm_json(prompt)
        content = json.dumps(obj)
        return self._make_reply(message, content, message_type="critique")


# ---- LiteratureAgent -------------------------------------------------------

_LITERATURE_SYSTEM = """\
You are a specialist literature review agent for biomedical research.
Given a research question or topic, you:
1. Identify key search terms for PubMed and bioRxiv.
2. Summarise relevant findings from the literature.
3. Identify contradictions between current findings and published work.
4. Suggest novel angles not yet explored.

Be thorough and cite specific concepts or known results where possible.
"""


class LiteratureAgent(_BaseAgent):
    def __init__(self, llm: LLMBackend) -> None:
        super().__init__("literature", _LITERATURE_SYSTEM, llm)

    def process(self, message: AgentMessage, context: dict[str, Any]) -> AgentMessage:
        knowledge_summary = context.get("knowledge_summary", "")
        prompt = (
            f"Research question / topic:\n{message.content}\n\n"
            f"Current knowledge state:\n{knowledge_summary[:2000]}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "search_terms": ["..."],\n'
            '  "key_findings": ["..."],\n'
            '  "contradictions": ["..."],\n'
            '  "novel_angles": ["..."],\n'
            '  "methodology_notes": "..."\n'
            "}\n"
        )
        obj = self._call_llm_json(prompt)
        return self._make_reply(message, json.dumps(obj), message_type="response")


# ---- ExperimentalistAgent --------------------------------------------------

_EXPERIMENTALIST_SYSTEM = """\
You are an expert experimentalist for computational biology.  Given a
hypothesis, you design rigorous experiments, suggest controls, propose
replication strategies, and recommend appropriate analysis methods
(ML vs traditional statistics).

Consider:
- Proper positive and negative controls
- Sample size / statistical power
- Cross-validation and held-out test sets for ML
- Batch-effect correction
- Multiple-hypothesis correction
- Independent replication with different methods
"""


class ExperimentalistAgent(_BaseAgent):
    def __init__(self, llm: LLMBackend) -> None:
        super().__init__("experimentalist", _EXPERIMENTALIST_SYSTEM, llm)

    def process(self, message: AgentMessage, context: dict[str, Any]) -> AgentMessage:
        hypothesis = context.get("hypothesis", "")
        available_data = context.get("available_data", [])
        prompt = (
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Available data keys: {available_data}\n\n"
            f"Current evidence:\n{message.content}\n\n"
            "Return JSON:\n"
            "{\n"
            '  "experimental_design": "...",\n'
            '  "controls": ["..."],\n'
            '  "replication_strategy": "...",\n'
            '  "recommended_methods": ["..."],\n'
            '  "ml_vs_stats": "...",\n'
            '  "power_considerations": "...",\n'
            '  "alternative_experiments": ["..."]\n'
            "}\n"
        )
        obj = self._call_llm_json(prompt)
        return self._make_reply(message, json.dumps(obj), message_type="suggestion")


# ---- SynthesisAgent --------------------------------------------------------

_SYNTHESIS_SYSTEM = """\
You are a synthesis agent that integrates perspectives from multiple
specialist agents.  You:
1. Consolidate findings across agents and research cycles.
2. Identify emerging patterns and consistent themes.
3. Flag contradictions that need resolution.
4. Produce a balanced, integrated summary.
5. Adjust confidence based on level of agreement among agents.

Weigh evidence carefully: strong agreement across independent analyses
increases confidence; unresolved objections decrease it.
"""


class SynthesisAgent(_BaseAgent):
    def __init__(self, llm: LLMBackend) -> None:
        super().__init__("synthesis", _SYNTHESIS_SYSTEM, llm)

    def process(self, message: AgentMessage, context: dict[str, Any]) -> AgentMessage:
        agent_outputs = context.get("agent_outputs", {})
        prompt = (
            "Integrate the following agent perspectives into a coherent synthesis.\n\n"
            f"Topic / hypothesis:\n{message.content}\n\n"
        )
        for agent_name, output in agent_outputs.items():
            prompt += f"--- {agent_name} ---\n{output[:1500]}\n\n"

        prompt += (
            "Return JSON:\n"
            "{\n"
            '  "integrated_summary": "...",\n'
            '  "consensus_points": ["..."],\n'
            '  "disagreements": ["..."],\n'
            '  "confidence_adjustment": 0.0,\n'
            '  "emerging_patterns": ["..."],\n'
            '  "recommended_next_steps": ["..."]\n'
            "}\n"
        )
        obj = self._call_llm_json(prompt)
        return self._make_reply(message, json.dumps(obj), message_type="response")


# ---------------------------------------------------------------------------
# AgentCoordinator
# ---------------------------------------------------------------------------


@dataclass
class MultiAgentConfig:
    enabled: bool = False
    critique_temperature: float = 0.4
    debate_rounds: int = 1
    require_falsification: bool = True


class AgentCoordinator:
    """Coordinates multiple specialist agents in structured protocols."""

    def __init__(
        self,
        llm: LLMBackend,
        knowledge_store: KnowledgeStore,
        config: MultiAgentConfig | None = None,
    ) -> None:
        self.llm = llm
        self.knowledge = knowledge_store
        self.config = config or MultiAgentConfig()
        self.agents: dict[str, SpecialistAgent] = {}
        self.message_log: list[AgentMessage] = []

        self.register_agent(CriticAgent(llm))
        self.register_agent(LiteratureAgent(llm))
        self.register_agent(ExperimentalistAgent(llm))
        self.register_agent(SynthesisAgent(llm))

    # -- registration -------------------------------------------------------

    def register_agent(self, agent: SpecialistAgent) -> None:
        self.agents[agent.name] = agent

    # -- helpers ------------------------------------------------------------

    def _send(
        self,
        sender: str,
        recipient: str,
        content: str,
        message_type: str = "request",
        context: dict[str, Any] | None = None,
        **meta: Any,
    ) -> AgentMessage:
        msg = AgentMessage(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=message_type,
            metadata=meta,
        )
        self.message_log.append(msg)

        agent = self.agents.get(recipient)
        if agent is None:
            logger.warning("No agent registered as %r; skipping", recipient)
            return msg

        reply = agent.process(msg, context or {})
        self.message_log.append(reply)
        return reply

    def _parse_json_content(self, msg: AgentMessage) -> dict[str, Any]:
        try:
            return json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            return extract_json_object(msg.content)

    # -- Critique Protocol --------------------------------------------------

    def run_critique_protocol(
        self,
        hypothesis: str,
        evidence: dict[str, Any],
    ) -> CritiqueResult:
        """Three-step critique: experimentalist -> critic -> synthesis."""
        evidence_str = json.dumps(evidence, default=str)[:3000]
        ctx_base: dict[str, Any] = {"hypothesis": hypothesis, "extra": evidence}

        # Step 1: Experimentalist reviews the evidence
        exp_reply = self._send(
            "coordinator",
            "experimentalist",
            evidence_str,
            context={**ctx_base, "available_data": list(evidence.keys())},
        )

        # Step 2: Critic challenges the conclusion
        critic_reply = self._send(
            "coordinator",
            "critic",
            evidence_str,
            context=ctx_base,
        )

        # Step 3: Synthesis integrates perspectives
        synth_reply = self._send(
            "coordinator",
            "synthesis",
            hypothesis,
            context={
                "agent_outputs": {
                    "experimentalist": exp_reply.content,
                    "critic": critic_reply.content,
                },
            },
        )

        critic_obj = self._parse_json_content(critic_reply)
        synth_obj = self._parse_json_content(synth_reply)

        objections = [str(o) for o in critic_obj.get("objections", [])]
        severity = str(critic_obj.get("severity", "moderate"))
        falsification = [str(f) for f in critic_obj.get("falsification_experiments", [])]
        if not falsification and self.config.require_falsification:
            falsification = ["Replicate with independent dataset or method"]

        critic_adj = float(critic_obj.get("confidence_adjustment", -0.05))
        synth_adj = float(synth_obj.get("confidence_adjustment", 0.0))
        combined_adj = float(np.clip(critic_adj + synth_adj, -0.4, 0.1))

        summary_parts = [
            synth_obj.get("integrated_summary", ""),
            f"Severity: {severity}",
            f"Objections: {len(objections)}",
        ]

        return CritiqueResult(
            objections=objections,
            severity=severity,
            falsification_experiments=falsification,
            confidence_adjustment=combined_adj,
            summary=" | ".join(summary_parts),
            agent_messages=[exp_reply, critic_reply, synth_reply],
        )

    # -- Hypothesis Debate ---------------------------------------------------

    def run_hypothesis_debate(
        self,
        hypotheses: list[dict[str, Any]],
    ) -> DebateOutcome:
        """Multi-agent debate to refine and rank hypotheses.

        1. Each agent scores hypotheses from their perspective
        2. Critic raises objections
        3. Synthesis produces final ranking
        """
        hyp_text = "\n".join(
            f"{i+1}. {h.get('statement', h.get('id', '?'))}"
            for i, h in enumerate(hypotheses)
        )

        scores: dict[str, list[float]] = {}
        all_objections: dict[str, list[str]] = {}

        for agent_name in ("experimentalist", "literature", "critic"):
            agent = self.agents.get(agent_name)
            if agent is None:
                continue

            msg = AgentMessage(
                sender="coordinator",
                recipient=agent_name,
                content=f"Score each hypothesis 0-10 from your perspective.\n\n{hyp_text}",
                message_type="request",
            )
            self.message_log.append(msg)

            ctx: dict[str, Any] = {
                "hypothesis": hyp_text,
                "knowledge_summary": self.knowledge.summary_blob(),
            }
            reply = agent.process(msg, ctx)
            self.message_log.append(reply)

            obj = self._parse_json_content(reply)

            if agent_name == "critic":
                for key, val in obj.items():
                    if isinstance(val, list) and key == "objections":
                        for i, h in enumerate(hypotheses):
                            hid = h.get("id", str(i))
                            all_objections.setdefault(hid, []).extend(
                                str(o) for o in val
                            )
            raw_scores = obj.get("scores", [])
            if isinstance(raw_scores, list):
                scores[agent_name] = [
                    float(s) if isinstance(s, (int, float)) else 5.0
                    for s in raw_scores
                ]

        n = len(hypotheses)
        averaged: list[float] = []
        for i in range(n):
            vals = [s[i] for s in scores.values() if i < len(s)]
            averaged.append(float(np.mean(vals)) if vals else 5.0)

        ranked_indices = list(np.argsort(averaged)[::-1])
        ranked = []
        for idx in ranked_indices:
            h = hypotheses[idx]
            ranked.append({
                **h,
                "multi_agent_score": round(averaged[idx], 2),
            })

        synth_reply = self._send(
            "coordinator",
            "synthesis",
            hyp_text,
            context={
                "agent_outputs": {
                    name: json.dumps(s) for name, s in scores.items()
                },
            },
        )
        synth_obj = self._parse_json_content(synth_reply)

        return DebateOutcome(
            ranked_hypotheses=ranked,
            objections=all_objections,
            consensus_notes=str(synth_obj.get("integrated_summary", "")),
            agent_messages=list(self.message_log[-len(scores) * 2 - 2:]),
        )

    # -- Deep Review ---------------------------------------------------------

    def run_deep_review(self, topic: str) -> ReviewResult:
        """Coordinated deep literature review across four agents.

        1. Literature agent searches
        2. Experimentalist evaluates methodology
        3. Critic identifies gaps
        4. Synthesis produces structured review
        """
        knowledge_summary = self.knowledge.summary_blob()

        # Step 1: Literature search
        lit_reply = self._send(
            "coordinator",
            "literature",
            topic,
            context={"knowledge_summary": knowledge_summary},
        )
        lit_obj = self._parse_json_content(lit_reply)

        # Step 2: Experimentalist evaluates methodology
        exp_reply = self._send(
            "coordinator",
            "experimentalist",
            f"Evaluate the methodology described in these findings:\n{lit_reply.content}",
            context={
                "hypothesis": topic,
                "available_data": [],
            },
        )

        # Step 3: Critic identifies gaps
        critic_reply = self._send(
            "coordinator",
            "critic",
            f"Identify gaps in this literature review:\n{lit_reply.content}",
            context={"hypothesis": topic},
        )
        critic_obj = self._parse_json_content(critic_reply)

        # Step 4: Synthesis
        synth_reply = self._send(
            "coordinator",
            "synthesis",
            topic,
            context={
                "agent_outputs": {
                    "literature": lit_reply.content,
                    "experimentalist": exp_reply.content,
                    "critic": critic_reply.content,
                },
            },
        )
        synth_obj = self._parse_json_content(synth_reply)

        exp_obj = self._parse_json_content(exp_reply)
        methodology = exp_obj.get(
            "experimental_design",
            exp_obj.get("ml_vs_stats", ""),
        )

        return ReviewResult(
            structured_review=str(synth_obj.get("integrated_summary", "")),
            papers_found=len(lit_obj.get("key_findings", [])),
            methodology_assessment=str(methodology),
            gaps=[str(g) for g in critic_obj.get("objections", [])],
            novel_angles=[str(a) for a in lit_obj.get("novel_angles", [])],
            agent_messages=[lit_reply, exp_reply, critic_reply, synth_reply],
        )

    # -- Replication Protocol ------------------------------------------------

    def run_replication_protocol(
        self,
        finding: str,
        experiment_log: dict[str, Any],
    ) -> ReplicationResult:
        """Adversarial replication assessment.

        1. Experimentalist designs alternative experiments
        2. Critic compares methodology
        3. Synthesis assesses reproducibility
        """
        log_str = json.dumps(experiment_log, default=str)[:3000]

        # Step 1: Experimentalist designs alternatives
        exp_reply = self._send(
            "coordinator",
            "experimentalist",
            f"Design alternative experiments to replicate this finding:\n{finding}\n\nOriginal log:\n{log_str}",
            context={
                "hypothesis": finding,
                "available_data": list(experiment_log.keys()),
            },
        )
        exp_obj = self._parse_json_content(exp_reply)

        # Step 2: Critic compares
        critic_reply = self._send(
            "coordinator",
            "critic",
            f"Compare the original experiment with proposed alternatives.\nFinding: {finding}\nOriginal: {log_str}\nAlternatives: {exp_reply.content}",
            context={"hypothesis": finding},
        )
        critic_obj = self._parse_json_content(critic_reply)

        # Step 3: Synthesis
        synth_reply = self._send(
            "coordinator",
            "synthesis",
            finding,
            context={
                "agent_outputs": {
                    "experimentalist": exp_reply.content,
                    "critic": critic_reply.content,
                },
            },
        )
        synth_obj = self._parse_json_content(synth_reply)

        severity = critic_obj.get("severity", "moderate")
        severity_scores = {"minor": 0.85, "moderate": 0.6, "major": 0.35, "fatal": 0.1}
        repro_base = severity_scores.get(str(severity), 0.5)
        synth_adj = float(synth_obj.get("confidence_adjustment", 0.0))
        repro_score = float(np.clip(repro_base + synth_adj, 0.0, 1.0))

        alt_experiments = exp_obj.get("alternative_experiments", [])
        if isinstance(alt_experiments, list):
            alt_experiments = [
                {"description": str(e)} if not isinstance(e, dict) else e
                for e in alt_experiments
            ]
        else:
            alt_experiments = []

        return ReplicationResult(
            reproducibility_score=repro_score,
            alternative_experiments=alt_experiments,
            discrepancies=[str(d) for d in critic_obj.get("objections", [])],
            assessment=str(synth_obj.get("integrated_summary", "")),
            agent_messages=[exp_reply, critic_reply, synth_reply],
        )

    # -- Convenience for orchestrator integration ----------------------------

    def critique_analysis(
        self,
        hypothesis_statement: str,
        analysis_dict: dict[str, Any],
        prior_confidence: float,
    ) -> dict[str, Any]:
        """Run the critique protocol and return a dict suitable for the
        orchestrator decision log, including an adjusted posterior.
        """
        result = self.run_critique_protocol(hypothesis_statement, analysis_dict)

        adjusted_posterior = float(
            np.clip(prior_confidence + result.confidence_adjustment, 0.02, 0.98)
        )

        return {
            "step": "multi_agent_critique",
            "objections": result.objections,
            "severity": result.severity,
            "falsification_experiments": result.falsification_experiments,
            "confidence_adjustment": result.confidence_adjustment,
            "adjusted_posterior": adjusted_posterior,
            "summary": result.summary,
            "n_messages": len(result.agent_messages),
        }
