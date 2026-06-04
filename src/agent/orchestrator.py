from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from src.agent.adversarial import adversarial_review, apply_adversarial_adjustment
from src.agent.experiment_executor import execute_plan
from src.agent.experiment_planner import plan_experiment, sanitize_plan
from src.agent.hypothesis_generator import generate_hypotheses
from src.agent.multi_agent import AgentCoordinator, MultiAgentConfig
from src.agent.result_analyzer import analyze_results
from src.agent.schemas import Hypothesis
from src.config import AppConfig, LLMBackend, get_config
from src.memory.experiment_log import ExperimentLog, ExperimentRecord
from src.memory.knowledge_store import (
    Finding,
    HypothesisRecord,
    KnowledgeStore,
    OpenQuestion,
    new_id,
)
from src.memory.paper_store import PaperStore
from src.memory.reproducibility import ReproducibilityLog, capture_bundle
from src.memory.retriever import MemoryRetriever
from src.planning.evaluation import evaluate_hypothesis_portfolio
from src.planning.strategy import assess_progress
from src.tools.data_analysis import ToolContext
from src.tools.literature import (
    PaperRecord,
    fetch_pubmed_abstracts,
    extract_paper_insights,
    literature_scan,
    search_pubmed,
)


@dataclass
class ResearchBudget:
    max_cycles: int = 5
    max_wall_time_s: float | None = None
    max_consecutive_pivots: int = 2


@dataclass
class OrchestratorConfig:
    research_question: str
    available_data_keys: list[str]
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    strategy_every_n_cycles: int = 2
    literature_review_every_n_cycles: int = 2
    literature_enabled: bool = True
    max_papers_per_scan: int = 3
    organism: str | None = None
    tissue: str | None = None
    condition: str | None = None
    use_multi_agent: bool = False
    auto_checkpoint: bool = True
    checkpoint_dir: str | None = None
    use_multi_agent: bool = False


@dataclass
class SessionState:
    session_id: str
    cycle: int
    decision_log: list[dict[str, Any]]
    research_question: str
    pivot_count: int = 0
    suggested_focus: str | None = None


class ResearchOrchestrator:
    def __init__(
        self,
        llm: LLMBackend,
        knowledge: KnowledgeStore,
        experiments: ExperimentLog,
        tool_ctx_factory: Callable[[], ToolContext],
        orch_cfg: OrchestratorConfig,
        dataset_summary_provider: Callable[[], dict[str, Any]],
        config: AppConfig | None = None,
        paper_store: PaperStore | None = None,
    ) -> None:
        self.llm = llm
        self.knowledge = knowledge
        self.experiments = experiments
        self.retriever = MemoryRetriever(knowledge, experiments, config)
        self.tool_ctx_factory = tool_ctx_factory
        self.orch_cfg = orch_cfg
        self.dataset_summary_provider = dataset_summary_provider
        self.config = config or get_config()
        self.paper_store = paper_store
        self.decision_log: list[dict[str, Any]] = []
        self.session_id = new_id("sess")
        self.cycle = 0
        self._start = time.time()
        self.pivot_count = 0
        self._suggested_focus: str | None = None
        self.repro_log = ReproducibilityLog(
            Path(self.orch_cfg.checkpoint_dir or "./outputs") / "reproducibility.json"
        )

        self._agent_coordinator: AgentCoordinator | None = None
        if orch_cfg.use_multi_agent:
            self._agent_coordinator = AgentCoordinator(
                llm, knowledge, MultiAgentConfig(enabled=True)
            )

    def _budget_exhausted(self) -> bool:
        b = self.orch_cfg.budget
        if self.cycle >= b.max_cycles:
            return True
        if b.max_wall_time_s is not None and (time.time() - self._start) >= b.max_wall_time_s:
            return True
        if self.pivot_count >= b.max_consecutive_pivots:
            return True
        return False

    def _persist_hypothesis(self, h: Hypothesis) -> None:
        self.knowledge.upsert_hypothesis(
            HypothesisRecord(
                id=h.id,
                statement=h.statement,
                rationale=h.rationale,
                testable_prediction=h.testable_prediction,
                required_data=h.required_data,
                confidence_prior=h.confidence_prior,
                novelty_score=h.novelty_score,
                status=h.status,
            )
        )

    def _build_lit_query(self) -> str:
        """Build a literature search query from the research question, recent findings, and dataset metadata."""
        _FILLER_WORDS = {
            "what", "which", "how", "does", "do", "are", "is", "the", "a", "an",
            "of", "in", "to", "and", "or", "from", "that", "this", "for", "with",
            "on", "by", "be", "it", "its", "our", "we", "can", "may", "between",
        }

        cfg = self.orch_cfg
        metadata_terms: list[str] = []
        if cfg.condition:
            metadata_terms.append(cfg.condition)
        if cfg.tissue:
            metadata_terms.append(cfg.tissue)
        if cfg.organism:
            metadata_terms.append(cfg.organism)

        if metadata_terms:
            question_tokens = cfg.research_question.split()
            key_terms = [t for t in question_tokens if t.lower().strip("?.,") not in _FILLER_WORDS]
            key_phrase = " ".join(key_terms[:6])
            parts = metadata_terms + [key_phrase]
        else:
            parts = [cfg.research_question]

        recent_findings = self.knowledge.all_findings()[-3:]
        for f in recent_findings:
            parts.append(f.statement[:100])
        return " ".join(parts)[:300]

    def _run_literature_review(self) -> dict[str, Any]:
        """Scan literature, extract insights, store papers, return summary for decision log."""
        if self.paper_store is None or not self.orch_cfg.literature_enabled:
            return {"skipped": True, "reason": "literature review disabled or no paper store"}

        query = self._build_lit_query()
        scan = literature_scan(
            query=query,
            research_context=self.knowledge.summary_blob(),
            llm=self.llm,
            config=self.config,
            max_papers=self.orch_cfg.max_papers_per_scan,
        )

        papers_added = 0
        for i, paper_dict in enumerate(scan.get("papers", [])):
            paper = PaperRecord(**paper_dict) if isinstance(paper_dict, dict) else paper_dict
            if self.paper_store.already_read(paper.paper_id):
                continue
            insights_list = scan.get("insights", [])
            if i < len(insights_list):
                from src.tools.literature import PaperInsights
                ins_data = insights_list[i]
                if isinstance(ins_data, dict):
                    ins = PaperInsights(**ins_data)
                else:
                    ins = ins_data
                self.paper_store.add_paper(paper, ins, research_context=query)
                papers_added += 1

        self.paper_store.save()

        return {
            "query": query,
            "papers_found": scan.get("papers_found", 0),
            "papers_analyzed": scan.get("papers_analyzed", 0),
            "new_papers_stored": papers_added,
            "suggested_hypotheses": scan.get("all_suggested_hypotheses", [])[:5],
            "methods_discovered": scan.get("all_methods_seen", [])[:5],
        }

    def run_cycle(self) -> dict[str, Any]:
        self.cycle += 1
        cycle_entry: dict[str, Any] = {
            "cycle": self.cycle,
            "session_id": self.session_id,
            "decisions": [],
        }

        # Phase 0: Literature review (periodic)
        if (
            self.orch_cfg.literature_enabled
            and self.paper_store is not None
            and (self.cycle == 1 or self.cycle % self.orch_cfg.literature_review_every_n_cycles == 0)
        ):
            lit_result = self._run_literature_review()
            cycle_entry["decisions"].append({"step": "literature_review", **lit_result})

        # Phase 1: Review knowledge state
        summary = self.knowledge.summary_blob()
        cycle_entry["decisions"].append(
            {"step": "review_knowledge", "summary_excerpt": summary[-1200:]}
        )

        # Phase 2: Generate hypotheses (now literature-aware)
        ds = self.dataset_summary_provider()
        hypotheses = generate_hypotheses(
            self.llm,
            self.orch_cfg.research_question,
            ds,
            self.knowledge,
            self.retriever,
            self.orch_cfg.available_data_keys,
            paper_store=self.paper_store,
            suggested_focus=self._suggested_focus,
        )
        self._suggested_focus = None
        for h in hypotheses:
            self._persist_hypothesis(h)
            if h.literature_grounded and h.source_papers and self.paper_store:
                for sp in h.source_papers:
                    self.paper_store.link_hypothesis(sp, h.id)

        # Phase 3: Rank and select
        ranked = evaluate_hypothesis_portfolio(hypotheses, set(self.orch_cfg.available_data_keys))
        cycle_entry["decisions"].append(
            {"step": "rank_hypotheses", "ranking": [asdict(x) for x in ranked]}
        )
        if not ranked:
            cycle_entry["decisions"].append({"step": "abort", "reason": "no hypotheses"})
            self.decision_log.append(cycle_entry)
            return cycle_entry

        top_id = ranked[0].hypothesis_id
        hyp = next(h for h in hypotheses if h.id == top_id)
        self.knowledge.update_hypothesis_status(hyp.id, "testing", posterior=None)

        # Phase 4: Plan experiment
        plan = sanitize_plan(
            plan_experiment(self.llm, hyp, ds, self.retriever),
            hyp.id,
        )
        cycle_entry["decisions"].append({"step": "plan", "plan": asdict(plan)})

        # Phase 5: Execute
        ctx = self.tool_ctx_factory()
        trace, aggregated = execute_plan(plan, ctx)
        cycle_entry["decisions"].append(
            {"step": "execute", "n_steps": len(trace), "trace": trace}
        )

        # Phase 5b: Capture reproducibility bundle
        try:
            bundle = capture_bundle(
                experiment_id=new_id("exp"),
                session_id=self.session_id,
                cycle=self.cycle,
                expression=ctx.expression,
                groups=ctx.groups,
                config={"research_question": self.orch_cfg.research_question},
                tool_sequence=[s.tool for s in plan.steps],
            )
            self.repro_log.add(bundle)
        except Exception:
            pass

        # Phase 6: Analyze results
        analysis = analyze_results(self.llm, hyp, aggregated)
        analysis_dict = {f.name: getattr(analysis, f.name) for f in fields(analysis)}
        cycle_entry["decisions"].append({"step": "analyze", "analysis": analysis_dict})

        # Phase 6b: Multi-agent critique (optional)
        if self._agent_coordinator is not None:
            critique_entry = self._agent_coordinator.critique_analysis(
                hypothesis_statement=hyp.statement,
                analysis_dict=analysis_dict,
                prior_confidence=analysis.posterior,
            )
            cycle_entry["decisions"].append(critique_entry)
            analysis.posterior = float(critique_entry["adjusted_posterior"])

        # Phase 6c: Adversarial hypothesis testing
        try:
            review = adversarial_review(self.llm, hyp, analysis, trace)
            analysis = apply_adversarial_adjustment(analysis, review)
            cycle_entry["decisions"].append({
                "step": "adversarial_review",
                "severity": review.severity,
                "recommendation": review.recommendation,
                "confidence_adjustment": review.confidence_adjustment,
                "objections": review.objections,
                "n_falsification_experiments": len(review.falsification_experiments),
            })
        except Exception as e:
            cycle_entry["decisions"].append({
                "step": "adversarial_review",
                "skipped": True,
                "reason": str(e),
            })

        # Phase 6d: SOTA-aware baseline comparison for model-building experiments
        ml_digest = analysis.raw.get("ml_models") or []
        if not ml_digest:
            from src.agent.result_analyzer import _numeric_digest
            nd = _numeric_digest(aggregated)
            ml_digest = nd.get("ml_models", [])

        baseline_comparison: dict[str, Any] | None = None
        if ml_digest:
            try:
                from src.agent.sota_scanner import collect_baselines
                bl = collect_baselines(
                    task=hyp.statement[:120],
                    ctx=self.tool_ctx_factory(),
                    dataset_name=self.orch_cfg.research_question[:60],
                )
                if bl:
                    baseline_comparison = {
                        "baselines": [
                            {"name": b.name, "metrics": b.metrics, "source": b.source}
                            for b in bl
                        ],
                        "novel_models": ml_digest,
                    }
                    cycle_entry["decisions"].append({
                        "step": "baseline_comparison",
                        "n_baselines": len(bl),
                        "n_novel_models": len(ml_digest),
                    })
            except Exception as e:
                cycle_entry["decisions"].append({
                    "step": "baseline_comparison",
                    "skipped": True,
                    "reason": str(e),
                })

        # Phase 7: Update knowledge
        exp_id = new_id("exp")
        interpretation: dict[str, Any] = {
            "verdict": analysis.verdict,
            "summary": analysis.summary,
            "confidence": analysis.confidence,
            "posterior": analysis.posterior,
            "follow_ups": analysis.follow_ups,
            "literature_grounded": hyp.literature_grounded,
            "source_papers": hyp.source_papers,
        }
        if baseline_comparison:
            interpretation["baseline_comparison"] = baseline_comparison

        record = ExperimentRecord(
            id=exp_id,
            hypothesis_id=hyp.id,
            plan=asdict(plan),
            execution_trace=trace,
            results=aggregated,
            interpretation=interpretation,
        )
        self.experiments.append(record)

        status_map = {
            "supported": "supported",
            "refuted": "refuted",
            "inconclusive": "inconclusive",
        }
        self.knowledge.update_hypothesis_status(
            hyp.id,
            status_map.get(analysis.verdict, "inconclusive"),
            posterior=analysis.posterior,
        )

        finding = Finding(
            id=new_id("find"),
            statement=analysis.summary,
            evidence_strength=analysis.evidence_strength,
            experiment_id=exp_id,
            hypothesis_id=hyp.id,
            provenance={
                "verdict": analysis.verdict,
                "details": analysis.raw,
                "literature_grounded": hyp.literature_grounded,
            },
            entities=[],
        )
        self.knowledge.add_finding(finding)

        for fu in analysis.follow_ups[:3]:
            self.knowledge.add_open_question(
                OpenQuestion(
                    id=new_id("q"),
                    question=str(fu),
                    priority=float(analysis.posterior),
                    linked_hypotheses=[hyp.id],
                )
            )

        self.knowledge.save()

        # Phase 8: Strategy check (periodic)
        if self.cycle % self.orch_cfg.strategy_every_n_cycles == 0:
            strat = assess_progress(
                self.llm,
                self.orch_cfg.research_question,
                self.knowledge.summary_blob(),
                self.cycle,
                self.config,
            )
            if strat.continue_thread:
                self.pivot_count = 0
            else:
                self.pivot_count += 1
            if strat.suggested_focus:
                self._suggested_focus = strat.suggested_focus
            cycle_entry["decisions"].append(
                {
                    "step": "strategy",
                    "continue_thread": strat.continue_thread,
                    "reason": strat.reason,
                    "suggested_focus": strat.suggested_focus,
                    "pivot_count": self.pivot_count,
                }
            )

        self.decision_log.append(cycle_entry)
        return cycle_entry

    def run(self) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        while not self._budget_exhausted():
            outcomes.append(self.run_cycle())
            if self.orch_cfg.auto_checkpoint and self.orch_cfg.checkpoint_dir:
                try:
                    self.checkpoint(self.orch_cfg.checkpoint_dir)
                except Exception as e:
                    print(f"Auto-checkpoint failed (cycle {self.cycle}): {e}")
        return outcomes

    def save_session(self, session_dir: Path | str) -> Path:
        root = Path(session_dir)
        root.mkdir(parents=True, exist_ok=True)
        state = SessionState(
            session_id=self.session_id,
            cycle=self.cycle,
            decision_log=self.decision_log,
            research_question=self.orch_cfg.research_question,
            pivot_count=self.pivot_count,
            suggested_focus=self._suggested_focus,
        )
        path = root / "orchestrator_state.json"
        path.write_text(json.dumps(asdict(state), indent=2))
        self.knowledge.save()
        if self.paper_store:
            self.paper_store.save()
        return path

    def checkpoint(self, session_dir: Path | str) -> Path:
        """Save full orchestrator state for later resumption.

        Persists everything ``save_session`` does, plus the orchestrator
        config and timing info so ``resume()`` can reconstruct the object.
        """
        root = Path(session_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.save_session(root)

        checkpoint_data = {
            "session_id": self.session_id,
            "cycle": self.cycle,
            "pivot_count": self.pivot_count,
            "suggested_focus": self._suggested_focus,
            "elapsed_before_checkpoint_s": time.time() - self._start,
            "orch_cfg": asdict(self.orch_cfg),
            "decision_log": self.decision_log,
        }
        ckpt_path = root / "checkpoint.json"
        ckpt_path.write_text(json.dumps(checkpoint_data, indent=2))
        print(f"Checkpoint saved at cycle {self.cycle} → {ckpt_path}")
        return ckpt_path

    @classmethod
    def resume(
        cls,
        session_dir: Path | str,
        llm: LLMBackend,
        knowledge: KnowledgeStore,
        experiments: ExperimentLog,
        tool_ctx_factory: Callable[[], ToolContext],
        dataset_summary_provider: Callable[[], dict[str, Any]],
        config: AppConfig | None = None,
        paper_store: PaperStore | None = None,
    ) -> "ResearchOrchestrator":
        """Restore an orchestrator from a checkpoint and continue running.

        The ``KnowledgeStore`` and ``ExperimentLog`` should be opened from
        the same paths used in the original session (they self-persist).
        """
        root = Path(session_dir)
        ckpt_path = root / "checkpoint.json"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. "
                f"Use save_session() for metadata-only snapshots, "
                f"or checkpoint() for resumable state."
            )

        ckpt = json.loads(ckpt_path.read_text())
        orch_cfg_raw = ckpt["orch_cfg"]

        budget_raw = orch_cfg_raw.pop("budget", {})
        budget = ResearchBudget(**budget_raw)
        orch_cfg = OrchestratorConfig(budget=budget, **orch_cfg_raw)

        inst = cls(
            llm=llm,
            knowledge=knowledge,
            experiments=experiments,
            tool_ctx_factory=tool_ctx_factory,
            orch_cfg=orch_cfg,
            dataset_summary_provider=dataset_summary_provider,
            config=config,
            paper_store=paper_store,
        )

        inst.session_id = ckpt["session_id"]
        inst.cycle = ckpt["cycle"]
        inst.pivot_count = ckpt.get("pivot_count", 0)
        inst._suggested_focus = ckpt.get("suggested_focus")
        inst.decision_log = ckpt.get("decision_log", [])
        elapsed_prior = ckpt.get("elapsed_before_checkpoint_s", 0.0)
        inst._start = time.time() - elapsed_prior

        print(
            f"Resumed session {inst.session_id} at cycle {inst.cycle} "
            f"({elapsed_prior:.0f}s elapsed prior)"
        )
        return inst

    @staticmethod
    def load_session_metadata(session_dir: Path | str) -> SessionState:
        path = Path(session_dir) / "orchestrator_state.json"
        raw = json.loads(path.read_text())
        return SessionState(
            session_id=raw["session_id"],
            cycle=raw["cycle"],
            decision_log=raw["decision_log"],
            research_question=raw["research_question"],
            pivot_count=raw.get("pivot_count", 0),
            suggested_focus=raw.get("suggested_focus"),
        )
