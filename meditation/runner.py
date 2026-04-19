"""Meditation pipeline runner — harvest → bucket → consolidation."""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

MEDITATION_HOME = get_hermes_home() / "meditation"


class MeditationPipelineRunner:
    """Run the full meditation cycle: extract, bucket, consolidate, write."""

    def __init__(
        self,
        db,
        lookback_hours: int = 24,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.db = db
        self.lookback_hours = lookback_hours
        self.model = model
        self.provider = provider
        from meditation.episode_extractor import EpisodeExtractor
        self.extractor = EpisodeExtractor(db, lookback_hours=lookback_hours)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self, tick_at: Optional[float] = None) -> Dict[str, Any]:
        tick_at = tick_at or time.time()
        date_str = time.strftime("%Y-%m-%d", time.localtime(tick_at))
        output_dir = MEDITATION_HOME / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Meditation started for %s (lookback=%dh)", date_str, self.lookback_hours)

        # 1. Harvest
        episodes = self.extractor.extract_all()
        if not episodes:
            logger.info("No friction episodes found in the last %dh. Silent exit.", self.lookback_hours)
            self._write_silent(output_dir)
            return {"status": "silent", "episodes": 0, "buckets": 0, "rules": 0}

        logger.info("Harvested %d episodes", len(episodes))
        ep_dicts = [self._episode_to_dict(e) for e in episodes]

        # 2. Bucket
        buckets = self._bucket_episodes(ep_dicts)
        logger.info("Formed %d buckets", len(buckets))

        # 3. Consolidate
        rules = self._consolidate_buckets(buckets)
        logger.info("Generated %d candidate rules", len(rules))

        # 4. Write outputs
        self._write_outputs(output_dir, date_str, ep_dicts, buckets, rules)

        return {
            "status": "ok",
            "episodes": len(episodes),
            "buckets": len(buckets),
            "rules": len(rules),
            "output_dir": str(output_dir),
        }

    # ------------------------------------------------------------------
    # Bucket stage
    # ------------------------------------------------------------------

    def _bucket_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from meditation.prompts import BUCKET_SYSTEM_PROMPT, build_bucket_prompt
        # If too many episodes, down-sample to keep prompt size reasonable
        if len(episodes) > 30:
            logger.info("Too many episodes (%d); sampling down to ~30 for bucket analysis", len(episodes))
            # Stratified sample: preserve at least one of each type, then fill by severity
            by_type: Dict[str, List[Dict]] = {}
            for ep in episodes:
                by_type.setdefault(ep.get("episode_type", "unknown"), []).append(ep)
            sampled = []
            for t, eps in by_type.items():
                # Take the most severe (high tool_call_count / abnormal end) first
                eps_sorted = sorted(eps, key=lambda e: e.get("tool_call_count", 0), reverse=True)
                sampled.extend(eps_sorted[:3])  # cap per type
            # Fill remainder up to 30
            remaining = [e for e in episodes if e not in sampled]
            remaining.sort(key=lambda e: e.get("tool_call_count", 0), reverse=True)
            sampled.extend(remaining[: max(0, 30 - len(sampled))])
            episodes = sampled

        prompt_text = build_bucket_prompt(episodes)
        messages = [
            {"role": "system", "content": BUCKET_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        raw = self._call_llm(messages, max_tokens=4096)
        buckets = self._safe_json_parse(raw, default=[])
        if not isinstance(buckets, list):
            buckets = []

        # Attach full episode objects to each bucket for consolidation
        for b in buckets:
            indices = b.get("episode_indices", [])
            b["episodes"] = [episodes[i] for i in indices if 0 <= i < len(episodes)]
        return buckets

    # ------------------------------------------------------------------
    # Consolidation stage
    # ------------------------------------------------------------------

    def _consolidate_buckets(self, buckets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from meditation.prompts import CONSOLIDATION_SYSTEM_PROMPT, build_consolidation_prompt
        rules = []
        for bucket in buckets:
            if not bucket.get("episodes"):
                continue
            prompt_text = build_consolidation_prompt(bucket)
            messages = [
                {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
            raw = self._call_llm(messages, max_tokens=2048)
            rule = self._safe_json_parse(raw, default=None)
            if rule and isinstance(rule, dict) and rule.get("rule"):
                rule["bucket_name"] = bucket.get("name", "unknown")
                rules.append(rule)
        return rules

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        try:
            from agent.auxiliary_client import call_llm
            # Default to alibaba/qwen-plus for meditation analysis because
            # kimi-k2.5 produces enormous reasoning_content that exhausts
            # max_tokens before emitting any content.
            provider = self.provider or "alibaba"
            model = self.model or "qwen-plus"
            resp = call_llm(
                task="meditation",
                messages=messages,
                model=model,
                provider=provider,
                max_tokens=max_tokens,
                temperature=0.3,
                timeout=120,
            )
            content = resp.choices[0].message.content
            # Fallback: some reasoning models return empty content but have
            # reasoning_content. We can't parse that reliably, so we prefer
            # non-reasoning providers for meditation.
            return content or ""
        except Exception as e:
            logger.warning("Meditation LLM call failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_to_dict(ep) -> Dict[str, Any]:
        d = asdict(ep)
        # Prune heavy fields for prompt sizing
        d["messages"] = [
            {
                "role": m.get("role"),
                "content": (m.get("content") or "")[:500],
                "tool_name": m.get("tool_name"),
            }
            for m in d.get("messages", [])
        ]
        return d

    @staticmethod
    def _safe_json_parse(text: str, default: Any = None) -> Any:
        if not text:
            return default
        # Strip markdown fences
        t = text.strip()
        if t.startswith("```json"):
            t = t[7:]
        elif t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON from LLM output: %r", text[:200])
            return default

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_outputs(
        self,
        output_dir: Path,
        date_str: str,
        episodes: List[Dict],
        buckets: List[Dict],
        rules: List[Dict],
    ):
        from meditation.prompts import render_overview

        # overview.md
        overview = render_overview(
            date=date_str,
            lookback_hours=self.lookback_hours,
            session_count=len({e["session_id"] for e in episodes}),
            episode_count=len(episodes),
            buckets=buckets,
            rules=rules,
        )
        (output_dir / "overview.md").write_text(overview, encoding="utf-8")

        # pending_rules.md
        lines = ["# Pending Rules\n"]
        for r in rules:
            lines.append(f"## {r.get('bucket_name', 'Unknown')}")
            lines.append(f"- **Target:** {r.get('target', 'uncertain')}")
            lines.append(f"- **Confidence:** {r.get('confidence', 'medium')}")
            lines.append(f"- **Rule:** {r.get('rule', '')}")
            lines.append(f"- **Rationale:** {r.get('rationale', '')}")
            if r.get("skill_name"):
                lines.append(f"- **Suggested skill:** `{r['skill_name']}` ({r.get('skill_category', 'unknown')})")
            lines.append("")
        (output_dir / "pending_rules.md").write_text("\n".join(lines), encoding="utf-8")

        # raw_episodes.json
        (output_dir / "raw_episodes.json").write_text(
            json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # buckets.json
        # Strip full episode messages to keep it readable
        buckets_light = []
        for b in buckets:
            bl = dict(b)
            bl["episodes"] = [
                {"session_id": e["session_id"], "episode_type": e["episode_type"], "description": e["description"]}
                for e in b.get("episodes", [])
            ]
            buckets_light.append(bl)
        (output_dir / "buckets.json").write_text(
            json.dumps(buckets_light, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info("Meditation outputs written to %s", output_dir)

    def _write_silent(self, output_dir: Path):
        (output_dir / "overview.md").write_text(
            "# Meditation Report — No friction detected\n\n"
            "No episodes were extracted in the lookback window. "
            "This is a good sign — sessions ran smoothly.\n",
            encoding="utf-8",
        )
