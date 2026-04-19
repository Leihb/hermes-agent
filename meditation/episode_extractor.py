"""Friction episode extraction from Hermes session history."""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Keywords that suggest user correction or dissatisfaction
_CORRECTION_KEYWORDS = [
    "不对", "错了", "应该", "不要", "别", "重新", "不是", "不对了",
    "incorrect", "wrong", "should not", "don't", "do not", "instead",
    "不是", "不行", "不太好", "不对", "错了", "应该", "不要", "别",
]

# Patterns that indicate tool / execution failure in tool response
_FAILURE_PATTERNS = [
    re.compile(r"\b(error|exception|failed|failure|traceback)\b", re.IGNORECASE),
    re.compile(r"\b(non-zero exit|exit code \d+)\b", re.IGNORECASE),
    re.compile(r"\b(command not found|no such file)\b", re.IGNORECASE),
]


@dataclass
class Episode:
    session_id: str
    session_source: str
    session_title: Optional[str]
    started_at: float
    ended_at: Optional[float]
    episode_type: str
    description: str
    messages: List[Dict[str, Any]]
    signal_index: int
    tool_call_count: int = 0
    message_count: int = 0


class EpisodeExtractor:
    """Extract friction episodes from recent Hermes sessions."""

    def __init__(
        self,
        db,
        lookback_hours: int = 24,
        max_episodes_per_session: int = 3,
        context_window: int = 6,
        min_tool_calls: int = 3,
        max_message_context: int = 40,
    ):
        self.db = db
        self.lookback_hours = lookback_hours
        self.max_episodes_per_session = max_episodes_per_session
        self.context_window = context_window
        self.min_tool_calls = min_tool_calls
        self.max_message_context = max_message_context

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self) -> List[Episode]:
        """Harvest episodes from all sessions in the lookback window."""
        sessions = self._fetch_recent_sessions()
        all_episodes: List[Episode] = []
        for sess in sessions:
            eps = self._extract_from_session(sess)
            all_episodes.extend(eps)
        # Deduplicate by session_id + signal_index
        seen = set()
        unique = []
        for ep in all_episodes:
            key = (ep.session_id, ep.signal_index)
            if key not in seen:
                seen.add(key)
                unique.append(ep)
        # Sort by time
        unique.sort(key=lambda e: e.started_at, reverse=True)
        return unique

    # ------------------------------------------------------------------
    # Session fetching
    # ------------------------------------------------------------------

    def _fetch_recent_sessions(self) -> List[Dict[str, Any]]:
        """Pull sessions started within the lookback window.

        Excludes cron and meditation sessions to prevent recursive analysis.
        """
        cutoff = time.time() - (self.lookback_hours * 3600)
        try:
            with self.db._lock:
                cursor = self.db._conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE started_at >= ?
                      AND source NOT IN ('cron', 'meditation')
                    ORDER BY started_at DESC
                    """,
                    (cutoff,),
                )
                rows = [dict(r) for r in cursor.fetchall()]
            logger.info("Fetched %d sessions in last %dh", len(rows), self.lookback_hours)
            return rows
        except Exception as e:
            logger.warning("Failed to fetch recent sessions: %s", e)
            return []

    # ------------------------------------------------------------------
    # Per-session extraction
    # ------------------------------------------------------------------

    def _extract_from_session(self, sess: Dict[str, Any]) -> List[Episode]:
        session_id = sess["id"]
        messages = self.db.get_messages(session_id)
        if not messages:
            return []

        episodes: List[Episode] = []
        session_meta = {
            "session_id": session_id,
            "session_source": sess.get("source", "unknown"),
            "session_title": sess.get("title"),
            "started_at": sess.get("started_at", 0),
            "ended_at": sess.get("ended_at"),
            "tool_call_count": sess.get("tool_call_count", 0),
            "message_count": len(messages),
        }

        # 1. Tool-failure episodes
        episodes.extend(self._find_tool_failures(messages, session_meta))

        # 2. User-correction episodes
        episodes.extend(self._find_user_corrections(messages, session_meta))

        # 3. Session-level signals (only one per session)
        if sess.get("tool_call_count", 0) >= self.min_tool_calls * 3:
            ep = self._make_high_iteration_episode(messages, session_meta)
            if ep:
                episodes.append(ep)

        if sess.get("end_reason") and sess["end_reason"] not in (None, "completed", "user_exit"):
            ep = self._make_abnormal_end_episode(messages, session_meta)
            if ep:
                episodes.append(ep)

        # Cap per session
        if len(episodes) > self.max_episodes_per_session:
            # Keep the most "severe" ones: tool failures first, then corrections
            episodes.sort(key=lambda e: 0 if e.episode_type == "tool_failure" else 1)
            episodes = episodes[: self.max_episodes_per_session]

        return episodes

    # ------------------------------------------------------------------
    # Signal detectors
    # ------------------------------------------------------------------

    def _find_tool_failures(self, messages: List[Dict], meta: Dict) -> List[Episode]:
        episodes = []
        current_cluster: List[int] = []  # indices of consecutive failures
        current_tool_name = "unknown"

        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            if isinstance(content, dict):
                content = json.dumps(content)
            if not self._looks_like_failure(content):
                # Flush any accumulated failure cluster
                if current_cluster:
                    signal_idx = current_cluster[len(current_cluster) // 2]
                    ep = self._build_episode(
                        messages,
                        signal_idx=signal_idx,
                        episode_type="tool_failure",
                        description=f"Tool '{current_tool_name}' failed with error output",
                        meta=meta,
                    )
                    if ep:
                        episodes.append(ep)
                    current_cluster = []
                    current_tool_name = "unknown"
                continue

            # This is a failure
            if not current_cluster:
                current_tool_name = self._resolve_tool_name(msg, messages, i)
            current_cluster.append(i)

        # Flush trailing cluster
        if current_cluster:
            signal_idx = current_cluster[len(current_cluster) // 2]
            ep = self._build_episode(
                messages,
                signal_idx=signal_idx,
                episode_type="tool_failure",
                description=f"Tool '{current_tool_name}' failed with error output",
                meta=meta,
            )
            if ep:
                episodes.append(ep)
        return episodes

    def _find_user_corrections(self, messages: List[Dict], meta: Dict) -> List[Episode]:
        episodes = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content") or ""
            if not self._looks_like_correction(content):
                continue
            ep = self._build_episode(
                messages,
                signal_idx=i,
                episode_type="user_correction",
                description="User corrected or redirected the agent",
                meta=meta,
            )
            if ep:
                episodes.append(ep)
        return episodes

    def _make_high_iteration_episode(self, messages: List[Dict], meta: Dict) -> Optional[Episode]:
        # Use the last meaningful assistant message as the signal point
        signal_idx = len(messages) - 1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                signal_idx = i
                break
        return self._build_episode(
            messages,
            signal_idx=signal_idx,
            episode_type="high_iteration",
            description=f"High tool-use session ({meta['tool_call_count']} tool calls)",
            meta=meta,
        )

    def _make_abnormal_end_episode(self, messages: List[Dict], meta: Dict) -> Optional[Episode]:
        signal_idx = len(messages) - 1
        return self._build_episode(
            messages,
            signal_idx=signal_idx,
            episode_type="abnormal_end",
            description=f"Session ended abnormally: {meta.get('end_reason')}",
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _looks_like_failure(self, content: str) -> bool:
        if not content:
            return False
        # 1. If it's a JSON response from common tools, parse it precisely
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                # Explicit success markers -> not a failure
                if data.get("success") is True:
                    return False
                if data.get("exit_code") in (0, "0", None):
                    # Even with exit_code 0, check for a top-level error field
                    if not data.get("error") and not data.get("status") == "error":
                        return False
                # Explicit failure markers
                if data.get("success") is False:
                    return True
                if data.get("status") == "error":
                    return True
                if data.get("exit_code") not in (None, 0, "0"):
                    return True
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # 2. Strong textual signals (must be at line start or clear patterns)
        lower = content.lower()
        strong = [
            "traceback (most recent call last):",
            "error:",
            "exception:",
            "fatal error",
            "command failed",
            "request timed out",
        ]
        for sig in strong:
            if sig in lower:
                return True

        # 3. Fallback to regex patterns, but be more conservative
        for pat in _FAILURE_PATTERNS:
            if pat.search(content):
                return True
        return False

    def _resolve_tool_name(self, msg: Dict[str, Any], messages: List[Dict], idx: int) -> str:
        """Walk backwards from a tool message to find the tool name."""
        tool_name = msg.get("tool_name")
        if tool_name:
            return tool_name
        tcid = msg.get("tool_call_id")
        if not tcid:
            return "unknown"
        for prior in reversed(messages[:idx]):
            if prior.get("role") == "assistant":
                tool_calls = prior.get("tool_calls") or []
                for tc in tool_calls:
                    if isinstance(tc, dict) and tc.get("id") == tcid:
                        return tc.get("function", {}).get("name", "unknown")
                    if isinstance(tc, str):
                        try:
                            tc_obj = json.loads(tc)
                            if tc_obj.get("id") == tcid:
                                return tc_obj.get("function", {}).get("name", "unknown")
                        except (json.JSONDecodeError, TypeError):
                            pass
        return "unknown"

    def _looks_like_correction(self, content: str) -> bool:
        if not content:
            return False
        lower = content.lower()
        for kw in _CORRECTION_KEYWORDS:
            if kw.lower() in lower:
                return True
        return False

    def _build_episode(
        self,
        messages: List[Dict],
        signal_idx: int,
        episode_type: str,
        description: str,
        meta: Dict,
    ) -> Optional[Episode]:
        start = max(0, signal_idx - self.context_window)
        end = min(len(messages), signal_idx + self.context_window + 1)
        subset = messages[start:end]
        # If the session is very long, clamp total context
        if len(messages) > self.max_message_context and len(subset) > self.max_message_context // 2:
            subset = subset[: self.max_message_context // 2]
        return Episode(
            session_id=meta["session_id"],
            session_source=meta["session_source"],
            session_title=meta["session_title"],
            started_at=meta["started_at"],
            ended_at=meta["ended_at"],
            episode_type=episode_type,
            description=description,
            messages=subset,
            signal_index=signal_idx,
            tool_call_count=meta.get("tool_call_count", 0),
            message_count=meta.get("message_count", 0),
        )
