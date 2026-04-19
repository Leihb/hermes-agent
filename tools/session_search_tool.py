#!/usr/bin/env python3
"""
Session Search Tool - Long-Term Conversation Recall (Lightweight)

Searches past session transcripts via FTS5 (+ optional semantic embedding)
and returns the actual matching message snippets with context. No LLM calls.

Flow:
  1. FTS5 search finds matching messages with surrounding context
  2. Optional semantic embedding search for recall augmentation
  3. Hybrid scoring ranks sessions
  4. Returns raw snippets grouped by session — fast, cheap, deterministic.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Per-session snippet budget to keep output readable.
_MAX_SNIPPET_CHARS_PER_SESSION = 3_000
_MAX_MATCHES_PER_SESSION = 3


def _get_semantic_config() -> Dict[str, Any]:
    """Read session_search semantic config from ~/.hermes/config.yaml"""
    cfg = {}
    try:
        import yaml
        from hermes_cli.config import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cfg = data.get("session_search", {})
    except Exception:
        pass
    return cfg


def _compute_semantic_weight(query: str, cfg: Dict[str, Any]) -> float:
    """
    Heuristic dynamic weight for hybrid scoring.
    Higher weight = trust semantic embedding more; lower = trust FTS5 keywords more.
    """
    q = query.lower().strip()
    if not q:
        return float(cfg.get("semantic_weight", 0.5))

    # Patterns that indicate precise keyword/entity matches are more important
    keyword_heavy_patterns = [
        r"\b(error|err)\s*\d+",            # error codes
        r"\d+\.\d+\.\d+",                   # version numbers
        r"/[a-zA-Z0-9_\-./]+",              # file paths
        r"\b0x[0-9a-fA-F]+\b",             # hex values
        r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}",  # UUID-ish
        r"`[^`]+`",                         # inline code
        r"\b[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\b",  # module.func patterns
    ]
    keyword_score = sum(1 for p in keyword_heavy_patterns if re.search(p, query))
    if keyword_score >= 2:
        return 0.25
    if keyword_score == 1:
        return 0.35

    # Semantic/conceptual question patterns
    semantic_boost_keywords = [
        "怎么", "如何", "为什么", "是什么", "什么意思",
        "原理", "机制", "方案", "思路", "建议", "优缺点",
        "区别", "差异", "对比", "哪个好", "有没有",
        "how to", "what is", "why", "difference between",
        "compare", "pros and cons", "approach", "idea",
    ]
    if any(kw in q for kw in semantic_boost_keywords):
        return 0.7

    # Default: balanced hybrid
    return float(cfg.get("semantic_weight", 0.5))


def _format_timestamp(ts: Union[int, float, str, None]) -> str:
    """Convert a Unix timestamp (float/int) or ISO string to a human-readable date."""
    if ts is None:
        return "unknown"
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime
            dt = datetime.fromtimestamp(ts)
            return dt.strftime("%B %d, %Y at %I:%M %p")
        if isinstance(ts, str):
            if ts.replace(".", "").replace("-", "").isdigit():
                from datetime import datetime
                dt = datetime.fromtimestamp(float(ts))
                return dt.strftime("%B %d, %Y at %I:%M %p")
            return ts
    except (ValueError, OSError, OverflowError):
        pass
    return str(ts)


def _format_message_short(msg: Dict[str, Any]) -> str:
    """Format a single message dict for snippet display, truncating long outputs."""
    role = msg.get("role", "unknown").upper()
    content = msg.get("content") or ""
    tool_name = msg.get("tool_name")

    # Truncate very long content (tool outputs, code dumps, etc.)
    if len(content) > 800:
        content = content[:400] + "\n...[truncated]...\n" + content[-200:]

    if role == "TOOL" and tool_name:
        return f"[TOOL:{tool_name}]: {content}"
    return f"[{role}]: {content}"


def _build_snippet_text(matches: List[Dict[str, Any]], query: str) -> str:
    """
    Build a readable snippet text from a session's FTS5 matches.
    Uses the pre-fetched 'context' and 'snippet' fields from search_messages.
    """
    parts = []
    total_chars = 0

    for idx, match in enumerate(matches[:_MAX_MATCHES_PER_SESSION], 1):
        block_lines = [f"--- Match {idx} ---"]

        # Context before (from DB's pre-fetched surrounding messages)
        ctx = match.get("context", [])
        match_idx = -1
        for i, cmsg in enumerate(ctx):
            if cmsg.get("content", "").strip() == match.get("snippet", "").replace(">>>", "").replace("<<<", "").strip():
                match_idx = i
                break

        # If we can't locate by content, assume the middle item is the match
        if match_idx == -1 and ctx:
            match_idx = len(ctx) // 2

        for i, cmsg in enumerate(ctx):
            line = _format_message_short(cmsg)
            if i == match_idx:
                # Highlight the matched line using the FTS5 snippet markers
                snippet = match.get("snippet", "")
                if snippet and (">>>" in snippet or "<<<" in snippet):
                    line = snippet
                elif snippet:
                    line = f">>> {snippet} <<<"
                else:
                    line = f">>> {line} <<<"
            block_lines.append(line)

        block_text = "\n".join(block_lines)
        if total_chars + len(block_text) > _MAX_SNIPPET_CHARS_PER_SESSION and parts:
            parts.append(f"... ({len(matches) - idx + 1} more matches in this session) ...")
            break

        parts.append(block_text)
        total_chars += len(block_text)

    return "\n\n".join(parts)


def _load_semantic_snippets(
    db,
    semantic_results_map: Dict[str, float],
    seen_sessions: Dict[str, Dict[str, Any]],
    current_lineage_root: Optional[str],
    current_session_id: Optional[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For sessions found only by semantic search (not FTS5), load a preview
    of their recent messages so we have something to show.
    Returns {session_id: [match_dicts]}.
    Also fills in missing session metadata (started_at, source, model).
    """
    semantic_only_sessions = {}
    for sid in semantic_results_map:
        if sid not in seen_sessions:
            continue  # should not happen, but be safe
        # Check if this session already has FTS5 matches
        has_fts = bool(seen_sessions[sid].get("_fts_matches"))
        if not has_fts:
            try:
                msgs = db.get_messages_as_conversation(sid)
                if not msgs:
                    continue
                # Take the last ~5 messages as a preview
                preview_msgs = msgs[-5:]
                semantic_only_sessions[sid] = [{
                    "snippet": "",
                    "context": preview_msgs,
                    "_semantic_only": True,
                }]
                # Backfill metadata from the DB
                s = db.get_session(sid)
                if s:
                    if not seen_sessions[sid].get("session_started"):
                        seen_sessions[sid]["session_started"] = s.get("started_at")
                    if seen_sessions[sid].get("source") == "unknown":
                        seen_sessions[sid]["source"] = s.get("source", "unknown")
                    if not seen_sessions[sid].get("model"):
                        seen_sessions[sid]["model"] = s.get("model")
            except Exception:
                pass
    return semantic_only_sessions


# Sources that are excluded from session browsing/searching by default.
_HIDDEN_SESSION_SOURCES = ("tool",)


def _list_recent_sessions(db, limit: int, current_session_id: str = None) -> str:
    """Return metadata for the most recent sessions (no LLM calls)."""
    try:
        sessions = db.list_sessions_rich(limit=limit + 5, exclude_sources=list(_HIDDEN_SESSION_SOURCES))

        # Resolve current session lineage to exclude it
        current_root = None
        if current_session_id:
            try:
                sid = current_session_id
                visited = set()
                while sid and sid not in visited:
                    visited.add(sid)
                    s = db.get_session(sid)
                    parent = s.get("parent_session_id") if s else None
                    sid = parent if parent else None
                current_root = max(visited, key=len) if visited else current_session_id
            except Exception:
                current_root = current_session_id

        results = []
        for s in sessions:
            sid = s.get("id", "")
            if current_root and (sid == current_root or sid == current_session_id):
                continue
            if s.get("parent_session_id"):
                continue
            results.append({
                "session_id": sid,
                "title": s.get("title") or None,
                "source": s.get("source", ""),
                "started_at": s.get("started_at", ""),
                "last_active": s.get("last_active", ""),
                "message_count": s.get("message_count", 0),
                "preview": s.get("preview", ""),
            })
            if len(results) >= limit:
                break

        return json.dumps({
            "success": True,
            "mode": "recent",
            "results": results,
            "count": len(results),
            "message": f"Showing {len(results)} most recent sessions. Use a keyword query to search specific topics.",
        }, ensure_ascii=False)
    except Exception as e:
        logging.error("Error listing recent sessions: %s", e, exc_info=True)
        return tool_error(f"Failed to list recent sessions: {e}", success=False)


def session_search(
    query: str,
    role_filter: str = None,
    limit: int = 3,
    db=None,
    current_session_id: str = None,
) -> str:
    """
    Search past sessions and return matching message snippets with context.

    Uses FTS5 (+ optional semantic embedding) to find matches, then returns
    the raw conversation fragments around each hit. No LLM summarization.
    """
    if db is None:
        return tool_error("Session database not available.", success=False)

    # Defensive limit coercion
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 3
    limit = max(1, min(limit, 5))

    # Recent sessions mode: when query is empty, return metadata for recent sessions.
    if not query or not query.strip():
        return _list_recent_sessions(db, limit, current_session_id)

    query = query.strip()

    try:
        # Parse role filter
        role_list = None
        if role_filter and role_filter.strip():
            role_list = [r.strip() for r in role_filter.split(",") if r.strip()]

        # Resolve child sessions to their parent
        def _resolve_to_parent(session_id: str) -> str:
            visited = set()
            sid = session_id
            while sid and sid not in visited:
                visited.add(sid)
                try:
                    session = db.get_session(sid)
                    if not session:
                        break
                    parent = session.get("parent_session_id")
                    if parent:
                        sid = parent
                    else:
                        break
                except Exception:
                    break
            return sid

        current_lineage_root = (
            _resolve_to_parent(current_session_id) if current_session_id else None
        )

        # ── 1. FTS5 search ──
        raw_results = db.search_messages(
            query=query,
            role_filter=role_list,
            exclude_sources=list(_HIDDEN_SESSION_SOURCES),
            limit=50,
            offset=0,
        )

        # Group matches by resolved session ID, keeping FTS5 rank order
        session_matches: Dict[str, List[Dict[str, Any]]] = {}
        seen_sessions: Dict[str, Dict[str, Any]] = {}

        for result in raw_results:
            raw_sid = result["session_id"]
            resolved_sid = _resolve_to_parent(raw_sid)
            if current_lineage_root and resolved_sid == current_lineage_root:
                continue
            if current_session_id and raw_sid == current_session_id:
                continue

            if resolved_sid not in session_matches:
                session_matches[resolved_sid] = []
                seen_sessions[resolved_sid] = {
                    "session_id": resolved_sid,
                    "source": result.get("source", "unknown"),
                    "session_started": result.get("session_started", None),
                    "model": result.get("model"),
                    "_fts_matches": True,
                }
            session_matches[resolved_sid].append(result)

        # ── 2. Semantic search ──
        semantic_cfg = _get_semantic_config()
        semantic_weight = _compute_semantic_weight(query, semantic_cfg)
        semantic_results_map: Dict[str, float] = {}

        if semantic_cfg.get("semantic_enabled", False):
            try:
                from agent.session_embeddings import embed_text
                query_emb = embed_text(query)
                if query_emb:
                    semantic_results = db.search_messages_semantic(
                        query_embedding=query_emb,
                        candidate_message_ids=None,
                        top_k=limit * 10,
                    )
                    for r in semantic_results:
                        sid = r["session_id"]
                        resolved_sid = _resolve_to_parent(sid)
                        if current_lineage_root and resolved_sid == current_lineage_root:
                            continue
                        if current_session_id and sid == current_session_id:
                            continue
                        sim = r.get("similarity", 0.0)
                        semantic_results_map[resolved_sid] = max(
                            semantic_results_map.get(resolved_sid, 0.0), sim
                        )
                        if resolved_sid not in seen_sessions:
                            seen_sessions[resolved_sid] = {
                                "session_id": resolved_sid,
                                "source": r.get("source", "unknown"),
                                "session_started": r.get("session_started", None),
                                "_fts_matches": False,
                            }
            except Exception as e:
                logging.warning("Semantic recall failed: %s", e)

        if not seen_sessions:
            return json.dumps({
                "success": True,
                "query": query,
                "results": [],
                "count": 0,
                "sessions_searched": 0,
                "message": "No matching sessions found.",
            }, ensure_ascii=False)

        # ── 3. Hybrid scoring & re-ranking ──
        if semantic_cfg.get("semantic_enabled", False) and semantic_results_map:
            session_scores: Dict[str, float] = {}

            if raw_results:
                candidate_ids = [r["id"] for r in raw_results if "id" in r]
                if candidate_ids:
                    try:
                        cand_semantic = db.search_messages_semantic(
                            query_embedding=query_emb,
                            candidate_message_ids=candidate_ids,
                            top_k=len(candidate_ids),
                        )
                        sim_map = {r["message_id"]: r["similarity"] for r in cand_semantic}
                    except Exception:
                        sim_map = {}

                    ranks = [r.get("rank", 0) for r in raw_results if "id" in r]
                    min_rank = min(ranks) if ranks else 0
                    max_rank = max(ranks) if ranks else 0
                    rank_range = max_rank - min_rank if max_rank != min_rank else 1.0

                    for r in raw_results:
                        msg_id = r.get("id")
                        sid = r.get("session_id")
                        if not msg_id or not sid:
                            continue
                        rank = r.get("rank", min_rank)
                        fts_norm = (max_rank - rank) / rank_range
                        sim = sim_map.get(msg_id, 0.0)
                        combined = fts_norm * (1 - semantic_weight) + sim * semantic_weight
                        session_scores[sid] = max(session_scores.get(sid, 0.0), combined)

            # Merge pure-semantic sessions
            for sid, sim in semantic_results_map.items():
                if sid in session_scores:
                    session_scores[sid] = max(session_scores[sid], sim) * 1.05
                else:
                    session_scores[sid] = sim

            seen_sessions = dict(sorted(
                seen_sessions.items(),
                key=lambda item: (
                    session_scores.get(item[0], 0.0)
                    + (0.15 if item[1].get("_fts_matches") else 0.0)
                ),
                reverse=True,
            )[:limit])
        else:
            # FTS5 only — already grouped by session, take top limit
            seen_sessions = dict(list(seen_sessions.items())[:limit])

        # ── 4. Build snippets for each session (NO LLM) ──
        # Load semantic-only previews if needed
        semantic_only = _load_semantic_snippets(
            db, semantic_results_map, seen_sessions, current_lineage_root, current_session_id
        )

        summaries = []
        for session_id, match_info in seen_sessions.items():
            try:
                if session_id in session_matches:
                    # FTS5 matches available — build snippets from them
                    matches = session_matches[session_id]
                    snippet_text = _build_snippet_text(matches, query)
                elif session_id in semantic_only:
                    # Semantic-only session — show recent message preview
                    snippet_text = _build_snippet_text(semantic_only[session_id], query)
                    snippet_text = "[Semantic match — showing recent messages]\n\n" + snippet_text
                else:
                    snippet_text = "No preview available."

                entry = {
                    "session_id": session_id,
                    "when": _format_timestamp(match_info.get("session_started")),
                    "source": match_info.get("source", "unknown"),
                    "model": match_info.get("model"),
                    "summary": snippet_text,
                }
                summaries.append(entry)
            except Exception as e:
                logging.warning("Failed to build snippet for session %s: %s", session_id, e)

        return json.dumps({
            "success": True,
            "query": query,
            "results": summaries,
            "count": len(summaries),
            "sessions_searched": len(seen_sessions),
        }, ensure_ascii=False)

    except Exception as e:
        logging.error("Session search failed: %s", e, exc_info=True)
        return tool_error(f"Search failed: {str(e)}", success=False)


def check_session_search_requirements() -> bool:
    """Requires SQLite state database."""
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search your long-term memory of past conversations, or browse recent sessions. This is your recall -- "
        "every past session is searchable, and this tool returns the actual matching message snippets.\n\n"
        "TWO MODES:\n"
        "1. Recent sessions (no query): Call with no arguments to see what was worked on recently. "
        "Returns titles, previews, and timestamps. Zero LLM cost, instant. "
        "Start here when the user asks what were we working on or what did we do recently.\n"
        "2. Keyword search (with query): Search for specific topics across all past sessions. "
        "Returns matching message snippets with surrounding context.\n\n"
        "USE THIS PROACTIVELY when:\n"
        "- The user says 'we did this before', 'remember when', 'last time', 'as I mentioned'\n"
        "- The user asks about a topic you worked on before but don't have in current context\n"
        "- The user references a project, person, or concept that seems familiar but isn't in memory\n"
        "- You want to check if you've solved a similar problem before\n"
        "- The user asks 'what did we do about X?' or 'how did we fix Y?'\n\n"
        "Don't hesitate to search when it is actually cross-session -- it's fast and cheap. "
        "Better to search and confirm than to guess or ask the user to repeat themselves.\n\n"
        "Search syntax: keywords joined with OR for broad recall (elevenlabs OR baseten OR funding), "
        "phrases for exact match (\"docker networking\"), boolean (python NOT java), prefix (deploy*). "
        "IMPORTANT: Use OR between keywords for best results — FTS5 defaults to AND which misses "
        "sessions that only mention some terms. If a broad OR query returns nothing, try individual "
        "keyword searches in parallel. Returns matching message snippets from the top sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — keywords, phrases, or boolean expressions to find in past sessions. Omit this parameter entirely to browse recent sessions instead (returns titles, previews, timestamps with no LLM cost).",
            },
            "role_filter": {
                "type": "string",
                "description": "Optional: only search messages from specific roles (comma-separated). E.g. 'user,assistant' to skip tool outputs.",
            },
            "limit": {
                "type": "integer",
                "description": "Max sessions to return (default: 3, max: 5).",
                "default": 3,
            },
        },
        "required": [],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=lambda args, **kw: session_search(
        query=args.get("query") or "",
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id")),
    check_fn=check_session_search_requirements,
    emoji="🔍",
)
