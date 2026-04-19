"""Tests for tools/session_search_tool.py — helper functions and search dispatcher."""

import json
import time
import pytest

from tools.session_search_tool import (
    _format_timestamp,
    _HIDDEN_SESSION_SOURCES,
    SESSION_SEARCH_SCHEMA,
)


# =========================================================================
# Tool schema guidance
# =========================================================================

class TestHiddenSessionSources:
    """Verify the _HIDDEN_SESSION_SOURCES constant used for third-party isolation."""

    def test_tool_source_is_hidden(self):
        assert "tool" in _HIDDEN_SESSION_SOURCES

    def test_standard_sources_not_hidden(self):
        for src in ("cli", "telegram", "discord", "slack", "cron"):
            assert src not in _HIDDEN_SESSION_SOURCES


class TestSessionSearchSchema:
    def test_keeps_cross_session_recall_guidance_without_current_session_nudge(self):
        description = SESSION_SEARCH_SCHEMA["description"]
        assert "past conversations" in description
        assert "recent turns of the current session" not in description


# =========================================================================
# _format_timestamp
# =========================================================================

class TestFormatTimestamp:
    def test_unix_float(self):
        ts = 1700000000.0  # Nov 14, 2023
        result = _format_timestamp(ts)
        assert "2023" in result or "November" in result

    def test_unix_int(self):
        result = _format_timestamp(1700000000)
        assert isinstance(result, str)
        assert len(result) > 5

    def test_iso_string(self):
        result = _format_timestamp("2024-01-15T10:30:00")
        assert isinstance(result, str)

    def test_none_returns_unknown(self):
        assert _format_timestamp(None) == "unknown"

    def test_numeric_string(self):
        result = _format_timestamp("1700000000.0")
        assert isinstance(result, str)
        assert "unknown" not in result.lower()


# =========================================================================
# session_search (dispatcher)
# =========================================================================

class TestSessionSearch:
    def test_no_db_returns_error(self):
        from tools.session_search_tool import session_search
        result = json.loads(session_search(query="test"))
        assert result["success"] is False
        assert "not available" in result["error"].lower()

    def test_empty_query_returns_error(self):
        from tools.session_search_tool import session_search
        mock_db = object()
        result = json.loads(session_search(query="", db=mock_db))
        assert result["success"] is False

    def test_whitespace_query_returns_error(self):
        from tools.session_search_tool import session_search
        mock_db = object()
        result = json.loads(session_search(query="   ", db=mock_db))
        assert result["success"] is False

    def test_current_session_excluded(self):
        """session_search should never return the current session."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        current_sid = "20260304_120000_abc123"

        # Simulate FTS5 returning matches only from the current session
        mock_db.search_messages.return_value = [
            {"session_id": current_sid, "content": "test match", "source": "cli",
             "session_started": 1709500000, "model": "test"},
        ]
        mock_db.get_session.return_value = {"parent_session_id": None}

        result = json.loads(session_search(
            query="test", db=mock_db, current_session_id=current_sid,
        ))
        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []

    def test_current_session_excluded_keeps_others(self):
        """Other sessions should still be returned when current is excluded."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        current_sid = "20260304_120000_abc123"
        other_sid = "20260303_100000_def456"

        mock_db.search_messages.return_value = [
            {"session_id": current_sid, "content": "match 1", "source": "cli",
             "session_started": 1709500000, "model": "test"},
            {"session_id": other_sid, "content": "match 2", "source": "telegram",
             "session_started": 1709400000, "model": "test"},
        ]
        mock_db.get_session.return_value = {"parent_session_id": None}
        mock_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        result = json.loads(session_search(
            query="test", db=mock_db, current_session_id=current_sid,
        ))

        assert result["success"] is True
        # Current session should be skipped, only other_sid should appear
        assert result["sessions_searched"] == 1
        assert current_sid not in [r.get("session_id") for r in result.get("results", [])]

    def test_current_child_session_excludes_parent_lineage(self):
        """Compression/delegation parents should be excluded for the active child session."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {"session_id": "parent_sid", "content": "match", "source": "cli",
             "session_started": 1709500000, "model": "test"},
        ]

        def _get_session(session_id):
            if session_id == "child_sid":
                return {"parent_session_id": "parent_sid"}
            if session_id == "parent_sid":
                return {"parent_session_id": None}
            return None

        mock_db.get_session.side_effect = _get_session

        result = json.loads(session_search(
            query="test", db=mock_db, current_session_id="child_sid",
        ))

        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["sessions_searched"] == 0

    def test_limit_none_coerced_to_default(self):
        """Model sends limit=null → should fall back to 3, not TypeError."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        result = json.loads(session_search(
            query="test", db=mock_db, limit=None,
        ))
        assert result["success"] is True

    def test_limit_type_object_coerced_to_default(self):
        """Model sends limit as a type object → should fall back to 3, not TypeError."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        result = json.loads(session_search(
            query="test", db=mock_db, limit=int,
        ))
        assert result["success"] is True

    def test_limit_string_coerced(self):
        """Model sends limit as string '2' → should coerce to int."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        result = json.loads(session_search(
            query="test", db=mock_db, limit="2",
        ))
        assert result["success"] is True

    def test_limit_clamped_to_range(self):
        """Negative or zero limit should be clamped to 1."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = []

        result = json.loads(session_search(
            query="test", db=mock_db, limit=-5,
        ))
        assert result["success"] is True

        result = json.loads(session_search(
            query="test", db=mock_db, limit=0,
        ))
        assert result["success"] is True

    def test_current_root_session_excludes_child_lineage(self):
        """Delegation child hits should be excluded when they resolve to the current root session."""
        from unittest.mock import MagicMock
        from tools.session_search_tool import session_search

        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {"session_id": "child_sid", "content": "match", "source": "cli",
             "session_started": 1709500000, "model": "test"},
        ]

        def _get_session(session_id):
            if session_id == "root_sid":
                return {"parent_session_id": None}
            if session_id == "child_sid":
                return {"parent_session_id": "root_sid"}
            return None

        mock_db.get_session.side_effect = _get_session

        result = json.loads(session_search(
            query="test", db=mock_db, current_session_id="root_sid",
        ))

        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []
        assert result["sessions_searched"] == 0
