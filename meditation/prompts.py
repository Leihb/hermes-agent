"""LLM prompts for the meditation pipeline."""

import json
from typing import Any, Dict, List


def _format_episode(ep: Dict[str, Any], idx: int) -> str:
    """Compact string representation of an episode for prompts."""
    lines = [
        f"## Episode {idx}",
        f"- Type: {ep.get('episode_type', 'unknown')}",
        f"- Session: {ep.get('session_source', 'unknown')} | {ep.get('session_title') or ep.get('session_id', '?')}",
        f"- Description: {ep.get('description', 'N/A')}",
        f"- Tool calls in session: {ep.get('tool_call_count', 0)}",
    ]
    # Include a truncated message preview
    messages = ep.get("messages", [])
    if messages:
        lines.append("- Key turns:")
        for m in messages[-4:]:  # last 4 turns are usually most informative
            role = m.get("role", "?")
            content = m.get("content") or ""
            if isinstance(content, dict):
                content = json.dumps(content)
            content = content.replace("\n", " ")[:150]
            tool = m.get("tool_name", "")
            prefix = f"  [{role}]"
            if tool:
                prefix += f" tool={tool}"
            lines.append(f"{prefix}: {content}")
    lines.append("")
    return "\n".join(lines)


BUCKET_SYSTEM_PROMPT = (
    "You are a pattern-matching analyst. Your job is to group friction episodes "
    "from an AI agent's recent sessions into thematic buckets.\n\n"
    "Rules:\n"
    "- Each bucket should capture a single recurring pattern or theme.\n"
    "- Episodes can belong to multiple buckets if they touch on multiple themes.\n"
    "- If an episode is a one-off with no pattern, put it in a 'misc' bucket.\n"
    "- Focus on user-visible friction, not just internal errors.\n"
    "- Keep bucket names short and specific (2-4 words).\n\n"
    "Output a JSON list of buckets. Each bucket must have:\n"
    "  'name': str,\n"
    "  'theme': str (1-sentence summary),\n"
    "  'episode_indices': List[int] (0-based indices from the input list),\n"
    "  'severity': str ('low' | 'medium' | 'high'),\n"
    "  'observation': str (what you noticed, 1-2 sentences)\n"
    "\n"
    "If there are no meaningful patterns, return an empty list."
)


def build_bucket_prompt(episodes: List[Dict[str, Any]]) -> str:
    if not episodes:
        return "No episodes to analyze."
    parts = [f"Analyze {len(episodes)} friction episodes and group them into buckets.\n"]
    for i, ep in enumerate(episodes):
        parts.append(_format_episode(ep, i))
    parts.append("\nReturn ONLY the JSON list of buckets. No markdown fences, no commentary.")
    return "\n".join(parts)


CONSOLIDATION_SYSTEM_PROMPT = (
    "You are a rule-authoring assistant. You receive a bucket of related friction episodes "
    "and write a concise, actionable rule that would prevent the friction in future sessions.\n\n"
    "Rules for the output:\n"
    "1. Start from user friction, not tool logs.\n"
    "2. Be specific: mention the exact tool, file type, or workflow where applicable.\n"
    "3. Write the rule as an imperative instruction the agent could follow.\n"
    "4. If the fix belongs in a skill, suggest a skill name and category.\n"
    "5. If the fix belongs in memory (user preference), say so.\n"
    "6. If you are unsure, say 'uncertain' and explain why.\n\n"
    "Output JSON with:\n"
    "  'rule': str (the actionable instruction),\n"
    "  'rationale': str (why this rule addresses the friction),\n"
    "  'target': str ('skill' | 'memory' | 'uncertain'),\n"
    "  'skill_name': str | null (suggested skill name if target=='skill'),\n"
    "  'skill_category': str | null (e.g. 'software-development', 'devops'),\n"
    "  'confidence': str ('low' | 'medium' | 'high')\n"
    "\n"
    "Return ONLY the JSON object. No markdown fences, no commentary."
)


def build_consolidation_prompt(bucket: Dict[str, Any]) -> str:
    parts = [
        f"Bucket: {bucket.get('name', 'Unnamed')}",
        f"Theme: {bucket.get('theme', 'N/A')}",
        f"Severity: {bucket.get('severity', 'medium')}",
        f"Observation: {bucket.get('observation', 'N/A')}",
        "",
        "Episodes in this bucket:",
    ]
    for ep in bucket.get("episodes", []):
        parts.append(_format_episode(ep, 0))
    parts.append("\nReturn ONLY the JSON object described in your instructions.")
    return "\n".join(parts)


OVERVIEW_TEMPLATE = """# Meditation Report — {date}

**Lookback:** {lookback_hours}h  
**Sessions scanned:** {session_count}  
**Episodes extracted:** {episode_count}  
**Buckets formed:** {bucket_count}  
**Candidate rules:** {rule_count}

---

## Buckets

{buckets_section}

---

## Candidate Rules

{rules_section}

---

## Raw Episodes

See `raw_episodes.json` for the full unredacted episode data.
"""


def render_overview(
    date: str,
    lookback_hours: int,
    session_count: int,
    episode_count: int,
    buckets: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
) -> str:
    def _bucket_line(b: Dict[str, Any]) -> str:
        sev = b.get("severity", "medium")
        icon = "🔴" if sev == "high" else "🟡" if sev == "medium" else "🟢"
        return f"- {icon} **{b.get('name', 'Unnamed')}** ({sev}) — {b.get('theme', '')}"

    def _rule_line(r: Dict[str, Any]) -> str:
        conf = r.get("confidence", "medium")
        icon = "✅" if conf == "high" else "⚠️" if conf == "medium" else "❓"
        target = r.get("target", "uncertain")
        return f"- {icon} **{target.upper()}** [{conf}] — {r.get('rule', '')}"

    return OVERVIEW_TEMPLATE.format(
        date=date,
        lookback_hours=lookback_hours,
        session_count=session_count,
        episode_count=episode_count,
        bucket_count=len(buckets),
        rule_count=len(rules),
        buckets_section="\n".join(_bucket_line(b) for b in buckets) or "_No buckets formed._",
        rules_section="\n".join(_rule_line(r) for r in rules) or "_No candidate rules._",
    )
