"""LLM card builder (Claude). Two steps, so facts and format never get mixed up:

1. **Research**: Claude searches the web for the player and writes sourced notes. It is told to
   separate evidence from opinion and to cite a URL for every claim.
2. **Draft**: Claude turns the notes (and any Cricsheet numbers) into a :class:`BatterCard`
   using structured outputs, so the card is valid against the schema by construction. It may
   only use the card's coach vocabulary; claims without a source must be marked "assumption".

The analyst then reviews the YAML. The LLM never writes reward code or simulator parameters.
Bring your own key: set ``ANTHROPIC_API_KEY`` (or log in with ``ant auth login``).
"""

from __future__ import annotations

import json

from .cards import BatterCard

MODEL = "claude-opus-5-5"
_BETAS = ["server-side-fallback-2026-07-01"]

RESEARCH_SYSTEM = """You are a cricket analyst's research assistant. Research one batter and write notes a coach can trust.
Rules:
- Cite a source URL for every factual claim. If you cannot find support, say "no source" and label it as opinion.
- Prefer ball-tracking analysis, coaching breakdowns and dismissal records over general praise.
- Cover: batting hand, level, scoring style and tempo, how they start an innings, how they defend the straight ball,
  whether they leave outside off, their main attacking shot, and specific weaknesses (length, line, pace,
  movement, height; and how they get out) and specific strengths (where they score).
- Be concise: bullet points, each with its source."""

DRAFT_SYSTEM = """Convert research notes into a batter card for a cricket simulator.
Rules:
- Use only the allowed vocabulary in the schema. Map each weakness/strength to the closest zone words.
- Every weakness and strength needs evidence. kind="source" with the URL when the notes cite one; kind="data" only
  for numbers from the supplied Cricsheet profile; kind="assumption" for anything unsupported. Never invent URLs.
- Include 1-4 weaknesses and 1-4 strengths, most important first. Do not include a zone unless the notes support it
  or you mark it as an assumption.
- Set uncertainty to "high" if most claims are assumptions, "low" only with strong data."""


def _client():
    import anthropic

    return anthropic.Anthropic()


def _text(response) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text")


def research(player: str, context: str = "", client=None) -> str:
    """Web research notes with citations (uses Claude's server-side web search)."""
    client = client or _client()
    messages = [{"role": "user", "content": f"Batter: {player}\n{context}".strip()}]
    parts = []
    for _ in range(4):  # server-side web search can pause a long turn; resume it
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=_BETAS,
            fallbacks="default",
            system=RESEARCH_SYSTEM,
            output_config={"effort": "high"},
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}],
            messages=messages,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined the research request.")
        parts.append(_text(response))
        if response.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": response.content})
    return "\n".join(p for p in parts if p)


def draft_card(player: str, notes: str, profile: dict | None = None, client=None) -> BatterCard:
    """Structured-output draft of a batter card from research notes and optional Cricsheet numbers."""
    client = client or _client()
    content = f"Batter: {player}\n\nResearch notes:\n{notes}"
    if profile:
        content += "\n\nCricsheet ball-by-ball profile (use kind='data', source='cricsheet'):\n" + json.dumps(profile, indent=1)
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        betas=_BETAS,
        fallbacks="default",
        system=DRAFT_SYSTEM,
        output_config={"effort": "medium"},
        output_format=BatterCard,
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"Could not draft a card (stop_reason={response.stop_reason}).")
    return response.parsed_output


def build_card(player: str, context: str = "", profile: dict | None = None, client=None) -> tuple[BatterCard, str]:
    client = client or _client()
    notes = research(player, context, client)
    return draft_card(player, notes, profile, client), notes
