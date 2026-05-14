"""Report generator: creates structured, multi-section reports.

Uses a three-stage pipeline:
1. Outline — LLM creates a structured ToC with per-section drafting memos
   and maps each section to source research threads
2. Draft — each section written individually with sliding context
   (outline + previous section ending + raw data from vault + blackboard)
3. Summary — executive summary written last with full knowledge of content

When a research vault is available (from a prior investigate call), each
section writer receives the raw evidence for its mapped threads — not a
compressed summary. This preserves specific data points, quotes, and
nuances that make the report authoritative.
"""

from __future__ import annotations

import json
import time

import llm
import research_vault as rv
from config import MAX_REPORT_SECTIONS, SECTION_TARGET_WORDS, RESEARCH_CONTEXT_LIMIT

# ---------------------------------------------------------------------------
# Outline tool
# ---------------------------------------------------------------------------

_OUTLINE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_outline",
            "description": "Create a structured report outline with sections and drafting memos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Report title",
                    },
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Section number like 1, 2, 3"},
                                "title": {"type": "string", "description": "Section heading"},
                                "memo": {
                                    "type": "string",
                                    "description": (
                                        "50-word instruction on what this section must cover, "
                                        "what data points to include, and what argument to make"
                                    ),
                                },
                                "source_threads": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": (
                                        "IDs of research threads whose raw data should be used "
                                        "for this section (e.g. [\"A\", \"B\"])"
                                    ),
                                },
                            },
                            "required": ["id", "title", "memo", "source_threads"],
                        },
                        "description": "Ordered list of report sections (excluding executive summary)",
                    },
                },
                "required": ["title", "sections"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_OUTLINE_PROMPT = """\
You are a report architect. Given a topic and research data, create a \
detailed report outline.

Rules:
- Create {max_sections} sections maximum (including Introduction and Conclusion)
- Each section needs a clear title and a drafting memo
- The drafting memo is a 50-word instruction telling a writer exactly what \
the section must cover, what data points to include, and what argument to make
- Do NOT include an "Executive Summary" section — it will be written separately
- Order sections logically: Introduction first, Conclusion last
- Make each section self-contained but building on previous sections
- Map each section to the research threads it should draw from (source_threads)
- Introduction and Conclusion may draw from all threads

{thread_info}

You MUST call create_outline with your outline.\
"""

_OUTLINE_PROMPT_NO_VAULT = """\
You are a report architect. Given a topic and research data, create a \
detailed report outline.

Rules:
- Create {max_sections} sections maximum (including Introduction and Conclusion)
- Each section needs a clear title and a drafting memo
- The drafting memo is a 50-word instruction telling a writer exactly what \
the section must cover, what data points to include, and what argument to make
- Do NOT include an "Executive Summary" section — it will be written separately
- Order sections logically: Introduction first, Conclusion last
- Make each section self-contained but building on previous sections
- For source_threads, use ["ALL"] since thread data is not available

You MUST call create_outline with your outline.\
"""

_SECTION_PROMPT = """\
You are writing one section of a detailed report.

## Report Outline
{outline_text}

## Your Assignment
Write section {section_id}: "{section_title}"

Drafting memo: {memo}

## Context
{context_block}

## Research Data
{research_data}

## Instructions
- Write approximately {target_words} words for this section
- Include specific data points, statistics, and examples from the research data
- Reference sources where possible
- Start with a clear topic sentence
- End with a natural transition to the next section (if not the conclusion)
- Do NOT repeat information already covered in previous sections
- Write in a professional, analytical tone\
"""

_SUMMARY_PROMPT = """\
You are writing the Executive Summary for a report titled: "{title}"

## What the report covers
{blackboard}

## Instructions
- Write a concise executive summary (200-300 words)
- Highlight the most important findings from each section
- State the key conclusions and recommendations
- This will appear at the top of the report, so it should stand alone
- Do not introduce new information — only summarize what the sections cover\
"""

# ---------------------------------------------------------------------------
# Internal functions
# ---------------------------------------------------------------------------


def _build_thread_info(vault: rv.ResearchVault) -> str:
    """Build a description of available research threads for the outline prompt."""
    lines = ["## Available Research Threads"]
    for tid, topic in vault.thread_topics.items():
        keys = vault.keys_for_threads([tid])
        summary = vault.thread_summaries.get(tid, "")
        # Include first 200 chars of summary so the outliner knows what data is available
        summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
        lines.append(f"\n### Thread {tid}: {topic}")
        lines.append(f"Data entries: {', '.join(keys) if keys else '(none)'}")
        if summary_preview:
            lines.append(f"Summary: {summary_preview}")
    return "\n".join(lines)


def _create_outline_with_vault(topic: str, vault: rv.ResearchVault) -> dict:
    """Create outline using vault data (thread info + summaries)."""
    thread_info = _build_thread_info(vault)

    messages = [
        {"role": "system", "content": _OUTLINE_PROMPT.format(
            max_sections=MAX_REPORT_SECTIONS,
            thread_info=thread_info,
        )},
        {"role": "user", "content": f"Topic: {topic}\n\nSynthesis:\n{vault.synthesis}"},
    ]

    msg = llm.call(messages, tools=_OUTLINE_TOOLS)

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "create_outline":
                return json.loads(tc.function.arguments)

    print("[report] Warning: LLM didn't use create_outline tool, using minimal outline")
    all_threads = list(vault.thread_topics.keys())
    return {
        "title": f"Report: {topic}",
        "sections": [
            {"id": "1", "title": "Introduction", "memo": f"Introduce {topic}.", "source_threads": all_threads},
            {"id": "2", "title": "Analysis", "memo": f"Analyze key aspects of {topic}.", "source_threads": all_threads},
            {"id": "3", "title": "Conclusion", "memo": f"Summarize findings on {topic}.", "source_threads": all_threads},
        ],
    }


def _create_outline_from_text(topic: str, research_data: str) -> dict:
    """Fallback: create outline from a plain research_data string (no vault)."""
    rd = research_data[:RESEARCH_CONTEXT_LIMIT] if research_data else "(No research data provided)"

    messages = [
        {"role": "system", "content": _OUTLINE_PROMPT_NO_VAULT.format(max_sections=MAX_REPORT_SECTIONS)},
        {"role": "user", "content": f"Topic: {topic}\n\nResearch data:\n{rd}"},
    ]

    msg = llm.call(messages, tools=_OUTLINE_TOOLS)

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "create_outline":
                return json.loads(tc.function.arguments)

    print("[report] Warning: LLM didn't use create_outline tool, using minimal outline")
    return {
        "title": f"Report: {topic}",
        "sections": [
            {"id": "1", "title": "Introduction", "memo": f"Introduce {topic}.", "source_threads": ["ALL"]},
            {"id": "2", "title": "Analysis", "memo": f"Analyze key aspects of {topic}.", "source_threads": ["ALL"]},
            {"id": "3", "title": "Conclusion", "memo": f"Summarize findings on {topic}.", "source_threads": ["ALL"]},
        ],
    }


def _format_outline(outline: dict) -> str:
    """Format outline as compact text for inclusion in section prompts."""
    lines = [f"# {outline.get('title', 'Report')}"]
    for s in outline.get("sections", []):
        lines.append(f"  {s['id']}. {s['title']}")
    return "\n".join(lines)


def _get_research_for_section(
    section: dict,
    vault: rv.ResearchVault | None,
    research_data: str,
) -> str:
    """Get the research data for a specific section.

    If vault is available, pulls raw data + thread summaries for the
    section's mapped threads. Otherwise falls back to the research_data string.
    """
    if not vault:
        return research_data[:RESEARCH_CONTEXT_LIMIT] if research_data else "(No research data available)"

    thread_ids = section.get("source_threads", [])

    # "ALL" means use all threads
    if "ALL" in thread_ids:
        thread_ids = list(vault.thread_topics.keys())

    if not thread_ids:
        return "(No source threads mapped to this section)"

    # Build research context: thread summaries + raw data
    parts = []

    # Thread summaries (concise overview)
    summaries = vault.summaries_for_threads(thread_ids)
    if summaries:
        parts.append("### Thread Summaries\n\n" + summaries)

    # Raw evidence from vault
    raw = vault.content_for_threads(thread_ids, limit=RESEARCH_CONTEXT_LIMIT)
    if raw:
        parts.append("### Raw Evidence\n\n" + raw)

    return "\n\n---\n\n".join(parts) if parts else "(No data found for mapped threads)"


def _write_section(
    section: dict,
    outline: dict,
    prev_paragraph: str,
    research_context: str,
    blackboard: list[str],
) -> str:
    """Write a single report section with sliding context."""
    outline_text = _format_outline(outline)

    # Build context block
    context_parts = []
    if blackboard:
        context_parts.append(
            "### Sections already written:\n" + "\n".join(f"- {b}" for b in blackboard)
        )
    if prev_paragraph:
        context_parts.append(f'### Previous section ended with:\n"{prev_paragraph}"')
    context_block = "\n\n".join(context_parts) if context_parts else "(This is the first section)"

    messages = [
        {"role": "system", "content": _SECTION_PROMPT.format(
            outline_text=outline_text,
            section_id=section["id"],
            section_title=section["title"],
            memo=section.get("memo", "Write this section thoroughly."),
            context_block=context_block,
            research_data=research_context,
            target_words=SECTION_TARGET_WORDS,
        )},
        {"role": "user", "content": f"Write section {section['id']}: {section['title']}"},
    ]

    msg = llm.call(messages, max_tokens=2048)
    clean, thinking = llm.parse_thinking(msg)
    clean = clean or "[Section generation failed]"
    if thinking:
        print(f"  [report] Section {section['id']} thinking: {thinking[:100]}...")
    return clean


def _write_executive_summary(title: str, blackboard: list[str]) -> str:
    """Write the executive summary based on all section summaries."""
    messages = [
        {"role": "system", "content": _SUMMARY_PROMPT.format(
            title=title,
            blackboard="\n".join(f"- {b}" for b in blackboard),
        )},
        {"role": "user", "content": "Write the executive summary."},
    ]

    msg = llm.call(messages, max_tokens=1024)
    clean, thinking = llm.parse_thinking(msg)
    clean = clean or "[Executive summary generation failed]"
    if thinking:
        print(f"  [report] Executive summary thinking: {thinking[:100]}...")
    return clean


def _get_last_paragraph(text: str) -> str:
    """Extract the last non-empty paragraph from a section."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[-1][:300]
    return text[-300:] if text else ""


def _get_section_summary(section_id: str, title: str, text: str) -> str:
    """Create a one-line blackboard entry for a written section."""
    first_sentence = text.split(". ")[0] if ". " in text else text[:150]
    return f"Section {section_id} ({title}): {first_sentence.strip()}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(topic: str, research_data: str = "") -> str:
    """Generate a structured, multi-section report.

    Uses the research vault (from a prior investigate call) if available.
    Each section receives raw evidence for its mapped threads, preserving
    specific data points that would be lost in a summary.

    Falls back to the research_data string if no vault is available.

    Args:
        topic: The report topic/question.
        research_data: Fallback research text (used only if no vault).

    Returns:
        The assembled report as a markdown string.
    """
    t_start = time.time()
    vault = rv.current_vault

    if vault:
        print(f"\n[report] Using vault: {len(vault.entries)} raw entries, "
              f"{len(vault.thread_topics)} threads")
    else:
        print(f"\n[report] No vault available, using research_data string "
              f"({len(research_data)} chars)")

    # Stage 1: OUTLINE
    print(f"\n[report] === OUTLINE ===")
    print(f"[report] Topic: {topic}")
    try:
        if vault:
            outline = _create_outline_with_vault(topic, vault)
        else:
            outline = _create_outline_from_text(topic, research_data)
    except Exception as e:
        print(f"[report] Outline failed: {e}")
        import traceback; traceback.print_exc()
        return f"[Report generation failed during outlining: {e}]"

    title = outline.get("title", f"Report: {topic}")
    sections = outline.get("sections", [])[:MAX_REPORT_SECTIONS]

    print(f"[report] Title: {title}")
    print(f"[report] {len(sections)} sections planned:")
    for s in sections:
        threads = s.get("source_threads", [])
        print(f"  {s['id']}. {s['title']}  ← threads: {threads}")

    # Stage 2: DRAFT (section by section with sliding context)
    print(f"\n[report] === DRAFT ===")
    written_sections = []
    blackboard = []
    prev_paragraph = ""

    for section in sections:
        print(f"\n[report] Writing section {section['id']}/{len(sections)}: {section['title']}")

        # Get research data for this section (from vault or fallback)
        research_context = _get_research_for_section(section, vault, research_data)
        print(f"[report] Research context: {len(research_context)} chars")

        try:
            text = _write_section(
                section=section,
                outline=outline,
                prev_paragraph=prev_paragraph,
                research_context=research_context,
                blackboard=blackboard,
            )
        except Exception as e:
            print(f"[report] Section {section['id']} failed: {e}")
            text = f"[Section generation failed: {e}]"

        written_sections.append({
            "id": section["id"],
            "title": section["title"],
            "text": text,
        })

        # Update sliding context
        prev_paragraph = _get_last_paragraph(text)
        blackboard.append(_get_section_summary(section["id"], section["title"], text))

        word_count = len(text.split())
        print(f"[report] Section {section['id']} done ({word_count} words)")

    # Stage 3: EXECUTIVE SUMMARY (written last, informed by all sections)
    print(f"\n[report] === EXECUTIVE SUMMARY ===")
    try:
        exec_summary = _write_executive_summary(title, blackboard)
    except Exception as e:
        print(f"[report] Executive summary failed: {e}")
        exec_summary = "[Executive summary generation failed]"

    # Stage 4: ASSEMBLE
    print(f"\n[report] === ASSEMBLE ===")
    parts = [f"# {title}\n"]
    parts.append(f"## Executive Summary\n\n{exec_summary}\n")

    for ws in written_sections:
        parts.append(f"## {ws['id']}. {ws['title']}\n\n{ws['text']}\n")

    report = "\n---\n\n".join(parts)

    total_words = len(report.split())
    elapsed = time.time() - t_start
    print(f"\n[report] Report complete: {total_words} words, {len(written_sections)} sections, {elapsed:.1f}s")

    return report
