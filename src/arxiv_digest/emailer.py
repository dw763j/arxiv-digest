from __future__ import annotations

from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import smtplib
from typing import Any


def _paper_link(paper: dict[str, Any]) -> str:
    title = paper.get("title", paper.get("paper_title", "论文"))
    link = paper.get("link", "")
    if link:
        return f'<a href="{link}">{title}</a>'
    return title


def _first_paper_link(papers: list[dict[str, Any]]) -> str:
    if not papers:
        return ""
    return f' <span class="paper-ref">({_paper_link(papers[0])})</span>'


def _render_research_areas_compact(
    areas: list[dict[str, Any]], *, limit: int = 8
) -> str:
    if not areas:
        return ""
    parts = ['<div class="section"><div class="section-title">研究领域</div><ul class="compact">']
    for area in areas[:limit]:
        name = area.get("name", "")
        desc = area.get("description", area.get("trend", ""))
        subtopics = area.get("subtopics", [])
        meta = f' <span class="muted">[{", ".join(subtopics)}]</span>' if subtopics else ""
        parts.append(
            f"<li><strong>{name}</strong> — {desc}{meta}"
            f"{_first_paper_link(area.get('papers', []))}</li>"
        )
    if len(areas) > limit:
        parts.append(f'<li class="muted">… 另有 {len(areas) - limit} 个方向，见本地数据</li>')
    parts.append("</ul></div>")
    return "\n".join(parts)


def _render_insights_compact(
    items: list[dict[str, Any]],
    *,
    section_title: str,
    limit: int = 6,
    show_why: bool = False,
) -> str:
    if not items:
        return ""
    parts = [
        f'<div class="section"><div class="section-title">{section_title}</div>'
        f'<ul class="compact">'
    ]
    for item in items[:limit]:
        title = item.get("title", item.get("pattern_name", ""))
        framing = item.get(
            "problem_framing",
            item.get("how_problems_are_found", ""),
        )
        insight = item.get("insight", item.get("typical_insight", ""))
        why = item.get("why_it_matters", "")
        papers = item.get("papers", [])
        if not papers and item.get("paper"):
            papers = [item["paper"]]
        examples = item.get("examples", [])
        if not papers and examples:
            papers = examples

        line = f"<li>"
        if title:
            line += f"<strong>{title}</strong>："
        if insight:
            line += insight
        elif framing:
            line += framing
        if show_why and why:
            line += f' <span class="muted">({why})</span>'
        line += _first_paper_link(papers)
        line += "</li>"
        parts.append(line)
    if len(items) > limit:
        parts.append(f'<li class="muted">… 另有 {len(items) - limit} 条，见本地数据</li>')
    parts.append("</ul></div>")
    return "\n".join(parts)


def _render_overall_html(summary: dict[str, Any]) -> str:
    sections = [
        '<div class="card">',
        f'<div class="card-meta">模型：{summary.get("model", "—")}</div>',
    ]

    landscape = summary.get("research_landscape")
    discovery = summary.get("problem_discovery")

    if landscape or discovery:
        if isinstance(landscape, dict):
            overview = landscape.get("overview", "")
            if overview:
                sections.append(f'<p class="overview">{overview}</p>')
            sections.append(
                _render_research_areas_compact(landscape.get("areas", []), limit=8)
            )
        if isinstance(discovery, dict):
            overview = discovery.get("overview", "")
            if overview:
                sections.append(f'<p class="overview">{overview}</p>')
            sections.append(
                _render_insights_compact(
                    discovery.get("patterns", []),
                    section_title="问题发现模式",
                    limit=5,
                )
            )
            sections.append(
                _render_insights_compact(
                    discovery.get("standout_insights", []),
                    section_title="值得关注的问题重构",
                    limit=6,
                    show_why=True,
                )
            )
    else:
        legacy_summary = summary.get("summary", "")
        if legacy_summary:
            sections.append(f'<p class="overview">{legacy_summary}</p>')
        sections.append(
            _render_research_areas_compact(summary.get("research_areas", []), limit=6)
        )
        sections.append(
            _render_insights_compact(
                summary.get("problem_insights", []),
                section_title="问题发现与 Insight",
                limit=5,
            )
        )

    keywords = summary.get("keywords", [])
    if keywords:
        shown = keywords[:12]
        suffix = f" 等 {len(keywords)} 个" if len(keywords) > 12 else ""
        sections.append(
            f'<p class="keywords"><strong>关键词：</strong>{", ".join(shown)}{suffix}</p>'
        )

    sections.append("</div>")
    return "\n".join(sections)


def _build_plain_text(
    date_str: str,
    overall_summary: dict[str, Any] | None,
    category_counts: dict[str, int] | None,
    chunk_count: int,
) -> str:
    lines = [f"arXiv 每日论文摘要 - {date_str}", ""]

    if category_counts:
        total = sum(category_counts.values())
        cats = ", ".join(f"{k}:{v}" for k, v in category_counts.items())
        lines.append(f"论文：{total} 篇 ({cats})")
        lines.append("")

    if not overall_summary:
        lines.append(f"整体总结尚未生成。本地已有 {chunk_count} 个分块摘要。")
        return "\n".join(lines)

    landscape = overall_summary.get("research_landscape", {})
    discovery = overall_summary.get("problem_discovery", {})

    if isinstance(landscape, dict) and landscape.get("overview"):
        lines.append(landscape["overview"])
        lines.append("")

    areas = landscape.get("areas", []) if isinstance(landscape, dict) else []
    if areas:
        lines.append("【研究领域】")
        for area in areas[:8]:
            name = area.get("name", "")
            desc = area.get("description", area.get("trend", ""))
            lines.append(f"- {name}：{desc}")
        lines.append("")

    if isinstance(discovery, dict) and discovery.get("overview"):
        lines.append(discovery["overview"])
        lines.append("")

    patterns = discovery.get("patterns", []) if isinstance(discovery, dict) else []
    if patterns:
        lines.append("【问题发现模式】")
        for item in patterns[:5]:
            title = item.get("pattern_name", item.get("title", ""))
            insight = item.get("typical_insight", item.get("insight", ""))
            lines.append(f"- {title}：{insight}")
        lines.append("")

    standout = discovery.get("standout_insights", []) if isinstance(discovery, dict) else []
    if standout:
        lines.append("【值得关注】")
        for item in standout[:6]:
            lines.append(f"- {item.get('insight', '')}")

    keywords = overall_summary.get("keywords", [])
    if keywords:
        lines.append("")
        lines.append(f"关键词：{', '.join(keywords[:12])}")

    return "\n".join(lines)


def _build_html(
    date_str: str,
    overall_summary: dict[str, Any] | None,
    *,
    chunk_count: int,
    category_counts: dict[str, int] | None,
) -> str:

    sections = [
        "<html><head><style>"
        "body{font-family:Arial,Helvetica,sans-serif;background:#f6f8fb;color:#1f2937;}"
        ".container{max-width:720px;margin:0 auto;padding:20px;}"
        "h2{margin:0 0 12px;font-size:20px;}"
        ".card{background:#fff;border-radius:10px;padding:16px;margin:10px 0;"
        "box-shadow:0 1px 4px rgba(15,23,42,0.06);}"
        ".card-meta{font-size:12px;color:#64748b;margin-bottom:10px;}"
        ".meta{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px;}"
        ".tag{background:#e2e8f0;border-radius:999px;padding:3px 9px;font-size:12px;}"
        ".section{margin-top:14px;}"
        ".section-title{font-size:14px;font-weight:700;color:#0f172a;margin-bottom:6px;}"
        ".overview{color:#334155;line-height:1.55;margin:0 0 8px;}"
        "ul.compact{margin:0;padding-left:18px;}"
        "ul.compact li{margin:6px 0;line-height:1.45;color:#374151;}"
        ".muted{color:#64748b;font-size:13px;}"
        ".paper-ref{font-size:13px;}"
        ".keywords{font-size:13px;color:#475569;margin-top:12px;}"
        ".notice{color:#64748b;font-size:14px;line-height:1.5;}"
        "a{color:#2563eb;text-decoration:none;}"
        "a:hover{text-decoration:underline;}"
        '</style></head><body><div class="container">',
        f"<h2>arXiv 每日论文摘要 - {date_str}</h2>",
    ]

    if category_counts:
        total = sum(category_counts.values())
        sections.append('<div class="meta">')
        sections.append(f'<span class="tag">总计 {total} 篇</span>')
        for category, count in category_counts.items():
            sections.append(f'<span class="tag">{category}: {count}</span>')
        sections.append("</div>")

    if overall_summary:
        sections.append(_render_overall_html(overall_summary))
    else:
        sections.append(
            '<div class="card"><p class="notice">'
            f"整体总结尚未生成。本地已有 {chunk_count} 个分块摘要，"
            "请重跑任务或查看 data 目录。"
            "</p></div>"
        )

    sections.append("</div></body></html>")
    return "\n".join(sections)


def send_email(
    *,
    host: str,
    port: int,
    username: str,
    password: str,
    sender: str,
    recipients: list[str],
    subject: str,
    summaries: list[dict[str, Any]],
    overall_summary: dict[str, Any] | None = None,
    category_counts: dict[str, int] | None = None,
    date_str: str,
) -> None:
    if not recipients:
        raise ValueError("No SMTP recipients configured.")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid()

    text_body = _build_plain_text(
        date_str, overall_summary, category_counts, len(summaries)
    )
    html_body = _build_html(
        date_str,
        overall_summary,
        chunk_count=len(summaries),
        category_counts=category_counts,
    )

    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL(host, port) as server:
        server.login(username, password)
        server.send_message(message)
