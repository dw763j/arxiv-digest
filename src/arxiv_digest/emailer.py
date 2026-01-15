from __future__ import annotations

from email.message import EmailMessage
import json
import smtplib
from typing import Any, Iterable


def _render_theme_html(theme: dict[str, Any]) -> str:
    name = theme.get("name", "")
    description = theme.get("description", "")
    papers = theme.get("papers", []) or []
    parts = [f"<div class=\"theme-title\">{name}</div>"]
    if description:
        parts.append(f"<div class=\"theme-desc\">{description}</div>")
    if papers:
        parts.append("<div class=\"theme-links\">")
        parts.append("<div class=\"theme-links-title\">相关论文：</div>")
        parts.append("<ul>")
        for paper in papers:
            title = paper.get("title", "Untitled")
            link = paper.get("link", "")
            if link:
                parts.append(f"<li><a href=\"{link}\">{title}</a></li>")
            else:
                parts.append(f"<li>{title}</li>")
        parts.append("</ul>")
        parts.append("</div>")
    return "\n".join(parts)


def _render_summary_html(title: str, summary: dict[str, Any]) -> str:
    sections = [f"<div class=\"card\"><h3>{title}</h3>"]
    sections.append(f"<p><strong>总结：</strong>{summary.get('summary', '')}</p>")
    keywords = summary.get("keywords", [])
    if keywords:
        sections.append(f"<p><strong>关键词：</strong>{', '.join(keywords)}</p>")
    themes = summary.get("themes", [])
    if themes:
        sections.append("<div class=\"themes\">")
        for theme in themes:
            sections.append("<div class=\"theme\">")
            sections.append(_render_theme_html(theme))
            sections.append("</div>")
        sections.append("</div>")
    sections.append("</div>")
    return "\n".join(sections)


def _build_html(
    date_str: str,
    overall_summary: dict[str, Any] | None,
    summaries: Iterable[dict[str, Any]],
) -> str:
    sections = [
        "<html><head><style>"
        "body{font-family:Arial,Helvetica,sans-serif;background:#f6f8fb;color:#1f2937;}"
        ".container{max-width:880px;margin:0 auto;padding:24px;}"
        "h2{margin:0 0 16px;font-size:22px;}"
        "h3{margin:0 0 10px;font-size:18px;color:#0f172a;}"
        ".card{background:#ffffff;border-radius:12px;padding:16px 18px;margin:12px 0;"
        "box-shadow:0 2px 8px rgba(15,23,42,0.06);}"
        ".themes{margin-top:10px;border-top:1px dashed #e5e7eb;padding-top:10px;}"
        ".theme{padding:8px 0;border-bottom:1px solid #f1f5f9;}"
        ".theme:last-child{border-bottom:none;}"
        ".theme-title{font-weight:700;color:#111827;}"
        ".theme-desc{margin:4px 0 6px;color:#374151;}"
        ".theme-links-title{font-weight:600;color:#334155;margin-bottom:4px;}"
        "a{color:#2563eb;text-decoration:none;}"
        "a:hover{text-decoration:underline;}"
        "ul{padding-left:18px;margin:6px 0;}"
        "</style></head><body><div class=\"container\">",
        f"<h2>arXiv 每日摘要 - {date_str}</h2>",
    ]
    if overall_summary:
        sections.append(_render_summary_html("整体总结", overall_summary))
    for idx, summary in enumerate(summaries, start=1):
        sections.append(_render_summary_html(f"分块摘要 {idx}", summary))
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
    date_str: str,
) -> None:
    if not recipients:
        raise ValueError("No SMTP recipients configured.")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject

    payload = {
        "date": date_str,
        "overall": overall_summary or {},
        "chunks": summaries,
    }
    text_body = json.dumps(payload, ensure_ascii=False, indent=2)
    html_body = _build_html(date_str, overall_summary, summaries)

    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    with smtplib.SMTP_SSL(host, port) as server:
        server.login(username, password)
        server.send_message(message)
