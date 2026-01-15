from __future__ import annotations

from datetime import date
import json
import re
from typing import Any, Callable

from loguru import logger
from openai import OpenAI

from .models import Paper


def _chunk(items: list[Paper], size: int) -> list[list[Paper]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response.")
    return json.loads(match.group(0))


def _build_prompt(target_date: date, papers: list[Paper]) -> str:
    lines = [
        "你是一名科研情报分析师，请基于以下 arXiv 论文列表输出结构化摘要。",
        "要求：",
        "1) 用中文输出。",
        "2) 给出主题分类（3-8 个主题），每个主题提供简短说明与代表论文。",
        "3) 提炼整体关键词（10-20 个）。",
        "4) 输出一个简要的整体总结（不超过 8 句）。",
        "5) 返回 JSON 对象，不要添加额外文字。",
        "",
        f"日期：{target_date.isoformat()}",
        "",
        "论文列表：",
    ]
    for idx, paper in enumerate(papers, start=1):
        lines.append(
            f"{idx}. [{paper.category}] {paper.title}\n"
            f"   Authors: {', '.join(paper.authors)}\n"
            f"   Abstract: {paper.summary}\n"
            f"   Link: {paper.link}\n"
        )

    lines.append(
        "JSON 结构示例："
        '{"summary": "...", "keywords": ["..."], "themes": '
        '[{"name": "...", "description": "...", "papers": [{"title": "...", "link": "..."}]}]}'
    )
    return "\n".join(lines)


def summarize_papers(
    client: OpenAI,
    model: str,
    target_date: date,
    papers: list[Paper],
    chunk_size: int = 20,
) -> list[dict[str, Any]]:
    return [
        payload
        for _, payload in summarize_papers_stream(
            client,
            model,
            target_date,
            papers,
            chunk_size=chunk_size,
            existing_chunks=None,
        )
    ]


def summarize_papers_stream(
    client: OpenAI,
    model: str,
    target_date: date,
    papers: list[Paper],
    *,
    chunk_size: int = 20,
    existing_chunks: dict[int, dict[str, Any]] | None = None,
    on_response: Callable[[int, str], None] | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    if not papers:
        return []

    outputs: list[tuple[int, dict[str, Any]]] = []
    chunks = _chunk(papers, chunk_size)
    existing_chunks = existing_chunks or {}

    for chunk_index, chunk in enumerate(chunks, start=1):
        if chunk_index in existing_chunks:
            logger.info("Using cached summary for chunk {}/{}", chunk_index, len(chunks))
            outputs.append((chunk_index, existing_chunks[chunk_index]))
            continue
        prompt = _build_prompt(target_date, chunk)
        logger.info(
            "Summarizing chunk {}/{} ({} papers)", chunk_index, len(chunks), len(chunk)
        )
        text_output = _call_model_with_fallback(client, model, prompt)
        if on_response:
            on_response(chunk_index, text_output)
        try:
            payload = json.loads(text_output)
        except json.JSONDecodeError:
            logger.warning("Failed to parse direct JSON. Attempting fallback parsing.")
            payload = _extract_json(text_output)
        outputs.append((chunk_index, payload))

    return outputs


def _build_overall_prompt(target_date: date, summaries: list[dict[str, Any]]) -> str:
    lines = [
        "你是一名科研情报分析师，请基于分块摘要进行一次更凝练的整体总结。",
        "要求：",
        "1) 用中文输出。",
        "2) 给出主题分类（3-6 个主题），每个主题提供一句话说明与代表论文。",
        "3) 提炼整体关键词（8-15 个）。",
        "4) 输出一个简要的整体总结（不超过 5 句）。",
        "5) 返回 JSON 对象，不要添加额外文字。",
        "",
        f"日期：{target_date.isoformat()}",
        "",
        "分块摘要 JSON：",
        json.dumps(summaries, ensure_ascii=False),
        "",
        "JSON 结构示例："
        '{"summary": "...", "keywords": ["..."], "themes": '
        '[{"name": "...", "description": "...", "papers": [{"title": "...", "link": "..."}]}]}',
    ]
    return "\n".join(lines)


def summarize_overall(
    client: OpenAI,
    model: str,
    target_date: date,
    summaries: list[dict[str, Any]],
    *,
    on_response: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not summaries:
        return {}
    prompt = _build_overall_prompt(target_date, summaries)
    logger.info("Summarizing overall digest with {} chunks", len(summaries))
    text_output = _call_model_with_fallback(client, model, prompt)
    if on_response:
        on_response(text_output)
    try:
        return json.loads(text_output)
    except json.JSONDecodeError:
        logger.warning("Failed to parse overall JSON. Attempting fallback parsing.")
        return _extract_json(text_output)


def _call_model_with_fallback(client: OpenAI, model: str, prompt: str) -> str:
    use_responses = model.lower().startswith("gpt")
    if use_responses:
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=0.2,
            )
            return response.output_text
        except Exception as exc:  # fallback for models that don't support /responses
            logger.warning(
                "Responses API failed for model {}, falling back to chat.completions: {}",
                model,
                exc,
            )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content or ""
