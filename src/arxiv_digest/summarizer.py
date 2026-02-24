from __future__ import annotations

from datetime import date
import json
import re
import time
from typing import Any, Callable

from loguru import logger
from openai import OpenAI
from openai import (
    AuthenticationError,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
)

from .models import Paper


class SummarizationError(Exception):
    """摘要生成过程中的错误。"""
    pass


def _is_retryable_error(exc: Exception) -> bool:
    """判断错误是否可重试。"""
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        # 5xx 服务器错误可重试，4xx 客户端错误通常不可重试
        return exc.status_code >= 500
    return False


def _get_retry_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
    """计算指数退避延迟。"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    return delay


def _log_api_error(exc: Exception, model: str, api_type: str) -> None:
    """记录 API 错误详情。"""
    if isinstance(exc, AuthenticationError):
        logger.error(
            "[{}] 认证失败 (401) - 请检查 OPENAI_API_KEY 和 OPENAI_BASE_URL 是否正确。model={}",
            api_type,
            model,
        )
    elif isinstance(exc, RateLimitError):
        logger.error(
            "[{}] 速率限制 (429) - 请求过于频繁，请稍后重试。model={}",
            api_type,
            model,
        )
    elif isinstance(exc, APITimeoutError):
        logger.error(
            "[{}] 请求超时 - 网络连接超时或服务器响应过慢。model={}",
            api_type,
            model,
        )
    elif isinstance(exc, APIConnectionError):
        logger.error(
            "[{}] 连接错误 - 无法连接到 API 服务器，请检查网络或 BASE_URL 配置。model={}",
            api_type,
            model,
        )
    elif isinstance(exc, InternalServerError):
        logger.error(
            "[{}] 服务器内部错误 (500) - 服务端问题，请稍后重试。model={}",
            api_type,
            model,
        )
    elif isinstance(exc, APIStatusError):
        logger.error(
            "[{}] API 状态错误 - 状态码：{}, model={}",
            api_type,
            exc.status_code,
            model,
        )
    else:
        logger.error(
            "[{}] 未知错误 - {}: {}",
            api_type,
            type(exc).__name__,
            exc,
        )


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
    on_response: Callable[[int, dict[str, Any]], None] | None = None,
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
        try:
            text_output, raw_payload = _call_model_with_fallback(client, model, prompt)
        except SummarizationError as exc:
            logger.error(
                "Chunk {}/{} 摘要生成失败，跳过该分块：{}",
                chunk_index,
                len(chunks),
                exc,
            )
            # 返回一个空的摘要结构，避免中断整个流程
            outputs.append((chunk_index, {"summary": "摘要生成失败", "keywords": [], "themes": []}))
            continue
        if on_response:
            on_response(chunk_index, raw_payload)
        try:
            if text_output.startswith("```json") and text_output.endswith("```"):
                text_output = text_output[len("```json"):-len("```")]
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
    on_response: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not summaries:
        return {}
    prompt = _build_overall_prompt(target_date, summaries)
    logger.info("Summarizing overall digest with {} chunks", len(summaries))
    try:
        text_output, raw_payload = _call_model_with_fallback(client, model, prompt)
    except SummarizationError as exc:
        logger.error("整体摘要生成失败：{}", exc)
        # 返回一个空的总结结构
        return {"summary": "整体摘要生成失败", "keywords": [], "themes": []}
    if on_response:
        on_response(raw_payload)
    try:
        return json.loads(text_output)
    except json.JSONDecodeError:
        logger.warning("Failed to parse overall JSON. Attempting fallback parsing.")
        return _extract_json(text_output)


def _call_model_with_fallback(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> tuple[str, dict[str, Any]]:
    """
    调用 LLM API 进行摘要生成，支持重试和降级。

    Args:
        client: OpenAI 客户端
        model: 模型名称
        prompt: 提示词
        max_retries: 最大重试次数
        base_delay: 基础重试延迟（秒）

    Returns:
        (文本输出，原始响应 payload)

    Raises:
        SummarizationError: 当所有重试和降级都失败时
    """
    use_responses = model.lower().startswith("gpt")
    last_error: Exception | None = None

    # 尝试 responses API（仅 GPT 模型）
    if use_responses:
        for attempt in range(max_retries):
            try:
                logger.debug(
                    "Calling responses API (attempt {}/{}): model={}",
                    attempt + 1,
                    max_retries,
                    model,
                )
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    temperature=0.2,
                )
                return response.output_text, _to_payload(response)
            except Exception as exc:
                last_error = exc
                if _is_retryable_error(exc) and attempt < max_retries - 1:
                    delay = _get_retry_delay(attempt, base_delay)
                    logger.warning(
                        "Responses API 失败，{} 后重试 (attempt {}/{}): {}",
                        f"{delay}秒",
                        attempt + 1,
                        max_retries,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                _log_api_error(exc, model, "responses")
                logger.warning(
                    "Responses API 失败，降级到 chat.completions: model={}",
                    model,
                )
                break

    # 降级到 chat.completions API
    for attempt in range(max_retries):
        try:
            logger.debug(
                "Calling chat.completions API (attempt {}/{}): model={}",
                attempt + 1,
                max_retries,
                model,
            )
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = completion.choices[0].message.content
            if not content:
                logger.warning("chat.completions 返回空内容，model={}", model)
                content = ""
            return content, _to_payload(completion)
        except Exception as exc:
            last_error = exc
            if _is_retryable_error(exc) and attempt < max_retries - 1:
                delay = _get_retry_delay(attempt, base_delay)
                logger.warning(
                    "chat.completions 失败，{} 后重试 (attempt {}/{}): {}",
                    f"{delay}秒",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                time.sleep(delay)
                continue
            _log_api_error(exc, model, "chat.completions")
            # 非重试错误或达到最大重试次数
            if not _is_retryable_error(exc):
                break

    # 所有尝试都失败
    raise SummarizationError(
        f"LLM API 调用失败 (model={model}): {type(last_error).__name__}: {last_error}"
    ) from last_error


def _to_payload(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {"raw": str(obj)}
