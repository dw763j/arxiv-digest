# arXiv Digest Crawler

一个面向 arXiv 的每日论文摘要爬虫与邮件推送工具：按分类抓取新论文，分块摘要与整体总结，关键词提取，断点续跑，邮件推送。

## 核心功能
- **多分类抓取**：支持 `cs.SE/cs.CV/cs.AI/cs.CR/cs.LG/cs.RO` 等类别
- **定时运行**：按时区每日定时执行
- **分块摘要 + 整体总结**：大批量论文分块总结后再整体凝练
- **断点恢复**：分块结果与原始输出落盘，重跑时复用
- **邮件推送**：HTML 美化邮件，整体总结在前，主题含论文链接
- **数据归档**：按日期目录保存原始数据/摘要/模型输出

## 安装
```bash
git clone https://github.com/dw763j/arxiv-digest.git
cd arxiv-digest
uv sync
```

## 运行
- 单次执行（默认抓取昨天）
```bash
uv run arxiv-digest --once --log-path /path/to/arxiv-digest.log --env-file /path/to/.env.local
```

- 指定日期
```bash
uv run arxiv-digest --once --date 2026-01-15
```

- 定时执行
```bash
uv run arxiv-digest --schedule
```

## 环境变量（.env）

可保存为 .env.local 文件，然后运行时指定 `--env-file /path/to/.env.local`。

```env
# arXiv 分类
ARXIV_CATEGORIES=cs.SE,cs.CV,cs.AI,cs.CR,cs.LG,cs.RO  # 可自定义选择分类

# 调度
APP_TIMEZONE=Asia/Shanghai
APP_DAILY_TIME=09:00

# 数据目录
APP_DATA_DIR=./data
APP_RETENTION_DAYS=30

# OpenAI 兼容 API
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
OPENAI_CHUNK_MODEL=gpt-4o-mini  # 分块摘要模型
OPENAI_OVERALL_MODEL=claude-sonnet-4-5-20250929  # 整体总结模型，使用更强大的模型

# SMTP(S)
SMTP_HOST=...
SMTP_PORT=465
SMTP_USER=...
SMTP_PASSWORD=...
SMTP_FROM=...
SMTP_TO=foo@bar.com,bar@foo.com
```

## 数据结构
```
data/
  YYYY-MM-DD/
    raw/papers.jsonl
    summaries/summary_partXX.json
    summaries/summary_overall.json
    responses/response_partXX.txt
    responses/response_overall.txt
  state/state.json
```

## 运行示例
```text
$ uv run arxiv-digest --once --date 2026-01-14
2026-01-15 09:00:01 | INFO | Running digest task for 2026-01-14
2026-01-15 09:00:02 | INFO | Loaded 291 papers from storage for 2026-01-14
2026-01-15 09:00:02 | INFO | Found 3/3 summary chunks for 2026-01-14. Reprocessing missing.
2026-01-15 09:00:05 | INFO | Summarizing chunk 1/3 (20 papers)
2026-01-15 09:01:18 | INFO | Summarizing overall digest with 3 chunks
2026-01-15 09:01:22 | INFO | Email sent to foo@bar.com
```

## 常见问题
**Q: 为什么重复运行没有重新总结？**  
A: 默认会复用已生成的分块摘要与整体总结，只有缺失时才补生成。可删除对应日期目录或使用 `--migrate` 后重新跑以重建。

**Q: 邮件没有收到整体总结？**  
A: 需要 `summary_overall.json` 存在；若缺失会自动补生成。请确认 `OPENAI_OVERALL_MODEL` 与 `OPENAI_API_KEY` 有效。

**Q: 没有发送邮件？**  
A: 需配置 `SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD/SMTP_TO`；缺失会跳过发送。

## 说明
- 非 GPT 系列模型默认走 `chat.completions`，GPT 系列优先走 `responses`，失败自动回退。
- 断点续跑：已生成的分块不会重复调用模型；整体总结缺失则补生成。

---

<details>
<summary><b>关键词</b></summary>

`arXiv` `crawler` `paper-digest` `daily-summary` `LLM` `OpenAI` `responses` `chat-completions` `email` `SMTPS` `scheduler` `apscheduler` `loguru` `python` `uv` `爬虫` `论文摘要` `每日摘要`

</details>
