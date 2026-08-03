# Badcase → SFT → 线上评测 数据闭环

本目录说明 `allcallall-agent-runtime` 中从线上失败样本回流到模型迭代训练、再自动评测验证的完整闭环（Part 1–3 已落地，Part 4 为接线与文档）。

## 闭环总览

```
[线上 run_workflow]
      │  (opt-in: enable_badcase_capture=true)
      ▼
[Part 1] Badcase 定义：classify_badcase 复用现有失败信号做确定性判定，落 BadcaseStore
      │  人工标注台补 label_corrected_response（金标准）→ sft_eligible=True
      ▼
[Part 2] SFT 回流：build_sft_dataset 把 eligible 样本转标准 messages 格式，export_sft_dataset 导出 JSONL
      │  交付外部训练平台微调 → 产出新 ModelVersion
      ▼
[Part 3] 线上评测：run_online_eval 在 golden 集（evals/cases.json）+ 抽样 live 样本上
      对比新旧 ModelVersion 的 eval_runner 15 维指标 + IR 指标，验证目标 Badcase 类别改善
```

## Part 1 — Badcase 定义

- `badcase.py`：`classify_badcase(request, response)` 优先级 `APPROVAL_BYPASS > REVIEW_REJECT > RETRIEVAL_MISS > TIMEOUT > RUNTIME_ERROR`；`BadcaseStore` 复用 checkpoint 的 sqlite3 单连接模式。
- 判定信号全部来自现有字段：`OutputDecision.final_verdict`、`status`/`stop_reason`/grounding、写工具 `approval_required`。零新增埋点。
- 分类：`RETRIEVAL_MISS / HALLUCINATION / ROUTE_ERROR / APPROVAL_BYPASS / TIMEOUT / REVIEW_REJECT / USER_DECLINE / UNSUPPORTED_MISHANDLE / RUNTIME_ERROR`。
- 接线：`harness.py` opt-in；`config.py` 的 `enable_badcase_capture` / `badcase_sqlite_path`。

## Part 2 — SFT 回流

- `sft_dataset.py`：`build_sft_sample` 仅对 `sft_eligible` 且已 `label` 且含 `label_corrected_response` 的 badcase 构造样本。
- 类别→金标准映射（构造 assistant 内容时校验不变量）：
  - `APPROVAL_BYPASS`：修正后所有写工具仍 `approval_required=True`，否则丢弃该样本。
  - `UNSUPPORTED_MISHANDLE`：修正后 `proposed_tool_calls=[]`，否则丢弃。
  - 其余：用修正后的 `summary` + 真实 `citations` 组装。
- `export_sft_dataset(records, out_path)` 导出 JSONL（每行为 `SFTSample`），供外部训练平台直接消费。
- 当前 `provider` 为 `rules` 占位、无自有可微调模型，本模块只产出标准数据集，训练在外部完成。

## Part 3 — 线上评测

- `online_eval.py`：复用 `eval_runner` 的 15 维断言与 `engineering_harness` 的 IR 指标（镜像 Go `rag_eval.go`）。
- `compare_eval_runs(baseline, candidate, target_categories)` 计算逐项 `delta`，`improved = 目标类别指标未回退 AND 整体通过率未回退`。
- `build_eval_run` + `EvalRunStore`（SQLite）持久化每次评测结果，支持 `latest_for_version` 取基线。
- `rag_metrics_from_cases` 用 `RagEvalHarness` 计算 `hit_rate_at_5 / mrr / ndcg_at_5`，与 Go 侧口径一致。

## CI / 命令行

```bash
# 1. 捕获 badcase（运行时设环境变量 / 配置 enable_badcase_capture=true）
# 2. 标注后导出 SFT 数据集
make sft-dataset           # 等价: python scripts/export_sft.py --out evals/sft_dataset.jsonl
# 3. 训练平台用 sft_dataset.jsonl 微调，得到新 ModelVersion
# 4. 跑线上评测（candidate 自动跑 eval_runner；baseline 取 eval_runs.db 中同版本最新一次）
make online-eval           # 等价: python scripts/run_online_eval.py --baseline-db eval_runs.db \
                          #        --model-version <new> --baseline-version <base> \
                          #        --target-categories retrieval_miss,hallucination
```

约束（仓库守则）：ruff `<0.16`、mypy strict、契约变更须 `make contracts` 重新生成并 `contracts-check`；推送仅 `ssh://git@ssh.github.com:443/XianingY/allcallall-agent-runtime.git`（SSH-443）。

## 后续（Phase 2，未在本轮实现）

- `USER_FEEDBACK`：需在 Go 后端加用户采纳/点踩落库端点，经既有 tool bridge 回传 `badcase_store`。
- 真实模型接入：把外部微调产出的权重接入 `provider` 配置，使 `run_online_eval` 可切换 candidate provider 而非仅对比报告。
- live 样本回收：线上抽样样本的自动入库与评测。
