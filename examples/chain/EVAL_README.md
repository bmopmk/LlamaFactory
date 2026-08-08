# 评估说明（Qwen3-0.6B pt → sft → dpo 链路）

本目录下三个训练配置已开启**在训评估**；本文件补充**训练后基准评估**的可用方案。

---

## 1. 在训评估（已配置，自动生效）

`qwen3_pt.yaml` / `qwen3_sft.yaml` / `qwen3_dpo.yaml` 均已加入：

```yaml
val_size: 0.1          # DPO 为 0.05
eval_strategy: steps
eval_steps: 50
per_device_eval_batch_size: 1
```

训练时会周期性输出到 TensorBoard（`saves/qwen3-0.6b/lora/*/runs`）：

- **PT / SFT**：`eval_loss`（PT 可换算 perplexity = exp(eval_loss)）。
- **DPO**：`eval_rewards/accuracies`（应 >0.5 并上升）、`eval_rewards/margins`（应为正）、`eval_logps/*`。

---

## 2. SFT 生成质量：BLEU / ROUGE（配置已备好）

用 `qwen3_predict.yaml` 对 SFT 产物做批量生成并算指标：

```bash
llamafactory-cli train examples/chain/qwen3_predict.yaml
```

结果（含 predict_bleu-4、predict_rouge-1/2/l）写到 `saves/qwen3-0.6b/lora/predict/`。

> 批量生成慢。更快的做法：先用 `scripts/vllm_infer.py` 生成，再用 `scripts/eval_bleu_rouge.py` 算分：
> ```bash
> python scripts/vllm_infer.py \
>   --model_name_or_path /home/genghaoyu/llm/vllm/models/Qwen3-0.6B \
>   --adapter_name_or_path saves/qwen3-0.6b/lora/sft \
>   --template qwen3_nothink \
>   --dataset alpaca_gpt4_en_local \
>   --save_name saves/qwen3-0.6b/lora/predict/generated.jsonl
> python scripts/eval_bleu_rouge.py saves/qwen3-0.6b/lora/predict/generated.jsonl
> ```

---

## 3. MMLU / CMMLU / C-Eval 选择题基准

> ⚠️ 本仓库内置的 `llamafactory-cli eval` 已废弃：
> `src/llamafactory/launcher.py` 中 `command == "eval"` 直接
> `raise NotImplementedError("Evaluation will be deprecated in the future.")`。
> 因此**不要**再用 `llamafactory-cli eval`。

推荐改用外部评测框架，对训练产物做标准化跑分。两条路：

### 3a. 先合并 LoRA 再评测（推荐）

```bash
# 把 dpo/sft 的 LoRA 合并进基座，导出成完整模型
llamafactory-cli export \
  --model_name_or_path /home/genghaoyu/llm/vllm/models/Qwen3-0.6B \
  --adapter_name_or_path saves/qwen3-0.6b/lora/dpo \
  --template qwen3_nothink \
  --finetuning_type lora \
  --export_dir saves/qwen3-0.6b/merged/dpo
```

### 3b. 用 lm-evaluation-harness 跑 MMLU / CMMLU / C-Eval

```bash
pip install lm-eval
lm_eval --model hf \
  --model_args pretrained=saves/qwen3-0.6b/merged/dpo,trust_remote_code=True \
  --tasks mmlu,cmmlu,ceval-valid \
  --num_fewshot 5 \
  --batch_size 8 \
  --output_path saves/qwen3-0.6b/eval_bench
```

离线环境需提前把 harness 的任务数据缓存到服务器（同数据集下载逻辑）。

> 注：0.6B 小模型在 MMLU/CMMLU 上分数会很低（接近随机 25%），主要用于观察相对提升，不宜与大模型横比。
