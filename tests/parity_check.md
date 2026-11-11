# Parity check: `--batch_size 1` vs `--batch_size 16`

Purpose: confirm that batched, left-padded generation in `evaluate.py` gives the
same answers as one-at-a-time generation before using `--batch_size 16` for the
full benchmark runs. Run it once per base model (LLaMA-family and, if used, Qwen2),
for one adapter. It takes a few minutes on one A100.

Run everything from the repo root, on a GPU node.

## 0. Variables

```bash
BASE=openlm-research/open_llama_3b_v2           # base model
ADAPTER=./trained_models/openllama-3b-lora-math7k  # an adapter trained with finetune.py
OUT=./parity                                     # scratch output dir (delete afterwards)
mkdir -p "$OUT"
```

Use a fresh `$OUT` directory: `evaluate.py` resumes from existing JSONL files
and would otherwise skip every example.

## 1. Run batch size 1 twice and batch size 16 once

```bash
for DS in AddSub AQuA; do
  python evaluate.py --dataset $DS --adapter LoRA --base_model "$BASE" --lora_weights "$ADAPTER" \
      --limit 50 --batch_size 1  --out_file "$OUT/${DS}_bs1_a.jsonl"; echo "exit=$?"
  python evaluate.py --dataset $DS --adapter LoRA --base_model "$BASE" --lora_weights "$ADAPTER" \
      --limit 50 --batch_size 1  --out_file "$OUT/${DS}_bs1_b.jsonl"; echo "exit=$?"
  python evaluate.py --dataset $DS --adapter LoRA --base_model "$BASE" --lora_weights "$ADAPTER" \
      --limit 50 --batch_size 16 --out_file "$OUT/${DS}_bs16.jsonl"; echo "exit=$?"
done
```

Every run must print `exit=0`.

## 2. Compare

```bash
python - "$OUT" <<'EOF'
import json, sys, os
out = sys.argv[1]
def load(p):
    return {r["index"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
for ds in ["AddSub", "AQuA"]:
    a, b, c = (load(os.path.join(out, f"{ds}_{s}.jsonl")) for s in ["bs1_a", "bs1_b", "bs16"])
    assert sorted(a) == sorted(b) == sorted(c) == list(range(50)), f"{ds}: missing indices"
    def cmp(x, y, name):
        same_text = sum(x[i]["output"] == y[i]["output"] for i in x)
        same_pred = sum(x[i]["prediction"] == y[i]["prediction"] for i in x)
        acc_x = sum(x[i]["correct"] for i in x); acc_y = sum(y[i]["correct"] for i in y)
        print(f"{ds:7s} {name:14s} same output {same_text}/50 | same prediction {same_pred}/50 | "
              f"correct {acc_x} vs {acc_y}")
        for i in sorted(x):
            if x[i]["prediction"] != y[i]["prediction"]:
                print(f"    idx {i}: {x[i]['prediction']!r} vs {y[i]['prediction']!r} "
                      f"(tokens {x[i]['generated_tokens']} vs {y[i]['generated_tokens']})")
    cmp(a, b, "bs1 vs bs1")
    cmp(a, c, "bs1 vs bs16")
    print(f"{ds:7s} hit_max_tokens (bs1): {sum(r['hit_max_tokens'] for r in a.values())}/50")
EOF
```

## 3. Pass criteria

| Check | Expected | If not |
|---|---|---|
| `bs1 vs bs1` | 50/50 identical outputs | Eval is non-deterministic: check that the log prints `training=False` (dropout off) and that nothing else changed between runs. Don't continue until this passes. |
| `bs1 vs bs16`, predictions | 50/50 identical is typical; 48–49/50 is acceptable | Left padding changes fp16 rounding (different matmul shapes, padded attention), so beam search can occasionally take a different path. Only a few near-tie examples should flip. |
| `bs1 vs bs16`, correct count | differs by at most 1 per 50 | A larger gap points to a padding bug. Confirm the log prints `padding_side=left` and the expected `pad_token_id` (0 for llama models, 151643 for Qwen2.5), and look at the flipped examples' `output` text. |
| `exit=` | 0 for every run | See the traceback; exit 2 means an argument error. |

If `bs1 vs bs16` fails the criteria, use `--batch_size 1` for the reported
numbers (it is still much faster than before because `use_cache=True`).

For the report, use one batch size for all models and adapters, and state which.

## 4. Clean up

```bash
rm -rf "$OUT"
```
