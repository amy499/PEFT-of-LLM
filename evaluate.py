import json
import math
import os
import re
import sys
import argparse

import torch

# Vendored peft fork: put it FIRST on sys.path, resolved relative to this file
# (not the cwd), so a pip-installed peft can never shadow it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "peft", "src"))
import peft  # noqa: E402
print(f"Using peft from: {peft.__file__}")
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

from prompts import generate_prompt

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main():
    args = parse_args()
    print(f"precision: {args.precision}")

    dataset = load_data(args)
    if args.limit is not None:
        dataset = dataset[: args.limit]
    total = len(dataset)

    # Resume: skip example indices already present in the JSONL output.
    done = load_done_indices(args.out_file)
    todo = [idx for idx in range(total) if idx not in done]
    print(f"{len(done)} example(s) already in {args.out_file}; {len(todo)} of {total} to evaluate.")
    if not todo:
        report_accuracy(args.out_file, total)
        return

    tokenizer, model = load_model(args)
    pad_token_id = configure_padding(tokenizer, model)
    eos_ids = get_eos_ids(tokenizer, model)

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)  # left-padded
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        results = []
        prompt_len = input_ids.shape[1]
        for s in generation_output.sequences:
            output = tokenizer.decode(s, skip_special_tokens=True)
            output = output.split("### Response:")[1].strip()
            n_generated, hit_max = count_generated(s[prompt_len:].tolist(), eos_ids, pad_token_id, max_new_tokens)
            results.append((output, n_generated, hit_max))
        return results

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    os.makedirs(out_dir, exist_ok=True)
    ensure_trailing_newline(args.out_file)

    correct = 0
    pbar = tqdm(total=len(todo))
    for start in range(0, len(todo), args.batch_size):
        batch_idx = todo[start: start + args.batch_size]
        batch = [dataset[idx] for idx in batch_idx]
        outputs = evaluate(
            [data.get('instruction') for data in batch],
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        records = []
        for idx, data, (output, n_generated, hit_max) in zip(batch_idx, batch, outputs):
            label = data.get('answer')
            predict, flag = score(args, label, output)
            correct += int(flag)
            records.append({
                "index": idx,
                "instruction": data.get('instruction'),
                "label": label,
                "output": output,
                # float('inf') (no number found) is not valid JSON; stored as null.
                "prediction": predict if not (isinstance(predict, float) and not math.isfinite(predict)) else None,
                "correct": flag,
                "generated_tokens": n_generated,
                "hit_max_tokens": hit_max,
            })
            print(' ')
            print('---------------')
            print(output)
            print('prediction:', predict)
            print('label:', label)
            print('---------------')
        # Append-only: one JSON object per line, flushed after every batch.
        with open(args.out_file, 'a', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        pbar.update(len(batch_idx))
        print(f'\rthis run: {start + len(batch_idx)}/{len(todo)} | accuracy {correct}  '
              f'{correct / (start + len(batch_idx))}')
    pbar.close()
    print('\n')
    print('test finished')
    report_accuracy(args.out_file, total)


def score(args, label, outputs):
    """Original scoring logic, unchanged: first-letter extraction for AQuA,
    last number with absolute tolerance 0.001 for everything else."""
    miss = 0.001
    flag = False
    if args.dataset.lower() in ['aqua']:
        predict = extract_answer_letter(args, outputs)
        if label == predict:
            flag = True
    else:
        if isinstance(label, str):
            label = float(label)
        predict = extract_answer_number(args, outputs)
        if abs(label - predict) <= miss:
            flag = True
    return predict, flag


def count_generated(new_tokens, eos_ids, pad_token_id, max_new_tokens):
    """Number of generated tokens (up to and including the first EOS) and whether
    generation stopped because it hit max_new_tokens rather than EOS."""
    for i, tok in enumerate(new_tokens):
        if tok in eos_ids:
            return i + 1, False
    # No EOS: drop trailing padding that generate() adds to shorter sequences.
    n = len(new_tokens)
    while n > 0 and new_tokens[n - 1] == pad_token_id:
        n -= 1
    return n, n >= max_new_tokens


def get_eos_ids(tokenizer, model):
    eos_ids = set()
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for e in (gen_eos, tokenizer.eos_token_id):
        if e is None:
            continue
        if isinstance(e, (list, tuple)):
            eos_ids.update(e)
        else:
            eos_ids.add(e)
    return eos_ids


def configure_padding(tokenizer, model):
    """Left padding for batched generation. pad = 0 for llama (<unk>, as in
    finetune.py); otherwise the tokenizer's own pad token, else EOS."""
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type == "llama":
        pad_token_id = 0
    elif tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token_id = pad_token_id
    tokenizer.padding_side = "left"
    print(f"model_type={model_type} pad_token_id={pad_token_id} padding_side=left")
    return pad_token_id


def load_done_indices(out_file):
    done = set()
    if not os.path.exists(out_file):
        return done
    with open(out_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["index"])
            except (json.JSONDecodeError, KeyError):
                print(f"warning: ignoring malformed line {line_no} in {out_file} (will be re-evaluated)")
    return done


def ensure_trailing_newline(out_file):
    """If a previous run was killed mid-write, make sure new records start on a fresh line."""
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        with open(out_file, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with open(out_file, 'a', encoding='utf-8') as f:
                f.write("\n")


def report_accuracy(out_file, total):
    records = {}
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("index", -1) < total:
                    records[rec["index"]] = rec
    n = len(records)
    n_correct = sum(1 for r in records.values() if r.get("correct"))
    n_max = sum(1 for r in records.values() if r.get("hit_max_tokens"))
    if n:
        print(f"accuracy over {n}/{total} recorded examples: {n_correct}/{n} = {n_correct / n:.4f} "
              f"| hit max_new_tokens: {n_max}/{n}")
    else:
        print("no records found")


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B'], default="")
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'],
                        default=None, help='label only; no longer used for the output filename')
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', default=None,
                        help='adapter directory (required unless --no_adapter)')
    parser.add_argument('--no_adapter', action='store_true', default=False,
                        help='evaluate the base model without loading adapter weights')
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--out_file', required=True,
                        help='append-only JSONL results file; existing indices are skipped (resume)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None,
                        help='only evaluate the first N examples (smoke tests)')
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'], default='fp16',
                        help='torch_dtype for the base model and adapter on CUDA')

    args = parser.parse_args()
    if not args.no_adapter and not args.lora_weights:
        parser.error('--lora_weights is required unless --no_adapter is given')
    if args.batch_size < 1:
        parser.error('--batch_size must be >= 1')
    return args


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights
    use_adapter = not args.no_adapter

    load_8bit = args.load_8bit
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
        if use_adapter:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch_dtype,
                device_map={"":0}
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if use_adapter:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if use_adapter:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    # Inference mode in EVERY branch: freshly created LoRA layers are built in
    # train mode, so without this lora_dropout stays active during generation.
    model.eval()
    print(f"adapter: {'none (base model)' if not use_adapter else lora_weights} | training={model.training}")

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    main()
