import os
import json
import torch
from unsloth import FastLanguageModel
from transformers import GenerationConfig

EVAL_DIR = "data/evaluation"
MODEL_DIR = "finetuned_model"

def format_pair(inp, out=None):
    """Match training format."""
    text = f"Input:\n{inp}\nOutput:\n"
    if out is not None:
        text += str(out)
    return text

def load_eval_files(eval_dir):
    for fname in os.listdir(eval_dir):
        if fname.endswith(".json"):
            with open(os.path.join(eval_dir, fname), "r") as f:
                yield fname, json.load(f)

def main():
    # Load finetuned model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_DIR,
        max_seq_length=2048,
        dtype=None, # auto
        load_in_4bit=True,
    )

    # Patch forward to avoid num_items crash
    _orig_forward = model.forward
    def safe_forward(*args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("n_items", None)
        return _orig_forward(*args, **kwargs)
    model.forward = safe_forward

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    total = 0
    correct = 0

    for fname, data in load_eval_files(EVAL_DIR):
        print(f"\n[Evaluating {fname}]")
        for ex in data.get("test", []):
            inp = ex["input"]
            expected = ex["output"]

            # Prepare prompt
            prompt = format_pair(inp)

            inputs = tokenizer(prompt, return_tensors="pt", max_length=2048).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract model’s completion after "Output:\n"
            if "Output:" in decoded:
                pred_text = decoded.split("Output:", 1)[-1].strip()
            else:
                pred_text = decoded.strip()

            # crude parse: try to json-load, else fallback to string
            try:
                pred = json.loads(pred_text)
            except Exception:
                pred = pred_text

            # Compare with ground truth
            is_correct = (pred == expected)
            print(f"Input: {inp}\nExpected: {expected}\nPred: {pred}\nCorrect: {is_correct}\n{'-'*40}")

            total += 1
            if is_correct:
                correct += 1

    print(f"\nDone. Accuracy: {correct}/{total} = {correct/total:.2%}")

if __name__ == "__main__":
    main()
