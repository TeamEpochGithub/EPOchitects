import torch
from unsloth import FastLanguageModel

def gpu_memory_report():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU Memory Free: {free/1e9:.2f} GB / Total: {total/1e9:.2f} GB")

def try_load_model(model_name, max_seq_length=2048, load_in_4bit=True):
    try:
        print(f"Trying model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        del model
        del tokenizer
        torch.cuda.empty_cache()
        print(f"SUCCESS: {model_name} fits in GPU memory.")
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"OOM: {model_name} too large for this GPU.")
            torch.cuda.empty_cache()
            return False
        else:
            raise e

def main():
    gpu_memory_report()

    # Candidate models, ordered small → large
    candidate_models = [
        "unsloth/Meta-Llama-3-1B-bnb-4bit",
        "unsloth/Meta-Llama-3-3B-bnb-4bit",
        "unsloth/Meta-Llama-3-7B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-13B-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    ]

    largest_model = None
    for model_name in candidate_models:
        if try_load_model(model_name):
            largest_model = model_name
        else:
            break

    if largest_model:
        print(f"\nLargest model that fits: {largest_model}")
    else:
        print("\nNo listed models fit in GPU memory.")

if __name__ == "__main__":
    main()
