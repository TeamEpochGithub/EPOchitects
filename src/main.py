import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils.json_loader import JsonDataLoader


def main():
    model_name = "nvidia/Mistral-NeMo-Minitron-8B-Base"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        max_seq_length=2048,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    # ---- Patch forward to strip bad kwargs ----
    _orig_forward = model.forward
    def safe_forward(*args, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("n_items", None)
        return _orig_forward(*args, **kwargs)
    model.forward = safe_forward
    # -------------------------------------------

    # Load data and turn into Dataset
    pairs = JsonDataLoader("data/training").get_all_train_pairs()
    dataset = Dataset.from_dict({
        "text": [f"Input:\n{inp}\nOutput:\n{out}" for inp, out in pairs]
    })

    tokenized = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, max_length=2048),
        batched=True,
        remove_columns=["text"],
    )

    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        save_strategy="no",
        remove_unused_columns=True,  # strips stray dataset columns
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained("finetuned_model")


if __name__ == "__main__":
    main()
