import torch
from unsloth import FastLanguageModel


model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,
    max_seq_length=2048
)

FastLanguageModel.for_inference(model)

outputs = model.generate(
    prompt,
    streamer=streamer,
    temperature=0.7,
    max_new_tokens=200
)

prompt = "You are a helpful assistant. Hello! How are you?"

def output(prompt = prompt):

    streamer = TextStreamer(tokenizer)

    outputs = model.generate(
        prompt,
        streamer=streamer,
        temperature=0.7,
        max_new_tokens=200
    )   

    print("Model says:", outputs)


def main():
    output()


if __name__ == "__main__":
    main()
