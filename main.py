from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
TARGET_WORDS = 200


def build_prompt() -> str:
    return (
        "Write a sincere professional apology email in about 200 words. "
        "Context: I missed an important client deadline due to poor planning. "
        "The email should acknowledge responsibility, explain briefly without excuses, "
        "and include a clear corrective action plan and request to rebuild trust."
    )


def main() -> None:
    prompt = build_prompt()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if device == "cpu":
        # Keep one core free for system responsiveness on CPU inference.
        cpu_threads = max(1, (torch.get_num_threads() or 1) - 1)
        torch.set_num_threads(cpu_threads)
        print(f"Using CPU threads: {cpu_threads}")

    messages = [{"role": "user", "content": prompt}]

    # Tokenize the chat-formatted prompt and report token count.
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    token_count = model_inputs["input_ids"].shape[1]
    print(f"Prompt token count: {token_count}")

    # 200 words is roughly 220-260 tokens for most English prose.
    max_new_tokens = 240 if device == "cpu" else 320
    start = time.perf_counter()
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed_s = time.perf_counter() - start

    new_tokens = generated_ids[0, token_count:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    generated_tokens = new_tokens.shape[0]
    tps = generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
    print(
        f"Generated tokens: {generated_tokens} in {elapsed_s:.1f}s "
        f"({tps:.2f} tokens/sec, target words: {TARGET_WORDS})"
    )
    print("\nGenerated output:\n")
    print(output_text)


if __name__ == "__main__":
    main()
