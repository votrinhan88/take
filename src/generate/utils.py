from copy import deepcopy
from typing import Callable

from transformers import GenerationConfig


def generate(
    model,
    tokenizer,
    num_samples: int,
    prompt: str,
    genconfig: GenerationConfig,
    ensure_bos_token: bool = True,
    validation_fn: Callable[[str], bool] | None = None,
    max_attempts: int | None = None,
    device: str = "auto",
) -> list[str]:
    genconfig = deepcopy(genconfig)
    genconfig.num_return_sequences = min(num_samples, genconfig.num_return_sequences)
    genconfig.eos_token_id = tokenizer.eos_token_id
    genconfig.pad_token_id = tokenizer.pad_token_id

    if ensure_bos_token and tokenizer.bos_token and not prompt.startswith(tokenizer.bos_token):
        prompt = tokenizer.bos_token + prompt
    if device == "auto":
        device = model.device

    # model.generate() expects Tensor inputs
    prompt_tok = tokenizer(text=prompt, add_special_tokens=False, return_tensors="pt")
    attempts = 0
    texts_out = []
    while len(texts_out) < num_samples:
        attempts += 1
        if max_attempts is not None and attempts > max_attempts:
            raise RuntimeError(
                f"Exceeded max_attempts={max_attempts}. Generated {len(texts_out)}/{num_samples}."
            )

        outputs = model.generate(
            input_ids=prompt_tok["input_ids"].to(device),
            attention_mask=prompt_tok["attention_mask"].to(device),
            generation_config=genconfig,
        )
        generated_text = tokenizer.batch_decode(
            sequences=outputs,
            skip_special_tokens=True,
        )
        if validation_fn is not None:
            generated_text = [t for t in generated_text if validation_fn(t)]
        texts_out.extend(generated_text)

    texts_out = texts_out[:num_samples]
    return texts_out
