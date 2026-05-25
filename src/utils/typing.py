from typing import Protocol


class Tokenizer(Protocol):
    bos_token: str
    eos_token: str
    pad_token: str
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    vocab_size: int

    def tokenize(self, text: str) -> list[str]: ...
    def decode(self, token_ids, skip_special_tokens: bool = ...) -> str: ...
    def batch_decode(self, sequences, skip_special_tokens: bool = ...) -> list[str]: ...

    def __call__(
        self,
        text,
        *,
        truncation: bool = ...,
        padding: bool | str = ...,
        return_tensors: str = ...,
        add_special_tokens: bool = ...,
        max_length: int = ...,
    ) -> dict: ...
