import re
from typing import overload

from src.metadata import DatasetMetadata


class TextTemplate:
    def __init__(self, dataset: str, default_task: str | None = None):
        super().__init__()
        self._validate_args(default_task=default_task)
        self.dataset = dataset
        self.default_task = default_task

        self.classes = DatasetMetadata(dataset=dataset).classes
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        class_pattern = "|".join(re.escape(k) for k in self.classes)

        # fmt: off
        self.patterns = {
            "generation": r"^Task: Generation\nLabel: (" + class_pattern + r")\n(?i:text):\s*(.*)$",
            "inference": r"^Task: Inference\nText:\s*(.*)\n(?i:label):\s*(" + class_pattern + r")$",
        }
        # fmt: on
        self.patterns = {k: re.compile(v) for k, v in self.patterns.items()}

    def _validate_args(self, default_task: str | None = None):
        if default_task not in ["generation", "inference", None]:
            raise ValueError("`default_task` must be one of ['generation', 'inference', None]")

    def _infer_task(self, task: str | None) -> str:
        if task is None:
            if self.default_task is None:
                raise ValueError(f"{self} does not have `default_task`, `task` must be specified.")
            task = self.default_task
        return task

    def validate_single(self, string: str, task: str) -> bool:
        matched = re.match(pattern=self.patterns[task], string=string)
        return matched is not None

    @overload
    def validate(self, string: str, task: str | None = None) -> bool: ...
    @overload
    def validate(self, string: list[str], task: str | None = None) -> list[bool]: ...
    def validate(self, string: str | list[str], task: str | None = None) -> bool | list[bool]:
        task = self._infer_task(task)
        if isinstance(string, str):
            return self.validate_single(string=string, task=task)
        else:
            return [self.validate_single(string=s, task=task) for s in string]

    def parse_task(self, string: str) -> str | None:
        for t in self.patterns.keys():
            if f"Task: {t.title()}" in string:
                return t

    def parse_single(
        self, string: str, task: str | None = None, label_as_int: bool = False, strict: bool = True
    ) -> dict:
        if task is None:
            task = self.parse_task(string)

        matched = re.match(pattern=self.patterns[task], string=string)
        if not matched:
            if strict:
                raise ValueError("Text does not match the template")
            else:
                return {"text": None, "label": None}

        if task == "generation":
            label, text = matched.group(1), matched.group(2)
        else:
            label, text = matched.group(2), matched.group(1)

        if label_as_int:
            label = self.class_index[label]
        return {"text": text, "label": label}

    def parse(
        self,
        strings: str | list[str],
        task: str | None = None,
        label_as_int: bool = False,
        strict: bool = True,
    ) -> dict:
        if isinstance(strings, str):
            return self.parse_single(
                string=strings, task=task, label_as_int=label_as_int, strict=strict
            )

        out = {"text": [], "label": []}
        for s in strings:
            parsed = self.parse_single(
                string=s, task=task, label_as_int=label_as_int, strict=strict
            )
            out["text"].append(parsed["text"])
            out["label"].append(parsed["label"])
        return out

    def __call__(
        self,
        text: str | None = None,
        label: str | int | None = None,
        task: str | None = None,
    ) -> str:
        task = self._infer_task(task)

        if task == "generation":
            return self.template_generation(text=text, label=label)
        else:
            return self.template_inference(text=text, label=label)

    def template_generation(self, text: str | None = None, label: str | int | None = None) -> str:
        string = f"Task: Generation\nLabel:"
        if label is not None:
            if isinstance(label, int):
                label = self.classes[label]
            string += f" {label}\nText:"
        else:
            return string

        if text is not None:
            string += f"{text}"
        return string

    def template_inference(self, text: str | None = None, label: str | int | None = None) -> str:
        string = f"Task: Inference\nText:"
        if text is not None:
            string += f" {text}\nLabel:"
        if label is not None:
            if isinstance(label, int):
                label = self.classes[label]
            string += f" {label}"
        return string

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(dataset={self.dataset}, default_task={self.default_task})"
        )


if __name__ == "__main__":

    def test_TextTemplate(verbose: bool = True):
        def print_if_verbose(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        samples = {
            "agnews": [
                {"label": "World", "text": "The stock market crashed today."},
                {"label": "Sports", "text": "The local team won their game."},
                {"label": "Business", "text": "A new startup disrupted the industry."},
                {"label": "Sci/Tech", "text": "A new smartphone was released."},
            ],
            "imdb": [
                {"label": "neg", "text": "The worst film I've ever seen."},
                {"label": "pos", "text": "I loved this movie!"},
            ],
        }

        print_if_verbose(" test_TextValidator ".center(88, "=") + "\n")
        for dataset in samples.keys():
            template = TextTemplate(dataset=dataset)

            for task in ["generation", "inference"]:
                print_if_verbose(f"Testing dataset {dataset} - task {task}".center(88, "="))

                strings = []
                for sample in samples[dataset]:
                    if task == "generation":
                        string = (
                            template.template_generation(label=sample["label"])
                            + " "
                            + sample["text"]
                        )
                    else:
                        string = (
                            template.template_inference(text=sample["text"]) + " " + sample["label"]
                        )
                    strings.append(string)
                    print_if_verbose(string + "\n")

                template.validate(string=strings[0], task=task)
                batch_parsed = template.parse(strings=strings, task=task)

                for i in range(len(samples[dataset])):
                    text_gt = samples[dataset][i]["text"]
                    label_gt = samples[dataset][i]["label"]
                    text_parsed = batch_parsed["text"][i]
                    label_parsed = batch_parsed["label"][i]
                    assert text_gt == text_parsed, f"Text mismatch: '{text_gt}' != '{text_parsed}'"
                    assert label_gt == label_parsed, (
                        f"Label mismatch: '{label_gt}' != '{label_parsed}'"
                    )
                    print_if_verbose(f"parse: (label: '{label_gt}', text: '{text_gt}')")

            print_if_verbose("\n\n")

        print(" test_TextValidator: PASSED ".center(88, "="))

    test_TextTemplate(verbose=False)
