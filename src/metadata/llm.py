class LLMMetadata:
    supported = ["gemma3_270m", "gemma3_1b", "gpt2", "llama32_1b", "qwen3_600m", "albert"]
    supported_cls = ["albert"]
    supported_nli = ["albert"]
    CONTEXT_LENGTH = {
        "gemma3_270m": 131072,
        "gemma3_1b": 131072,
        "gpt2": 1024,
        "llama32_1b": 131072,
        "llama32_3b": 131072,
        "qwen3_600m": 32768,
        "albert": 512,
    }

    def __init__(self, model: str):
        self.model = self._validate_args("model", model)
        self.context_length = self.CONTEXT_LENGTH[self.model]

    def _validate_args(self, arg, value):
        if arg == "model":
            if value not in self.supported:
                msg = f"Model {value} not supported. Supported: {self.supported}"
                raise ValueError(msg)
            return value

    def get_preset_model(self) -> dict:
        if self.model == "gemma3_270m":
            preset_metadata = {
                "Class": "eval:AutoModelForCausalLM.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "google/gemma-3-270m",
                    "cache_dir": "./models/pretrained/llms/",
                    "device_map": "auto",
                },
                "set_eos_as_pad": False,
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
            }
            return preset_metadata

        elif self.model == "gemma3_1b":
            preset_metadata = {
                "Class": "eval:AutoModelForCausalLM.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "google/gemma-3-1b-pt",
                    "cache_dir": "./models/pretrained/llms/",
                    "device_map": "auto",
                },
                "set_eos_as_pad": False,
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
            }
            return preset_metadata

        elif self.model == "gpt2":
            preset_metadata = {
                "Class": "eval:AutoModelForCausalLM.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "gpt2",
                    "cache_dir": "./models/pretrained/llms/",
                    "device_map": "auto",
                },
                "set_eos_as_pad": True,
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["c_attn"],
                },
            }
            return preset_metadata

        elif self.model == "llama32_1b":
            preset_metadata = {
                "Class": "eval:AutoModelForCausalLM.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B",
                    "cache_dir": "./models/pretrained/llms/",
                    "device_map": "auto",
                },
                "set_eos_as_pad": True,
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
            }
            return preset_metadata

        elif self.model == "qwen3_600m":
            preset_metadata = {
                "Class": "eval:AutoModelForCausalLM.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "Qwen/Qwen3-0.6B",
                    "cache_dir": "./models/pretrained/llms/",
                    "device_map": "auto",
                },
                "set_eos_as_pad": False,
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
            }
            return preset_metadata

        elif self.model == "albert":
            preset_metadata = {
                "Class": "eval:AutoModelForSequenceClassification.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "albert-base-v2",
                    "cache_dir": "./models/pretrained/seqclf/",
                    "device_map": "auto",
                },
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "SEQ_CLS",
                    "target_modules": ["query", "key", "value", "dense"],
                    "modules_to_save": ["classifier"],
                },
            }
            return preset_metadata

        else:
            msg = f"Model {self.model} not supported. Supported: {self.supported}"
            raise ValueError(msg)

    def get_preset_tokenizer(self) -> dict:
        if self.model == "gemma3_270m":
            preset_metadata = {
                "Class": "eval:AutoTokenizer.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "google/gemma-3-270m",
                    "cache_dir": "./models/pretrained/llms/",
                    "padding_side": "right",
                },
                "set_eos_as_pad": False,
            }
            return preset_metadata

        elif self.model == "gemma3_1b":
            preset_metadata = {
                "Class": "eval:AutoTokenizer.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "google/gemma-3-1b-pt",
                    "cache_dir": "./models/pretrained/llms/",
                    "padding_side": "right",
                },
                "set_eos_as_pad": False,
            }
            return preset_metadata

        elif self.model == "gpt2":
            preset_metadata = {
                "Class": "eval:AutoTokenizer.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "gpt2",
                    "cache_dir": "./models/pretrained/llms/",
                    "padding_side": "right",
                },
                "set_eos_as_pad": True,
            }
            return preset_metadata

        elif self.model == "llama32_1b":
            preset_metadata = {
                "Class": "eval:AutoTokenizer.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B",
                    "cache_dir": "./models/pretrained/llms/",
                    "padding_side": "right",
                },
                "set_eos_as_pad": True,
            }
            return preset_metadata

        elif self.model == "qwen3_600m":
            preset_metadata = {
                "Class": "eval:AutoTokenizer.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "Qwen/Qwen3-0.6B",
                    "cache_dir": "./models/pretrained/llms/",
                    "padding_side": "right",
                },
                "set_eos_as_pad": False,
            }
            return preset_metadata
        
        elif self.model == "albert":
            preset_metadata = {
                "Class": "eval:AutoTokenizer.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "albert-base-v2",
                    "cache_dir": "./models/pretrained/seqclf/",
                },
            }
            return preset_metadata

        else:
            msg = f"Model {self.model} not supported. Supported: {self.supported}"
            raise ValueError(msg)
