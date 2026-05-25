class EncoderMetadata:
    supported = ["e5", "glove", "jina_nano", "jina_small", "minilm", "tfidf"]

    def __init__(self, model: str):
        self.model = self._validate_args("model", model)

    def _validate_args(self, arg, value):
        if arg == "model":
            if value not in self.supported:
                msg = f"Model {value} not supported. Supported: {self.supported}."
                raise ValueError(msg)
            return value

    def get_preset(self) -> dict:
        if self.model == "e5":
            preset_metadata = {
                "abbrev": "e5",
                "Class": "eval:SentenceTransformer",
                "kwargs": {
                    "model_name_or_path": "intfloat/e5-base-v2",
                    "cache_folder": "./models/pretrained/encoders/",
                },
                "wrap": {"Class": "eval:E5Wrapper", "kwargs": {}},
                "embed_dim": 768,
            }

        elif self.model == "glove":
            preset_metadata = {
                "abbrev": "glove",
                "Class": "eval:GloVeEncoder",
                "kwargs": {"embed_dim": 100, "frozen": True, "embed_level": "token"},
                "embed_dim": 100,
            }

        elif self.model == "jina_nano":
            preset_metadata = {
                "abbrev": "jina_nano",
                "Class": "eval:AutoModel.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "jinaai/jina-embeddings-v5-text-nano",
                    "cache_folder": "./models/pretrained/encoders/",
                    "trust_remote_code": True,
                    "dtype": "eval:torch.bfloat16",
                },
                "wrap": {"Class": "eval:JinaWrapper", "kwargs": {}},
                "embed_dim": 768,
            }

        elif self.model == "jina_small":
            preset_metadata = {
                "abbrev": "jina_small",
                "Class": "eval:AutoModel.from_pretrained",
                "kwargs": {
                    "pretrained_model_name_or_path": "jinaai/jina-embeddings-v5-text-small",
                    "cache_folder": "./models/pretrained/encoders/",
                    "trust_remote_code": True,
                    "dtype": "eval:torch.bfloat16",
                },
                "wrap": {"Class": "eval:JinaWrapper", "kwargs": {}},
                "embed_dim": 1024,
            }

        elif self.model == "minilm":
            preset_metadata = {
                "abbrev": "minilm",
                "Class": "eval:SentenceTransformer",
                "kwargs": {
                    "model_name_or_path": "all-MiniLM-L6-v2",
                    "cache_folder": "./models/pretrained/encoders/",
                },
                "wrap": {"Class": "eval:MiniLMWrapper", "kwargs": {}},
                "embed_dim": 384,
            }

        elif self.model == "tfidf":
            preset_metadata = {
                "abbrev": "tfidf",
                "Class": "eval:Tfidf",
                "kwargs": {"embed_dim": 3072, "sparse": True},
                "embed_dim": 3072,
            }

        else:
            raise ValueError(f"Model {self.model} not supported. Supported: {self.supported}.")

        return preset_metadata
