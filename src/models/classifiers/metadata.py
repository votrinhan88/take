class ClassifierMetadata:
    supported = ["logistic", "nbayes", "svm", "textcnn", "textrnn", "siamlog"]

    def __init__(self, model: str):
        self.model = self._validate_args("model", model)

    def _validate_args(self, arg, value):
        if arg == "model":
            if value not in self.supported:
                msg = f"Model {value} not supported. Supported: {self.supported}"
                raise ValueError(msg)
            return value

    def get_preset(self) -> dict:
        if self.model == "logistic":
            preset_metadata = {
                "abbrev": "logistic",
                "Class": "eval:LogisticRegression",
                "kwargs": {"loss_fn": "ce", "return_logits": True},
            }
            return preset_metadata

        elif self.model == "nbayes":
            preset_metadata = {
                "abbrev": "nbayes",
                "Class": "eval:NaiveBayes",
                "kwargs": {"epsilon": 1e-8},
            }
            return preset_metadata

        elif self.model == "svm":
            preset_metadata = {
                "abbrev": "svm",
                "Class": "eval:SupportVectorMachine",
                "kwargs": {"loss_fn": "hinge"},
            }
            return preset_metadata

        elif self.model == "textcnn":
            preset_metadata = {
                "abbrev": "textcnn",
                "Class": "eval:TextCNN",
                "kwargs": {
                    "num_channels": 100,
                    "kernel_sizes": [3, 4, 5],
                    "p_dropout": 0.5,
                    "mask_strategy": "trim_zero",
                    "return_logits": True,
                    "loss_fn": "ce",
                },
            }
            return preset_metadata

        elif self.model == "textrnn":
            preset_metadata = {
                "abbrev": "textrnn",
                "Class": "eval:TextRNN",
                "kwargs": {
                    "hidden_dim": 100,
                    "num_layers": 2,
                    "bidirectional": False,
                    "p_dropout": 0.5,
                    "mask_strategy": "trim_zero",
                    "return_logits": True,
                    "loss_fn": "ce",
                },
            }
            return preset_metadata

        elif self.model == "siamlog":
            preset_metadata = {
                "abbrev": "siamlog",
                "Class": "eval:SiameseLogistic",
                "kwargs": {"loss_fn": "ce", "return_logits": True},
            }
            return preset_metadata

        else:
            raise ValueError(f"Model {self.model} not supported.")
