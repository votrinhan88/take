class ClassifierMetadata:
    supported = ["logistic", "nbayes", "svm", "textcnn", "textrnn", "siamlog"]
    supported_cls = ["logistic", "nbayes", "svm", "textcnn", "textrnn"]
    supported_nli = ["siamlog"]

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
            return {
                "abbrev": "logistic",
                "task": "cls",
                "Class": "eval:LogisticRegression",
                "kwargs": {"loss_fn": "ce", "return_logits": True},
            }

        elif self.model == "nbayes":
            return {
                "abbrev": "nbayes",
                "task": "cls",
                "Class": "eval:NaiveBayes",
                "kwargs": {"epsilon": 1e-8},
            }

        elif self.model == "svm":
            return {
                "abbrev": "svm",
                "task": "cls",
                "Class": "eval:SupportVectorMachine",
                "kwargs": {"loss_fn": "hinge"},
            }

        elif self.model == "textcnn":
            return {
                "abbrev": "textcnn",
                "task": "cls",
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

        elif self.model == "textrnn":
            return {
                "abbrev": "textrnn",
                "task": "cls",
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

        elif self.model == "siamlog":
            return {
                "abbrev": "siamlog",
                "task": "nli",
                "Class": "eval:SiameseLogistic",
                "kwargs": {"loss_fn": "ce", "return_logits": True},
            }

        else:
            raise ValueError(f"Model {self.model} not supported.")
