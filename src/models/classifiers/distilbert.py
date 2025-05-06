from torch import nn, Tensor
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    DistilBertConfig,
)


class DistilBertClassifier(nn.Module):
    MODEL_NAME = "distilbert-base-uncased"

    def __init__(self, num_classes: int, return_logits: bool = False):
        super(DistilBertClassifier, self).__init__()
        self.num_classes = num_classes
        self.return_logits = return_logits

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_NAME, cache_dir="./pretrained/"
        )
        self.model_config = DistilBertConfig.from_pretrained(
            pretrained_model_name_or_path=self.MODEL_NAME,
            num_labels=self.num_classes,
            cache_dir="./pretrained/",
        )
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            config=self.model_config,
            cache_dir="./pretrained/",
        )
        if not self.return_logits:
            if self.num_classes == 1:
                self.act = nn.Sigmoid()
            elif self.num_classes > 1:
                self.act = nn.Softmax(dim=1)

    def forward(self, inputs):
        tokens = self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        )
        tokens = {
            k: v.to(self.device) for k, v in tokens.items() if isinstance(v, Tensor)
        }
        outputs = self.model(**tokens)
        x = outputs.logits
        if not self.return_logits:
            x = self.act(x)
        return x

    @property
    def device(self):
        return next(self.model.parameters()).device
