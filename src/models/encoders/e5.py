from sentence_transformers import SentenceTransformer
import torch
from torch import nn, Tensor


class E5Wrapper(nn.Module):
    def __init__(self, model, normalize: bool = True):
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.device = next(self.model.parameters()).device

    def forward(self, *args, **kwargs) -> Tensor:
        embeddings = self.model.encode(*args, **kwargs, normalize_embeddings=self.normalize)
        embeddings = torch.from_numpy(embeddings).to(self.device)
        return embeddings


if __name__ == "__main__":
    model = SentenceTransformer(
        model_name_or_path="intfloat/e5-base-v2",
        cache_folder="./pretrained/encoders/",
    )
    encoder = E5Wrapper(model)

    input_texts = [
        "query: how much protein should a female eat",
        "query: summit define",
        "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
        "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    ]
    embeddings = encoder(input_texts)
    print(embeddings.shape)
