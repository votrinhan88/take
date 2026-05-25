# Download link obtained from https://github.com/stanfordnlp/GloVe
# If download fails, try:
#   + https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
#   + https://nlp.stanford.edu/data/wordvecs/glove.6B.zip
#   + https://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.6B.zip
curl --resolve -LO https://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.6B.zip --output ./models/pretrained/encoders/glove/glove.6B.zip
unzip ./models/pretrained/encoders/glove/glove.6B.zip -d ./models/pretrained/encoders/glove/