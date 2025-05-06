# Download link obtained from https://github.com/stanfordnlp/GloVe
# If download fails, try:
#   + https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
#   + https://nlp.stanford.edu/data/wordvecs/glove.6B.zip
#   + https://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.6B.zip
curl --resolve -LO https://downloads.cs.stanford.edu/nlp/data/wordvecs/glove.6B.zip --output ./pretrained/encoders/glove/glove.6B.zip
unzip ./pretrained/encoders/glove/glove.6B.zip -d ./pretrained/encoders/glove/