from .dcr import DistanceToClosestRecord
from .distinctn import DistinctN
from .perplexity import Perplexity
from .selfbleu import SelfBLEU
from .ttr import TypeTokenRatio
from .infogain import InformationGain, DeterminantalPointProcess, AverageSimilarityGain, NearestNeighborDissimilarity
from .similarity import (
    CosineSimilarity, ExponentialCosineSimilarity, NormalizedCosineSimilarity,
    InnerProductSimilarity, JaccardSimilarity, GeneralizedJaccardSimilarity,
    RBFKernelSimilarity, CosineDissimilarity, CDist,
)

del dcr
del distinctn
del perplexity
del selfbleu
del ttr
del infogain
del similarity
