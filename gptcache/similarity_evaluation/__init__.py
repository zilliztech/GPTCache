from typing import Dict, Any

from gptcache.similarity_evaluation.similarity_evaluation import SimilarityEvaluation

__all__ = [
    "SimilarityEvaluation",
    "OnnxModelEvaluation",
    "NumpyNormEvaluation",
    "SearchDistanceEvaluation",
    "ExactMatchEvaluation",
    "KReciprocalEvaluation",
    "CohereRerankEvaluation",
    "SequenceMatchEvaluation",
    "TimeEvaluation",
    "SbertCrossencoderEvaluation"
]

from gptcache.utils.lazy_import import LazyImport

onnx = LazyImport("onnx", globals(), "gptcache.similarity_evaluation.onnx")
np = LazyImport("np", globals(), "gptcache.similarity_evaluation.np")
distance = LazyImport("simple", globals(), "gptcache.similarity_evaluation.distance")
exact_match = LazyImport(
    "exact_match", globals(), "gptcache.similarity_evaluation.exact_match"
)
kreciprocal = LazyImport(
    "kreciprocal", globals(), "gptcache.similarity_evaluation.kreciprocal"
)
cohere = LazyImport(
    "cohere", globals(), "gptcache.similarity_evaluation.cohere_rerank"
)
sequence_match = LazyImport(
    "sequence_match", globals(), "gptcache.similarity_evaluation.sequence_match"
)
time = LazyImport(
    "time", globals(), "gptcache.similarity_evaluation.time"
)

time = LazyImport(
    "time", globals(), "gptcache.similarity_evaluation.time"
)

sbert_crossencoder = LazyImport(
    "sbert_crossencoder", globals(), "gptcache.similarity_evaluation.sbert_crossencoder"
)

def OnnxModelEvaluation(model="GPTCache/albert-duplicate-onnx"):
    return onnx.OnnxModelEvaluation(model)


def NumpyNormEvaluation(enable_normal: bool = False, **kwargs):
    return np.NumpyNormEvaluation(enable_normal, **kwargs)


def SearchDistanceEvaluation(max_distance=4.0, positive=False):
    return distance.SearchDistanceEvaluation(max_distance, positive)


def ExactMatchEvaluation():
    return exact_match.ExactMatchEvaluation()


def KReciprocalEvaluation(vectordb, top_k=3, max_distance=4.0, positive=False):
    return kreciprocal.KReciprocalEvaluation(vectordb, top_k, max_distance, positive)


def CohereRerankEvaluation(model: str = "rerank-english-v2.0", api_key: str = None):
    return cohere.CohereRerank(model=model, api_key=api_key)


def SequenceMatchEvaluation(weights, embedding_extractor, embedding_config: Dict[str, Any] = None):
    return sequence_match.SequenceMatchEvaluation(weights, embedding_extractor, embedding_config=embedding_config)


def TimeEvaluation(evaluation: str, evaluation_config: Dict[str, Any], time_range: float = 86400.0):
    return time.TimeEvaluation(evaluation, evaluation_config=evaluation_config, time_range=time_range)

def SbertCrossencoderEvaluation(model: str = "cross-encoder/quora-distilroberta-base"):
    return sbert_crossencoder.SbertCrossencoderEvaluation(model)
