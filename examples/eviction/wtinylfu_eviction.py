from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.manager.eviction import EvictionBase


def wtinylfu_basic_example():
    """
    Basic W-TinyLFU eviction example.

    Uses the default settings: 1% window, 20/80 probation/protected split,
    cost-aware admission enabled. The policy combines frequency-based
    admission filtering (TinyLFU) with cost-weighted eviction decisions,
    preferring to retain expensive-to-regenerate cache entries.
    """
    onnx = Onnx()
    data_manager = get_data_manager(
        cache_base=CacheBase("sqlite"),
        vector_base=VectorBase("faiss", dimension=onnx.dimension),
        eviction_base=EvictionBase(
            "wtinylfu",
            maxsize=200,
            clean_size=50,
        ),
    )

    cache = Cache()
    cache.init(data_manager=data_manager)
    question = "What is github?"
    answer = "Online platform for version control and code collaboration."
    embedding = onnx.to_embeddings(question)
    cache.import_data([question], [answer], [embedding])


def wtinylfu_custom_params_example():
    """
    W-TinyLFU with custom parameters.

    Tunable parameters:
    - window_pct: window cache as % of total capacity (default: 1.0)
    - probation_pct: probation segment as % of main cache (default: 20.0)
    - cost_aware: enable cost-weighted admission (default: True)
    - cms_width_multiplier: Count-Min Sketch width scaling (default: 1)
    - reset_multiplier: CMS aging interval as multiple of capacity (default: 10)
    """
    onnx = Onnx()
    data_manager = get_data_manager(
        cache_base=CacheBase("sqlite"),
        vector_base=VectorBase("faiss", dimension=onnx.dimension),
        eviction_base=EvictionBase(
            "wtinylfu",
            maxsize=500,
            clean_size=100,
            window_pct=2.0,
            probation_pct=25.0,
            cost_aware=True,
        ),
    )

    cache = Cache()
    cache.init(data_manager=data_manager)
    question = "Explain quantum computing"
    answer = "Quantum computing uses quantum bits (qubits) that can exist in superposition..."
    embedding = onnx.to_embeddings(question)
    cache.import_data([question], [answer], [embedding])


def wtinylfu_no_cost_example():
    """
    W-TinyLFU without cost awareness (pure frequency-based admission).

    When cost_aware=False, the admission decision uses only the TinyLFU
    frequency estimate, equivalent to Caffeine's default policy.
    """
    onnx = Onnx()
    data_manager = get_data_manager(
        cache_base=CacheBase("sqlite"),
        vector_base=VectorBase("faiss", dimension=onnx.dimension),
        eviction_base=EvictionBase(
            "wtinylfu",
            maxsize=200,
            clean_size=50,
            cost_aware=False,
        ),
    )

    cache = Cache()
    cache.init(data_manager=data_manager)
    question = "What is machine learning?"
    answer = "A subset of AI that enables systems to learn from data."
    embedding = onnx.to_embeddings(question)
    cache.import_data([question], [answer], [embedding])
