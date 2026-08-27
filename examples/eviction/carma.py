"""Configure GPTCache with the opt-in CARMA eviction policy."""

from gptcache.manager import manager_factory


def create_carma_manager(data_dir="./carma-cache", dimension=768):
    """Create a SQLite/FAISS manager with online semantic-aware eviction."""
    return manager_factory(
        "sqlite,faiss",
        data_dir=data_dir,
        vector_params={"dimension": dimension, "top_k": 1},
        eviction_params={
            "eviction": "CARMA",
            "max_size": 100,
            "clean_size": 1,
            "policy_params": {
                "topic_threshold": 0.70,
                "cell_threshold": 0.97,
                "demand_half_life": 500,
                "quota_strength": 1.0,
                "ghost_support_threshold": 1.5,
                "admission_margin": 1.05,
            },
        },
    )


if __name__ == "__main__":
    manager = create_carma_manager(dimension=2)
    print(manager.eviction_base.policy)
    manager.close()
