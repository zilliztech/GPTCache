import datetime

from gptcache.manager.scalar_data.base import CacheData
from gptcache.similarity_evaluation import TimeEvaluation


def test_evaluation_time():
    eval = TimeEvaluation("distance", {}, time_range=2)
    assert eval.range() == (0.0, 4.0)

    similarity = eval.evaluation({}, {"search_result": (3.5, None)})
    assert similarity == 0.0

    similarity = eval.evaluation(
        {}, {"search_result": (3.5, None), "cache_data": CacheData("a", "b")}
    )
    assert similarity == 0.0

    similarity = eval.evaluation(
        {},
        {
            "search_result": (3.5, None),
            "cache_data": CacheData("a", "b", create_on=datetime.datetime(2022, 1, 1)),
        },
    )
    assert similarity == 0.0

    similarity = eval.evaluation(
        {},
        {
            "search_result": (3.5, None),
            "cache_data": CacheData("a", "b", create_on=datetime.datetime.now()),
        },
    )
    assert similarity == 0.5
