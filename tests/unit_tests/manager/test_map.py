import os

from gptcache.manager.data_manager import MapDataManager

data_map_path = "data_map.txt"


def test_map():
    if os.path.isfile(data_map_path):
        os.remove(data_map_path)

    data_manager = MapDataManager(data_map_path, 3)
    a = "a"
    for i in range(4):
        data_manager.save(chr(ord(a) + i), str(i), chr(ord(a) + i))
    assert len(data_manager.search("a")) == 0
    question, answer, emb, _ = data_manager.search("b")[0]
    assert question == "b", question
    assert answer == "1", answer
    assert emb == "b", emb
    data_manager.close()
