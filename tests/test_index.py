import numpy as np

from pricematch.index.vector_index import build_index


def test_index_search():
    idx = build_index(3)
    idx.add(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))
    scores, ids = idx.search(np.array([[0.9, 0.1, 0.0]]), k=2)
    assert ids[0][0] == 0
    assert scores[0][0] > scores[0][1]
