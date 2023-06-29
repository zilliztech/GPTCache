import numpy as np


def softmax(x: list):
    x = np.array(x)
    assert len(x.shape) == 1, f"Expect to get a shape of (len,) but got {x.shape}, x value: {x}."
    max_val = x.max()
    e_x = np.exp(x - max_val)
    return e_x / e_x.sum()
