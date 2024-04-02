import numpy as np
def softmax(x: np.ndarray, axis=1) -> np.ndarray:
    temp = np.exp(x)
    return temp/np.sum(temp, axis=axis, keepdims=True)


def VJP(Z: np.ndarray, y_bar: np.ndarray) -> np.ndarray:
    y = softmax(Z)
    Z_bar = y * (y_bar - np.sum(y * y_bar, axis=1, keepdims=True))
    return Z_bar
