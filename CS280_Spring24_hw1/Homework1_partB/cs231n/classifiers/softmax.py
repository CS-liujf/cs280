from builtins import range
import numpy as np
from random import shuffle

from scipy.special import softmax

def softmax(x: np.ndarray, axis=1) -> np.ndarray:
    x_max = np.amax(x, axis=axis, keepdims=True)
    temp = np.exp(x-x_max)
    return temp/np.sum(temp, axis=axis, keepdims=True)


def softmax_loss_naive(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    scores = X.dot(W)
    probabilities = softmax(scores)
    for i in range(N):
        probability = probabilities[i]
        loss -= np.log(probability[y[i]])
        for d in range(D):
            for c in range(C):
                if(c==y[i]):
                    dW[d, c] += 1.0 * X[i, d] * \
                        (probability[c] -1)
                else:
                    dW[d, c] += 1.0 * X[i, d] * probability[c]
                    
    loss = loss/N + reg * np.sum(W * W)
    dW = dW/N + 2*reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float) -> tuple[float, np.ndarray]:
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    temp: np.ndarray = np.dot(X, W)  # shape (N, C)
    output = softmax(temp, axis=1)
    temp = output[np.arange(N), y]  # shape (N, )
    cross_entropy_loss: float = np.sum(-np.log(temp))/N
    regularization = reg*np.sum(W**2)
    loss = cross_entropy_loss+regularization

    labels_onehot = np.zeros(output.shape)
    labels_onehot[np.arange(N), y] = 1.0
    dW: np.ndarray = np.dot(X.T, output-labels_onehot)/N+2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
