import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
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
  
  for n in xrange(X.shape[0]):
    Z=X[n,:]@W
    Z-=np.max(Z)
    loss-=Z[y[n]]-np.log(np.sum(np.exp(Z)))
    for c in xrange(W.shape[1]):
         softmax = np.exp(Z[c])/np.sum(np.exp(Z))
         if c == y[n]:
             dW[:,c] += (-1 + softmax)*X[n] 
         else: 
             dW[:,c] += softmax*X[n]
  dW = dW/X.shape[0] + reg*W
  loss = loss/X.shape[0] + reg*np.sum(W*W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
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
  Z=X@W
  Z-=np.max(Z)
  softmax=np.exp(Z.T)/(np.sum(np.exp(Z),axis=1)+1e-9)
  loss-=np.sum(np.log(softmax.T[range(X.shape[0]), list(y)]))
  loss=loss/X.shape[0] + reg*np.sum(W*W)

  dW=softmax.T
  dW[xrange(X.shape[0]), list(y)]-=1
  dW=X.T@dW
  dW=dW/X.shape[0] + reg* W 
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

