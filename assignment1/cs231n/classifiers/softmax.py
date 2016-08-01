import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_dims = X.shape[1]
  num_classes = W.shape[1]
  for i in xrange(num_train):
      scores = X[i].dot(W)
      exp_scores = np.exp(scores)
      exp_sum = np.sum(exp_scores)
      loss += -np.log(exp_scores[y[i]] / exp_sum)

      for j in xrange(num_classes):
          if y[i] == j:
              dW[:, j] += X[i] * ((exp_scores[y[i]] - exp_sum) / exp_sum)
          else:
              dW[:, j] += X[i] * (exp_scores[j] / exp_sum)

  loss /= num_train
  dW /= num_train
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
  num_train = X.shape[0]
  num_dims = X.shape[1]
  num_classes = W.shape[1]

  scores = X.dot(W)
  scores_exp = np.exp(scores)
  scores_exp_sum = np.sum(scores_exp, axis = 1)
  scores_exp_minus_sum = scores_exp.copy() - scores_exp_sum[:, None]
  correct_class_exp = scores_exp[np.arange(num_train), y]
  loss = np.sum(-np.log(correct_class_exp / scores_exp_sum)) / num_train

  # Set 1.0 wherever for the correct classes.
  correct_class_flag = np.zeros([num_train, num_classes])
  incorrect_class_flag = np.ones([num_train, num_classes])
  correct_class_flag[np.arange(correct_class_flag.shape[0]), y] = 1.0
  incorrect_class_flag[np.arange(incorrect_class_flag.shape[0]), y] = 0.0

  # This matrix would have the weights to multiply X[i] with.
  coeff = np.zeros_like(correct_class_flag)
  coeff += np.multiply(incorrect_class_flag, (scores_exp))
  coeff += np.multiply(correct_class_flag, (scores_exp_minus_sum))
  coeff /= scores_exp_sum[:, None]
  dW = X.T.dot(coeff)
  dW /= num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
