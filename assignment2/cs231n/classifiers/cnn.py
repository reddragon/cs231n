import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    '''
    W1 = N * F * s * s
    b1 = F
    W2 = N * F *
    '''
    c, h, w = input_dim
    C = num_classes
    F = num_filters
    s = filter_size
    D = hidden_dim

    self.conv_param = {'stride': 1, 'pad': (s - 1) / 2}
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    # Output of the conv layer will be of size N * F * conv_dim1h * conv_dim1w
    conv_dim1h = 1 + (h + 2 * self.conv_param['pad'] - s) / self.conv_param['stride']
    conv_dim1w = 1 + (w + 2 * self.conv_param['pad'] - s) / self.conv_param['stride']

    # Output of the pool layer will be of size N * F * pool_dim1h * pool_dim1w
    pool_dim1h = 1 + (conv_dim1h - self.pool_param['pool_height']) / self.pool_param['stride']
    pool_dim1w = 1 + (conv_dim1w - self.pool_param['pool_width']) / self.pool_param['stride']

    #print 'F: ', F, 'dim1h: ', pool_dim1h, 'dim1w: ', pool_dim1w

    self.params['W1'] = np.random.normal(0, weight_scale, (F, c, s, s))
    self.params['b1'] = np.random.normal(0, weight_scale, F)

    # Output of the conv-relu-pool would be N * F * pool_dim1h * pool_dim1w
    # We need to resize it to be able to do a matrix multiplication.
    self.params['W2'] = np.random.normal(0, weight_scale, (F * pool_dim1h * pool_dim1w, D))
    self.params['b2'] = np.random.normal(0, weight_scale, D)

    self.params['W3'] = np.random.normal(0, weight_scale, (D, C))
    self.params['b3'] = np.random.normal(0, weight_scale, C)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    (conv1, c1_cache) = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
    #print 'Conv-Relu-Pool o/p is of shape: ', conv1.shape

    # Resize conv1 to be able to do matrix-mul.
    # Keep the original copy, since you dont want to lose the shape and
    # do another resize later.
    conv1r = conv1.copy()
    conv1r = conv1r.reshape(conv1.shape[0], conv1.shape[1] * conv1.shape[2] * conv1.shape[3])
    #print 'Reshaped conv1: ', conv1r.shape

    (affine1, a1_cache) = affine_relu_forward(conv1r, W2, b2)
    #print 'Size of affine1: ', affine1.shape

    (affine2, a2_cache) = affine_forward(affine1, W3, b3)
    #print 'Size of affine2: ', affine2.shape
    scores = affine2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss = 0
    grads = {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dxf = softmax_loss(scores, y)
    rc = 0.5
    loss += rc * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    # Computing the gradients
    # dl3i = differential input for layer 3. Forgive my naming.
    dl3i = dxf
    #print 'Starting with dl3i: ', dl3i.shape
    dl2i, grads['W3'], grads['b3'] = affine_backward(dl3i, a2_cache)

    #print 'Input for layer 2, dl2i: ', dl2i.shape
    dl1i, grads['W2'], grads['b2'] = affine_relu_backward(dl2i, a1_cache)

    #print 'Initial un-shaped input for layer 1, dl1i: ', dl1i.shape
    dl1i = dl1i.reshape(*conv1.shape)
    #print 'Shaped input for layer 1, dl1i: ', dl1i.shape
    dl0i, grads['W1'], grads['b1'] = conv_relu_pool_backward(dl1i, c1_cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
