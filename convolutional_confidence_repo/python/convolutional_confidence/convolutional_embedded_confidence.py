# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""Contains the convolutional layer classes and their functional aliases.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
# from tensorflow.python import framework

from tensorflow.python.ops import nn
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import initializers

from tensorflow.contrib.layers.python.layers import utils as contrib_utils
from tensorflow.contrib.layers.python.layers import convolution
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.training import moving_averages

from tensorflow.python.framework.tensor_shape import Dimension
from tensorflow.python.framework.tensor_shape import TensorShape

def _build_variable_getter(rename=None):
  """Build a model variable getter that respects scope getter and renames."""
  # VariableScope will nest the getters
  def layer_variable_getter(getter, *args, **kwargs):
    kwargs['rename'] = rename
    return _model_variable_getter(getter, *args, **kwargs)
  return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
  """Adds variable (or all its parts) to all collections with that name."""
  collections = contrib_utils.get_variable_collections(
      collections_set, collections_name) or []
  variables_list = [variable]
  if isinstance(variable, tf_variables.PartitionedVariable):
    variables_list = [v for v in variable]
  for collection in collections:
    for var in variables_list:
      if var not in ops.get_collection(collection):
        ops.add_to_collection(collection, var)


def _model_variable_getter(getter, name, shape=None, dtype=None,
                           initializer=None, regularizer=None, trainable=True,
                           collections=None, caching_device=None,
                           partitioner=None, rename=None, use_resource=None,
                           **_):
  """Getter that uses model_variable for compatibility with core layers."""
  short_name = name.split('/')[-1]
  if rename and short_name in rename:
    name_components = name.split('/')
    name_components[-1] = rename[short_name]
    name = '/'.join(name_components)
  return variables.model_variable(
      name, shape=shape, dtype=dtype, initializer=initializer,
      regularizer=regularizer, collections=collections, trainable=trainable,
      caching_device=caching_device, partitioner=partitioner,
      custom_getter=getter, use_resource=use_resource)


# def conv2d_wnorm(*args, **kwargs):
#     return conv2d(Conv2D_WeightNorm, *args, **kwargs)

@add_arg_scope
def conv2d_conf(*args, **kwargs):
    return conv2d(Conv2D_Confidence, *args, **kwargs)


# TODO(jbms): change `rate` parameter to `dilation_rate` for consistency with
# underlying op.
@add_arg_scope
def conv2d(
        layer_class,
        inputs,
        num_outputs,
        kernel,
        kernel_size,
        stride=1,
        padding='SAME',
        data_format=None,
        rate=1,
        activation_fn=nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
  """ Modifying `convolution` from tensorflow/contrib/layers/python/layers/layers.py.
  """
  if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
    raise ValueError('Invalid data_format: %r' % (data_format,))

  layer_variable_getter = _build_variable_getter(
      {'bias': 'biases', 'kernel': 'weights'})

  with variable_scope.variable_scope(
      scope, 'Conv', [inputs], reuse=reuse,
      custom_getter=layer_variable_getter) as sc:
    inputs = ops.convert_to_tensor(inputs)
    input_rank = inputs.get_shape().ndims

    if input_rank != 4:
      raise ValueError('Convolution not supported for input with rank',
                       input_rank)

    df = ('channels_first' if data_format and data_format.startswith('NC')
          else 'channels_last')
    layer = layer_class(filters=num_outputs,
                        kernel=kernel,
                        kernel_size=kernel_size,
                        strides=stride,
                        padding=padding,
                        data_format=df,
                        dilation_rate=rate,
                        activation=None,
                        use_bias=not normalizer_fn and biases_initializer,
                        kernel_initializer=weights_initializer,
                        bias_initializer=biases_initializer,
                        kernel_regularizer=weights_regularizer,
                        bias_regularizer=biases_regularizer,
                        activity_regularizer=None,
                        trainable=trainable,
                        name=sc.name,
                        dtype=inputs.dtype.base_dtype,
                        _scope=sc,
                        _reuse=reuse)
    outputs = layer.apply(inputs)

    # Add variables to collections.
    _add_variable_to_collections(layer.kernel, variables_collections, 'weights')
    if layer.use_bias:
      _add_variable_to_collections(layer.bias, variables_collections, 'biases')

    if normalizer_fn is not None:
      normalizer_params = normalizer_params or {}
      outputs = normalizer_fn(outputs, **normalizer_params)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return contrib_utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)


class Conv2D_Confidence(base.Layer):
  """ 
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/layers/convolutional.py#L32
  2D convolutional layer measuring "confidence".
      To be paired with a regular convolutional layer.
  """

  def __init__(self, filters,
               kernel,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               trainable=True,
               name=None,
               rank=2,
               **kwargs):
    super(Conv2D_Confidence, self).__init__(trainable=trainable,
                                name=name, **kwargs)
    self.rank = rank
    self.filters = filters  # number of filters
    self.kernel = kernel
    self.kernel_size = utils.normalize_tuple(kernel_size, rank, 'kernel_size')
    self.strides = utils.normalize_tuple(strides, rank, 'strides')
    self.padding = utils.normalize_padding(padding)
    self.data_format = utils.normalize_data_format(data_format)
    self.dilation_rate = utils.normalize_tuple(
        dilation_rate, rank, 'dilation_rate')
    self.activation = activation
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.activity_regularizer = activity_regularizer
    self.input_spec = base.InputSpec(ndim=self.rank + 2)

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = input_shape[channel_axis].value
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    # To-do: refactor out kernel parameters
    # self.kernel = self.add_variable(name='kernel',  # do share the kernel
    #                                 shape=kernel_shape,
    #                                 initializer=self.kernel_initializer,
    #                                 regularizer=self.kernel_regularizer,
    #                                 trainable=True,
    #                                 dtype=self.dtype)
    self.sum_kernel = tf.ones_like(self.kernel, name='sum_kernel')
    shape = TensorShape([Dimension(1), Dimension(1), Dimension(1), Dimension(1)])

    self.kernel_id = tf.ones(shape, name='sum_kernel')

    self.kernel_norm = tf.nn.l2_normalize(self.kernel, [0,1,2])
    if self.use_bias:
      self.bias = self.add_variable(name='bias',  # don't share bias with normal convolution
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    trainable=True,
                                    dtype=self.dtype)
    else:
      self.bias = None

    self.input_spec = base.InputSpec(ndim=self.rank + 2,
                                     axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    inputs2 = tf.square(inputs)
    input_norm = nn.convolution(  # sum of squares within kernel
        input=inputs2,
        filter=self.sum_kernel,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, self.rank + 2))
    input_norm = tf.sqrt(input_norm)
 
    outputs_1 = nn.convolution(
        input=inputs,  # normalized in next step
        filter=self.kernel_norm,
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=self.padding.upper(),
        data_format=utils.convert_data_format(self.data_format, self.rank + 2))

    outputs = tf.div(outputs_1, tf.add(input_norm, 1e-8))

    if self.bias is not None:
      if self.data_format == 'channels_first':
        if self.rank == 1:
          # nn.bias_add does not accept a 1D input tensor.
          bias = array_ops.reshape(self.bias, (1, self.filters, 1))
          outputs += bias
        if self.rank == 2:
          outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
        if self.rank == 3:
          # As of Mar 2017, direct addition is significantly slower than
          # bias_add when computing gradients. To use bias_add, we collapse Z
          # and Y into a single dimension to obtain a 4D input tensor.
          outputs_shape = outputs.shape.as_list()
          outputs_4d = array_ops.reshape(outputs,
                                         [outputs_shape[0], outputs_shape[1],
                                          outputs_shape[2] * outputs_shape[3],
                                          outputs_shape[4]])
          outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
          outputs = array_ops.reshape(outputs_4d, outputs_shape)
      else:
        outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def _compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      space = input_shape[1:-1]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      import pdb
      pdb.set_trace()
      return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                      [self.filters])
    else:
      space = input_shape[2:]
      new_space = []
      for i in range(len(space)):
        new_dim = utils.conv_output_length(
            space[i],
            self.kernel_size[i],
            padding=self.padding,
            stride=self.strides[i],
            dilation=self.dilation_rate[i])
        new_space.append(new_dim)
      import pdb
      pdb.set_trace()
      return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                      new_space)
