"""Brings all inception models under one namespace."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
from convolutional_confidence.inception_resnet_v2_confidence_v0 import inception_resnet_v2_conf_v0 as resnet_conf_v0
from convolutional_confidence.inception_resnet_v2_confidence_v0 import arg_scope as resnet_conf_v0_arg_scope

from convolutional_confidence.inception_resnet_v2_confidence_v1_mul import inception_resnet_v2_conf_v1_mul as resnet_conf_v1_mul
from convolutional_confidence.inception_resnet_v2_confidence_v1_mul import arg_scope as resnet_conf_v1_mul_arg_scope

from convolutional_confidence.inception_resnet_v2_confidence_v2_mul import network_model as resnet_conf_v2_mul
from convolutional_confidence.inception_resnet_v2_confidence_v2_mul import arg_scope as resnet_conf_v2_mul_arg_scope

# pylint: enable=unused-import
