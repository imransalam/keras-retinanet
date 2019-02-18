"""
Copyright 2018 vidosits (https://github.com/vidosits/)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from keras.applications import nasnet
from keras.utils import get_file

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


allowed_backbones = {
    'nasnet_large': nasnet.NASNetLarge,
    'nasnet_mobile': nasnet.NASNetMobile
}


class NasNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return nasnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Download pre-trained weights for the specified backbone name.
        This name is in the format {backbone}_weights_tf_dim_ordering_tf_kernels_notop
        where backbone is the nasnet_ + large | mobile.
        For more info check the explanation from the keras densenet script itself:
            https://github.com/keras-team/keras/blob/master/keras/applications/nasnet.py
        """
        origin    = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/'
        file_name = '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'

        # load weights
        if keras.backend.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_first" format are not available.')

        weights_url = origin + file_name.format(self.backbone)
        return get_file(file_name.format(self.backbone), weights_url, cache_subdir='models')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones.keys()))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')


def nasnet_retinanet(num_classes, backbone='nasnet_large', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a nasnet backbone.
    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('densenet121', 'densenet169', 'densenet201')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
    Returns
        RetinaNet model with a NasNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input((None, None, 3))

    creator = allowed_backbones[backbone]
    model = creator(input_tensor=inputs, include_top=False, pooling=None, weights=None)

    
    layer_names = ['concatenate_1', 'concatenate_2', 'concatenate_3', 'concatenate_4']
    layer_outputs = [backbone.get_layer(name).output for name in layer_names]

    # create the nasnet backbone
    model = keras.models.Model(inputs=inputs, outputs=layer_outputs, name=model.name)

    # invoke modifier if given
    if modifier:
        model = modifier(model)

    # create the full model
    model = retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=model.outputs, **kwargs)

    return model