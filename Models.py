from Utilities import *

###############################################################################
#                                   FusionNet 2D
###############################################################################
@layer_register(log_shape=True)
def residual(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        inputs = x
        x = tf.pad(x, name='pad1', mode='REFLECT', paddings=[[0,0], 
                                                             [1*(kernel_shape//2),1*(kernel_shape//2)], 
                                                             [1*(kernel_shape//2),1*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv2D('conv1', x, chan, padding='VALID', dilation_rate=1)
        x = tf.pad(x, name='pad2', mode='REFLECT', paddings=[[0,0], 
                                                             [2*(kernel_shape//2),2*(kernel_shape//2)], 
                                                             [2*(kernel_shape//2),2*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv2D('conv2', x, chan, padding='VALID', dilation_rate=2)
        x = tf.pad(x, name='pad3', mode='REFLECT', paddings=[[0,0], 
                                                             [4*(kernel_shape//2),4*(kernel_shape//2)], 
                                                             [4*(kernel_shape//2),4*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv2D('conv3', x, chan, padding='VALID', dilation_rate=4, activation=tf.identity)             
        # x = tf.pad(x, name='pad4', mode='REFLECT', paddings=[[0,0], [8*(kernel_shape//2),8*(kernel_shape//2)], [8*(kernel_shape//2),8*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv4', x, chan, padding='VALID', dilation_rate=8) 
        x = InstanceNorm('inorm', x) + inputs
        return x


@layer_register(log_shape=True)
def residual_enc(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        x = tf.pad(x, name='pad_i', mode='REFLECT', paddings=[[0,0], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [0,0]])
        x = Conv2D('enc_i', x, chan, stride=2) 
        x = residual('enc_r', x, chan, first=True)
        x = tf.pad(x, name='pad_o', mode='REFLECT', paddings=[[0,0], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [0,0]])
        x = Conv2D('enc_o', x, chan, stride=1) #, activation=tf.identity) 
        #x = InstanceNorm('enc_n', x)
        return x


@layer_register(log_shape=True)
def residual_dec(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        x  = Deconv2D('dec_i', x, chan, stride=1) 
        x  = residual('dec_r', x, chan, first=True)
        x = Deconv2D('dec_o', x, chan, stride=2, activation=tf.identity) 
        # x1 = Deconv2D('dec_o', x, chan, stride=2, activation=tf.identity) 
        # x2 = BilinearUpSample('upsample', x, 2)
        # x  = InstanceNorm('dec_n', (x1+x2)/2.0)
        x  = InstanceNorm('dec_n', x)
        return x

###############################################################################
def arch_fusionnet_encoder_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e0 = residual_enc('e0', img, nb_filters*1)
            e0 = Dropout('f0', e0, 0.5)
            e1 = residual_enc('e1',  e0, nb_filters*2)
            e1 = Dropout('f1', e1, 0.5)
            e2 = residual_enc('e2',  e1, nb_filters*4)
            e2 = Dropout('f2', e2, 0.5)
            e3 = residual_enc('e3',  e2, nb_filters*8)
            e3 = Dropout('f3', e3, 0.5)
            return e3, [e2, e1, e0]
           

def arch_fusionnet_decoder_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e2, e1, e0 = feats 
            d4 = img 
            # d4 = Dropout('r4', d4, 0.5)

            d3 = residual_dec('d3', d4, nb_filters*4)
            # d3 = Dropout('r3', d3, 0.5)
            d3 = d3+e2 if e2 is not None else d3 
            
            d2 = residual_dec('d2', d3, nb_filters*2)
            # d2 = Dropout('r2', d2, 0.5)
            d2 = d2+e1 if e1 is not None else d2 
            
            d1 = residual_dec('d1', d2, nb_filters*1)
            # d1 = Dropout('r1', d1, 0.5)
            d1 = d1+e0 if e0 is not None else d1  
            
            d0 = residual_dec('d0', d1, nb_filters*1) 
            # d0 = Dropout('r0', d0, 0.5)
            
            dp = tf.pad( d0, name='pad_o', mode='REFLECT', paddings=[[0,0], 
                                                                     [7//2,7//2], 
                                                                     [7//2,7//2], 
                                                                     [0,0]])
            dd = Conv2D('convlast', dp, last_dim, kernel_shape=7, stride=1, padding='VALID', nl=nl, use_bias=True) 
            return dd, [d1, d2, d3]
###############################################################################
def arch_fusionnet_translator_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    enc, feat_enc = arch_fusionnet_encoder_2d(img, feats, last_dim, nl, nb_filters)
    dec, feat_dec = arch_fusionnet_decoder_2d(enc, feat_enc, last_dim, nl, nb_filters)
    return dec

def arch_fusionnet_classifier_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    with argscope(Conv2D, activation=INLReLU, kernel_size=4, strides=2):
        lin = (LinearWrap(img)
                 .Conv2D('conv0', nb_filters, activation=tf.nn.leaky_relu)
                 .Conv2D('conv1', nb_filters * 2)
                 .Conv2D('conv2', nb_filters * 4)
                 .Conv2D('conv3', nb_filters * 8, strides=1)
                 .Conv2D('conv4', 1, strides=1, activation=tf.identity, use_bias=True)())
        return lin

###############################################################################











###############################################################################
#                                   Tiramisu 2D
# https://github.com/0bserver07/One-Hundred-Layers-Tiramisu/blob/master/model-dynamic.py
###############################################################################
# from __future__ import absolute_import
# from __future__ import print_function
# import os


import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, Conv2DTranspose

from keras import backend as K

import cv2
import numpy as np
import json

K.set_image_dim_ordering('tf')

# weight_decay = 0.0001
from keras.regularizers import l2
 
class Tiramisu():
    def __init__(self, last_dim=1, nl=INLReLU, nb_filters=32):
        self.last_dim   =   last_dim
        self.nl         =   nl
        self.nb_filters =   nb_filters

        # self.create()
        

        
    def DenseBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            # model.add(BatchNormalization(mode=0, #axis=1,
            #                              gamma_regularizer=l2(0.0001),
            #                              beta_regularizer=l2(0.0001)))
            model.add(Activation('relu'))
            model.add(Conv2D(filters,   kernel_size=(3, 3), padding='same',
                                        kernel_initializer="he_uniform",
                                        data_format='channels_last'))
            model.add(Dropout(0.2))

    def TransitionDown(self,filters):
        model = self.model
        # model.add(BatchNormalization(mode=0, #axis=1,
        #                              gamma_regularizer=l2(0.0001),
        #                              beta_regularizer=l2(0.0001)))
        model.add(Activation('relu'))
        model.add(Conv2D(filters, kernel_size=(1, 1), padding='same', strides=(2, 2),
                                  kernel_initializer="he_uniform"))
        model.add(Dropout(0.2))
        # model.add(MaxPooling2D( pool_size=(2, 2),
        #                         strides=(2, 2),
        #                         data_format='channels_last'))

    def TransitionUp(self,filters,input_shape,output_shape):
        model = self.model
        model.add(Conv2DTranspose(filters,  kernel_size=(3, 3), strides=(2, 2),
                                            padding='same',
                                            output_shape=output_shape,
                                            input_shape=input_shape,
                                            kernel_initializer="he_uniform",
                                            data_format='channels_last'))


    # def create(self):
    #     self.model = models.Sequential()
    #     # cropping
    #     # model.add(Cropping2D(cropping=((68, 68), (128, 128)), input_shape=(3, 360,480)))

    #     self.model.add(Conv2D(48, kernel_size=(3, 3), padding='same', 
    #                          input_shape=(512, 512, 1),
    #                         kernel_initializer="he_uniform",
    #                         kernel_regularizer = l2(0.0001),
    #                         data_format='channels_last'))

    #     # (5 * 4)* 2 + 5 + 5 + 1 + 1 +1
    #     # growth_m = 4 * 12
    #     # previous_m = 48

    #     self.DenseBlock(4,112) # 4*16 = 64 + 48 = 112
    #     self.TransitionDown(112)
    #     self.DenseBlock(5,192) # 5*16 = 80 + 112 = 192
    #     self.TransitionDown(192)
    #     self.DenseBlock(7,304) # 7 * 16 = 112 + 192 = 304
    #     self.TransitionDown(304)
    #     self.DenseBlock(10,464)
    #     self.TransitionDown(464)
    #     self.DenseBlock(12,656)
    #     self.TransitionDown(656)

    #     self.DenseBlock(15,896) # m = 656 + 15x16 = 896

    #     # upsampling part, m[B] is the sum of 3 terms
    #     # 1. the m value corresponding to same resolution in the downsampling part (skip connection)
    #     # 2. the number of feature maps from the upsampled block (n_layers[B-1] * growth_rate)
    #     # 3. the number of feature maps in the new block (n_layers[B] * growth_rate)
    #     #
    #     self.TransitionUp(1088, (1088, 16, 16), (None, 1088, 32, 32))  # m = 656 + 15x16 + 12x16 = 1088.
    #     self.DenseBlock(12,1088)

    #     self.TransitionUp(816, (816, 32, 32), (None, 816, 64, 64)) #m = 464 + 12x16 + 10x16 = 816
    #     self.DenseBlock(10,816)

    #     self.TransitionUp(576, (576, 64, 64), (None, 576, 128, 128)) # m = 304 + 10x16 + 7x16 = 576
    #     self.DenseBlock(7,576)

    #     self.TransitionUp(384, (384, 128, 128), (None, 384, 256, 256)) # m = 192 + 7x16 + 5x16 = 384
    #     self.DenseBlock(5,384)

    #     self.TransitionUp(256, (256, 256, 256), (None, 256, 512, 512)) # m = 112 + 5x16 + 4x16 = 256
    #     self.DenseBlock(4,256)

    #     self.model.add(Conv2D(1, kernel_size=(1,1), 
    #                          padding='same',
    #                          kernel_initializer="he_uniform",
    #                          kernel_regularizer = l2(0.0001),
    #                          data_format='channels_last'))
        
    #     # model.add(Reshape((1, 512 * 512)))
    #     # model.add(Permute((2, 1)))
    #     # model.add(Activation('softmax'))
    #     self.model.summary()

    #     with open('tiramisu_fc_dense103_model.json', 'w') as outfile:
    #         outfile.write(json.dumps(json.loads(self.model.to_json()), indent=3))


    #     return self.model
    def gfactorCounterDown(self,growth_factor,block_size,previous_conv_size,block_count=5):
        for i in range(block_count):
            m = block_size * growth_factor + previous_conv_size
            self.DenseBlock(growth_factor,m)
            self.TransitionDown(m)

    def gfactorCounterUp(self,growth_factor,block_size,previous_block_size,previous_conv_size,block_count=5):
        # previous_conv_size = 288, since:
        # self.DenseBlock(4,288) # 4*12 = 48 + 288 = 336
        # self.TransitionDown(288) 
        
        for i in range(block_count):
            m = block_size * growth_factor + previous_block_size * growth_factor + previous_conv_size
            self.DenseBlock(growth_factor,m)
            self.TransitionUp(m)

    def create(self):
        self.model = models.Sequential()
        self.model.add(Conv2D(48, kernel_size=(3, 3), 
                    padding='same', 
                    input_shape=(512,512,1), 
                    init="he_uniform", 
                    W_regularizer = l2(0.0001), 
                    data_format='channels_last'))

        ############# 56
        # self.DenseBlock(4,96) # 4*12 = 48 + 48 = 96
        # self.TransitionDown(96)
        # self.DenseBlock(4,144) # 4*12 = 48 + 96 = 144
        # self.TransitionDown(144)
        # self.DenseBlock(4,192) # 4*12 = 48 + 144 = 192
        # self.TransitionDown(192)
        # self.DenseBlock(4,240)# 4*12 = 48 + 192 = 240
        # self.TransitionDown(240)
        # self.DenseBlock(4,288) # 4*12 = 48 + 288 = 336
        # self.TransitionDown(288)

        # self.DenseBlock(15,336) # 4 * 12 = 48 + 288 = 336

        # self.TransitionUp(384, (384, 16, 16), (None, 384, 32, 32))  # m = 288 + 4x12 + 4x12 = 384.
        # self.DenseBlock(4,384)

        # self.TransitionUp(336, (336, 32, 32), (None, 336, 64, 64)) #m = 240 + 4x12 + 4x12 = 336
        # self.DenseBlock(4,336)

        # self.TransitionUp(288, (288, 64, 64), (None, 288, 128, 128)) # m = 192 + 4x12 + 4x12 = 288
        # self.DenseBlock(4,288)

        # self.TransitionUp(240, (240, 128, 128), (None, 240, 256, 256)) # m = 144 + 4x12 + 4x12 = 240
        # self.DenseBlock(4,240)

        # self.TransitionUp(192, (192, 256, 256), (None, 192, 512, 512)) # m = 96 + 4x12 + 4x12 = 192
        # self.DenseBlock(4,192)
        ############# 56
        ############# 67
        self.DenseBlock(5,108) # 5*12 = 60 + 48 = 108
        self.TransitionDown(108)
        self.DenseBlock(5,168) # 5*12 = 60 + 108 = 168
        self.TransitionDown(168)
        self.DenseBlock(5,228) # 5*12 = 60 + 168 = 228
        self.TransitionDown(228)
        self.DenseBlock(5,288)# 5*12 = 60 + 228 = 288
        self.TransitionDown(288)
        self.DenseBlock(5,348) # 5*12 = 60 + 288 = 348
        self.TransitionDown(348)

        self.DenseBlock(15,408) # m = 348 + 5*12 = 408


        self.TransitionUp(468, (468, 16, 16), (None, 468, 32, 32))  # m = 348 + 5x12 + 5x12 = 468.
        self.DenseBlock(5,468)

        self.TransitionUp(408, (408, 32, 32), (None, 408, 64, 64)) # m = 288 + 5x12 + 5x12 = 408
        self.DenseBlock(5,408)

        self.TransitionUp(348, (348, 64, 64), (None, 348, 128, 128)) # m = 228 + 5x12 + 5x12 = 348
        self.DenseBlock(5,348)

        self.TransitionUp(288, (288, 128, 128), (None, 288, 256, 256)) # m = 168 + 5x12 + 5x12 = 288
        self.DenseBlock(5,288)

        self.TransitionUp(228, (228, 256, 256), (None, 228, 512, 512)) # m = 108 + 5x12 + 5x12 = 228
        self.DenseBlock(5,228)
        ############# 67
        self.model.add(Conv2D(1, kernel_size=(1,1), 
                     padding='same',
                     kernel_initializer="he_uniform",
                     kernel_regularizer = l2(0.0001),
                     data_format='channels_last'))

        self.model.summary()

        with open('tiramisu_fc_dense_67_model.json', 'w') as outfile:
            outfile.write(json.dumps(json.loads(self.model.to_json()), indent=3))
        return self.model







###############################################################################
###############################################################################
def shape3d (a):
    if type(a) == int:
        return [a,a,a]
    if isinstance (a, (list, tuple)):
        assert len (a) == 3
        return list (a)
    raise RuntimeError("Illegal shape: {}".format(a))

def shape5d (a, data_format='NDHWC'):
    return [1] + shape3d (a) + [1]


from tensorpack.models.common import layer_register, VariableHolder
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format
from tensorpack.models.tflayer import rename_get_variable, convert_to_tflayer_args

@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv3D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1):
    """
    A wrapper around `tf.layers.Conv3D`.
    Some differences to maintain backward-compatibility:
    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group conv.
    Variable Names:
    * ``W``: weights
    * ``b``: bias
    """
    if split == 1:
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv3D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                _reuse=tf.get_variable_scope().reuse)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias

    else:
        # group conv implementation
        pass
    return ret


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size', 'strides'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Deconv3D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1, 1),
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.
    Some differences to maintain backward-compatibility:
    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'
    Variable Names:
    * ``W``: weights
    * ``b``: bias
    """

    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv3DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            _reuse=tf.get_variable_scope().reuse)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret


###################################################################################################

@layer_register()
def InstanceNorm3D(x, epsilon=1e-5, use_affine=True, gamma_init=None, data_format='NDHWC'):
    """
    Instance Normalization, as in the paper:
    `Instance Normalization: The Missing Ingredient for Fast Stylization
    <https://arxiv.org/abs/1607.08022>`_.
    Args:
        x (tf.Tensor): a 4D tensor.
        epsilon (float): avoid divide-by-zero
        use_affine (bool): whether to apply learnable affine transformation
    """
    shape = x.get_shape().as_list()
    assert len(shape) == 5, "Input of InstanceNorm has to be 4D!"

    if data_format == 'NDHWC':
        axis = [1, 2, 3]
        ch = shape[4]
        new_shape = [1, 1, 1, 1, ch]
    else:
        axis = [2, 3, 4]
        ch = shape[1]
        new_shape = [1, ch, 1, 1, 1]
    assert ch is not None, "Input of InstanceNorm require known channel!"

    mean, var = tf.nn.moments(x, axis, keep_dims=True)

    if not use_affine:
        return tf.divide(x - mean, tf.sqrt(var + epsilon), name='output')

    beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)
    if gamma_init is None:
        gamma_init = tf.constant_initializer(1.0)
    gamma = tf.get_variable('gamma', [ch], initializer=gamma_init)
    gamma = tf.reshape(gamma, new_shape)
    return tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')

###################################################################################################
def INReLU3D(x, name=None):
    x = InstanceNorm3D('inorm', x)
    return tf.nn.relu(x, name=name)

def INLReLU3D(x, name=None):
    x = InstanceNorm3D('inorm', x)
    return tf.nn.leaky_relu(x, name=name)

def INELU3D(x, name=None):
    x = InstanceNorm3D('inorm', x)
    return tf.nn.elu (x, name=name)

def INELU2D(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.elu (x, name=name)


























###############################################################################
#                                   FusionNet 3D aniso
###############################################################################
@layer_register(log_shape=True)
def residual_3d(x, chan, first=False, kernel_shape=3):
    with argscope([Conv3D], nl=INLReLU3D, stride=1, kernel_shape=kernel_shape):
        inputs = x
        x = tf.pad(x, name='pad1', mode='REFLECT', paddings=[[0,0], 
                                                             [1*(kernel_shape//2),1*(kernel_shape//2)], 
                                                             [1*(kernel_shape//2),1*(kernel_shape//2)], 
                                                             [1*(kernel_shape//2),1*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv3D('conv1', x, chan, padding='VALID', dilation_rate=1)
        x = tf.pad(x, name='pad2', mode='REFLECT', paddings=[[0,0], 
                                                             [2*(kernel_shape//2),2*(kernel_shape//2)], 
                                                             [2*(kernel_shape//2),2*(kernel_shape//2)], 
                                                             [2*(kernel_shape//2),2*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv3D('conv2', x, chan, padding='VALID', dilation_rate=2)
        x = tf.pad(x, name='pad3', mode='REFLECT', paddings=[[0,0], 
                                                             [4*(kernel_shape//2),4*(kernel_shape//2)], 
                                                             [4*(kernel_shape//2),4*(kernel_shape//2)], 
                                                             [4*(kernel_shape//2),4*(kernel_shape//2)], 
                                                             [0,0]])
        x = Conv3D('conv3', x, chan, padding='VALID', dilation_rate=4, activation=tf.identity)             
        # x = tf.pad(x, name='pad4', mode='REFLECT', paddings=[[0,0], [8*(kernel_shape//2),8*(kernel_shape//2)], [8*(kernel_shape//2),8*(kernel_shape//2)], [0,0]])
        # x = Conv3D('conv4', x, chan, padding='VALID', dilation_rate=8) 
        x = InstanceNorm3D('inorm', x) + inputs
        return x


@layer_register(log_shape=True)
def residual_enc_3d(x, chan, first=False, kernel_shape=3):
    with argscope([Conv3D], nl=INLReLU3D, stride=1, kernel_shape=kernel_shape):
        x = tf.pad(x, name='pad_i', mode='REFLECT', paddings=[[0,0], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [0,0]])
        x = Conv3D('enc_i', x, chan, stride=1)
        x = tf.layers.max_pooling3d(x, pool_size=(1,2,2), strides=(1,2,2), name='max_p') 
        x = residual_3d('enc_r', x, chan, first=True)
        x = tf.pad(x, name='pad_o', mode='REFLECT', paddings=[[0,0], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [0,0]])
        x = Conv3D('enc_o', x, chan, stride=1) #, activation=tf.identity) 
        #x = InstanceNorm('enc_n', x)
        return x

@layer_register(log_shape=True)
def residual_dec_3d(x, chan, kernel_shape=3):
    with argscope([Conv3D, Deconv3D], nl=INLReLU3D, stride=1, kernel_shape=kernel_shape):
                
        x = (LinearWrap(x)
            .Deconv3D('deconv_i', chan, stride=1, kernel_shape=(3,3,3)) 
            .residual_3d('res2_', chan, kernel_shape=kernel_shape)
            .Deconv3D('deconv_o', chan, stride=(1,2,2)) 
            # .Dropout('drop', 0.75)
            ())
        return x


###############################################################################
def arch_fusionnet_encoder_3d(img, feats=[None, None, None], last_dim=1, nl=INLReLU3D, nb_filters=32):
    with argscope([Conv3D], nl=INLReLU3D, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU3D, kernel_shape=3, stride=2, padding='SAME'):
            e0 = residual_enc_3d('e0', img, nb_filters*1)
            e0 = Dropout('f0', e0, 0.5)
            e1 = residual_enc_3d('e1',  e0, nb_filters*2)
            e1 = Dropout('f1', e1, 0.5)
            e2 = residual_enc_3d('e2',  e1, nb_filters*4)
            e2 = Dropout('f2', e2, 0.5)
            e3 = residual_enc_3d('e3',  e2, nb_filters*8)
            e3 = Dropout('f3', e3, 0.5)
            return e3, [e2, e1, e0]
           

def arch_fusionnet_decoder_3d(img, feats=[None, None, None], last_dim=1, nl=INLReLU3D, nb_filters=32):
    with argscope([Conv3D], nl=INLReLU3D, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv3D], nl=INLReLU3D, kernel_shape=3, stride=2, padding='SAME'):
            e2, e1, e0 = feats 
            d4 = img 
            # d4 = Dropout('r4', d4, 0.5)

            d3 = residual_dec_3d('d3', d4, nb_filters*4)
            # d3 = Dropout('r3', d3, 0.5)
            d3 = d3+e2 if e2 is not None else d3 
            
            d2 = residual_dec_3d('d2', d3, nb_filters*2)
            # d2 = Dropout('r2', d2, 0.5)
            d2 = d2+e1 if e1 is not None else d2 
            
            d1 = residual_dec_3d('d1', d2, nb_filters*1)
            # d1 = Dropout('r1', d1, 0.5)
            d1 = d1+e0 if e0 is not None else d1  
            
            d0 = residual_dec_3d('d0', d1, nb_filters*1) 
            # d0 = Dropout('r0', d0, 0.5)
            
            dp = tf.pad( d0, name='pad_o', mode='REFLECT', paddings=[[0,0], 
                                                                     [7//2,7//2], 
                                                                     [7//2,7//2], 
                                                                     [7//2,7//2], 
                                                                     [0,0]])
            dd = Conv3D('convlast', dp, last_dim, kernel_shape=7, stride=1, padding='VALID', nl=nl, use_bias=True) 
            return dd, [d1, d2, d3]
###############################################################################
def arch_fusionnet_translator_3d(img, feats=[None, None, None], last_dim=1, nl=INLReLU3D, nb_filters=32):
    enc, feat_enc = arch_fusionnet_encoder_3d(img, feats, last_dim, nl, nb_filters)
    dec, feat_dec = arch_fusionnet_decoder_3d(enc, feat_enc, last_dim, nl, nb_filters)
    return dec























###############################################################################
#                                   FusionNet 3D iso
###############################################################################



@layer_register(log_shape=True)
def residual_enc_3d_iso(x, chan, first=False, kernel_shape=3):
    with argscope([Conv3D], nl=INLReLU3D, stride=1, kernel_shape=kernel_shape):
        x = tf.pad(x, name='pad_i', mode='REFLECT', paddings=[[0,0], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [0,0]])
        x = Conv3D('enc_i', x, chan, stride=2)
        #x = tf.layers.max_pooling3d(x, pool_size=(2,2,2), strides=(2,2,2), name='max_p') 
        x = residual_3d('enc_r', x, chan, first=True)
        x = tf.pad(x, name='pad_o', mode='REFLECT', paddings=[[0,0], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [kernel_shape//2,kernel_shape//2], 
                                                              [0,0]])
        x = Conv3D('enc_o', x, chan, stride=1) #, activation=tf.identity) 
        #x = InstanceNorm('enc_n', x)
        return x

@layer_register(log_shape=True)
def residual_dec_3d_iso(x, chan, kernel_shape=3):
    with argscope([Conv3D, Deconv3D], nl=INLReLU3D, stride=1, kernel_shape=kernel_shape):
                
        x = (LinearWrap(x)
            .Deconv3D('deconv_i', chan, stride=1, kernel_shape=(3,3,3)) 
            .residual_3d('res2_', chan, kernel_shape=kernel_shape)
            .Deconv3D('deconv_o', chan, stride=(2,2,2)) 
            # .Dropout('drop', 0.75)
            ())
        return x


###############################################################################
def arch_fusionnet_encoder_3d_iso(img, feats=[None, None, None], last_dim=1, nl=INLReLU3D, nb_filters=32):
    with argscope([Conv3D], nl=INLReLU3D, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU3D, kernel_shape=3, stride=2, padding='SAME'):
            e0 = residual_enc_3d_iso('e0', img, nb_filters*1)
            #e0 = Dropout('f0', e0, 0.5)
            e1 = residual_enc_3d_iso('e1',  e0, nb_filters*2)
            #e1 = Dropout('f1', e1, 0.5)
            e2 = residual_enc_3d_iso('e2',  e1, nb_filters*4)
            #e2 = Dropout('f2', e2, 0.5)
            e3 = residual_enc_3d_iso('e3',  e2, nb_filters*8)
            #e3 = Dropout('f3', e3, 0.5)
            return e3, [e2, e1, e0]
           

def arch_fusionnet_decoder_3d_iso(img, feats=[None, None, None], last_dim=1, nl=INLReLU3D, nb_filters=32):
    with argscope([Conv3D], nl=INLReLU3D, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv3D], nl=INLReLU3D, kernel_shape=3, stride=2, padding='SAME'):
            e2, e1, e0 = feats 
            d4 = img 
            # d4 = Dropout('r4', d4, 0.5)

            d3 = residual_dec_3d_iso('d3', d4, nb_filters*4)
            # d3 = Dropout('r3', d3, 0.5)
            d3 = d3+e2 if e2 is not None else d3 
            
            d2 = residual_dec_3d_iso('d2', d3, nb_filters*2)
            # d2 = Dropout('r2', d2, 0.5)
            d2 = d2+e1 if e1 is not None else d2 
            
            d1 = residual_dec_3d_iso('d1', d2, nb_filters*1)
            # d1 = Dropout('r1', d1, 0.5)
            d1 = d1+e0 if e0 is not None else d1  
            
            d0 = residual_dec_3d_iso('d0', d1, nb_filters*1) 
            # d0 = Dropout('r0', d0, 0.5)
            
            dp = tf.pad( d0, name='pad_o', mode='REFLECT', paddings=[[0,0], 
                                                                     [3//2,3//2], 
                                                                     [3//2,3//2], 
                                                                     [3//2,3//2], 
                                                                     [0,0]])
            dd = Conv3D('convlast', dp, last_dim, kernel_shape=3, stride=1, padding='VALID', nl=nl, use_bias=True) 
            return dd, [d1, d2, d3]
###############################################################################
def arch_fusionnet_translator_3d_iso(img, feats=[None, None, None], last_dim=1, nl=INLReLU3D, nb_filters=32):
    enc, feat_enc = arch_fusionnet_encoder_3d_iso(img, feats, last_dim, nl, nb_filters)
    dec, feat_dec = arch_fusionnet_decoder_3d_iso(enc, feat_enc, last_dim, nl, nb_filters)
    return dec











###############################################################################
# tflearn
import tflearn
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_3d, conv_3d_transpose, max_pool_3d
from tflearn.layers.core import dropout
from tflearn.layers.merge_ops import merge
from tflearn.activations import linear, sigmoid, tanh, elu
from tensorflow.python.framework import ops

def tf_bottleneck(inputs, nb_filter, name="bottleneck"):
    with tf.variable_scope(name):
            original  = tf.identity(inputs, name="identity")

            with tf.contrib.framework.arg_scope([conv_3d, conv_3d_transpose], strides=[1, 1, 1, 1, 1], activation='leaky_relu'):
                    shape = original.get_shape().as_list()
                    conv_4x4i = original #conv_3d(incoming=original,    name="conv_4x4i", filter_size=4, nb_filter=nb_filter) # From 256 to 64 in Residual pape, bias=Falser
                    # original  = tf.nn.dropout(original, keep_prob=0.5)
                    conv_4x4i = conv_3d(incoming=conv_4x4i, name="conv_4x4i", filter_size=4, nb_filter=nb_filter, bias=False) # From 256 to 64 in Residual paper
                    # conv_4x4i = tf.nn.dropout(conv_4x4i, keep_prob=0.5)
                    conv_4x4m = conv_3d(incoming=conv_4x4i, name="conv_4x4m", filter_size=4, nb_filter=nb_filter, bias=False)
                    # conv_4x4o = tf.nn.dropout(conv_4x4o, keep_prob=0.5)
                    conv_4x4o = conv_3d(incoming=conv_4x4m, name="conv_4x4o", filter_size=4, nb_filter=nb_filter, bias=False, activation=tf.identity,
                                                                              # output_shape=[shape[1], shape[2], shape[3]]
                                                                              )
            summation = tf.add(original, conv_4x4o, name="summation")
            # summation = elu(summation)
            # return batch_normalization(summation)
            ret = InstanceNorm('bn', tf.squeeze(summation, axis=0))
            ret = tf.expand_dims(ret, axis=0)
            return ret


# In[10]:


#def arch_fusionnet_3d(img, last_dim=1, nl=INLReLU, nb_filters=32, name='fusion3d'):
def arch_fusionnet_translator_3d_iso_tflearn(img, feats=[None, None, None], last_dim=1, nl=INLReLU3D, nb_filters=32):
    
    # Add decorator to tflearn source code
    # sudo nano /usr/local/lib/python2.7/dist-packages/tflearn/layers/conv.py
    # @tf.contrib.framework.add_arg_scope
    with tf.contrib.framework.arg_scope([conv_3d], filter_size=4, strides=[1, 2, 2, 2, 1], activation='leaky_relu'):
        with tf.contrib.framework.arg_scope([conv_3d_transpose], filter_size=4, strides=[1, 2, 2, 2, 1], activation='leaky_relu'):
            shape = img.get_shape().as_list()
            dimb, dimz, dimy, dimx, dimc = shape
            e1a  = conv_3d(incoming=img,           name="e1a", nb_filter=nb_filters*1, bias=False)
            r1a  = tf_bottleneck(e1a,              name="r1a", nb_filter=nb_filters*1)
            r1a  = tf.nn.dropout(r1a,     keep_prob=0.5)

            e2a  = conv_3d(incoming=r1a,           name="e2a", nb_filter=nb_filters*1, bias=False)
            r2a  = tf_bottleneck(e2a,              name="r2a", nb_filter=nb_filters*1)
            r2a  = tf.nn.dropout(r2a,     keep_prob=0.5)

            e3a  = conv_3d(incoming=r2a,           name="e3a", nb_filter=nb_filters*2, bias=False)
            r3a  = tf_bottleneck(e3a,              name="r3a", nb_filter=nb_filters*2)
            r3a  = tf.nn.dropout(r3a,     keep_prob=0.5)

            e4a  = conv_3d(incoming=r3a,           name="e4a", nb_filter=nb_filters*2, bias=False)
            r4a  = tf_bottleneck(e4a,              name="r4a", nb_filter=nb_filters*2)
            r4a  = tf.nn.dropout(r4a,     keep_prob=0.5)

            e5a  = conv_3d(incoming=r4a,           name="e5a", nb_filter=nb_filters*4, bias=False)
            r5a  = tf_bottleneck(e5a,              name="r5a", nb_filter=nb_filters*4)
            r5a  = tf.nn.dropout(r5a,     keep_prob=0.5)

            # e6a  = conv_3d(incoming=r5a,           name="e6a", nb_filter=nb_filters*4, bias=False)
            # r6a  = tf_bottleneck(e6a,              name="r6a", nb_filter=nb_filters*4)

            # e7a  = conv_3d(incoming=r6a,           name="e7a", nb_filter=nb_filters*8)           , bias=False 
            # r7a  = tf_bottleneck(e7a,              name="r7a", nb_filter=nb_filters*8)
            # r7a  = dropout(incoming=r7a, keep_prob=0.5)
            print("In1 :", img.get_shape().as_list())
            print("E1a :", e1a.get_shape().as_list())
            print("R1a :", r1a.get_shape().as_list())
            print("E2a :", e2a.get_shape().as_list())
            print("R2a :", r2a.get_shape().as_list())
            print("E3a :", e3a.get_shape().as_list())
            print("R3a :", r3a.get_shape().as_list())
            print("E4a :", e4a.get_shape().as_list())
            print("R4a :", r4a.get_shape().as_list())
            print("E5a :", e5a.get_shape().as_list())
            print("R5a :", r5a.get_shape().as_list())
            
            r5b  = tf_bottleneck(r5a,              name="r5b", nb_filter=nb_filters*4)
            d4b  = conv_3d_transpose(incoming=r5b, name="d4b", nb_filter=nb_filters*2, output_shape=[-(-dimz//(2**4)), -(-dimy//(2**4)), -(-dimx/(2**4))], bias=False)
            a4b  = tf.add(d4b, r4a,            name="a4b")

            r4b  = tf_bottleneck(a4b,              name="r4b", nb_filter=nb_filters*2)
            d3b  = conv_3d_transpose(incoming=r4b, name="d3b", nb_filter=nb_filters*2, output_shape=[-(-dimz//(2**3)), -(-dimy//(2**3)), -(-dimx/(2**3))], bias=False)
            a3b  = tf.add(d3b, r3a,            name="a3b")


            r3b  = tf_bottleneck(a3b,              name="r3b", nb_filter=nb_filters*2)
            d2b  = conv_3d_transpose(incoming=r3b, name="d2b", nb_filter=nb_filters*1, output_shape=[-(-dimz//(2**2)), -(-dimy//(2**2)), -(-dimx/(2**2))], bias=False)
            a2b  = tf.add(d2b, r2a,            name="a2b")

            r2b  = tf_bottleneck(a2b,              name="r2b", nb_filter=nb_filters*1)
            d1b  = conv_3d_transpose(incoming=r2b, name="d1b", nb_filter=nb_filters*1, output_shape=[-(-dimz//(2**1)), -(-dimy//(2**1)), -(-dimx/(2**1))], bias=False)
            a1b  = tf.add(d1b, r1a,            name="a1b")

            out  = conv_3d_transpose(incoming=a1b, name="out", nb_filter=last_dim,
                                                            activation='tanh',
                                                            output_shape=[-(-dimz//(2**0)), -(-dimy//(2**0)), -(-dimx/(2**0))])


            # print("R7b :", r7b.get_shape().as_list())
            # print("D6b :", d6b.get_shape().as_list())
            # print("A6b :", a6b.get_shape().as_list())

            # print("R6b :", r6b.get_shape().as_list())
            # print("D5b :", d5b.get_shape().as_list())
            # print("A5b :", a5b.get_shape().as_list())

            print("R5b :", r5b.get_shape().as_list())
            print("D4b :", d4b.get_shape().as_list())
            print("A4b :", a4b.get_shape().as_list())

            print("R4b :", r4b.get_shape().as_list())
            print("D3b :", d3b.get_shape().as_list())
            print("A3b :", a3b.get_shape().as_list())

            print("R3b :", r3b.get_shape().as_list())
            print("D2b :", d2b.get_shape().as_list())
            print("A2b :", a2b.get_shape().as_list())

            print("R2b :", r2b.get_shape().as_list())
            print("D1b :", d1b.get_shape().as_list())
            print("A1b :", a1b.get_shape().as_list())

            print("Out :", out.get_shape().as_list())

            return out



