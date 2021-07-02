import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L


# class VGG_FeatureExtractor(K.Model):
#     """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

#     def __init__(self, input_channel, output_channel=512):
#         super(VGG_FeatureExtractor, self).__init__()
#         self.output_channel = [int(output_channel / 8), int(output_channel / 4),
#                                int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
#         self.ConvNet = K.Sequential([
#             L.Conv2D(self.output_channel[0], 3, 1, 1, activation='relu'), 
#             L.MaxPool2D(2, 2),  # 64x16x50
#             L.Conv2D(self.output_channel[1], 3, 1, 1, activation='relu'), 
#             L.MaxPool2D(2, 2),  # 128x8x25
#             L.Conv2D(self.output_channel[2], 3, 1, 1, activation='relu'),  # 256x8x25
#             L.Conv2D(self.output_channel[2], 3, 1, 1, activation='relu'), 
#             L.ReLU(),
#             L.MaxPool2D((2, 1), (2, 1)),  # 256x4x25
#             L.Conv2D(self.output_channel[3], 3, 1, 1, use_bias=False, activation='relu'),
#             L.BatchNormalization(), L.ReLU(),  # 512x4x25
#             L.Conv2D(self.output_channel[3], 3, 1, 1, use_bias=False, activation='relu'),
#             L.BatchNormalization(), L.ReLU(),
#             L.MaxPool2D((2, 1), (2, 1)),  # 512x2x25
#             L.Conv2D(self.output_channel[3], 2, 1, 0), activation='relu')  # 512x1x24
#         ])
            
#     def call(self, input):
#         return self.ConvNet(input)


# class RCNN_FeatureExtractor(K.Model):
#     """ FeatureExtractor of GRCNN (https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf) """

#     def __init__(self, input_channel, output_channel=512):
#         super(RCNN_FeatureExtractor, self).__init__()
#         self.output_channel = [int(output_channel / 8), int(output_channel / 4),
#                                int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
#         self.ConvNet = K.Sequential([
#             L.Conv2D(self.output_channel[0], 3, 1, 1, activation='relu'),
#             L.MaxPool2D(2, 2),  # 64 x 16 x 50
#             GRCL(self.output_channel[0], self.output_channel[0], num_iteration=5, kernel_size=3, pad=1),
#             L.MaxPool2D(2, 2),  # 64 x 8 x 25
#             GRCL(self.output_channel[0], self.output_channel[1], num_iteration=5, kernel_size=3, pad=1),
#             L.MaxPool2D(2, (2, 1), (0, 1)),  # 128 x 4 x 26
#             GRCL(self.output_channel[1], self.output_channel[2], num_iteration=5, kernel_size=3, pad=1),
#             L.MaxPool2D(2, (2, 1), (0, 1)),  # 256 x 2 x 27
#             L.Conv2D(self.output_channel[3], 2, 1, 0, use_bias=False),
#             L.BatchNormalization(), L.ReLU()
#             ])  # 512 x 1 x 26

#     def call(self, input):
#         return self.ConvNet(input)


class ResNet_FeatureExtractor(K.Model):
    """ FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def call(self, input):
        return self.ConvNet(input)


# # For Gated RCNN
# class GRCL(K.Model):

#     def __init__(self, input_channel, output_channel, num_iteration, kernel_size, pad):
#         super(GRCL, self).__init__()
#         self.wgf_u = L.Conv2D(output_channel, 1, 1, 0, use_bias=False)
#         self.wgr_x = L.Conv2D(output_channel, 1, 1, 0, use_bias=False)
#         self.wf_u = L.Conv2D(output_channel, kernel_size, 1, pad, use_bias=False)
#         self.wr_x = L.Conv2D(output_channel, kernel_size, 1, pad, use_bias=False)

#         self.BN_x_init = L.BatchNormalization()

#         self.num_iteration = num_iteration
#         self.GRCL = [GRCL_unit(output_channel) for _ in range(num_iteration)]
#         self.GRCL = K.Sequential(*self.GRCL)

#     def call(self, input):
#         """ The input of GRCL is consistant over time t, which is denoted by u(0)
#         thus wgf_u / wf_u is also consistant over time t.
#         """
#         wgf_u = self.wgf_u(input)
#         wf_u = self.wf_u(input)
#         x = F.relu(self.BN_x_init(wf_u))

#         for i in range(self.num_iteration):
#             x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))

#         return x


# class GRCL_unit(K.Model):

#     def __init__(self, output_channel):
#         super(GRCL_unit, self).__init__()
#         self.BN_gfu = L.BatchNormalization()
#         self.BN_grx = L.BatchNormalization()
#         self.BN_fu = L.BatchNormalization()
#         self.BN_rx = L.BatchNormalization()
#         self.BN_Gx = L.BatchNormalization()

#     def call(self, wgf_u, wgr_x, wf_u, wr_x):
#         G_first_term = self.BN_gfu(wgf_u)
#         G_second_term = self.BN_grx(wgr_x)
#         G = F.sigmoid(G_first_term + G_second_term)

#         x_first_term = self.BN_fu(wf_u)
#         x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
#         x = F.relu(x_first_term + x_second_term)

#         return x


class BasicBlock(K.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = L.BatchNormalization()
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = L.BatchNormalization()
        self.relu = L.ReLU()
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return L.Conv2D(out_planes, kernel_size=3, stride=stride,
                         padding=1, use_bias=False)

    def call(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(K.Model):

    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [int(output_channel / 4), int(output_channel / 2), output_channel, output_channel]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = L.Conv2D(int(output_channel / 16),
                                 kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn0_1 = L.BatchNormalization()
        self.conv0_2 = L.Conv2D(self.inplanes,
                                 kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn0_2 = L.BatchNormalization()
        self.relu = L.ReLU()

        self.maxpool1 = L.MaxPool2D(pool_size=(2, 2), stride=2, padding='valid')
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = L.Conv2D(self.output_channel_block[
                               0], kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn1 = L.BatchNormalization()

        self.maxpool2 = L.MaxPool2D(pool_size=(2, 2), stride=2, padding='valid')
        self.layer2 = self._make_layer(block, self.output_channel_block[1], layers[1], stride=1)
        self.conv2 = L.Conv2D(
            self.output_channel_block[1], 
            kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn2 = L.BatchNormalization()
        # TODO: self.bn2 output shape 확인 필요. 
        # padding=(0, 1) (height, width)
        _pad_h0_w1 = [[0, 0], [0, 0], [1, 1], [0, 0]] # data_format: NHWC
        self.maxpool3 = tf.nn.max_pool2d(ksize=(2, 2), strides=(2, 1), padding=_pad_h0_w1)
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = L.Conv2D(self.output_channel_block[
                               2], kernel_size=3, stride=1, padding=1, use_bias=False)
        self.bn3 = L.BatchNormalization()

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        
        self.conv4_1 = L.Conv2D(self.output_channel_block[
                                 3], kernel_size=2, stride=(2, 1), padding=(0, 1), use_bias=False)
        self.bn4_1 = L.BatchNormalization()
        self.conv4_2 = L.Conv2D(self.output_channel_block[
                                 3], kernel_size=2, stride=1, padding=0, use_bias=False)
        self.bn4_2 = L.BatchNormalization()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = K.Sequential([
                L.Conv2D(planes * block.expansion,
                          kernel_size=1, stride=stride, use_bias=False),
                L.BatchNormalization(),
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return K.Sequential([layers])

    def call(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x
