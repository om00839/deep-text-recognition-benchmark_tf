import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L

from dtrb_tf.module.grid_sample_tf import grid_sample_2d


class TPS_SpatialTransformerNetwork(K.Model):
    """ Rectification Network of RARE, namely TPS based STN """

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        """ Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def call(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = tf.reshape(build_P_prime, [build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        
        batch_I_r = grid_sample_2d(batch_I, build_P_prime_reshape, padding_mode='border', align_corners=True)
        
        return batch_I_r


class LocalizationNetwork(K.Model):
    """ Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, I_channel_num):
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = K.Sequential([
            L.Conv2D(64, kernel_size=3, stride=1, padding=1,
                      bias=False), 
            L.BatchNormalization(), 
            L.ReLU(), # input_shape: self.I_channel_num
            L.MaxPool2D(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            L.Conv2D(128, 3, 1, 1, bias=False), 
            L.BatchNormalization(), 
            L.ReLU(), # input_shape: 64
            L.MaxPool2D(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            L.Conv2D(256, 3, 1, 1, bias=False), 
            L.BatchNormalization(), 
            L.ReLU(), # input_shape: 128
            L.MaxPool2D(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            L.Conv2D(512, 3, 1, 1, bias=False), 
            L.BatchNormalization(), 
            L.ReLU(), # input_shape: 256
            L.GlobalAveragePooling2D()  # batch_size x 512
        ])

        self.localization_fc1 = K.Sequential([L.Dense(256), L.ReLU()])
        # Init fc2 in LocalizationNetwork
        self.localization_fc2 = L.Dense(self.F * 2, kernel_initializer='zeros', bias_initializer='zeros',)

        """ see RARE paper Fig. 6 (a) """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        # TODO: ???
        self.localization_fc2.bias.set_weights(tf.convert_to_tensor(initial_bias, dtype=tf.float32)) 
        # self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def call(self, batch_I):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.shape[0]
        features = self.conv(batch_I)
        features = tf.reshape(features, [batch_size, -1])
        batch_C_prime = self.localization_fc2(self.localization_fc1(features))
        batch_C_prime = tf.reshape(batch_C_prime, [batch_size, self.F, 2])
        return batch_C_prime


class GridGenerator(K.Model):
    """ Grid Generator of RARE, which produces P_prime by multipling T with P """

    def __init__(self, F, I_r_size):
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        ## for multi-gpu, you need register buffer
        # TODO: register_buffer와 같은 역할을 하는 tf function이 뭐가 있남?
        self.register_buffer("inv_delta_C", tf.convert_to_tensor(self._build_inv_delta_C(self.F, self.C)), dtype=tf.float32)  # F+3 x F+3
        self.register_buffer("P_hat", tf.convert_to_tensor(self._build_P_hat(self.F, self.C, self.P)), dtype=tf.float32)  # n x F+3
        ## for fine-tuning with different image width, you may use below instead of self.register_buffer
        #self.inv_delta_C = torch.tensor(self._build_inv_delta_C(self.F, self.C)).float().cuda()  # F+3 x F+3
        #self.P_hat = torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float().cuda()  # n x F+3

    def _build_C(self, F):
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """ Return inv_delta_C which is needed to calculate T """
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y),
            axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """ Generate Grid from batch_C_prime [batch_size x F x 2] """
        batch_size = batch_C_prime.shape[0]
        batch_inv_delta_C = tf.tile(self.inv_delta_C, [1, 1])
        batch_P_hat = tf.tile(self.P_hat, [1, 1])
        batch_C_prime_with_zeros = tf.concat([batch_C_prime, tf.zeros(
            [batch_size, 3, 2], dtype=tf.float32)], axis=1)  # batch_size x F+3 x 2
        batch_T = tf.linalg.matmul(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = tf.linalg.matmul(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2
