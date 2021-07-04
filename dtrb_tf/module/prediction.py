# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L


class Attention(K.Model):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = L.Dense(num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = tf.expand_dims(input_char, 1)
        batch_size = input_char.shape[0]
        # TODO: 확인필요
        one_hot = tf.one_hot(input_char, onehot_dim, dtype=tf.float32)
        one_hot = tf.tile(one_hot, [batch_size, 0, 0])
        # pytorch version
        # one_hot = tf.zeros([batch_size, onehot_dim], dtype=tf.float32)
        # one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def call(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = tf.zeros([batch_size, num_steps, self.hidden_size], dtype=tf.float32)
        # output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (
            tf.zeros([batch_size, self.hidden_size], dtype=tf.float32),
            tf.zeros([batch_size, self.hidden_size], dtype=tf.float32)
        )
        # hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
        #           torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = tf.zeros(batch_size, dtype=tf.float64)
            # targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = tf.zeros([batch_size, num_steps, self.num_classes], dtype=tf.float32)
            # probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(K.Model):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = L.Dense(hidden_size, use_bias=False)
        self.h2h = L.Dense(hidden_size)  # either i2i or h2h should have bias
        self.score = L.Dense(1, use_bias=False)
        
        self.rnn = L.LSTMCell(hidden_size) # input shape: (input_size + num_embeddings)
        self.hidden_size = hidden_size

    def call(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(K.activations.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = K.activations.softmax(e, dim=1)
        context = tf.linalg.matmul(tf.transpose(alpha, perm=[0, 2, 1]), batch_H)
        context = tf.squeeze(context, 1)  # batch_size x num_channel
        concat_context = tf.concat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
