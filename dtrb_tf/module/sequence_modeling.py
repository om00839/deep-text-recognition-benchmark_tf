# import torch.nn as nn

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers as L


class BidirectionalLSTM(K.Model):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        # TODO: return_sequence / return_state ??
        self.rnn = L.Bidirectional(L.LSTM(hidden_size, batch_first=True, return_sequences=False, return_state=False))
        self.linear = L.Dense(output_size) # input shape: hidden_size * 2, 

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        # TODO: flatten_parameters 이건 또 뭐람
        # self.rnn.flatten_parameters() 
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
