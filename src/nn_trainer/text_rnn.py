import tensorflow as tf
import numpy as np
import sys
sys.append("../..")
from src.nn_trainer import nn_util


class TextRNN(object):
    def __init__(self,
                 sequence_length,
                 vocab_embeddings,
                 cell_type,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 l2_reg_lambda=0,
                 additional_conf={},
                 ):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.Variable(np.array(vocab_embeddings, dtype='float32'), trainable=False)
            self.embedded_x = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.variable_scope("RNN"):
            rnn_output, rnn_states, hidden_size = _rnn(
                inputs=self.embedded_x,
                cell_type=cell_type,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout_keep_prob=self.dropout_keep_prob,
                bidirectional=bidirectional
            )

            # last one
            self.rnn_rep = nn_util.last_relevant(rnn_output, nn_util.get_data_length(self.embedded_x))

            # sum
            # self.rnn_rep = tf.reduce_sum(rnn_output, 1)

            # avg, 除以的是100，有问题
            self.rnn_rep = tf.reduce_mean(rnn_output, 1)

            self.rnn_output_size = hidden_size

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.rnn_rep, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("softmax"):

            # 加一个隐藏层
            hidden_layer_size = 100
            h_W = tf.get_variable(
                "h_W",
                shape=[self.rnn_output_size, hidden_layer_size],
                initializer=tf.contrib.layers.xavier_initializer())
            h_b = tf.Variable(tf.constant(0.1, shape=[hidden_layer_size]), name="h_b")
            self.hidden_layer_output = tf.nn.tanh(tf.nn.xw_plus_b(self.h_drop, h_W, h_b, name="hidden_layer_output"))

            W = tf.get_variable(
                "W",
                shape=[hidden_layer_size, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")

            self.predictions = tf.nn.tanh(tf.nn.xw_plus_b(self.hidden_layer_output, W, b, name="predictions"))

        # Calculate Mean Squared Error (MSE)
        with tf.name_scope("loss"):
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))

            # cosine_distance
            # self.loss = tf.contrib.losses.cosine_distance(self.predictions, self.input_y, 1)

            # self.loss = tf.contrib.distributions.kl(self.predictions, self.input_y)


def _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob):
    cell = None
    if cell_type == "BasicLSTM":
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    if cell_type == "LSTM":
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0)
    if cell_type == "GRU":
        cell = tf.contrib.rnn.GRUCell(hidden_size)

    if cell is None:
        raise ValueError("cell type: %s is incorrect!!" % (cell_type))

    # dropout
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    # multi-layer
    if num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
    return cell


def _rnn(inputs,
         cell_type,
         hidden_size,
         num_layers,
         dropout_keep_prob,
         bidirectional=False):
    if bidirectional:
        cell_fw = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)
        cell_bw = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=tf.to_int64(nn_util.get_data_length(inputs)),
            dtype=tf.float32)

        hidden_size *= 2
        outputs = tf.concat(outputs, 2)

    else:
        cell = _get_rnn_cell(cell_type, hidden_size, num_layers, dropout_keep_prob)
        outputs, state = tf.nn.dynamic_rnn(
            cell,
            inputs,
            dtype=tf.float32,
            sequence_length=nn_util.get_data_length(inputs),
        )

    return outputs, state, hidden_size
