import tensorflow as tf
import numpy as np

class rnn_Model():
    def __init__(self, rnn_size, num_classes, embedding_size, word_dict, attn_size, layer_size, sequence_length):
        self.rnn_size = rnn_size
        self.attn_size = attn_size
        self.embedding_size = embedding_size
        self.word_dict = word_dict
        self.layer_size = layer_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.encoder_inputs = tf.placeholder(tf.int32, [None, self.sequence_length], name="rnn_encoder_inputs")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="rnn_dropout_keep_prob")
        self.onehot = tf.placeholder(tf.int32, [None], name="rnn_onehot_in")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.y = tf.one_hot(self.onehot,20,1,0)
        self.y_out = tf.placeholder(tf.float32, [None, self.num_classes], name="rnn_y_out")
        
        with tf.name_scope("rnn_embedding"):
            embedded_const = tf.nn.embedding_lookup(self.word_dict, self.encoder_inputs)
            masked = tf.sign(tf.abs(tf.reduce_sum(embedded_const, -1)))
            #self.em = tf.Variable(self.word_dict, name="em")
            self.em = tf.Variable(
                tf.random_uniform([len(self.word_dict)+1, self.embedding_size], -1.0, 1.0),
                name="em")
            embedded_chars = tf.nn.embedding_lookup(self.em, self.encoder_inputs)
            passage_encoding = tf.concat((embedded_const, embedded_chars), axis = 2)

        with tf.name_scope('lstm'):
            lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=self.dropout_keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell([drop] * self.layer_size)
            initial_state = cell.zero_state(self.batch_size, tf.float32)
            #outputs, _ = tf.contrib.rnn.static_rnn(cell, passage_encoding, dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(cell, passage_encoding, initial_state=initial_state)
            
        with tf.name_scope('attention'):
            outputs2 = tf.reshape(outputs, (-1, self.rnn_size))
            attention_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.attn_size], stddev=0.1), name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.attn_size]), name='attention_b')
            u_t = tf.tanh(tf.matmul(outputs2, attention_w) + attention_b)
            v_w = tf.Variable(tf.truncated_normal([self.attn_size, 1], stddev=0.1), name='v_w')
            z_t = tf.matmul(u_t, v_w)
            z_t = tf.reshape(z_t, (-1, self.sequence_length))
            z_t = z_t * masked + ((1.0 - masked) * (-1e12))
            scores = tf.nn.softmax(z_t)
            context = tf.expand_dims(scores, -1) * outputs
            final_output = tf.reduce_sum(context, 1)
            
        fc_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([self.num_classes]), name='fc_b')
        logits = tf.matmul(final_output, fc_w) + fc_b
        prob = tf.nn.softmax(logits, name='rnn_prob')
        self.predictions = tf.argmax(prob, 1, name="rnn_predictions")
        self.loss = tf.losses.softmax_cross_entropy(self.y_out, logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_out, axis=1), self.predictions), tf.float32))