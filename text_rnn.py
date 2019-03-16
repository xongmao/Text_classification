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
        self.y = tf.one_hot(self.onehot,20,1,0)
        self.y_out = tf.placeholder(tf.float32, [None, self.num_classes], name="rnn_y_out")
        
        # 定义前向RNN Cell
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            print (tf.get_variable_scope().name)
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=self.dropout_keep_prob)

        # 定义反向RNN Cell
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            print (tf.get_variable_scope().name)
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(self.rnn_size) for _ in range(self.layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list), output_keep_prob=self.dropout_keep_prob)

        with tf.name_scope("rnn_embedding"):
            self.em = tf.Variable(self.word_dict, name="em")
            embedded_chars = tf.nn.embedding_lookup(self.em, self.encoder_inputs)
            embedded_chars = tf.transpose(embedded_chars, [1,0,2])
            embedded_chars = tf.reshape(embedded_chars, [-1, self.rnn_size])
            embedded_chars = tf.split(embedded_chars, self.sequence_length, 0)

        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, embedded_chars, dtype=tf.float32)
            
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2*self.rnn_size, self.attn_size], stddev=0.1), name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[self.attn_size]), name='attention_b')
            u_list = []
            for t in range(self.sequence_length):
                u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b) 
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([self.attn_size, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in range(self.sequence_length):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            alpha = tf.nn.softmax(attn_zconcat)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(alpha, [1,0]), [self.sequence_length, -1, 1])
            final_output = tf.reduce_sum(outputs * alpha_trans, 0)
            
        print (final_output.shape)
        # outputs shape: (sequence_length, batch_size, 2*rnn_size)
        fc_w = tf.Variable(tf.truncated_normal([2*self.rnn_size, self.num_classes], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([self.num_classes]), name='fc_b')
        logits = tf.matmul(final_output, fc_w) + fc_b
        prob = tf.nn.softmax(logits, name='rnn_prob')
        self.predictions = tf.argmax(prob, 1, name="rnn_predictions")
        self.loss = tf.losses.softmax_cross_entropy(self.y_out, logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_out, axis=1), self.predictions), tf.float32))