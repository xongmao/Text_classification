import tensorflow as tf
import numpy as np

class attn_Model():
    def __init__(self, num_classes, embedding_size, word_dict, sequence_length):
        self.embedding_size = embedding_size
        self.word_dict = word_dict
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.encoder_inputs = tf.placeholder(tf.int32, [None, self.sequence_length], name="attn_encoder_inputs")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="attn_dropout_keep_prob")
        self.onehot = tf.placeholder(tf.int32, [None], name="attn_onehot_in")
        self.y = tf.one_hot(self.onehot,20,1,0,name="attn_one_hot")
        self.y_out = tf.placeholder(tf.float32, [None, self.num_classes], name="attn_y_out")

        with tf.name_scope("attn_attention"):
            self.em = tf.Variable(self.word_dict, name="em")
            embedded_chars = tf.nn.embedding_lookup(self.em, self.encoder_inputs)
            embedded_chars1 = tf.nn.embedding_lookup(self.word_dict, self.encoder_inputs)
            masked = tf.sign(tf.abs(tf.reduce_sum(embedded_chars1, -1)))
            embedded_chars2 = tf.reshape(embedded_chars, (-1, self.embedding_size))
            q_w = tf.Variable(tf.truncated_normal([self.embedding_size, 100], stddev=0.1), name='q_w')
            q_b = tf.Variable(tf.constant(0.1, shape=[100]), name='q_b')
            u_t = tf.tanh(tf.matmul(embedded_chars2, q_w) + q_b)
            v_w = tf.Variable(tf.truncated_normal([100, 1], stddev=0.1), name='v_w')
            z_t = tf.matmul(u_t, v_w)
            z_t = tf.reshape(z_t, (-1, self.sequence_length))
            z_t = z_t * masked + ((1.0 - masked) * (-1e12))
            scores = tf.nn.softmax(z_t)
            context = tf.expand_dims(scores, -1) * embedded_chars
            final_output = tf.reduce_sum(context, 1)
            
        fc_w = tf.Variable(tf.truncated_normal([self.embedding_size, self.num_classes], stddev=0.1), name='attn_fc_w')
        fc_b = tf.Variable(tf.zeros([self.num_classes]), name='attn_fc_b')
        logits = tf.matmul(final_output, fc_w) + fc_b
        prob = tf.nn.softmax(logits, name='attn_prob')
        self.predictions = tf.argmax(prob, 1, name="attn_predictions")
        self.loss = tf.losses.softmax_cross_entropy(self.y_out, logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_out, axis=1), self.predictions), tf.float32))
        self.accuracy2 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.y_out, axis=1), self.predictions), tf.float32), 0)