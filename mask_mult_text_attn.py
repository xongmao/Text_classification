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
            #self.em = tf.Variable(self.word_dict, name="em")
            self.em = tf.Variable(
                tf.random_uniform([len(self.word_dict)+1, self.embedding_size], -1.0, 1.0),
                name="em")
            embedded_chars = tf.nn.embedding_lookup(self.em, self.encoder_inputs)
            embedded_chars1 = tf.nn.embedding_lookup(self.word_dict, self.encoder_inputs)
            masked = tf.sign(tf.abs(tf.reduce_sum(embedded_chars1, -1)))
            passage_encoding = tf.concat((embedded_chars1, embedded_chars), axis = 2)
            embedded_chars2 = tf.reshape(passage_encoding, (-1, self.embedding_size*2))
            mult_output = []
            for i in range(2):
                with tf.name_scope("mult-attn-%s" % i):
                    q_w = tf.Variable(tf.truncated_normal([self.embedding_size*2, 50], stddev=0.1), name='q_w')
                    q_b = tf.Variable(tf.constant(0.1, shape=[50]), name='q_b')
                    u_t = tf.tanh(tf.matmul(embedded_chars2, q_w) + q_b)
                    mult_output.append(u_t)
            attn_mult_output = tf.concat(mult_output, 1)
            v_w = tf.Variable(tf.truncated_normal([100, 1], stddev=0.1), name='v_w')
            z_t = tf.matmul(attn_mult_output, v_w)
            z_t = tf.reshape(z_t, (-1, self.sequence_length))
            z_t = z_t * masked + ((1.0 - masked) * (-1e12))
            scores = tf.nn.softmax(z_t)
            context = tf.expand_dims(scores, -1) * passage_encoding
            final_output = tf.reduce_sum(context, 1)
            
        fc_w = tf.Variable(tf.truncated_normal([self.embedding_size*2, self.embedding_size], stddev=0.1), name='attn_fc_w')
        fc_b = tf.Variable(tf.zeros([self.embedding_size]), name='attn_fc_b')
        fc_w2 = tf.Variable(tf.truncated_normal([self.embedding_size, self.num_classes], stddev=0.1), name='attn_fc_w2')
        fc_b2 = tf.Variable(tf.zeros([self.num_classes]), name='attn_fc_b2')
        out_prob = tf.nn.relu(tf.matmul(final_output, fc_w) + fc_b)
        prob_d = tf.nn.dropout(out_prob, keep_prob=self.dropout_keep_prob)
        logits = tf.matmul(prob_d, fc_w2) + fc_b2
        prob = tf.nn.softmax(logits, name='attn_prob')
        self.predictions = tf.argmax(prob, 1, name="attn_predictions")
        self.loss = tf.losses.softmax_cross_entropy(self.y_out, logits)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_out, axis=1), self.predictions), tf.float32))
        self.accuracy2 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.y_out, axis=1), self.predictions), tf.float32), 0)