import tensorflow as tf
import numpy as np

class cnn_Model():
    def __init__(self, num_classes, embedding_size, word_dict, num_filters, filter_sizes, sequence_length):
        self.embedding_size = embedding_size
        self.word_dict = word_dict
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_filters_total = self.num_filters * 3
        self.encoder_inputs = tf.placeholder(tf.int32, [None, self.sequence_length], name="cnn_encoder_inputs")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="cnn_dropout_keep_prob")
        self.onehot = tf.placeholder(tf.int32, [None], name="cnn_onehot_in")
        self.y = tf.one_hot(self.onehot,20,1,0,name="cnn_one_hot")
        self.y_out = tf.placeholder(tf.float32, [None, self.num_classes], name="cnn_y_out")
        l2_loss = tf.constant(0.0)

        with tf.name_scope("cnn_embedding"):
            self.em = tf.Variable(self.word_dict, name="em")
            embedded_chars = tf.nn.embedding_lookup(self.em, self.encoder_inputs)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                print (h.shape)
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print (pooled.shape)
                pooled_outputs.append(pooled)

        h_pool = tf.concat(pooled_outputs, 3)
        print (h_pool.shape)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
        print (h_pool_flat.shape)
        
        # Add dropout
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        
        W = tf.get_variable(
            "W",
            shape=[self.num_filters_total, self.num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")
            
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_out)
        self.loss = tf.reduce_mean(losses) + 0.5 * l2_loss 
            
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y_out, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")