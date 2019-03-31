import tensorflow as tf
import numpy as np

class cnn_Model():
    def __init__(self, num_classes, embedding_size, word_dict, num_filters, sequence_length):
        self.embedding_size = embedding_size
        self.word_dict = word_dict
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.num_filters_total = self.num_filters * 3
        self.encoder_inputs = tf.placeholder(tf.int32, [None, self.sequence_length], name="cnn_encoder_inputs")
        self.z_in1 = tf.placeholder(tf.float32, [None, self.sequence_length-1, self.num_filters], name="z_in1")
        self.z_in2 = tf.placeholder(tf.float32, [None, self.sequence_length-2, self.num_filters], name="z_in2")
        self.z_in3 = tf.placeholder(tf.float32, [None, self.sequence_length-3, self.num_filters], name="z_in3")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="cnn_dropout_keep_prob")
        self.onehot = tf.placeholder(tf.int32, [None], name="cnn_onehot_in")
        self.y = tf.one_hot(self.onehot,20,1,0,name="cnn_one_hot")
        self.y_out = tf.placeholder(tf.float32, [None, self.num_classes], name="cnn_y_out")
        l2_loss = tf.constant(0.0)

        with tf.name_scope("cnn_embedding"):
            self.em = tf.Variable(
                tf.random_uniform([len(self.word_dict)+1, self.embedding_size], -1.0, 1.0),
                name="em")
            #self.em = tf.Variable(self.word_dict, name="em")
            embedded_chars = tf.nn.embedding_lookup(self.word_dict, self.encoder_inputs)
            embedded_chars2 = tf.nn.embedding_lookup(self.em, self.encoder_inputs)
            passage_encoding = tf.concat((embedded_chars, embedded_chars2), axis = 2)
            embedded_chars_expanded = tf.expand_dims(passage_encoding, -1)

        pooled_outputs = []
        
        with tf.name_scope("conv-maxpool1"):
            filter_shape = [2, self.embedding_size*2, 1, self.num_filters]
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
            zz1 = tf.expand_dims(self.z_in1, 2)
            h = h * zz1 + ((1.0 - zz1) * tf.float32.min)
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_length - 2 + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            print (pooled.shape)
            pooled_outputs.append(pooled)
            
        with tf.name_scope("conv-maxpool2"):
            filter_shape = [3, self.embedding_size*2, 1, self.num_filters]
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
            zz2 = tf.expand_dims(self.z_in2, 2)
            print (zz2.shape)
            h = h * zz2 + ((1.0 - zz2) * tf.float32.min)
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_length - 3 + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            print (pooled.shape)
            pooled_outputs.append(pooled)
            
        with tf.name_scope("conv-maxpool3"):
            filter_shape = [4, self.embedding_size*2, 1, self.num_filters]
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
            zz3 = tf.expand_dims(self.z_in3, 2)
            print (zz3.shape)
            h = h * zz3 + ((1.0 - zz3) * tf.float32.min)
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, self.sequence_length - 4 + 1, 1, 1],
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
        with tf.name_scope("cnn_dropout"):
            self.h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            
        with tf.name_scope("cnn_output"):
            W = tf.get_variable(
                "W",
                shape=[self.num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y_out)
            self.loss = tf.reduce_mean(losses) + 0.5 * l2_loss 
            
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y_out, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")