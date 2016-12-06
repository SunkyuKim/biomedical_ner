import tensorflow as tf
import numpy as np

def variable_summaries(var, name):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)
def nn_layer(input_tensor, input_dim, output_dim, layer_name):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            # weights = tf.Variable(
            #     tf.truncated_normal([input_dim, output_dim], stddev=1))
            weights = tf.get_variable(layer_name + "_W", shape=[input_dim, output_dim],
                                      initializer=tf.contrib.layers.xavier_initializer())
            variable_summaries(weights, layer_name + "/weights")
        with tf.name_scope("biases"):
            bias = tf.Variable(
                tf.constant(0.1, shape=[output_dim]))
            variable_summaries(bias, layer_name + "/biases")
        with tf.name_scope("Wx_plus_b"):
            z = tf.matmul(input_tensor, weights) + bias
            tf.histogram_summary(layer_name + '/z', z)

        # hidden1 = tf.nn.dropout(relu, 0.5)
        l2loss = tf.nn.l2_loss(weights) + tf.nn.l2_loss(bias)

        return z, l2loss
def conv_layer(input_tensor, feature_size, filter_width, filter_height, num_filters, layer_name):
    # if there are multiple embeddings, we dont need to expand it
    input_tensor_expanded = tf.expand_dims(input_tensor, -1)

    with tf.name_scope(layer_name):
        filter_shape = [filter_width, filter_height, 1, num_filters]
        with tf.name_scope("weights"):
            weights = tf.get_variable(layer_name + "_W", shape=filter_shape,
                                      initializer=tf.contrib.layers.xavier_initializer())
            variable_summaries(weights, layer_name + "/weights")
        with tf.name_scope("biases"):
            bias = tf.Variable(
                tf.constant(0.1, shape=[num_filters]))
            variable_summaries(bias, layer_name + "/biases")
        with tf.name_scope("CONV"):
            conv = tf.nn.conv2d(
                input_tensor_expanded,
                weights,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv"
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
            # Maxpooling over the outputs
            z = tf.nn.max_pool(
                h,
                ksize=[1, feature_size - filter_width + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")

            tf.histogram_summary(layer_name + '/z', z)

        # make flat-dimension (?,1,1,num_filter) -> (?,num_filter)
        z_flat = tf.reshape(z, [-1, num_filters])
        return z_flat
def recurrent_layer(input_tensor, feature_size, hidden_size, layer_name):
    with tf.name_scope(layer_name):
        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
        _X = tf.unpack(tf.transpose(input_tensor, perm=[1, 0, 2]))
        # outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)
        outputs, _fw = tf.nn.rnn(lstm_fw_cell, _X, dtype=tf.float32)

        # output : n_step * batch_size * feature_size
        # rnn_o = outputs[-1]

        with tf.name_scope("max_pooling"):
            # "pack" makes list of tensor to tensor
            #
            outputs = tf.pack(outputs)
            rnn_o = tf.reduce_max(outputs, 0)

        return rnn_o
def embedding_layer(tf_dataset, vocab_size, embedding_dim, pre_trained_embedding, layer_name):
    with tf.name_scope(layer_name):
        x = np.asarray(range(vocab_size)) + 1
        e = tf_dataset
        # http://stackoverflow.com/questions/35842598/tensorflow-using-a-tensor-to-index-another-tensor
        xm = tf.mul(e, x)
        where = tf.not_equal(xm, 0)
        # indices = tf.pack([tf.reshape(tf.where(t), [-1])
        #                   for t in tf.unpack(where)])
        xm = tf.cast(xm, tf.int32)
        W = tf.Variable(
            tf.random_uniform([vocab_size + 1, embedding_dim], -1.0, 1.0),
            name="W", trainable=False)
        W.assign(pre_trained_embedding)

        tf_dataset_embedding = tf.nn.embedding_lookup(W, xm)

        return tf_dataset_embedding