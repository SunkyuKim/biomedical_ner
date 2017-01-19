import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class Model():
    def __init__(self, args):

        with tf.name_scope("Input_Layer"):
            self.args = args
            self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.targets = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.class_size])

        with tf.name_scope("Embedding_layer"):
            self.embedding = tf.placeholder(tf.float32, [args.vocab_size, args.rnn_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size], trainable=True)
                tf.assign(embedding, self.embedding)
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            # embedding vector of unused word is filled all of 0.
            # The sum of absolute values of whole values in a embedding vector of a word should be over 0 if this word is not padding word.
            # [batch_size, sequence_size]
            used = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
            # the number of values used words
            # [batch_size]
            self.length = tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)

        with tf.name_scope("RNN_layer"):
            fw_cell = rnn_cell.BasicLSTMCell(args.rnn_size)
            bw_cell = rnn_cell.BasicLSTMCell(args.rnn_size)
            # self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            inputs = tf.split(1, args.seq_length, inputs)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            # inputs => [batch_size, word_dim] * seqence_length
            # outputs => [batch_size, rnn_size] * seqence_length
            # outputs, states = rnn.rnn(cell, inputs, initial_state=self.initial_state)
            outputs, _, _ = rnn.bidirectional_rnn(fw_cell, bw_cell, inputs)

            #output => [batch_size*sequence_length, rnn_size]
            output = tf.reshape(tf.transpose(tf.pack(outputs), perm=[1,0,2]), [-1, args.rnn_size])

        with tf.name_scope("Softmax_layer"):
            softmax_w = tf.get_variable("softmax_x", [args.rnn_size*2, args.class_size])
            softmax_b = tf.get_variable("softmax_b", [args.class_size])
            #prediction => [batch_size*sequence_length, class_size]
            prediction = tf.nn.softmax(tf.matmul(output, softmax_w) + softmax_b)

            #self.prediction => [batch_size, sequence_length, class_size]
            self.prediction = tf.reshape(prediction, [-1, args.seq_length, args.class_size])

        with tf.name_scope("Optimizer"):
            self.loss = self.cost()
            optimizer = tf.train.AdamOptimizer(0.003)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            tf.scalar_summary("loss", self.loss)
            self.merged = tf.merge_all_summaries()

    # masking cross entropy values of unused input, output(padding values) to 0
    def cost(self):
        cross_entropy = self.targets * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.targets), reduction_indices=2))
        # cross entropy values of only used words are left.
        # [batch_size, sequence_length]
        cross_entropy *= mask
        # [batch_size]
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        # single value
        return tf.reduce_mean(cross_entropy)
