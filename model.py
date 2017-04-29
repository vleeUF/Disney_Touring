import sys
import tensorflow as tf
import numpy as np
# from sklearn.metrics import mean_absolute_error
from ops import cnn, batch_norm, dropout, _linear


class Model:
    def __init__(self, params):
        self.params = params
        self.dir = params.save_dir

        # Initialize data and training variables for later usage
        self.features = None
        self.wait_times = None
        self.dates = None
        self.times = None

        self.mean_error = None
        self.total_loss = None
        self.train_step = None

        self.is_training = None

        with tf.variable_scope('Model'):
            print("Building Model...")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.build()
            self.saver = tf.train.Saver()

    def build(self):
        params = self.params

        # Placeholders
        features = tf.placeholder('int32', shape=[params.batch_size], name='features')
        dates = tf.placeholder('int32', shape=[params.batch_size], name='dates')
        times = tf.placeholder('int32', shape=[params.batch_size], name='times')
        wait_times = tf.placeholder('int32', shape=[params.batch_size], name='wait_times')
        is_training = tf.placeholder(tf.bool)

        # Prepare parameters

        with tf.variable_scope("CNN"):
            cnn_output = cnn(features, params.feature_maps, params.kernels, dates)

            # Regularizations [batch norm and dropout]
            output = batch_norm(cnn_output, is_training=is_training)
            output = dropout(output, params.d_rate, is_training)
        """"
        with tf.variable_scope("LSTM") as scope:
            lstm_outputs = []
            loss = 0

            cell = tf.nn.rnn_cell.BasicLSTMCell(params.rnn_size)
            stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * params.layer_depth)

            outputs, _ = tf.nn.rnn(stacked_cell, output)
            for idx, (pred_y, true_y) in enumerate(zip(outputs, self.wait_times)):
                output = dropout(pred_y, params.d_rate, is_training)
                if idx != 0:
                    scope.reuse_variables()
                output = _linear(output, h)
                lstm_outputs.append(output)
                loss += tf.nn.sparse_softmax_cross_entropy_with_logits(lstm_outputs[idx], tf.squeeze(self.wait_times))
        """
        with tf.name_scope('Loss'):
            # Cross Entropy Loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output, wait_times)
            loss = tf.reduce_mean(cross_entropy)
            total_loss = loss + params.weight_decay * tf.add_n(tf.get_collection('l2'))

        # Optimization with ADAM
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        train_step = optimizer.minimize(total_loss, global_step=self.global_step)

        # Data variables
        self.features = features
        self.wait_times = wait_times
        self.dates = dates
        self.times = times
        self.is_training = is_training

        # Training variables
        self.mean_error = mean_error
        self.total_loss = total_loss
        self.train_step = train_step

    def get_feed_dict(self, batch, is_train):
        features, wait_times, dates, times = batch
        return {self.features: features,
                self.wait_times: wait_times,
                self.dates: dates,
                self.times: times,
                self.is_training: is_train}

    def train_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train=True)
        return sess.run([self.train_step, self.global_step], feed_dict=feed_dict)

    def test_batch(self, sess, batch):
        feed_dict = self.get_feed_dict(batch, is_train=False)
        return sess.run([self.mean_error, self.total_loss, self.global_step], feed_dict=feed_dict)

    def train(self, sess, train):
        params = self.params
        num_epochs = params.num_epochs
        num_batches = train.num_batches

        for epoch in range(num_epochs):
            for _ in range(num_batches):
                batch = train.next_batch()
                _, global_step = self.train_batch(sess, batch)
            train.reset()
            if (epoch + 1) % params.save_period == 0:
                self.save(sess)

        print("Training Completed.")

    def eval(self, sess, test):
        num_batches = test.num_batches
        mean_errors = []
        for _ in range(num_batches):
            batch = test.next_batch()
            cur_mean_error, cur_loss, global_step = self.test_batch(sess, batch)
            mean_errors.append(cur_mean_error)
        test.reset()
        with tf.name_scope('Accuracy'):
            # MAE
            mae = np.mean(mean_errors)
            print("MAE: %f.4" % mae)

    def save(self, sess):
        print("Saving model to %s" % self.dir)
        self.saver.save(sess, self.dir, self.global_step)

    def load(self, sess):
        print("Loading model...")
        checkpoint = tf.train.get_checkpoint_state(self.dir)
        if checkpoint is None:
            print("Error: No saved model available.")
            sys.exit()
        self.saver.restore(sess, checkpoint.model_checkpoint_path)