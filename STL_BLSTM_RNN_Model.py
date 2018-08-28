import os
path = os.path.dirname(os.path.abspath(__file__))

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

n_output_classes = {'chord_symbol': 25,
                    'key': 24,
                    'pri_deg': 21,
                    'sec_deg': 21,
                    'quality': 10,
                    'inversion': 4}

class STL_BLSTM_RNNModel(object):
    def __init__(self,
                 feature_size=1952,
                 n_steps=64,
                 n_hidden_units=1024,
                 learning_rate=1e-4,
                 L2_beta=1e-4,
                 dropout_rate=0.5,
                 use_crf=True,
                 task='chord_symbol'):

        self._feature_size = feature_size
        self._n_steps = n_steps
        self._hidden_size = n_hidden_units
        self._task = task
        try:
            self._n_classes = n_output_classes[self._task]
        except KeyError as e:
            print('Task Error:', e, 'Task should be one of the following: \'chord_symbol\', \'key\', \'pri_deg\', \'sec_deg\', \'quality\', \'inversion\'.')

        self._session = None
        self._graph = None

        # Summary for training visualization
        self._graph_path = path + "\\Training"
        self._train_writer = None
        self._valid_writer = None

        self._L2_bata = L2_beta
        self._dropout_rate = dropout_rate
        self._learning_rate = learning_rate
        self._use_crf = use_crf

        batch_in_shape = [None, self._n_steps, self._feature_size]
        batch_out_shape = [None, self._n_steps]
        self.batch_in = tf.placeholder(tf.float32, shape=batch_in_shape, name='batch_in')
        self.batch_out = tf.placeholder(tf.int32, shape=batch_out_shape, name='batch_out')
        self.is_dropout = tf.placeholder(tf.bool)

        self._optimizer = tf.train.AdamOptimizer(self._learning_rate)

    def load_variables(self, path='./stl_blstm_rnn_ckpt'):
        if self._session is None:
            self._session = tf.Session()
            saver = tf.train.Saver()
            print('loading variables...')
            saver.restore(self._session, path)

    def save_variables(self, path='./stl_blstm_rnn_ckpt', to_print=True):
        saver = tf.train.Saver()
        if to_print:
            print('saving variables...')
        saver.save(self._session, path)

    def _label_smoothing(self, inputs, epsilon=0.1):
        K = inputs.get_shape().as_list()[-1] # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / K)

    def network(self):

        # RNN cell
        with tf.name_scope('LSTM_cell'):
            encoder_cell_fw = LSTMCell(num_units=self._hidden_size)
            encoder_cell_bw = LSTMCell(num_units=self._hidden_size)

        with tf.name_scope('Dropout'):
            keep_prob = tf.cond(self. is_dropout, lambda: 1 - self._dropout_rate, lambda: 1.0)
            encoder_cell_fw = DropoutWrapper(encoder_cell_fw, output_keep_prob=keep_prob)
            encoder_cell_bw = DropoutWrapper(encoder_cell_bw, output_keep_prob=keep_prob)


        with tf.name_scope('LSTM_layer'):
            # bi-LSTM
            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
                                                                                           encoder_cell_bw,
                                                                                           self.batch_in,
                                                                                           time_major=False,
                                                                                           dtype=tf.float32)
            rnn_outputs = tf.concat([output_fw, output_bw], axis=-1)  # shape = [batch, n_steps, 2*n_hiddne_units]

        with tf.name_scope('Output_projection_layer'):
            logits = tf.layers.dense(rnn_outputs, self._n_classes)  # shape = [batch, n_steps, n_classes]
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)  # shape = [batch, n_steps]

        with tf.name_scope('loss'):
            if not self._use_crf:
                y_smoothed = self._label_smoothing(tf.one_hot(self.batch_out, depth=self._n_classes))
                # cross entropy
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_smoothed)
                loss_1 = tf.reduce_mean(cross_entropy)
            else:
                # CRF
                sequence_lens = tf.ones(tf.shape(self.batch_in)[0], dtype=tf.int32)*self._n_steps
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(inputs=logits, tag_indices=self.batch_out, sequence_lengths=sequence_lens)
                viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(potentials=logits, transition_params=transition_params, sequence_length=sequence_lens)
                loss_1 = tf.reduce_mean(-log_likelihood)

            # L2 norm regularization
            vars = tf.trainable_variables()
            L2_regularizer = self._L2_bata * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

            # loss
            loss = loss_1 + L2_regularizer
        tf.summary.scalar('Loss', loss)

        with tf.name_scope('accuracy'):
            if not self._use_crf:
                correct_predictions = tf.equal(predictions, self.batch_out) # use softmax
            else:
                correct_predictions = tf.equal(viterbi_sequence, self.batch_out) # use crf
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        tf.summary.scalar('Accuracy', accuracy)

        merged = tf.summary.merge_all() # merge all summaries collected in the graph

        # Apply gradient clipping
        gvs = self._optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train = self._optimizer.apply_gradients(capped_gvs)

        return train, loss, logits, predictions, accuracy, merged, L2_regularizer

    def train(self,
              batch_in,
              batch_out,
              variable_path='./stl_blstm_rnn_ckpt',
              load_vars_from_disk=False,
              save_vars_to_disk=False,
              step=0):
        """
        Train the network.

        params:
        ---
        batch_in: a 3-D numpy array. The dimensions should be [batch_size, num_timesteps, feature_size]
        batch_out: a 2-D dimensional numpy array. The dimensions should be [batch_size, n_steps, n_classes]
        variable_path: the path to which variable values will be saved and/or loaded
        load_vars_from_disk: bool, whether to load variables prior to training
        load_vars_from_disk: bool, whether to save variables after training
        """

        optim, loss, logits, predictions, accuracy, merged, L2_regularizer = self._get_graph()

        if not load_vars_from_disk:
            if self._session is None:
                self._session = tf.Session()
                print('Saving graph to: %s' % self._graph_path)
                self._train_writer = tf.summary.FileWriter(self._graph_path + '\\train', self._session.graph)
                self._valid_writer = tf.summary.FileWriter(self._graph_path + '\\valid', self._session.graph)
                init = tf.global_variables_initializer()
                self._session.run(init)
        else:
            self.load_variables(variable_path)

        ops = [optim, loss, predictions, merged, accuracy, L2_regularizer]
        feed_dict = {self.batch_in: batch_in, self.batch_out: batch_out, self.is_dropout: True}
        _, _loss, _predictions, _summary, _accuracy, _L2 = self._session.run(ops, feed_dict)

        if step == 0:
            print('CE = %.4f,  L2 = %.4f' % (_loss - _L2, _L2))

        if step % 200 == 0:
            self._train_writer.add_summary(_summary, step) # add to training log
            print("*------ iteration %d:  train_loss %.4f, train_accuracy %.4f ------*" % (step, _loss, _accuracy))

            # prediction result
            # print('  Label     > {}'.format(batch_out[0, :]))
            # print('  Prediction > {}'.format(_predictions[0, :]))
            print('Label'.ljust(10, ' '), ''.join([str(b).rjust(3, ' ') for b in batch_out[0, :]]))
            print('Prediction'.ljust(10, ' '), ''.join([str(b).rjust(3, ' ') for b in _predictions[0, :]]))

        if save_vars_to_disk:
            self.save_variables(variable_path, to_print=True)


    def predict(self, batch_in, batch_out, variable_path='./stl_blstm_rnn_ckpt', step=0, is_valid=False):
        """
        Make predictions.

        params:
        ---
        batch_in: batch for which to make predictions. should have dimensions [batch_size, n_steps, feature_size]
        batch_out: ground truth. should have dimensions [batch_size, n_steps]
        variable_path: string. If there is no active session in the network
            object (i.e. it has not yet been used to train or predict, or the
            tensorflow session has been manually closed), variables will be
            loaded from the provided path. Otherwise variables already present
            in the session will be used.

        returns:
        ---
        predictions for the batch
        """

        _, loss, _, predictions, accuracy, merged, _ = self._get_graph()

        self._load_vars(variable_path)

        ops = [loss, predictions, accuracy, merged]
        feed_dict = {self.batch_in: batch_in, self.batch_out: batch_out, self.is_dropout: False}
        _loss, _predictions, _accuracy, _summary = self._session.run(ops, feed_dict)

        if is_valid:
            self._valid_writer.add_summary(_summary, step)

        return _predictions, _loss, _accuracy

    def _get_graph(self):
        if self._graph is None:
            self._graph = self.network()
        return self._graph

    def _load_vars(self, variable_path):
        if self._session is None:
            try:
                self.load_variables(variable_path)
            except:
                raise RuntimeError('Session unitialized and no variables saved at provided path %s' % variable_path)


if __name__ == "__main__":

    import time
    import random
    from preprocessing import get_training_data

    # Prepare training data
    [x_train, x_valid, x_test, y_train, y_valid, y_test] = get_training_data(label_type='chord_symbol')
    n_sequences_train = x_train.shape[0]

    # create model
    tf.reset_default_graph()
    network = STL_BLSTM_RNNModel(feature_size=61,
                                 n_steps=256,
                                 n_hidden_units=1024,
                                 learning_rate=1e-4,
                                 L2_beta=1e-3,
                                 dropout_rate=0.5,
                                 use_crf=False,
                                 task='chord_symbol')

    variable_path = path + "\\Training\\training_model_ckpt"
    best_variable_path = path + "\\Training\\best_training_model_ckpt"
    n_epoches = 27 # number of training epochs
    bsize = 36 # batch size
    best_valid_acc, in_succession = 0.0, 0 # log for early stopping
    n_in_succession = 8 # number of accuracy drops before early stopping

    startTime = time.time()
    print('\nStart training......')
    for epoch in range(n_epoches):

        # shuffle training set
        new_order = random.sample(range(n_sequences_train), n_sequences_train)
        batches_indices = [new_order[x:x + bsize] for x in range(0, len(new_order), bsize)]

        for batch, indices in enumerate(batches_indices):
            save = False if batch != (len(batches_indices) -1) else True
            network.train(x_train[indices], y_train[indices], save_vars_to_disk=save, variable_path=variable_path, step=epoch * len(batches_indices) + batch)

        # validation
        valid_pred, valid_loss, valid_acc = network.predict(batch_in=x_valid, batch_out=y_valid, variable_path=variable_path, step=(epoch+1) * len(batches_indices), is_valid=True)
        print("======== epoch: %d  valid_loss = %4f, valid_accuracy = %.4f ========" % (epoch + 1, valid_loss, valid_acc))

        # prediction result
        sample_index = random.randint(0, y_valid.shape[0] - 1)
        print('Label'.ljust(10, ' '), ''.join([str(b).rjust(3, ' ') for b in y_valid[sample_index, :]]))
        print('Prediction'.ljust(10, ' '), ''.join([str(b).rjust(3, ' ') for b in valid_pred[sample_index, :]]))

        # check if early stop
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            in_succession = 0
            network.save_variables(best_variable_path, to_print=True)
        else:
            in_succession += 1
            if in_succession > n_in_succession:
                break
    elapsed_time = time.time() - startTime
    print('training time = %.2f hr' % (elapsed_time / 3600))

    # testing
    startTime = time.time()
    print('\nStart Testing......')
    network._session.close() # closs session
    network._session = None # force predict() to load variables from best_variable_path
    test_pred, test_loss, test_acc = network.predict(batch_in=x_test, batch_out=y_test, variable_path=best_variable_path, is_valid=False)
    print('testing accuracy = %.4f' % test_acc)

    elapsed_time = time.time() - startTime
    print('testing time = %.2f min\n' % (elapsed_time / 60))

