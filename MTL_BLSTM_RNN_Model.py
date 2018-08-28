import os
path = os.path.dirname(os.path.abspath(__file__))

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper

n_output_classes = {'key': 24,
                    'pri_deg': 21,
                    'sec_deg': 21,
                    'quality': 10,
                    'inversion': 4}

class MTL_BLSTM_RNNModel(object):
    def __init__(self,
                 feature_size=1952,
                 n_steps=64,
                 n_hidden_units=1024,
                 learning_rate=1e-4,
                 L2_beta=1e-4,
                 dropout_rate=0.5):

        self._feature_size = feature_size
        self._n_steps = n_steps
        self._hidden_size = n_hidden_units
        self._total_classes = sum(n_output_classes.values())

        self._session = None
        self._graph = None

        # Summary for training visualization
        self._graph_path = path + "\\Training"
        self._train_writer = None
        self._valid_writer = None

        self._L2_bata = L2_beta
        self._dropout_rate = dropout_rate
        self._learning_rate = learning_rate

        batch_in_shape = [None, self._n_steps, self._feature_size]
        batch_out_shape = [None, self._n_steps]
        self.batch_in = tf.placeholder(tf.float32, shape=batch_in_shape, name='batch_in')
        self.batch_out = {}
        for key in n_output_classes.keys():
            self.batch_out[key] = tf.placeholder(tf.int32, shape=batch_out_shape, name='batch_out_'+key)

        self.is_dropout = tf.placeholder(tf.bool)

        self._optimizer = tf.train.AdamOptimizer(self._learning_rate)

    def load_variables(self, path='./mtl_blstm_rnn_ckpt'):
        if self._session is None:
            self._session = tf.Session()
            saver = tf.train.Saver()
            print('loading variables...')
            saver.restore(self._session, path)

    def save_variables(self, path='./mtl_blstm_rnn_ckpt', to_print=True):
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
            keep_prob = tf.cond(self.is_dropout, lambda: 1 - self._dropout_rate, lambda: 1.0)
            encoder_cell_fw = DropoutWrapper(encoder_cell_fw, output_keep_prob=keep_prob)
            encoder_cell_bw = DropoutWrapper(encoder_cell_bw, output_keep_prob=keep_prob)

        with tf.name_scope('LSTM_layer'):
            # bi-LSTM
            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
                                                                                           encoder_cell_bw,
                                                                                           self.batch_in,
                                                                                           time_major=False,
                                                                                           dtype=tf.float32)
            rnn_outputs = tf.concat([output_fw, output_bw], axis=-1) # shape = [batch, n_steps, 2*n_hiddne_units]

        with tf.name_scope('Output_projection_layer'):
            logits = tf.layers.dense(rnn_outputs, self._total_classes) # shape = [batch, n_steps, total_classes]
            # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) # shape = [batch, n_steps]

        with tf.name_scope('Task_specific_layer'):
            logits_key = logits[:,:,:24]
            logits_pri_deg = logits[:, :, 24:45]
            logits_sec_deg = logits[:, :, 45:66]
            logits_quality = logits[:, :, 66:76]
            logits_inversion = logits[:, :, 76:]
            logits = {'key': logits_key,
                      'pri_deg': logits_pri_deg,
                      'sec_deg': logits_sec_deg,
                      'quality': logits_quality,
                      'inversion': logits_inversion}

            predictions = {}
            for key in logits.keys():
                predictions[key] = tf.argmax(logits[key], axis=-1, output_type=tf.int32)

        with tf.name_scope('loss'):
            # cross entropy
            cross_entropy = []
            for key in n_output_classes.keys():
                cross_entropy.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[key], labels=self.batch_out[key])))
            cross_entropy = tf.reduce_sum(cross_entropy)

            # L2 norm regularization
            vars = tf.trainable_variables()
            L2_regularizer = self._L2_bata * tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name])

            # loss
            loss = cross_entropy + L2_regularizer
        tf.summary.scalar('Loss', loss)

        with tf.name_scope('accuracy'):
            correct_predictions = {}
            for key in predictions.keys():
                correct_predictions[key] = tf.equal(predictions[key], self.batch_out[key])
            """keys in correct_predictions: key, pri_deg, sec_deg, quality, inversion"""

            correct_predictions['overall'] = tf.stack(list(correct_predictions.values()), axis=2)
            correct_predictions['overall'] = tf.reduce_all(correct_predictions['overall'], axis=2)
            """keys in correct_predictions: key, pri_deg, sec_deg, quality, inversion, overall"""

            correct_predictions['degree'] = tf.stack([correct_predictions['pri_deg'], correct_predictions['sec_deg']], axis=2)
            correct_predictions['degree'] = tf.reduce_all(correct_predictions['degree'], axis=2)
            """keys in correct_predictions: key, pri_deg, sec_deg, quality, inversion, overall, degree"""

            accuracy = {}
            for key in [k for k in correct_predictions.keys() if k != 'pri_deg']:
                accuracy[key] = tf.reduce_mean(tf.cast(correct_predictions[key], tf.float32))
                tf.summary.scalar(key, accuracy[key])
            """keys in accuracy: key, sec_deg, quality, inversion, overall, degree"""

        merged = tf.summary.merge_all() # merge all summaries collected in the graph

        # Apply gradient clipping
        gvs = self._optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train = self._optimizer.apply_gradients(capped_gvs)

        return train, loss, logits, predictions, accuracy, merged, L2_regularizer

    def train(self,
              batch_in,
              batch_out,
              variable_path='./mtl_blstm_rnn_ckpt',
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
        feed_dict = {self.batch_in: batch_in,
                     self.batch_out['key']: batch_out['key'],
                     self.batch_out['pri_deg']: batch_out['pri_deg'],
                     self.batch_out['sec_deg']: batch_out['sec_deg'],
                     self.batch_out['quality']: batch_out['quality'],
                     self.batch_out['inversion']: batch_out['inversion'],
                     self.is_dropout: True}
        _, _loss, _predictions, _summary, _accuracy, _L2 = self._session.run(ops, feed_dict)

        if step == 0:
            print('CE = %.4f,  L2 = %.4f' % (_loss - _L2, _L2))

        if step % 200 == 0:
            self._train_writer.add_summary(_summary, step) # add to training log
            print("*------ iteration %d:  train_loss %.4f, train_accuracy %.4f, %.4f, %.4f, %.4f, %.4f, %.4f ------*" % (step,
                                                                                                           _loss,
                                                                                                           _accuracy['key'],
                                                                                                           _accuracy['degree'],
                                                                                                           _accuracy['sec_deg'],
                                                                                                           _accuracy['quality'],
                                                                                                           _accuracy['inversion'],
                                                                                                           _accuracy['overall']))

            for key in _predictions.keys():
                print('$'+key+'$')
                print(' Label'.ljust(11, ' '), ''.join([str(b).rjust(3, ' ') for b in batch_out[0, :][key]]))
                print(' Prediction'.ljust(11, ' '), ''.join([str(b).rjust(3, ' ') for b in _predictions[key][0, :]]))

        if save_vars_to_disk:
            self.save_variables(variable_path, to_print=True)


    def predict(self, batch_in, batch_out, variable_path='./mtl_blstm_rnn_ckpt', step=0, is_valid=False):
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
        feed_dict = {self.batch_in: batch_in,
                     self.batch_out['key']: batch_out['key'],
                     self.batch_out['pri_deg']: batch_out['pri_deg'],
                     self.batch_out['sec_deg']: batch_out['sec_deg'],
                     self.batch_out['quality']: batch_out['quality'],
                     self.batch_out['inversion']: batch_out['inversion'],
                     self.is_dropout: False}
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
    import numpy as np
    # from preprocessing import get_training_data

    # Prepare training data
    [x_train, x_valid, x_test, y_train, y_valid, y_test] = get_training_data(label_type='chord_function')
    n_sequences_train = x_train.shape[0]

    # create model
    tf.reset_default_graph()
    network = MTL_BLSTM_RNNModel(feature_size=1952,
                                 n_steps=64,
                                 n_hidden_units=1024,
                                 learning_rate=1e-4,
                                 L2_beta=1e-3,
                                 dropout_rate=0.5)

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
        print("======== epoch: %d  valid_loss = %4f, valid_accuracy = %.4f, %.4f, %.4f, %.4f, %.4f %.4f ========" % (epoch + 1,
                                                                                                                valid_loss,
                                                                                                                valid_acc['key'],
                                                                                                                valid_acc['degree'],
                                                                                                                valid_acc['sec_deg'],
                                                                                                                valid_acc['quality'],
                                                                                                                valid_acc['inversion'],
                                                                                                                valid_acc['overall']))

        # prediction result
        sample_index = random.randint(0, y_valid.shape[0] - 1)
        for key in valid_pred.keys():
            print('$'+key+'$')
            print(' Label'.ljust(11, ' '), ''.join([str(b).rjust(3, ' ') for b in y_valid[sample_index, :][key]]))
            print(' Prediction'.ljust(11, ' '), ''.join([str(b).rjust(3, ' ') for b in valid_pred[key][sample_index, :]]))


        # check if early stop
        if valid_acc['overall'] > best_valid_acc:
            best_valid_acc = valid_acc['overall']
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
    print('testing accuracy = %.4f, %.4f, %.4f,%.4f ,%.4f ,%.4f' % (test_acc['key'],
                                                             test_acc['degree'],
                                                             test_acc['sec_deg'],
                                                             test_acc['quality'],
                                                             test_acc['inversion'],
                                                             test_acc['overall']))

    elapsed_time = time.time() - startTime
    print('testing time = %.2f min\n' % (elapsed_time / 60))
