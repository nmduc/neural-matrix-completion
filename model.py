import tensorflow as tf
from tensorflow.python import debug as tf_debug
import tensorflow.contrib.slim as slim
import tensorflow.contrib.losses as L
import tensorflow.contrib.keras as K
import numpy as np 

class NMC():
    def __init__(self, x_dim, y_dim, cfg, phase='train'):
        ''' Initialize network
        Inputs:
            - x_dim: (int) dimension of rows (number of column - M)
            - y_dim: (int) dimension of columns (number of row - N)
            - cfg: configurations
        '''
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.parse_model_configs(cfg)

        # define placeholders
        if phase == 'train':
            self.x = tf.placeholder(tf.float32, [self.batch_size_x, self.x_dim])
            self.y = tf.placeholder(tf.float32, [self.batch_size_y, self.y_dim])
            self.R = tf.placeholder(tf.float32, [self.batch_size_x, self.batch_size_y])
            self.mask = tf.placeholder(tf.float32, [self.batch_size_x, self.batch_size_y])
        elif phase == 'test':
            # to feed data with random batch size
            self.x = tf.placeholder(tf.float32, [None, self.x_dim])
            self.y = tf.placeholder(tf.float32, [None, self.y_dim])
            self.R = tf.placeholder(tf.float32, [None, None])
            self.mask = tf.placeholder(tf.float32, [None, None])
        else:
            assert False, 'Invalid phase'

        self.lr = tf.placeholder(tf.float32, [])    # learning rate
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')
        
        # initializer
        self.initializer = tf.contrib.layers.xavier_initializer 

        # encoder 
        with tf.variable_scope('user'):
            self.inp_x = self.x
            # add summarization layers (if specified in the configs)
            if self.summarization:
                self.inp_x = self.build_summary_conv(self.x, self.x_dim, 
                    self.u_summ_layer_sizes, self.n_u_summ_filters, 'summary')
            self.latent_x = self.build_embedder(self.inp_x, self.u_hidden_sizes, 'user')
        with tf.variable_scope('movie'):
            self.inp_y = self.y
            # add summarization layers (if specified in the configs)
            if self.summarization:
                self.inp_y = self.build_summary_conv(self.y, self.inp_y, 
                    self.v_summ_layer_sizes, self.n_v_summ_filters, 'summary')
            self.latent_y = self.build_embedder(self.inp_y, self.v_hidden_sizes, 'movie')

        self.recons = self.build_recon_cosine(self.latent_x, self.latent_y)

        # loss functions
        self.recon_loss = self.build_recon_loss(self.R, self.recons, self.mask)
        tf.contrib.losses.add_loss(self.recon_loss)
        self.total_loss = tf.contrib.losses.get_total_loss(add_regularization_losses=True)

        # build train_opt
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_opt = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
    
        # session
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=sess_config)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=10)
        return None

    def parse_model_configs(self, cfg):
        self.batch_size_x = cfg.batch_size_x
        self.batch_size_y = cfg.batch_size_y
        self.weight_decay = cfg.weight_decay
        self.u_hidden_sizes = self.string_to_array(cfg.u_hidden_sizes, dtype='int')
        self.v_hidden_sizes = self.string_to_array(cfg.v_hidden_sizes, dtype='int')
        self.keep_prob = None
        if cfg.dropout_keep_prob > 0 and cfg.dropout_keep_prob <= 1:
            self.keep_prob = cfg.dropout_keep_prob
        # summary layers
        self.summarization = cfg.summarization
        self.n_u_summ_filters = self.string_to_array(cfg.n_u_summ_filters)
        self.n_v_summ_filters = self.string_to_array(cfg.n_v_summ_filters)
        self.u_summ_layer_sizes = self.string_to_array(cfg.u_summ_layer_sizes)
        self.v_summ_layer_sizes = self.string_to_array(cfg.v_summ_layer_sizes)

        self.latent_dim = self.u_hidden_sizes[-1]
        self.use_bn = cfg.use_bn
        if cfg.activation_fn == 'relu':
            self.transfer = tf.nn.relu
        elif cfg.activation_fn == 'tanh':
            self.transfer = tf.nn.tanh
        elif cfg.activation_fn == 'sigmoid':
            self.transfer = tf.nn.sigmoid
        elif cfg.activation_fn == 'elu':
            self.transfer = tf.nn.elu
        elif cfg.activation_fn == 'relu6':
            self.transfer = tf.nn.relu6
        elif cfg.activation_fn == 'crelu':
            self.transfer = tf.nn.crelu
        else:
            assert False, 'Invalid activation function'


    def build_embedder(self, inp, hidden_sizes, scope):
        latent = inp
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected], 
                weights_initializer=self.initializer(),
                biases_initializer=tf.constant_initializer(0),
                activation_fn=None,
                weights_regularizer=slim.l2_regularizer(self.weight_decay)):              
                for i in range(len(hidden_sizes)):
                    latent = slim.fully_connected(latent, hidden_sizes[i], scope='fc%d'%i)
                    if self.use_bn: # and i == len(hidden_sizes) - 1:
                        latent = self.bn_layer(latent, scope='bn%d'%i)
                    if i < len(hidden_sizes) - 1: 
                        # do not put non-linearity after the last layer (i.e. latent features)
                        latent = self.transfer(latent)
                    if self.keep_prob and i < len(hidden_sizes) - 1:
                        latent = slim.dropout(latent, self.keep_prob, 
                            is_training=self.is_training, scope='dropout%d'%i)
        return latent

    def build_summary_conv(self, inp, inp_dim, summ_sizes, n_filters, scope):
        assert len(n_filters) == len(summ_sizes), 'Invalid'
        with tf.variable_scope(scope):
            output = tf.expand_dims(inp, 2)
            for i in range(len(summ_sizes)):
                output = K.layers.Convolution1D(filters=n_filters[i], kernel_size=summ_sizes[i], strides=summ_sizes[i]/2)(output)
                if self.use_bn:
                    output = self.bn_layer(output, scope='bn_%d'%i)
                output = self.transfer(output)
        output = tf.reshape(output, [-1, output.shape[1].value * n_filters[-1]])
        return output
    
    def build_recon_cosine(self, latent_x, latent_y):
        ''' use cosine similarity as reconstruction (range [-1,1])
        '''
        l2_norm_lx = tf.nn.l2_normalize(latent_x, dim=1)
        l2_norm_ly = tf.nn.l2_normalize(latent_y, dim=1)
        recon = tf.matmul(l2_norm_lx, l2_norm_ly, transpose_b=True) # N x M
        return recon

    def build_recon_loss(self, R, recon, mask):
        diff = R - recon
        sq_diff = tf.square(diff)
        loss = tf.reduce_sum(tf.multiply(mask, sq_diff)) / tf.reduce_sum(mask)
        loss = tf.sqrt(loss)
        return loss


    def partial_fit(self, x, y, R, mask, lr):
        step = self.sess.run(self.global_step)
        loss, recons, _ = self.sess.run([self.total_loss, self.recons, self.train_opt], 
            feed_dict={self.x:x, self.y:y, self.R:R, self.mask:mask, self.lr:lr, self.is_training:True})
        return loss, recons, step

    def embed_x(self, x):
        ''' transform given x to embedding space
        '''
        latent_x = self.sess.run(self.latent_x, feed_dict={self.x:x, self.is_training:False})
        return latent_x

    def embed_y(self, y):
        ''' transform given x and y to embedding space
        '''
        latent_y = self.sess.run(self.latent_y, feed_dict={self.y:y, self.is_training:False})
        return latent_y

    def calc_loss(self, x, y, R, mask):
        loss = self.sess.run(self.total_loss, 
            feed_dict={self.x:x, self.y:y, self.R:R, self.mask:mask, self.is_training:False})
        return loss

    #-----------------------------------------------------------------------------------------
    # Util functions 
    #-----------------------------------------------------------------------------------------
    def bn_layer(self, inputs, scope):
        bn = tf.contrib.layers.batch_norm(inputs, is_training=self.is_training, 
            center=True, fused=False, scale=True, updates_collections=None, decay=0.9, scope=scope)
        return bn

    def save(self, save_path):
        self.saver.save(self.sess, save_path, global_step=self.sess.run(self.global_step))

    def restore(self, save_path):
        self.saver.restore(self.sess, save_path)

    def string_to_array(self, str, dtype='int'):
        arr = str.strip().split(',')
        for i in range(len(arr)):
            if dtype == 'int':
                arr[i] = int(arr[i])
            elif dtype == 'float':
                arr[i] = float(arr[i])
        return arr
