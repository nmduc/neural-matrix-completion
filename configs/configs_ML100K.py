import tensorflow as tf

class ModelConfig():
    u_hidden_sizes = [2048,1024]    # hidden layers's size for the row branch
    v_hidden_sizes = [2048,1024]    # hidden layers's size for the column branch
    dropout_keep_prob = 0.1         # dropout rate = 1.0 - dropout_keep_prob
    use_bn = True                   # use batch normalization after fully-connected layer
    activation_fn = 'relu'          # activation function
    summarization = False           # user summarization layers 
    n_u_summ_filters = [32]         # no. conv filters in summarization layers in the row branch
    n_v_summ_filters = [32]         # no. conv filters in summarization layers in the column branch
    u_summ_layer_sizes = [20]       # conv filter sizes in summarization layers in the row branch
    v_summ_layer_sizes = [10]       # conv filter sizes in summarization layers in the column branch

class TrainConfig(object):
    """Sets the default training hyperparameters."""
    batch_size_x = 100
    batch_size_y = 170              # should be set accordingly to the ratio between row and column of the original matrix 

    initial_lr = 1e-2               # initial learning rate
    lr_decay_factor = 0.65          # learning rate decay factor
    num_epochs_per_decay = 50       # decay learning every ? epochs
    n_epochs = 1000                 # number of training epochs (1 epoch is one round passing through all the rows or columns)
    save_every_n_epochs = 500       # saving model every ? epochs
    log_every_n_steps = 20          # print training log every ? steps

    weight_decay = 0.0              # weight of the l2 regularization 

def arr_to_string(arr):
    for i in range(len(arr)):
        arr[i] = str(arr[i])
    return ','.join(arr)

# model configs
tf.flags.DEFINE_string('u_hidden_sizes', arr_to_string(ModelConfig.u_hidden_sizes),'')
tf.flags.DEFINE_string('v_hidden_sizes', arr_to_string(ModelConfig.v_hidden_sizes),'')
tf.flags.DEFINE_float('dropout_keep_prob', ModelConfig.dropout_keep_prob,'')
tf.flags.DEFINE_boolean('use_bn', ModelConfig.use_bn,'')
tf.flags.DEFINE_string('activation_fn', ModelConfig.activation_fn,'')
tf.flags.DEFINE_boolean('summarization', ModelConfig.summarization,'')
tf.flags.DEFINE_string('n_u_summ_filters', arr_to_string(ModelConfig.n_u_summ_filters),'')
tf.flags.DEFINE_string('n_v_summ_filters', arr_to_string(ModelConfig.n_v_summ_filters),'')
tf.flags.DEFINE_string('u_summ_layer_sizes', arr_to_string(ModelConfig.u_summ_layer_sizes),'')
tf.flags.DEFINE_string('v_summ_layer_sizes', arr_to_string(ModelConfig.v_summ_layer_sizes),'')

# training configs
tf.flags.DEFINE_integer('batch_size_x', TrainConfig.batch_size_x,'')
tf.flags.DEFINE_integer('batch_size_y', TrainConfig.batch_size_y,'')
tf.flags.DEFINE_float('initial_lr', TrainConfig.initial_lr,'')
tf.flags.DEFINE_float('lr_decay_factor', TrainConfig.lr_decay_factor,'')
tf.flags.DEFINE_integer('num_epochs_per_decay', TrainConfig.num_epochs_per_decay,'')
tf.flags.DEFINE_integer('n_epochs', TrainConfig.n_epochs,'')
tf.flags.DEFINE_integer('save_every_n_epochs', TrainConfig.save_every_n_epochs,'')
tf.flags.DEFINE_integer('log_every_n_steps', TrainConfig.log_every_n_steps,'')
tf.flags.DEFINE_float('weight_decay', TrainConfig.weight_decay,'')

CONFIGS = tf.app.flags.FLAGS
