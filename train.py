import tensorflow as tf
import numpy as np 
import configs.configs_ML100K as configs
from model import NMC
from data_loader import DataLoader
import time 

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./data/MovieLens100K/", "Data directory.")
tf.flags.DEFINE_string("output_basedir", "./outputs/", "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("pretrained_fname", "", "Name of the pretrained model checkpoints (to resume from)")
tf.flags.DEFINE_string("output_dir", "", "Model output directory.")
FLAGS.output_dir = FLAGS.output_basedir + 'snapshots/snapshot'

cfgs = configs.CONFIGS

def main(unused_argv):
    assert FLAGS.output_dir, "--output_dir is required"
    # Create training directory.
    output_dir = FLAGS.output_dir
    if not tf.gfile.IsDirectory(output_dir):
        tf.logging.info("Creating training directory: %s", output_dir)
        tf.gfile.MakeDirs(output_dir)

    dl = DataLoader(FLAGS.data_dir)
    dl.load_data()
    dl.split()

    x_dim = dl.get_X_dim()
    y_dim = dl.get_Y_dim()

    # Build the model.
    model = NMC(x_dim, y_dim, cfgs)

    if FLAGS.pretrained_fname:
        try:
            print('Resume from %s' %(FLAGS.pretrained_fname))
            model.restore(FLAGS.pretrained_fname)
        except:
            print('Cannot resume model... Training from scratch')
    
    lr = cfgs.initial_lr
    epoch_counter = 0
    ite = 0
    while True:
        start = time.time()
        x, y, R, mask, flag = dl.next_batch(cfgs.batch_size_x, cfgs.batch_size_y, 'train')
        if np.sum(mask) == 0:
            continue

        load_data_time = time.time() - start
        if flag: 
            epoch_counter += 1
        
        # some boolean variables    
        do_log = (ite % cfgs.log_every_n_steps == 0) or flag
        do_snapshot = flag and epoch_counter > 0 and epoch_counter % cfgs.save_every_n_epochs == 0

        # train one step
        start = time.time()
        loss, recons, ite = model.partial_fit(x, y, R, mask, lr)
        one_iter_time = time.time() - start
        
        # writing outs
        if do_log:
            print('Iteration %d, (lr=%f) training loss  : %f' %(ite, lr, loss))

        if do_snapshot:
            print('Snapshotting')
            model.save(FLAGS.output_dir)
        
        if flag: 
            # decay learning rate during training
            if epoch_counter % cfgs.num_epochs_per_decay == 0:
                lr = lr * cfgs.lr_decay_factor
                print('Decay learning rate to %f' %lr)
            if epoch_counter == FLAGS.n_epochs:
                if not do_snapshot:
                    print('Final snapshotting')
                    model.save(FLAGS.output_dir)
                break

if __name__ == '__main__':
    tf.app.run()


