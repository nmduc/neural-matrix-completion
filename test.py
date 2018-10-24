import tensorflow as tf
import numpy as np 
import time 
import scipy.sparse
import configs.configs_ML100K as configs
from model import NMC

FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "./data/MovieLens100K/", "Data directory.")
tf.flags.DEFINE_string("snapshot_dir", "./outputs/snapshots/", "Directory containing trained models.")
cfgs = configs.CONFIGS
embed_dim = int(cfgs.u_hidden_sizes.strip().split(',')[-1])

def normalize_minus_plus_one(data, min_val, max_val):
    mid = (max_val + min_val) / 2
    data = (data - mid) / (mid - min_val)
    return data

def renormalize_minus_plus_one(data, min_val, max_val):
    mid = (max_val + min_val) / 2
    data = data * (mid - min_val) + mid
    return data

def embed_x(model, X, dim, min_val, max_val, bs=1000):
    n_samples = X.shape[0]
    fv = np.zeros((n_samples, dim))
    start = 0
    while True:
        end = start + bs
        if end > n_samples:
            end = n_samples
        X_batch = X[start:end,:]
        fv[start:end,:] = model.embed_x(X_batch)
        if end == n_samples:
            break
        start = end
    return fv

def embed_y(model, Y, dim, min_val, max_val, bs=1000):
    n_samples = Y.shape[0]
    fv = np.zeros((n_samples, dim))
    start = 0
    while True:
        end = start + bs
        if end > n_samples:
            end = n_samples
        Y_batch = Y[start:end,:]
        fv[start:end,:] = model.embed_y(Y_batch)
        if end == n_samples:
            break
        start = end
    return fv

def reconstruct_cosine(latent_x, latent_y):
    l2_norm_lx = latent_x / np.linalg.norm(latent_x, axis=1, keepdims=True)
    l2_norm_ly = latent_y / np.linalg.norm(latent_y, axis=1, keepdims=True)
    recons = np.matmul(l2_norm_lx, l2_norm_ly.T)
    return recons

def RMSE_MAE(recon, ref, mask):
    ind1, ind2 = np.nonzero(ref.multiply(mask))
    ref_values = ref[ind1, ind2]
    values = recon[ind1, ind2]
    sum_sqr_diff = np.sum(np.square(values - ref_values))
    sum_abs_diff = np.sum(np.abs(values - ref_values))
    n_elements = len(ind1)
    rmse = np.sqrt(sum_sqr_diff / n_elements)
    mae = sum_abs_diff / n_elements
    return rmse, mae

def prepare_data(R, tr_mask, te_mask):
    X = R.multiply(tr_mask).todense()
    R_ = R.copy()
    return X, R_, tr_mask, te_mask

def main(unused_argv):
    # load data
    R = scipy.sparse.load_npz(FLAGS.data_dir + 'rating.npz')
    val_set = np.unique(R.data)
    min_val = float(val_set[0]) 
    max_val = float(val_set[-1])
    tr_mask = scipy.sparse.load_npz(FLAGS.data_dir + 'train_mask.npz')
    val_mask = scipy.sparse.load_npz(FLAGS.data_dir + 'val_mask.npz')
    te_mask = scipy.sparse.load_npz(FLAGS.data_dir + 'test_mask.npz')
    print('Finished loading data')
    count = np.sum((tr_mask + val_mask).multiply(te_mask))
    assert count == 0, 'Train and test overlap !!!'

    X, R_, tr_mask, te_mask = prepare_data(R, tr_mask, te_mask)
    print('Finished preparing data')

    # load model 
    model = NMC(X.shape[1], X.shape[0], cfgs, phase='test')

    snapshot_fname = tf.train.latest_checkpoint(FLAGS.snapshot_dir)
    assert snapshot_fname != None, 'No model found'
    model.restore(snapshot_fname)
    print('Restored from %s' %snapshot_fname)

    lX = embed_x(model, X, embed_dim, min_val, max_val, bs=1000)
    print('Finished embedding the rows')
    lY = embed_y(model, X.T, embed_dim, min_val, max_val, bs=1000) 
    print('Finished embedding the columns')
    recons = reconstruct_cosine(lX, lY)
    recons = renormalize_minus_plus_one(recons, min_val, max_val)
    print('Finished completion')
    rmse_tr, mae_tr = RMSE_MAE(recons, R_, tr_mask)
    rmse_te, mae_te = RMSE_MAE(recons, R_, te_mask)
    
    print('-------------RESULT-------------')
    print('Training')
    print('RMSE - MAE : %f - %f' %(rmse_tr, mae_tr))
    print('Testing')
    print('RMSE - MAE : %f - %f' %(rmse_te, mae_te))    
    print('--------------------------------')
    
if __name__ == '__main__':
    tf.app.run()
