



import pickle
import numpy as np



def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t



def predict_on_batch(sess, predict_func, test_x, test_t, test_g, batchsize=800):
    n_samples_test = test_x.shape[0]
    n_batch_test = n_samples_test//batchsize 
    test_pred = np.zeros(n_samples_test)
    for i_batch in range(n_batch_test):
        batch_x = test_x.iloc[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_t = test_t[i_batch*batchsize:(i_batch+1)*batchsize]
        batch_g = test_g[i_batch*batchsize:(i_batch+1)*batchsize]
        _pred = predict_func(sess, batch_x, batch_t, batch_g)
        test_pred[i_batch*batchsize:(i_batch+1)*batchsize] = _pred.reshape(-1)
    if n_batch_test*batchsize<n_samples_test:
        batch_x = test_x.iloc[n_batch_test*batchsize:]
        batch_t = test_t[n_batch_test*batchsize:]
        batch_g = test_g[n_batch_test*batchsize:]
        _pred = predict_func(sess, batch_x, batch_t, batch_g)
        test_pred[n_batch_test*batchsize:] = _pred.reshape(-1)
    return test_pred