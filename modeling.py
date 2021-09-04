



import tensorflow as tf
from tqdm import tqdm
from util import predict_on_batch, read_pkl
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import keras
from keras.preprocessing.sequence import pad_sequences

 
import torch.optim as optim
import torch.nn as nn
import torch
 

class Modeling: 
    def __init__(self, model, saver_path, BATCHSIZE, test_x_test, test_t_test, test_g_test):
        # init model
        self.model = model
        self.saver_path = saver_path
        self.batchsize = BATCHSIZE
        self.test_x_test = test_x_test
        self.test_t_test = test_t_test
        self.test_g_test = test_g_test
        self.alpha = model.alpha
        self.ME_lr = model.ME_lr
        self.warm_lr = model.warm_lr

        # loss function setting
        self.log_loss = nn.BCELoss()
        #self.log_loss = nn.BCEWithLogitsLoss()
        # opt func setting
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.warm_lr)

 
    def pre_train_base_model(self, train_x, train_g, train_y, train_t, test_y_test):
        """
        Pre-train the base model
        """
        n_samples = train_x.shape[0]
        n_batch = n_samples//self.batchsize
        for _ in range(1):
            for i_batch in tqdm(range(n_batch)):
                batch_x = train_x.iloc[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_t = train_t[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_g = train_g[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_y = train_y.iloc[i_batch*self.batchsize:(i_batch+1)*self.batchsize].values
                y_hat = self.model(batch_x, batch_t, batch_g, meta_ID_emb=None ,warm_or_cold='warm')
                # calculate loss 
                batch_y_tensor = torch.tensor(batch_y).view(-1,1).type(torch.FloatTensor)
                batch_loss = self.log_loss(y_hat, batch_y_tensor)
                # opt loss
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
        # test eval  
        test_pred_test = self.model(self.test_x_test, self.test_t_test, self.test_g_test, meta_ID_emb=None, warm_or_cold='warm')
        test_y_test_tensor = torch.tensor(test_y_test).view(-1,1).type(torch.FloatTensor)
        logloss_base_cold = test_loss_test = self.log_loss(test_pred_test, test_y_test_tensor)
        print("[pre-train]\n\ttest-test loss: {:.6f}".format(test_loss_test))
        test_pred_test_array = test_pred_test.detach().numpy()
        auc_base_cold = test_auc_test = roc_auc_score(test_y_test, test_pred_test_array)
        print("[pre-train]\n\ttest-test auc: {:.6f}".format(test_auc_test))
        print("COLD-START BASELINE:")
        print("\t Loss: {:.4f}".format(logloss_base_cold))
        print("\t AUC: {:.4f}".format(auc_base_cold))
        return logloss_base_cold, auc_base_cold


    def train_meta_embedding_module(self, n_epoch, test_y_test, ID_col, item_col,context_col):
        '''
        Train the Meta-Embedding generator
        n_epoch, test_y_test
        '''
        best_auc = 0
        best_loss = 10
        for i_epoch in range(n_epoch):
            # Read the few-shot training data of big ads
            if i_epoch==0:
                _train_a = read_pkl("data/train_oneshot_a.pkl")
                _train_b = read_pkl("data/train_oneshot_b.pkl")
            elif i_epoch==1:
                _train_a = read_pkl("data/train_oneshot_c.pkl")
                _train_b = read_pkl("data/train_oneshot_d.pkl")
            elif i_epoch==2:
                _train_a = read_pkl("data/train_oneshot_b.pkl")
                _train_b = read_pkl("data/train_oneshot_c.pkl")
            elif i_epoch==3:
                _train_a = read_pkl("data/train_oneshot_d.pkl")
                _train_b = read_pkl("data/train_oneshot_a.pkl")
            train_x_a = _train_a[[ID_col]+item_col+context_col]
            train_y_a = _train_a['y'].values
            train_t_a = pad_sequences(_train_a.Title, maxlen=8)
            train_g_a = pad_sequences(_train_a.Genres, maxlen=4)

            train_x_b = _train_b[[ID_col]+item_col+context_col]
            train_y_b = _train_b['y'].values
            train_t_b = pad_sequences(_train_b.Title, maxlen=8)
            train_g_b = pad_sequences(_train_b.Genres, maxlen=4)
            
            n_samples = train_x_a.shape[0]
            n_batch = n_samples//self.batchsize
            # Start training 
            for i_batch in tqdm(range(n_batch)):
                batch_x_a = train_x_a.iloc[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_t_a = train_t_a[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_g_a = train_g_a[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_y_a = train_y_a[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_x_b = train_x_b.iloc[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_t_b = train_t_b[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_g_b = train_g_b[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                batch_y_b = train_y_b[i_batch*self.batchsize:(i_batch+1)*self.batchsize]
                # first term 
                y_hat_first = self.model(batch_x_a, batch_t_a, batch_g_a, meta_ID_emb=None, warm_or_cold='cold')
                meta_ID_emb = self.model.meta_ID_emb
                # calculate loss (1)
                batch_y_a_tensor = torch.tensor(batch_y_a).view(-1,1).type(torch.FloatTensor)
                batch_loss_a = self.log_loss(y_hat_first, batch_y_a_tensor)
                #batch_loss_a.requires_grad_(True)
                # second term
                y_hat = self.model(batch_x_b, batch_t_b, batch_g_b, cold_loss_a=batch_loss_a, meta_ID_emb=meta_ID_emb, warm_or_cold='cold')
                # calculate loss (2)
                batch_y_b_tensor = torch.tensor(batch_y_b).view(-1,1).type(torch.FloatTensor)
                batch_loss_b = self.log_loss(y_hat, batch_y_b_tensor)
                # opt loss
                ME_loss = batch_loss_a * self.alpha + batch_loss_b * (1-self.alpha)
                self.optimizer.zero_grad()
                ME_loss.backward()
                self.optimizer.step()
            # on epoch end
            test_pred_test = self.model(self.test_x_test, self.test_t_test, self.test_g_test, warm_or_cold='cold')
            test_y_test_tensor = torch.tensor(test_y_test).view(-1,1).type(torch.FloatTensor)
            logloss_ME_cold = test_loss_test = self.log_loss(test_pred_test, test_y_test_tensor)
            print("[Meta-Embedding]\n\ttest-test loss: {:.6f}".format(test_loss_test))
            test_pred_test_array = test_pred_test.detach().numpy()
            auc_ME_cold = test_auc_test = roc_auc_score(test_y_test, test_pred_test_array)
            print("[Meta-Embedding]\n\ttest-test auc: {:.6f}".format(test_auc_test))
        save_path = None
        return save_path


