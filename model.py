
'''
self.model
-> train_warm : input trainX, trainY and then opt (old)
-> pred_warm : input X, output Y_hat (old)
-> train_ME : input trainX, trainY and then opt (new)
-> pred_ME : ...
-> get_meta_embedding : ??
-> assign_meta_embedding : ??
'''

 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Meta_Model(nn.Module):
    def __init__(self, ID_col, item_col, context_col, nb_words, model='FM',
                 emb_size=128, alpha=0.1,
                 warm_lr=1e-3, cold_lr=1e-4, ME_lr=1e-3):
        super(Meta_Model, self).__init__()
        """
        ID_col: string, the column name of the item ID
        item_col: list, the columns of item features
        context_col: list, the columns of other features
        nb_words: dict, nb of words in each of these columns
        """
        self.columns = [ID_col] + item_col + context_col
        self.ID_col = ID_col
        self.item_col = item_col
        self.cold_lr = cold_lr
        self.alpha = alpha
        self.warm_lr = warm_lr
        self.ME_lr = ME_lr
        self.emb_size = emb_size

        '''
        *CHOOSE THE BASE MODEL HERE*
        '''
        self.get_yhat = {
            "PNN": self.get_yhat_PNN, 
            "deepFM": self.get_yhat_deepFM
        }[model]

        # lookup embedding
        self.column2lookup_embedding_layer = dict()
        for col in self.columns:
            lookup_embedding_layer = nn.Embedding(nb_words[col], emb_size)#.to(device)
            lookup_embedding_layer.weight.data.normal_(0, 0.01)
            self.column2lookup_embedding_layer[col] = lookup_embedding_layer
        self.title_lookup_embedding_layer = nn.Embedding(20001, emb_size)#.to(device)
        self.genres_lookup_embedding_layer = nn.Embedding(21, emb_size)#.to(device)
 
        # layer
        # self.emb_pred_Dense = nn.Parameter(torch.rand((len(item_col)+ 2)*emb_size,emb_size), requires_grad=True).type(torch.FloatTensor)
        # self.register_parameter('emb_predictor' , self.emb_pred_Dense)
        self.emb_pred_Dense = nn.Linear((len(item_col)+ 2)*emb_size, emb_size)
 
        feature_num = len(self.columns) + 2
        #self.deep0_dense_layer = nn.Parameter(torch.rand(feature_num*emb_size,feature_num*emb_size), requires_grad=True).type(torch.FloatTensor)
        #self.register_parameter('deep-0' , self.deep0_dense_layer)        
        #self.deep1_dense_layer = nn.Parameter(torch.rand(feature_num*emb_size,feature_num*emb_size), requires_grad=True).type(torch.FloatTensor)
        #self.register_parameter('deep-1' , self.deep1_dense_layer)   

        self.deep0_dense_layer = nn.Linear(feature_num*emb_size, feature_num*emb_size)
        self.deep1_dense_layer = nn.Linear(feature_num*emb_size, feature_num*emb_size)

        #self.out_dense_layer = nn.Parameter(torch.rand((feature_num*emb_size)+len(self.columns),1), requires_grad=True).type(torch.FloatTensor)
        #self.register_parameter('out' , self.out_dense_layer)   
        
        self.out_dense_layer = nn.Linear((feature_num*emb_size)+feature_num, 1)

        # activation layer
        self.relu_layer = nn.ReLU()
        self.sigmoid_layer = nn.Sigmoid()
 

    def get_yhat_deepFM(self, ID_emb, item_embs, other_embs, **kwargs):
        embeddings = [ID_emb] + item_embs + other_embs
        embeddings_cat = torch.cat([emb.view(-1,1,self.emb_size) for emb in embeddings], 1) #torch.Size([200, 8, 128])
        #print('embeddings_cat : ',embeddings_cat.shape)
        sum_of_emb = torch.mean(embeddings_cat, 1) #torch.Size([200, 128])
        #print('sum_of_emb : ',sum_of_emb.shape)
        diff_of_emb = [sum_of_emb - x for x in embeddings]
        dot_of_emb = [torch.sum(embeddings[i]*diff_of_emb[i], axis=1).view(-1,1) for i in range(len(embeddings))]
        h = torch.cat(dot_of_emb, 1)  #torch.Size([200, 6])
        h2 = torch.cat(embeddings, 1) #torch.Size([200, 1024])
        h2 = self.relu_layer(self.deep0_dense_layer(h2)) #torch.Size([1024, 1024]) | torch.Size([200, 1024])
        h2 = self.relu_layer(self.deep1_dense_layer(h2)) #torch.Size([1024, 1024]) | torch.Size([200, 1024])
        h = torch.cat([h,h2], 1) #torch.Size([200, 1030])
        #y_hat = self.sigmoid_layer(h.mm(self.out_dense_layer)) #torch.Size([1030, 1]) | torch.Size([200, 1])
        y_hat = self.sigmoid_layer(self.out_dense_layer(h)) #torch.Size([1030, 1]) | torch.Size([200, 1])
        return y_hat
 

    def get_yhat_PNN(self):
        y_hat = None
        return y_hat
    

    def get_embeddings(self, batch_x, batch_t, batch_g):
        item_embs, other_embs = [], []
        for col in self.columns:
            lookup_embedding_layer = self.column2lookup_embedding_layer[col]
            input_tensor = torch.tensor(list(batch_x[col])).long()
            if col==self.ID_col:
                ID_emb = lookup_embedding_layer(input_tensor)
            elif col in self.item_col:
                item_embs.append(lookup_embedding_layer(input_tensor))
            else:
                other_embs.append(lookup_embedding_layer(input_tensor))
        batch_t_tensor = torch.tensor(batch_t).long()
        batch_g_tensor = torch.tensor(batch_g).long()
        title_emb = self.title_lookup_embedding_layer(batch_t_tensor)
        genre_emb = self.genres_lookup_embedding_layer(batch_g_tensor)
        item_embs.append(torch.mean(title_emb, axis=1))
        item_embs.append(torch.mean(genre_emb, axis=1))
        return ID_emb, item_embs, other_embs


    def generate_meta_emb(self, item_embs): 
        """
        This is the simplest architecture of the embedding generator,
        with only a dense layer.
        You can customize it if you want have a stronger performance, 
        for example, you can add an l2 regularization term or alter 
        the pooling layer. 
        """
        embs = torch.stack(item_embs, 1)
        item_h = torch.flatten(embs,1)
        #emb_pred = item_h.mm(self.emb_pred_Dense) / 5.
        emb_pred = self.emb_pred_Dense(item_h) / 5.
        return emb_pred


    def forward(self, batch_x, batch_t, batch_g, cold_loss_a=None,meta_ID_emb=None, warm_or_cold=str):
        # get lookup embedding
        ID_emb, item_embs, other_embs = self.get_embeddings(batch_x, batch_t, batch_g)
        # main model
        if warm_or_cold == 'warm':
            y_hat = self.get_yhat(ID_emb, item_embs, other_embs)
            return y_hat
        elif warm_or_cold == 'cold':
            # Meta-Embedding: step 1, cold-start, 
            #     use the generated meta-embedding to make predictions
            #     and calculate the cold-start loss_a
            if meta_ID_emb is None:
                meta_ID_emb = self.generate_meta_emb(item_embs)
                self.meta_ID_emb = meta_ID_emb
                cold_yhat_a = self.get_yhat(meta_ID_emb, item_embs, other_embs)
                return cold_yhat_a
            else:
                # Meta-Embedding: step 2, apply gradient descent once
                #     get the adapted embedding
                #cold_emb_grads = tf.gradients(cold_loss_a, meta_ID_emb)[0]
                cold_emb_grads = torch.autograd.grad(cold_loss_a, meta_ID_emb,retain_graph=True)[0]
                meta_ID_emb_new = meta_ID_emb - self.cold_lr * cold_emb_grads
                # Meta-Embedding: step 3, 
                #     use the adapted embedding to make prediction on another mini-batch 
                #     and calculate the warm-up loss_b
                cold_yhat_b = self.get_yhat(meta_ID_emb_new, item_embs, other_embs)
                return cold_yhat_b




