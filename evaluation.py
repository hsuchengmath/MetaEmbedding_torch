

'''
No matter which is based model or meta model, 
original tf version retrain model by test_a,b,c dataset.
Then, eval by test_test dataset
'''
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch



class Evaluation:
    def __init__(self, test_test_data=dict, test_a_data=dict, test_b_data=dict, test_c_data=dict, logloss_base_cold=None, auc_base_cold=None):
        # test data
        self.test_x_test = test_test_data['test_x_test']
        self.test_t_test = test_test_data['test_t_test']
        self.test_g_test = test_test_data['test_g_test']
        self.test_y_test = test_test_data['test_y_test']
        self.test_y_test_tensor = torch.tensor(self.test_y_test).view(-1,1).type(torch.FloatTensor)
        # # val data (a)
        # self.test_x_a = test_a_data['test_x_a']
        # self.test_t_a = test_a_data['test_t_a']
        # self.test_g_a = test_a_data['test_g_a']
        # self.test_y_a = test_a_data['test_y_a']
        # # val data (b)
        # self.test_x_b = test_b_data['test_x_b']
        # self.test_t_b = test_b_data['test_t_b']
        # self.test_g_b = test_b_data['test_g_b']
        # self.test_y_b = test_b_data['test_y_b']
        # # val data (c)
        # self.test_x_c = test_c_data['test_x_c']
        # self.test_t_c = test_c_data['test_t_c']
        # self.test_g_c = test_c_data['test_g_c']
        # self.test_y_c = test_c_data['test_y_c']
        
        # loss function setting
        self.log_loss = nn.BCELoss()

        # parameter
        self.logloss_base_cold = logloss_base_cold
        self.auc_base_cold = auc_base_cold




    def retrain_model(self, model, val_x, val_t, val_g, val_y):
        return model

    def eval_by_test_data(self, model, warm_or_cold=str):
        test_pred_test = model(self.test_x_test, self.test_t_test, self.test_g_test, warm_or_cold=warm_or_cold)
        logloss_base_batcha = test_loss_test = self.log_loss(test_pred_test, self.test_y_test_tensor) 
        print("[baseline]\n\ttest-test loss:\t{:.4f}, improvement: {:.2%}".format(
            test_loss_test, 1-test_loss_test/self.logloss_base_cold))
        test_pred_test_array = test_pred_test.detach().numpy()
        auc_base_batcha = test_auc_test = roc_auc_score(self.test_y_test, test_pred_test_array)
        print("[baseline]\n\ttest-test auc:\t{:.4f}, improvement: {:.2%}".format(
            test_auc_test, test_auc_test/self.auc_base_cold -1))



    def eval_base_model(self, model):
        # load model
        # retrain by val a
        # eval by test data
        # retrain by val b
        # eval by test data
        # retrain by val c
        # eval by test data
        self.eval_by_test_data(model, warm_or_cold='warm')




        
    def eval_meta_learning_model(self, model):
        self.eval_by_test_data(model, warm_or_cold='cold')



