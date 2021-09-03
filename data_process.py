


import pickle
from util import * 
from keras.preprocessing.sequence import pad_sequences

 
# training data of big ads
train = read_pkl("data/big_train_main.pkl")
# some pre-processing
num_words_dict = {
    'MovieID': 4000,
    'UserID': 6050,
    'Age': 7,
    'Gender': 2,
    'Occupation': 21,
    'Year': 83,
}
ID_col = 'MovieID'
item_col = ['Year']
context_col = ['Age', 'Gender', 'Occupation', 'UserID']
train_y = train['y']
train_x = train[[ID_col]+item_col+context_col]
train_t = pad_sequences(train.Title, maxlen=8)
train_g = pad_sequences(train.Genres, maxlen=4)



# few-shot data for the small ads
test_a = read_pkl("data/test_oneshot_a.pkl")
test_b = read_pkl("data/test_oneshot_b.pkl")
test_c = read_pkl("data/test_oneshot_c.pkl")
test_test = read_pkl("data/test_test.pkl")

test_x_a = test_a[[ID_col]+item_col+context_col]
test_y_a = test_a['y'].values
test_t_a = pad_sequences(test_a.Title, maxlen=8)
test_g_a = pad_sequences(test_a.Genres, maxlen=4)

test_x_b = test_b[[ID_col]+item_col+context_col]
test_y_b = test_b['y'].values
test_t_b = pad_sequences(test_b.Title, maxlen=8)
test_g_b = pad_sequences(test_b.Genres, maxlen=4)

test_x_c = test_c[[ID_col]+item_col+context_col]
test_y_c = test_c['y'].values
test_t_c = pad_sequences(test_c.Title, maxlen=8)
test_g_c = pad_sequences(test_c.Genres, maxlen=4)

test_x_test = test_test[[ID_col]+item_col+context_col]
test_y_test = test_test['y'].values
test_t_test = pad_sequences(test_test.Title, maxlen=8)
test_g_test = pad_sequences(test_test.Genres, maxlen=4)