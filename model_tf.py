




import tensorflow as tf
from tensorflow.layers import Dense
import keras
from keras.preprocessing.sequence import pad_sequences


 
class Meta_Model(object):
    def __init__(self, ID_col, item_col, context_col, nb_words, model='FM',
                 emb_size=128, alpha=0.1,
                 warm_lr=1e-3, cold_lr=1e-4, ME_lr=1e-3):
        """
        ID_col: string, the column name of the item ID
        item_col: list, the columns of item features
        context_col: list, the columns of other features
        nb_words: dict, nb of words in each of these columns
        """
        columns = [ID_col] + item_col + context_col
        
        def get_embeddings():
            inputs, tables = {}, []
            item_embs, other_embs = [], []
            for col in columns:
                inputs[col] = tf.placeholder(tf.int32, [None])
                table = tf.get_variable(
                    "table_{}".format(col), [nb_words[col], emb_size],
                    initializer=tf.random_normal_initializer(stddev=0.01))
                emb = tf.nn.embedding_lookup(table, inputs[col])
                if col==ID_col:
                    ID_emb = emb
                    ID_table = table
                elif col in item_col:
                    item_embs.append(emb)
                else:
                    other_embs.append(emb)

            inputs["title"] = tf.placeholder(tf.int32, [None, 8])
            inputs["genres"] = tf.placeholder(tf.int32, [None, 4])

            title_emb = tf.contrib.layers.embed_sequence(
                inputs["title"], 20001, emb_size, scope="word_emb")
            genre_emb = tf.contrib.layers.embed_sequence(
                inputs["genres"], 21, emb_size, scope="genre_table")
            item_embs.append(tf.reduce_mean(title_emb, axis=1))
            item_embs.append(tf.reduce_mean(genre_emb, axis=1))
            
            return inputs, ID_emb, item_embs, other_embs, ID_table
        
        def generate_meta_emb(item_embs):
            """
            This is the simplest architecture of the embedding generator,
            with only a dense layer.
            You can customize it if you want have a stronger performance, 
            for example, you can add an l2 regularization term or alter 
            the pooling layer. 
            """
            embs = tf.stop_gradient(tf.stack(item_embs, 1))
            item_h = tf.layers.flatten(embs)
            emb_pred_Dense = tf.layers.Dense(
                emb_size, activation=tf.nn.tanh, use_bias=False,
                name='emb_predictor') 
            emb_pred = emb_pred_Dense(item_h) / 5.
            ME_vars = emb_pred_Dense.trainable_variables
            return emb_pred, ME_vars

        def get_yhat_deepFM(ID_emb, item_embs, other_embs, **kwargs):
            embeddings = [ID_emb] + item_embs + other_embs
            sum_of_emb = tf.add_n(embeddings) # sum_of_emb :  (?, 128)
            diff_of_emb = [sum_of_emb - x for x in embeddings] # diff_of_emb[0] :  (?, 128) ; len = 6
            dot_of_emb = [tf.reduce_sum(embeddings[i]*diff_of_emb[i], 
                                        axis=1, keepdims=True) 
                          for i in range(len(columns))] # dot_of_emb[0] :  (?, 1) ; len = 6
            h = tf.concat(dot_of_emb, 1) # h :  (?, 6)
            h2 = tf.concat(embeddings, 1) # h2 :  (?, 1024)
            for i in range(2):
                h2 = tf.nn.relu(tf.layers.dense(h2, emb_size, name='deep-{}'.format(i)))
            h = tf.concat([h,h2], 1)
            y = tf.nn.sigmoid(tf.layers.dense(h, 1, name='out'))
            return y
        
        def get_yhat_PNN(ID_emb, item_embs, other_embs, **kwargs):
            embeddings = [ID_emb] + item_embs + other_embs
            sum_of_emb = tf.add_n(embeddings)
            diff_of_emb = [sum_of_emb - x for x in embeddings]
            dot_of_emb = [tf.reduce_sum(embeddings[i]*diff_of_emb[i], 
                                        axis=1, keepdims=True)
                          for i in range(len(columns))]
            dots = tf.concat(dot_of_emb, 1)
            h2 = tf.concat(embeddings, 1)
            h = tf.concat([dots,h2], 1)
            w = tf.get_variable('MLP_1/kernel', shape=(h.shape[1],emb_size))
            b = tf.get_variable('MLP_1/bias', shape=(emb_size,), 
                                initializer=tf.initializers.zeros)
            h = tf.nn.relu(tf.matmul(h,w)+b)
            w = tf.get_variable('MLP_2/kernel', shape=(h.shape[1],1))
            b = tf.get_variable('MLP_2/bias', shape=(1,), 
                                initializer=tf.initializers.constant(0.))
            y = tf.nn.sigmoid(tf.matmul(h,w)+b)
            return y
        '''
        *CHOOSE THE BASE MODEL HERE*
        '''
        get_yhat = {
            "PNN": get_yhat_PNN, 
            "deepFM": get_yhat_deepFM
        }[model]
        
        with tf.variable_scope("model"):
            # build the base model
            inputs, ID_emb, item_embs, other_embs, ID_table = get_embeddings()
            label = tf.placeholder(tf.float32, [None, 1])
            # outputs and losses of the base model
            yhat = get_yhat(ID_emb, item_embs, other_embs)
            warm_loss = tf.losses.log_loss(label, yhat)
            # Meta-Embedding: build the embedding generator
            meta_ID_emb, ME_vars = generate_meta_emb(item_embs)

        with tf.variable_scope("model", reuse=True):
            # Meta-Embedding: step 1, cold-start, 
            #     use the generated meta-embedding to make predictions
            #     and calculate the cold-start loss_a
            cold_yhat_a = get_yhat(meta_ID_emb, item_embs, other_embs)
            cold_loss_a = tf.losses.log_loss(label, cold_yhat_a) 
            # Meta-Embedding: step 2, apply gradient descent once
            #     get the adapted embedding
            cold_emb_grads = tf.gradients(cold_loss_a, meta_ID_emb)[0]
            meta_ID_emb_new = meta_ID_emb - cold_lr * cold_emb_grads
            # Meta-Embedding: step 3, 
            #     use the adapted embedding to make prediction on another mini-batch 
            #     and calculate the warm-up loss_b
            inputs_b, _, item_embs_b, other_embs_b, _ = get_embeddings()
            label_b = tf.placeholder(tf.float32, [None, 1])
            cold_yhat_b = get_yhat(meta_ID_emb_new, item_embs_b, other_embs_b)
            cold_loss_b = tf.losses.log_loss(label_b, cold_yhat_b)            
        
        # build the optimizer and update op for the original model
        warm_optimizer = tf.train.AdamOptimizer(warm_lr)
        warm_update_op = warm_optimizer.minimize(warm_loss)
        warm_update_emb_op = warm_optimizer.minimize(warm_loss, var_list=[ID_table])
        # build the optimizer and update op for meta-embedding
        # Meta-Embedding: step 4, calculate the final meta-loss
        ME_loss = cold_loss_a * alpha + cold_loss_b * (1-alpha)
        ME_optimizer = tf.train.AdamOptimizer(ME_lr)
        ME_update_op = ME_optimizer.minimize(ME_loss, var_list=ME_vars)
        
        ID_table_new = tf.placeholder(tf.float32, ID_table.shape)
        ME_assign_op = tf.assign(ID_table, ID_table_new)
        
        def predict_warm(sess, X, Title, Genres):
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            return sess.run(yhat, feed_dict)
        
        def predict_ME(sess, X, Title, Genres):
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            return sess.run(cold_yhat_a, feed_dict)
        
        def get_meta_embedding(sess, X, Title, Genres):
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            return sess.run(meta_ID_emb, feed_dict)
        
        def assign_meta_embedding(sess, ID, emb):
            # take the embedding matrix
            table = sess.run(ID_table)
            # replace the ID^th row by the new embedding
            table[ID, :] = emb
            return sess.run(ME_assign_op, feed_dict={ID_table_new: table})
        
        def train_warm(sess, X, Title, Genres, y, embedding_only=False): 
            # original training on batch
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            feed_dict[label] = y.reshape((-1,1))
            return sess.run([
                warm_loss, warm_update_emb_op if embedding_only else warm_update_op 
            ], feed_dict=feed_dict)
        
        def train_ME(sess, X, Title, Genres, y, 
                     X_b, Title_b, Genres_b, y_b):
            # train the embedding generator
            feed_dict = {inputs[col]: X[col] for col in columns}
            feed_dict = {inputs["title"]: Title,
                         inputs["genres"]: Genres,
                         **feed_dict}
            feed_dict[label] = y.reshape((-1,1))
            feed_dict_b = {inputs_b[col]: X_b[col] for col in columns}
            feed_dict_b = {inputs_b["title"]: Title_b,
                           inputs_b["genres"]: Genres_b,
                           **feed_dict_b}
            feed_dict_b[label_b] = y_b.reshape((-1,1))
            return sess.run([
                cold_loss_a, cold_loss_b, ME_update_op
            ], feed_dict={**feed_dict, **feed_dict_b})
        
        self.predict_warm = predict_warm
        self.predict_ME = predict_ME
        self.train_warm = train_warm
        self.train_ME = train_ME
        self.get_meta_embedding = get_meta_embedding
        self.assign_meta_embedding = assign_meta_embedding