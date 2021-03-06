import collections
import json
import numpy as np 
import tensorflow as tf

class PoetryMachine(tf.keras.Model):
    def __init__(self, lr = 1e-2, num_words = 5000, embeddings_dim = 128, units_each_layer = 128):
        super().__init__()
        self.num_words = num_words
        self.batch_size = 256
        
        self.embeddings = tf.keras.layers.Embedding(input_dim=num_words, output_dim=embeddings_dim, 
                                                    embeddings_initializer='normal')
        self.lstm_layer_1 = tf.keras.layers.LSTMCell(units=units_each_layer)
        self.lstm_layer_2 = tf.keras.layers.LSTMCell(units=units_each_layer)
        self.lstm_layer_3 = tf.keras.layers.LSTMCell(units=units_each_layer)
        
        self.stacked_lstm = tf.keras.layers.StackedRNNCells(cells=[self.lstm_layer_1, self.lstm_layer_2, self.lstm_layer_3])
        self.lstm_layer = tf.keras.layers.RNN(self.stacked_lstm, return_sequences=True, return_state=True)
        
        self.W_fc = tf.Variable(tf.random.normal(shape=[units_each_layer, self.num_words], dtype='float32'), trainable=True)
        self.b_fc = tf.Variable(tf.random.normal(shape=[self.num_words], dtype='float32'), trainable=True)
    
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    def load_file(self, filename = 'poetryTang.txt'):
        raw_poems = []
        try:
            f = open(filename)
            for each_line in f.readlines():
                if len(each_line) > 128 or len(each_line) < 12:
                    continue
                raw_poems.append('[' + each_line.split(':')[-1].strip('\n') + ']')

        except Exception:
            print('no such file found!')
        
        self.raw_poems = raw_poems

    def get_dictionary(self, raw_poems):
        dictionary = {}
        all_words = []
        for each_poem in raw_poems:
            all_words.extend(each_poem)

        word_count = collections.Counter(all_words).most_common(self.num_words - 1)
        words_set, _ = zip(*word_count)
        words_set += (' ', )
        self.words_set = words_set
        
        self.dictionary = dict(zip(words_set, range(len(words_set))))
        word_count.sort(key = lambda x : x[-1])

        self.word_to_id_fun = lambda x : self.dictionary.get(x, len(words_set) - 2)
        self.poems_vector = [([self.word_to_id_fun(word) for word in poem]) for poem in raw_poems]
    
    def load_train_data(self, batch_size = 256):
        print('loading train_data...')
        self.batch_size = batch_size
        self.X = []
        self.Y = []
        batch_num = (len(self.poems_vector) - 1) // self.batch_size 
        for i in range(batch_num):
            #batch
            batch = self.poems_vector[i * self.batch_size : (i + 1) * self.batch_size]
            max_len = max([len(vector) for vector in batch])
            temp = np.full((self.batch_size, max_len), self.word_to_id_fun(' '), np.int32)
            for j in range(model.batch_size):
                temp[j, :len(batch[j])] = batch[j]
            self.X.append(temp)
            temp2 = np.copy(temp)
            temp2[:, :-1] = temp[:, 1:]
            self.Y.append(temp2)
        print('done, batch_size: {}, batch_num of each epoch: {}, count of poems: {}'.format(str(self.batch_size), str(batch_num), str(len(self.poems_vector))))
        
    def forward(self, input_x, initial_state=None):
        embeddings = self.embeddings(input_x)
        if initial_state is not None:
            outputs, final_state_0, final_state_1, final_state_2 = self.lstm_layer(embeddings, initial_state=initial_state)
        else:
            outputs, final_state_0, final_state_1, final_state_2 = self.lstm_layer(embeddings)
        hidden = tf.reshape(outputs, [-1, outputs.shape[-1]])
        logits = tf.matmul(hidden, self.W_fc) + self.b_fc
        
        return logits, [final_state_0[-1], final_state_1[-1], final_state_2[-1]]
    
    
    def loss(self, logits, input_y):
        input_y = tf.reshape(input_y, [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_y, logits=logits)
        return loss
    
    
    def get_variables(self):
        return self.trainable_variables
    
    
    def train(self, batch_size=256, epochs=10, input=None):
        if input is not None:
            pass
        else:
            self.load_train_data(batch_size=batch_size)
        self.vars = self.get_variables()
        self.restore_weights()
        train_step = 0
        batch_num = len(self.X)
        for epoch in range(epochs):
            for x, y in zip(self.X, self.Y):
                with tf.GradientTape() as tape:
                    logits, states = self.forward(input_x=x)
                    loss = self.loss(logits=logits, input_y=y)
                
                grads = tape.gradient(loss, self.vars)
                self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.vars))
                train_step += 1
            
                print('Episode {}:  gloable_step {}, loss {:.3f}'.format(epoch, train_step, tf.reduce_sum(loss)/(self.batch_size*x[0].shape[0])))
            
            if epoch%10 == 0:
                self.save_my_weights()
                
        self.save_my_weights()
                
    def save_my_weights(self, path='./libai_weights'):
        try:
            self.save_weights(path)
            print('weights saved: {}'.format(path))
        except Exception as e:
            print(e)
            print('fail to save')
            
    
    def restore_weights(self, path='./libai_weights'):
        try:
            self.load_weights(path)
            print('weights restored from {}'.format(path))
        except:
            print('model weights not found...')
                        
    def write(self, generate_num = 10):
        poems = []
        for i in range(generate_num):
            x = np.array([[self.word_to_id_fun('[')]])
            prob1, state = self.forward(input_x=x)
            prob1 = tf.nn.softmax(prob1, axis=-1)
            word = self.probs_to_word(prob1, self.words_set)
            poem = ''
            loop = 0
            while word != ']' and word != ' ':
                loop += 1
                poem += word 
                if word == '。':
                    poem += '\n'
                x = np.array([[self.word_to_id_fun(word)]])	

                next_probs, state = self.forward(input_x=x, initial_state=state)
                next_probs = tf.nn.softmax(next_probs, axis=-1)
                word = self.probs_to_word(next_probs, self.words_set)
                if loop > 100:
                    loop = 0
                    break
            print(poem)
            poems.append(poem)
        return poems

    def probs_to_word(self, weights, words):
        t = np.cumsum(weights)
        s = np.sum(weights)
        coff = np.random.rand(1)
        index = int(np.searchsorted(t, coff * s))
        return words[index]
```
创建一个写诗机实例，载入数据集
model = PoetryMachine()
model.load_file()
model.get_dictionary(model.raw_poems)
训练模型
model.train()
写诗
model.write()
```
