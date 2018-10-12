'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

'''

from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, LSTM, Dropout, Masking
from keras.layers import Conv1D, MaxPooling1D, Embedding, Bidirectional, BatchNormalization
from keras.models import Model
from keras.metrics import binary_crossentropy
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

import sys

import tensorflow as tf
#tf.python.control_flow_ops = tf

from sklearn.metrics import precision_recall_fscore_support

BASE_DIR = ''
GLOVE_DIR = './'
MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

#seed random number geenrators for reproducible results
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
# gigaword glove
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
#common crawl glove
#f= open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    #print(np.linalg.norm(coefs))
    #exit()
    embeddings_index[word] = coefs
f.close()

for words in ['<unk>', '<timeref>', '<math>', '<mathfunc>', '<eop>', '<urlref>']:
    k=np.random.rand(1, EMBEDDING_DIM)
    k=7*k/np.linalg.norm(k)
    embeddings_index[words] = k
    #print(np.linalg.norm(k))
    #print('a')


print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts_train = []  # list of text samples
texts_test = []
labels_index = {'0':0 , '1':1}  # dictionary mapping label name to numeric id
labels_train = []  # list of label ids
labels_test = []
ids_test = []
ids_train = []
f1=open(sys.argv[1])

#Read line ids -->ids

for lines in f1:
	comps=lines.split('\t')
	texts_train.append(''.join(comps[2:]).strip().lower())
	labels_train.append(int(comps[1].strip()))
	ids_train.append(int(comps[0].strip()))
print('%s' %labels_train)
print(ids_train)

f2=open(sys.argv[2])
for lines in f2:
        comps=lines.split('\t')
        texts_test.append(''.join(comps[2:]).strip().lower())
        labels_test.append(int(comps[1].strip()))
        ids_test.append(int(comps[0].strip()))
print('%s' %labels_test)
print(ids_test)

class_ratio = np.bincount(labels_train)
#class_weight={ 0: (class_ratio[0] +  class_ratio[1])*1.0 /class_ratio[0],
#	               1: (class_ratio[0] +  class_ratio[1])*1.0 /class_ratio[1]}

#print(class_weight)

class_weight = { 0: 1.0,
		 1: class_ratio[0]*1.0 /class_ratio[1]
		}

print(class_weight)

print('Found %s train texts.' % len(texts_train))
print('Found %s test texts.' % len(texts_test))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts_train)
sequences_train = tokenizer.texts_to_sequences(texts_train)

tokenizer.fit_on_texts(texts_test)
sequences_test = tokenizer.texts_to_sequences(texts_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

labels_train = to_categorical(np.asarray(labels_train))
ids_train = np.asarray(ids_train, dtype='int')
print('Shape of data tensor:', data_train.shape)
print('Shape of label tensor:', labels_train.shape)

labels_test = to_categorical(np.asarray(labels_test))
ids_test = np.asarray(ids_test, dtype='int')
print('Shape of data tensor:', data_test.shape)
print('Shape of label tensor:', labels_test.shape)

# split the data into a training set and a validation set
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
data_train = data_train[indices]
labels_train = labels_train[indices]
#print(indices)
ids_train = ids_train[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data_train.shape[0])

data_t = data_train[:-nb_validation_samples]
labels_t = labels_train[:-nb_validation_samples]
ids_t = ids_train[:-nb_validation_samples]

nb_validation_t_samples = int(VALIDATION_SPLIT * data_t.shape[0])

x_train = data_t[:-nb_validation_t_samples]
y_train = labels_t[:-nb_validation_t_samples]
ids_train = ids_t[:-nb_validation_t_samples]

x_val = data_t[-nb_validation_t_samples:]
y_val = labels_t[-nb_validation_t_samples:]
ids_val = ids_t[-nb_validation_t_samples:]

x_test = data_test
y_test = labels_test
ids_test = ids_test

print('Preparing embedding matrix.')

## prepare embedding matrix
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
#print(word_index)
count_oov = 0
oov_words = [] 

for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = embeddings_index.get('<unk>')
        oov_words.append(word)
        count_oov += 1

print(len(embedding_matrix))
print('oov words %d' %count_oov)

#foov=open(sys.argv[1]+'_oov.txt','w')

#for i in oov_words:
#	foov.write(i+'\n')
#foov.close()
#sys.exit(0)

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.................')

## train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
#mask = Masking(mask_value=0)(embedded_sequences)
#x= Bidirectional(LSTM(150, dropout_W=0.2, dropout_U=0.2))(embedded_sequences)  # try using a GRU instead, for fun
#x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=regularizers.l2(0.01)))(mask) #(embedded_sequences) #(mask)
conv1d_1 = Conv1D(128, 5, activation='relu')
x = conv1d_1(embedded_sequences)
x = MaxPooling1D(25)(x)
x = BatchNormalization()(x)
x = Conv1D(32, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Bidirectional(LSTM(256, dropout=0.4, recurrent_dropout=0.2))(x)
#norm = BatchNormalization()
preds = Dense(len(labels_index), activation='sigmoid', kernel_initializer='random_uniform')(x)

#x = Conv1D(128, 5, activation='relu')(embedded_sequences)
#x = MaxPooling1D(25)(x)
#x = Conv1D(32, 5, activation='relu')(x)
#x = MaxPooling1D(25)(x)
#x = Conv1D(32, 5, activation='relu')(x)
#x = MaxPooling1D(25)(x)
#x = Flatten()(x)
#x = Dropout(0.4)(Dense(64, activation='relu')(x))
#preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)

adam_opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#optimizer='rmsprop',
model.compile(loss='categorical_crossentropy',
		optimizer=adam_opt,
              metrics=['acc', binary_crossentropy])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=0, min_delta=0, mode='auto'),
    ModelCheckpoint('./best.model', monitor='val_loss', save_best_only=True, verbose=0, mode='auto')
]

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=2, batch_size=16, class_weight=class_weight, callbacks=callbacks)

#reloads the best model weights that was saved during training (fit)
model.load_weights('./best.model')

scores = model.evaluate(x_test, y_test, verbose=0)
print('Predict')
print( model.predict(x_test).argmax(1))
print('True')
print(y_test.argmax(1))

a,b,c,d = precision_recall_fscore_support(model.predict(x_test).argmax(1), y_test.argmax(1))
print (a[1],  b[1], c[1])
print (a,b,c,d)


## Code to extract vectors from the last layer
"""
from keras import backend as K

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

## Testing
feature_array_file=open('../feats/'+sys.argv[1]+'.feats.numpy','a')
feature_array_file2=open('../feats/'+sys.argv[1]+'.feats.numpy.ids','a')
##test = np.random.random(input_shape)[np.newaxis,...]A

for itemsa,itemsb in [(x_train, ids_train), (x_val, ids_val) , (x_test, ids_test)]:
	layer_outs = functor([itemsa, 0.])
	abc = np.array(layer_outs[-2], dtype='float32')
	print (abc.shape)
	np.savetxt(feature_array_file, abc)
	np.savetxt(feature_array_file2, itemsb, fmt='%i')
"""

##	print(layer_outs[0].shape)
##	rows = layer_outs[0].shape[0]
##	for i in range(rows):
##		print(layer_outs[1][i])
##		#np.savetxt(feature_array_file,layer_outs[0][i])
##	#print(layer_outs[0][0].shape, end='\n-------------------\n')
##	#raw_input("Press Enter to continue...")

##import theano
##get_activations = theano.function([model.layers[0].input], model.layers[1].output(train=False), allow_input_downcast=True)
##print (get_activations(x_train))
##print (get_activations(x_val))
##print (get_activations(x_test))


#print("%s: %.2f%%, %.2f, %.2f" % (model.metrics_names, scores[1]*100, scores[2], scores[3]))
