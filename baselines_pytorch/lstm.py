import torchwordemb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.autograd import Variable
from torchvision import transforms, utils
import re
import os
#import torchtext

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
#from torchtext.data import Field
#from torchtext.data import TabularDataset, Iterator, BucketIterator

import pandas as pd
import numpy as np
#from visualize_attention import attentionDisplay
from sklearn import metrics
#import matplotlib.pyplot as plt
import spacy
from spacy.lang.en import English

import argparse

VALIDATION_SPLIT = 0.2
MINI_BATCH_SIZE = 16 
num_epochs = 1 
LEARNING_RATE = 1e-3

parser = argparse.ArgumentParser(description='BiLSTM model for predicting instructor internvention')
parser.add_argument('-d','--dim',help="dimension of the embedding. 50 or 300", default=300, required=False, type=int)
parser.add_argument('-e','--epochs',help="number of epochs. > 0", default=2, required=False, type=int)
parser.add_argument('-r','--lr',help="learning rate >0 but <100", default=1e-2, required=False, type=int)
parser.add_argument('-b','--bz',help="mini batch size. Usually in powers of 2; >=16", default=16, required=False, type=int)
parser.add_argument('-v','--val',help="validation split: a number between 0 to 1", default=0.2, required=False, type=int)
parser.add_argument('-c','--course',help="course id", required=True, type=str)
parser.add_argument('-i','--ver',help="verbose mode", required=False, type=bool)
parser.add_argument('-l','--load',help="load model", required=False, type=bool)

args = vars(parser.parse_args())
course = args['course']
EMBEDDING_DIM = args['dim']

input_path = '/diskA/muthu/Transact-Net/feats/in' + course + '_w2v'
print(input_path)

#vocab, vec = torchwordemb.load_glove_text("/diskA/animesh/glove/glove.6B.50d.txt")

if torch.cuda.is_available():
    torch.device('cuda')

#set seed for reproducibility of results
from numpy.random import seed
seed(1491)

torch.manual_seed(1491)
torch.cuda.manual_seed(1491)

def tokenize_and_clean(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #for math equations
    string = re.sub(r"\$\$.*?\$\$", " <MATH> ", string)
    string = re.sub(r"\(.*\(.*?\=.*?\)\)", " <MATH> ", string)
    string = re.sub(r"\\\(\\mathop.*?\\\)", " <MATH> ", string)
    string = re.sub(r"\\\[\\mathop.*?\\\]", " <MATH> ", string)
    string = re.sub(r"[A-Za-z]+\(.*?\)", " <MATH> ", string)    
    string = re.sub(r"[A-Za-z]+\[.*?\]", " <MATH> ", string)    
    string = re.sub(r"[0-9][\+\*\\\/\~][0-9]", " <MATH> ", string) 
    string = re.sub(r"<MATH>\s*[\+\-\*\\\/\~][0-9]", " <MATH> ", string) 
        
    string = re.sub(r"<MATH>\s*[\+\-\*\\\/\~\=]", " <MATH> ", string)
    string = re.sub(r"[\+\-\*\\\/\~\=]\s*<MATH>", " <MATH> ", string)
        
    string = re.sub(r"[\+\*\\\/\~]", " <MATH> ", string)    
    string = re.sub(r"(<MATH>\s*)+", " <MATH> ", string)

    #for time 
    string = re.sub(r"[0-9][0-9]?:[0-9][0-9]?", "<TIMEREF>", string)

    #for url's
    string = re.sub(r"https?\:\/\/[a-zA-Z0-9][a-zA-Z0-9\.\_\?\=\/\%\-\~\&]+", "<URLREF>", string)

    # for english sentences 
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    string =  string.strip().lower()
    
    #tokenize string, filter for stopwords and return
    nlp = spacy.load('en')
    tokenizer = English().Defaults.create_tokenizer(nlp)
    return [tok.text for tok in tokenizer(string) if not tok.is_stop]

#read inout files
df = pd.read_csv(input_path, sep='\t', header=None, encoding="ISO-8859-1")
train,test = train_test_split(df, test_size=0.2, random_state=1491, shuffle=True)

#X_train = (train.to_frame().T)
X_train = train[2]
y_train = train[1]
X_test = test[2]
y_test =  test[1]

## Load pretrained word vector
if args['dim'] == 50:
	print('loading 50d glove embedding')
	vocab, vec = torchwordemb.load_glove_text("/diskA/animesh/glove/glove.6B.50d.txt")
elif args['dim'] == 100:
	print('loading 100d glove embedding')
	vocab, vec = torchwordemb.load_glove_text("/diskA/animesh/glove/glove.6B.100d.txt")
elif args['dim'] == 200:
	print('loading 200d glove embedding')
	vocab, vec = torchwordemb.load_glove_text("/diskA/animesh/glove/glove.6B.200d.txt")
elif args['dim'] == 300:
	print('loading 300d glove embedding')
	vocab, vec = torchwordemb.load_glove_text("/diskA/animesh/glove/glove.6B.300d.txt")
else:
	print("Embedding dimension not available. Defaulting to 50 dimensions")
	vocab, vec = torchwordemb.load_glove_text("/diskA/animesh/glove/glove.6B.50d.txt")

vec = vec.cuda()
max_idx = len(vocab)

#add indices and random embeddings of important OOV words. We will learn these 
#embedding during traiing
for word in ['<unk>', 'timeref', 'math', 'mathfunc', 'eop', 'urlref', '<pad>']:
	k = np.random.rand(1, EMBEDDING_DIM)
	k = 7*k/np.linalg.norm(k)
	vocab[word]= max_idx
	k_tensor = torch.from_numpy(k).cuda()
	k_tensor = k_tensor.type(torch.cuda.FloatTensor)
	vec = torch.cat((k_tensor,vec), 0)
	max_idx += 1

if args['ver']:
    print(vec.size())

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        V = args['embed_num']
        D = args['embed_dim']
        C = args['class_num']
        Ci = 1

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args['vec'])
        self.embed.weight.requires_grad = True
        self.lstm = nn.LSTM(input_size=D, hidden_size=128, dropout=0.2)
        self.dropout = nn.Dropout(args['dropout'])
        self.fc1 = nn.Linear(128, C)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = Variable(x)
        x = x.permute(1,0,2)  # (W, N, D)
        x,_ = self.lstm(x)
        logit = self.fc1(self.dropout(x[-1]))  # concatenation and dropout
        return logit
        
lstm_args = {} 
lstm_args['embed_num'] = max_idx
lstm_args['vec'] = vec
lstm_args['class_num'] = 2
lstm_args['cuda'] = torch.cuda.is_available()
lstm_args['embed_dim'] = args['dim']
lstm_args['dropout'] = 0.4

lstm = Model(lstm_args)
lstm = lstm.cuda()

#test input
inp = torch.tensor([[vocab["hello"], vocab["world"], vocab["english"],vocab["hello"], vocab["world"], vocab["english"],
                     vocab["hello"], vocab["world"], vocab["english"],vocab["hello"], vocab["world"], vocab["english"],
                     vocab["hello"], vocab["world"], vocab["english"],vocab["hello"], vocab["world"], vocab["english"],
                     vocab["hello"], vocab["world"], vocab["english"],vocab["hello"], vocab["world"], vocab["english"]]],
                     dtype=torch.long).cuda()

#calculate class weights for use in classification to counter class imbalance
class_ratio = np.bincount(y_train.values)
class_weight_dict = { 0: 1.0,
                      1: class_ratio[0]*1.0 /class_ratio[1]
                    }
class_weights_tensor = torch.FloatTensor([class_weight_dict[0],class_weight_dict[1]]).cuda()

if args['ver']:
    print(class_weight_dict)
    print(class_weights_tensor)


optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)

print('Training instances for course', args['course'])
print('Training loss....')

#make minibatches
num_batches = len(X_train.index.values) // MINI_BATCH_SIZE
X_batches = np.array_split(X_train, num_batches)
y_batches = np.array_split(y_train, num_batches)

#initialise variable to count OOV words
oov_words = {}
in_vocab_words = {}

def get_sequences(X_batch, y_batch):
    '''
    turns words in pieces of text into padded 
    sequences of word indices correspodning to 
    the vocab
    '''
    X_batch_tkns = []
    for idx in (X_batch.index.values):
        X_batch_tkns.append(tokenize_and_clean(X_batch[idx]))
                
    # get the length of each thread in the batch 
    X_lengths = [len(text) for text in X_batch_tkns]

    # create an empty matrix with padding tokens
    pad_token = vocab['<pad>']
    #max_text_length = max(X_lengths)
    max_text_length = 500
    batch_size = len(X_batch)
    padded_X_batch = np.ones((batch_size, max_text_length), dtype=int) * pad_token
    
    X_batch_idxs = X_batch.values

    # copy over the actual sequences
    for i, x_len in enumerate(X_lengths):
        #sequence = np.array([vocab[w] if w in vocab else vocab['<unk>'] for w in X_batch_tkns[i]])
        sequence = np.array([vocab[w] if w in vocab else vocab['<unk>'] for w in X_batch_tkns[i]])
        if x_len > max_text_length:
            padded_X_batch[i, 0:x_len] = sequence[:max_text_length]
        else:
            padded_X_batch[i, 0:x_len] = sequence[:x_len]

    #count for OOVs here
    for i in range(len(X_lengths)): 
        for w in X_batch_tkns[i]:
            if w not in vocab:
                oov_words.update({w:1})
            else:
                in_vocab_words.update({w:1})

    targets = []
    for idx in (y_batch.index.values):
        targets.append(y_batch[idx])
    targets_np = np.array(targets)

    return padded_X_batch, targets_np, max_text_length

if args['load']:
    optimizer.load_state_dict(torch.load(filename))

def get_thread_length(X_batch):
    posts = X_batch.iloc[0].strip().split("<EOP>")
    if posts[-1] == '':
        del posts[-1]
    return len(posts)

for epoch in range(1,num_epochs+1):
    for batch_num in range(num_batches):
        print('batch num', batch_num)
        word_idxs, targets, max_seq_length = get_sequences(X_batches[batch_num], y_batches[batch_num])
        
        word_idxs_tensor = torch.LongTensor(word_idxs)

        targets_tensor = torch.LongTensor(targets)

        inp = Variable(word_idxs_tensor, requires_grad=False).cuda()
        target = Variable(targets_tensor, requires_grad=False).cuda()
        
        if args['ver']:
            print(inp.size())
          
        if args['ver']:
             print("target size" + str(target.size()))
             print(target)
             print(target.size())
        optimizer.zero_grad()
        logit = lstm(inp)

        loss = F.cross_entropy(logit, target, class_weights_tensor, size_average=True)
        
        print(epoch, batch_num, loss.item())
            
        # Zero the gradients before running the backward pass.

        loss.backward()
        optimizer.step()

torch.save(lstm.state_dict(), "./best_models/lstm_best_model_weights_"+course)
torch.save(optimizer.state_dict(), "./best_models/lstm_best_model_gradients_"+course)

y_preds = []
y_true = []
#Test Time
print('Test instances for course', args['course'])
for idx in list(X_test.index.values):
    x_tkns = tokenize_and_clean(X_test[idx])
    
    word_idxs = torch.tensor([vocab[w] if w in vocab else vocab['<unk>'] for w in x_tkns], dtype=torch.long)
    word_idxs = word_idxs.reshape(1,-1)
    inp = (Variable(word_idxs)).cuda()
        
    target = (Variable(torch.LongTensor([y_test[idx]]), requires_grad=False)).cuda()

    if args['ver']:
         print(inp.size())

    op = lstm(inp)
    
    _,prediction = op.max(dim=1)
    prediction  = prediction.item()
    if args['ver']:
        print('idx, op, prediction', idx, op, prediction)
    y_preds.append(prediction)
    y_true.append(y_test[idx])
        
    #test target
    #target = torch.rand_like(op)

    loss = F.cross_entropy(op, target, class_weights_tensor, size_average=True)    
    print(idx, loss.item())
 
#metric calculation
prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_preds, average=None, labels=['0', '1'])

print(oov_words)
print('No of training instances', len(X_train.index))
print('No of test instances', len(X_test.index))
print('Ground truth', y_true)
print('Predictions', y_preds)
print('Precision, Recall, F-score', prec[1], recall[1], fscore[1])
print(len(oov_words), len(in_vocab_words))
