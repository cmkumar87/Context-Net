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

MAX_SEQUENCE_LENGTH = 250 #250 words per post
MAX_THREAD_LENGTH = 15
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.2
MINI_BATCH_SIZE = 1 
num_epochs = 1 
LEARNING_RATE = 1e-3
CONTEXT = 999

#CORPUS = ['ml-005', 'rprog-003', 'calc1-003', 'compilers-004', 'smac-001', 'maththink-004', 'bioelectricity-002', 'gametheory2-001', 'musicproduction-006', 'medicalneuro-002','comparch-002', 'bioinfomethods1-001', 'casebasedbiostat-002']

parser = argparse.ArgumentParser(description='BiLSTM model for predicting instructor internvention')
parser.add_argument('-d','--dim',help="dimension of the embedding. 50 or 300", default=50, required=False, type=int)
parser.add_argument('-e','--epochs',help="number of epochs. > 0", default=2, required=False, type=int)
parser.add_argument('-r','--lr',help="learning rate >0 but <100", default=1e-2, required=False, type=int)
parser.add_argument('-b','--bz',help="mini batch size. Usually in powers of 2; >=16", default=16, required=False, type=int)
parser.add_argument('-v','--val',help="validation split: a number between 0 to 1", default=0.2, required=False, type=int)
parser.add_argument('-c','--course',help="specificy course id to match that in the input file name", required=True, type=str)
parser.add_argument('-i','--ver',help="verbose mode", required=False, type=bool)
parser.add_argument('-l','--load',help="load model", required=False, type=bool)
parser.add_argument('-ct','--con',help="thread context to use as model input", required=True, type=int)

args = vars(parser.parse_args())
course = args['course']
EMBEDDING_DIM = args['dim']
CONTEXT = args['con']

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
for word in ['<unk>', '<timeref>', '<math>', '<mathfunc>', '<eop>', '<urlref>', '<pad>']:
	k = np.random.rand(1, EMBEDDING_DIM)
	k = 7*k/np.linalg.norm(k)
	vocab[word]= max_idx
	k_tensor = torch.from_numpy(k).cuda()
	k_tensor = k_tensor.type(torch.cuda.FloatTensor)
	vec = torch.cat((k_tensor,vec), 0)
	max_idx += 1

if args['ver']:
    print(vec.size())

'''
Yoon Kim 2010 CNN sentence classifier
from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
'''
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        V = args['embed_num']
        D = args['embed_dim']
        C = args['class_num']
        Ci = 1
        hidden = args['hidden']

        self.embed = nn.Embedding(V, D)
        self.embed.weight = nn.Parameter(args['vec'])
        self.embed.weight.requires_grad = True
        self.lstm1 = nn.LSTM(input_size=D, hidden_size=hidden, dropout=0.2, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=hidden, dropout=0.2, bidirectional=False)
        self.dropout = nn.Dropout(args['dropout'])
        self.fc1 = nn.Linear(128, C)

    @staticmethod
    def conv_and_max_pool(x, conv):
        """Convolution and global max pooling layer"""
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = Variable(x)
        x = x.permute(1,0,2)  # (W, N, D)
        x,_ = self.lstm1(x)
        x = x[-1]
        x = x.unsqueeze(1)
        x,_ = self.lstm2(x)
        logit = self.fc1(self.dropout(x[-1]))  # concatenation and dropout
        return logit
        
lstm_args = {} 
lstm_args['embed_num'] = max_idx
lstm_args['vec'] = vec
lstm_args['class_num'] = 2
lstm_args['cuda'] = torch.cuda.is_available()
lstm_args['hidden'] = 128 
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

def get_sequences(X_batch, context):
    '''
    turns words in pieces of text into padded 
    sequences of word indices correspodning to 
    the vocab
    '''
    posts = X_batch.iloc[0].strip().split("<EOP>")
    if posts[-1] == '':
        del posts[-1]

    original_thread_length = len(posts) 

    if context < len(posts):
        posts = posts[-context:]
    
    #print(posts)
    #print(len(posts))
    #print(context)
    
    post_tkns = []
    for post in posts:
        post_tkns.append(tokenize_and_clean(post))
    #print (post_tkns)
    
    # get the length of each sentence
    post_lengths = [len(text) for text in post_tkns]
    #print (len(post_tkns))
    #print (post_lengths)

    # create an empty matrix with padding tokens
    pad_token = vocab['<pad>']
    max_post_length = max(post_lengths)
    thread_length = len(posts)
    padded_posts = np.ones((thread_length, max_post_length), dtype=int) * pad_token

    # copy over the actual sequences
    for i, post_length in enumerate(post_lengths):
        sequence = np.array([vocab[w] if w in vocab else vocab['<unk>'] for w in post_tkns[i]])
        padded_posts[i, 0:post_length] = sequence[:post_length]


    return padded_posts, original_thread_length


if args['load']:
    optimizer.load_state_dict(torch.load(filename))

#Training
for epoch in range(1,num_epochs+1):
    for batch_num in range(num_batches):
        print('batch num', batch_num)

        targets = []
        y_batch = y_batches[batch_num]
        for idx in (y_batch.index.values):
            targets.append(y_batch[idx])
        targets_np = np.array(targets)
        targets_tensor = torch.LongTensor(targets)
        target = Variable(targets_tensor, requires_grad=False).cuda()

        word_idxs, thread_length = get_sequences(X_batches[batch_num], context=CONTEXT)
        if word_idxs.size == 0:
            continue

        word_idxs_tensor = torch.LongTensor(word_idxs)#torch.from_numpy(word_idxs).long()
        
        inp = Variable(word_idxs_tensor, requires_grad=False).cuda()
        
        if args['ver']:
            print(inp.size())
          
        word_idxs2, thread_length2 = get_sequences(X_batches[batch_num], context=999)
        if word_idxs2.size == 0:
            continue

        word_idxs2_tensor = torch.LongTensor(word_idxs2)#torch.from_numpy(word_idxs).long()
        
        inp2 = Variable(word_idxs2_tensor, requires_grad=False).cuda()

        if args['ver']:
             print("target size" + str(target.size()))
             print(target)
             print(target.size())
        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
        
        # Forward pass
        #logit = lstm(inp)
        logit1 = lstm(inp)
        logit2 = lstm(inp2)
        #logit = logit1 - logit2
        #print (batch_num, logit)

        #loss_in_context = 1 - (CONTEXT/thread_length)
        loss = F.cross_entropy(logit1, target, class_weights_tensor, size_average=True) + F.cross_entropy(logit2, target, class_weights_tensor, size_average=True) 
        #print(loss.item(), loss_in_context, thread_length)
        #if loss_in_context > 0:
        #   loss = loss * loss_in_context

        #loss = F.cross_entropy(logit1, target, class_weights_tensor, size_average=False) - F.cross_entropy(logit2, target, class_weights_tensor, size_average=False)
        #torch.abs_(loss)

        #print(epoch, batch_num, loss)
 
        #loss = loss / len()
        print(epoch, batch_num, loss.item())

        loss.backward()
        optimizer.step()

torch.save(lstm.state_dict(), "./best_model_weights")
torch.save(optimizer.state_dict(), "./best_model_gradients")

y_preds = []
y_true = []

X_batches = np.array_split(X_test, len(X_test))
y_batches = np.array_split(y_test, len(y_test))

#Test Time
print('Test instances for course', args['course'])
for batch_num in range(0, len(X_test)):
#for idx in list(X_test.index.values):
    print(batch_num)
    word_idxs, thread_length = get_sequences(X_batches[batch_num], context=999)
    if word_idxs.size == 0:
        continue

    word_idxs_tensor = torch.LongTensor(word_idxs)#torch.from_numpy(word_idxs).long()
    inp = Variable(word_idxs_tensor, requires_grad=False).cuda()

    if args['ver']:
         print(inp.size())

    op = lstm(inp)
    
    _,prediction = op.max(dim=1)
    prediction  = prediction.item()
    if args['ver']:
        print('idx, op, prediction', idx, op, prediction)
    y_preds.append(prediction)
    y_batch = y_batches[batch_num]
    y_true.append(y_batch.iloc[0])
        
    #test target
    #target = torch.rand_like(op)
    targets = []
    for idx in (y_batch.index.values):
        targets.append(y_batch[idx])
    #targets_np = np.array(targets)
    #targets_tensor = torch.LongTensor(targets)
    #target = Variable(targets_tensor, requires_grad=False).cuda()
    targets = y_batch.iloc[0]
    targets_tensor = torch.LongTensor([targets])
    target = Variable(targets_tensor, requires_grad=False).cuda()
    loss = F.cross_entropy(op, target, class_weights_tensor, size_average=True)    
    print(idx, loss.item())
 
#metric calculation
prec, recall, fscore, _ = precision_recall_fscore_support(y_true, y_preds, average=None, labels=['0', '1'])

print('No of training instances', len(X_train.index))
print('No of test instances', len(X_test.index))
print('Ground truth', y_true)
print('Predictions', y_preds)
print('Precision, Recall, F-score', prec[1], recall[1], fscore[1])
