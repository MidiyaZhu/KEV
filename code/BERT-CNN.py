from sklearn import svm
import torch
from torchtext import data
import torch.optim as optim
import random
import torch.nn as nn
# from nnet.modules import EmbedLayer, Encoder, Dot_Attention, Node_Attention, Classifier, ContrastiveLoss
import torch.nn.functional as F
import time
import spacy
from numpy import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import joblib
from torch.autograd import Variable
import const
from pytorch_transformers import *
from transformers import BertTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer('vocab_update.txt', do_lower_case= True)
init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id


max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
print('max input length',max_input_length)
# exit()
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

def data_loader(path,train_file,valid_file,test_file,SEED):
    TEXT = data.Field(batch_first=True,
                      use_vocab=False,
                      tokenize=tokenize_and_cut,
                      preprocessing=tokenizer.convert_tokens_to_ids,
                      init_token=init_token_idx,
                      eos_token=eos_token_idx,
                      pad_token=pad_token_idx,
                      unk_token=unk_token_idx)
  
    LABEL = data.LabelField(dtype=torch.long)
 
    fields = [('text', TEXT), ('label', LABEL)]

    train_data, valid_data, test_data = data.TabularDataset.splits(path=path,
                                                                   train=train_file,
                                                                   validation=valid_file,
                                                                   test=test_file,
                                                                   format='csv',
                                                                   fields=fields,
                                                                   skip_header=True)


    LABEL.build_vocab(train_data)

    texicon = LABEL.vocab.itos
    print('label index: ',len(LABEL.vocab.stoi))
   
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    print(texicon)
    return TEXT,LABEL,train_data,valid_data, test_data,texicon

def build_dataiterator(TEXT,LABEL,train_data,valid_data,test_data,device,BATCH_SIZE):
   
    fields =  [('text', TEXT), ('label', LABEL)]
    train_data = data.Dataset(train_data, fields, filter_pred=None)
    valid_data = data.Dataset(valid_data, fields, filter_pred=None)
    test_data = data.Dataset(test_data,fields,filter_pred=None)



    print('traindata:',len(train_data),'validdata:',len(valid_data),'test: ',len(test_data))
  

    train_iterator, valid_iterator, test_iterator= data.BucketIterator.splits(
        (train_data, valid_data,test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=False,
        sort_key=lambda x: len(x.text),
        device=device)

    return train_iterator,valid_iterator,test_iterator

bert = BertModel.from_pretrained('bert-base-uncased')


class ClassificationBert_CNN(nn.Module):
    def __init__(self, num_labels,batch_size,hidden_dim,n_layers,dropout, n_filters, filter_sizes):
        super(ClassificationBert_CNN, self).__init__()
        # Load pre-trained bert model
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = bert
        self.linear = nn.Linear(hidden_dim , num_labels)

        self.embedding_dim = bert.config.to_dict()['hidden_size']

        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim,num_layers=n_layers,
                            dropout=dropout, batch_first=True)

        self.drop = nn.Dropout(dropout)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, self.embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_labels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
     
        with torch.no_grad():
            all_hidden, pooler = self.bert(x)

        embdim = all_hidden.shape[0]
        embedded = all_hidden.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat), cat



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    a=correct.sum()
    b=torch.FloatTensor([y.shape[0]]).cuda()
    c=max_preds
    return correct.sum().cuda(),max_preds
parm = {}
def parm_to_excel(excel_name,key_name,parm):
    with pd.ExcelWriter(excel_name) as writer:
        [output_num,input_num,filter_size,_]=parm[key_name].size()
        for i in range(output_num):
            for j in range(input_num):
                data=pd.DataFrame(parm[key_name][i,j,:,:].detach().numpy())
                #print(data)
                data.to_excel(writer,index=False,header=True,startrow=i*(filter_size+1),startcol=j*filter_size)

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    count=0
    for batch in iterator:

        optimizer.zero_grad()
        text = batch.text

        predictions, cat = model(text)

        pp_log = predictions
      
        loss = criterion(pp_log, batch.label)
        acc,final= categorical_accuracy(pp_log, batch.label)
        count+=len(batch.label)
        loss.backward()

        optimizer.step()
     
        epoch_loss += loss.item()
        epoch_acc += acc.item()

   
    return epoch_loss / len(iterator), epoch_acc /count

def evaluate(model, iterator, criterion,texicon):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()
  
    with torch.no_grad():
        count=0
   
        for batch in iterator:
            text = batch.text

            predictions, cat = model(text)

            prediction_log = predictions

            loss = criterion(prediction_log, batch.label)
            acc,final = categorical_accuracy(prediction_log, batch.label)
            count += len(batch.label)
          
    return epoch_loss / len(iterator), epoch_acc /count

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



if __name__=='__main__':
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
  
    data_path = r'data/'


    train_file = 'dataset2_train.csv'
    valid_file = 'dataset2_val.csv'
    test_file = 'dataset2_test.csv'
    TEXT, LABEL,train_data, valid_data,test_data, texicon = data_loader(data_path,train_file,valid_file,test_file,1234)


    N_EPOCHS =20
    lr = 5e-4

    FILTER_SIZES = [2]
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.5
    N_LAYERS = 2
    BIDIRECTIONAL = True

    use_cuda = True
    attention_size = 16
    sequence_length = 5000
    BATCH_SIZE = 32
    N_FILTERS = 100
    model = ClassificationBert_CNN(num_labels=OUTPUT_DIM, batch_size=BATCH_SIZE, hidden_dim=256,
                                   n_layers=N_LAYERS,
                                   dropout=DROPOUT, n_filters=N_FILTERS, filter_sizes=FILTER_SIZES)

    for name, param in model.named_parameters():
        if name.startswith('bert'):
            param.requires_grad = False

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_iterator, valid_iterator,test_iterator = build_dataiterator(TEXT, LABEL,
                                                      train_data,valid_data, test_data,
                                                     device,BATCH_SIZE)


    pathl = r'model.pt'


    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(),lr=lr)
 

    #Loss function
    criterion1=nn.CrossEntropyLoss()



    model = model.to(device)
  
    criterion1 = criterion1.to(device)
    best_valid_posacc=float(0)
    bestl_valid_acc = float(0)
    best_valid_negacc = float(0)
    bestl_valid_loss = float('inf')
 
    for epoch in range(N_EPOCHS):

        start_time = time.time()
     
        train_loss, train_ac= train(model, train_iterator,
                                                                        optimizer, criterion1)
        valid_loss, valid_acc= evaluate(
            model,
            valid_iterator,
            criterion1, texicon)
                                              
        train_accuracy.append(train_acc)
        valid_accuracy.append(valid_acc)
        train_losslist.append(train_loss)
        valid_losslist.append(valid_loss)
        iteration.append(epoch)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < bestl_valid_loss:

            torch.save(model.state_dict(), pathl)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
  
    model.load_state_dict(torch.load(pathl))
 

    test_loss, test_acc= evaluate(
        model, test_iterator, criterion1, texicon)
              