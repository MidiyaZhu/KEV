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
class LabelWordCompatLayer(nn.Module):
    def __init__(self, output_dim, batch_size, hidden_size, n_layers, dropout,ngram):
        nn.Module.__init__(self)
        # Load pre-trained bert model
        self.batch_size = batch_size
        self.hidden_size= hidden_size
        self.n_layers = n_layers
        self.output_dim = output_dim
      

        self.linear = nn.Linear(hidden_size, output_dim)

        self.embedding_dim = bert.config.to_dict()['hidden_size']

        self.lstm = nn.LSTM(self.embedding_dim, hidden_size, num_layers=n_layers,
                            dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        assert ngram % 2 == 1, "n-gram should be odd number {2r+1}"
        self.phrase_filter = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            padding=(ngram - 1) // 2,  # pad should be (filter - 1)/2
            kernel_size=(ngram, 1))
        self.phrase_extract = nn.MaxPool2d(
            kernel_size=(1, output_dim))  # extract the max similarity of all labels for a word
        self.dropout = nn.Dropout(0.3)

        self.init_c()

    def init_c(self):
        self.c = Variable(torch.rand(size=(self.output_dim, self.embedding_dim)), requires_grad=True).to(device)

    def batch_cosinesim(self, v, c):
        normalized_v = v / torch.norm(v, p=2, dim=2).unsqueeze(2).repeat(1, 1, self.hidden_size)
        # a=normalized_v.sum(0)
        normalized_c = c / torch.norm(c, p=2, dim=1).unsqueeze(1).repeat(1, self.hidden_size)

        # nan -> pad_idx(0) or not-aligned label part
        normalized_v[torch.isnan(normalized_v)] = 0  # [b, l, h]
        normalized_c[torch.isnan(normalized_c)] = 0  # [k, h]

        normalized_c = normalized_c.unsqueeze(0).repeat(normalized_v.shape[0], 1, 1).permute(0, 2, 1)  # [b,h,k]
        g = torch.bmm(normalized_v, normalized_c)
        return g

    def forward(self, text,labelvec,bert):
      
        embeds,_ = bert(text)
        cc,_=bert(labelvec)
        embdim = embeds.shape[0]
        c_embdim=cc.shape[0]
      )
        h_0 = Variable(torch.zeros(self.n_layers, embdim, self.hidden_size).to(device))
        c_0 = Variable(torch.zeros(self.n_layers, embdim, self.hidden_size).to(device))
        hc_0 = Variable(torch.zeros(self.n_layers, c_embdim, self.hidden_size).to(device))
        cc_0 = Variable(torch.zeros(self.n_layers, c_embdim, self.hidden_size).to(device))
        '''final hidden state is the feature of each sentence'''
        '''output is the feature of each word in each sentence'''
        h_output, (final_hidden_state, final_cell_state) = self.lstm(embeds, (h_0, c_0))
        c_output, (final_hidden_state_c, final_cell_state_c) = self.lstm(cc, (hc_0, cc_0))
        c_output=c_output.squeeze(0)
        # g = self.dropout(self.batch_cosinesim(v, self.c))  # [b, l, k]
        g = self.dropout(self.batch_cosinesim(h_output, c_output))
       
        m = self.dropout(self.phrase_extract(g))
     
        b = torch.softmax(m, dim=1)  # [b, l, 1] #word attention weight with max similarity label

        return b, h_output





class LEAM(nn.Module):
    def __init__(self, batch_size,output_dim,hid_dim, nlayers,dropout, ngram):
        nn.Module.__init__(self)

        self.compat_model = LabelWordCompatLayer(
            hidden_size=hid_dim,
            n_layers=nlayers,
            ngram=ngram,
            dropout=dropout,
            output_dim=output_dim,
            batch_size=batch_size
        )
        self.bert = bert
        self.fc = nn.Linear(hid_dim, output_dim)
        self.batch_size = batch_size
        self.device = device
        self.dropout = nn.Dropout(dropout)
        self.output_dim=output_dim

    def forward(self, text,labelvector):

        labelvec=Variable(torch.ones(size=[1,self.output_dim],dtype=torch.int64,device=device))
        for i in range(len(labelvec)):
            labelvec[i]=labelvector[i]

        weight, embed = self.compat_model(text,labelvec,self.bert)  #weight is the word weight with the most possible label

        container = torch.full(size=embed.shape, fill_value=np.nan, device=self.device)

        for idx in range(weight.shape[0]):

            tmp = weight[idx] * embed[idx]
            container[idx] = tmp
        weighted_embed = container.sum(1)  #SENTENCE EMBEDDING

        z = self.dropout(self.fc(weighted_embed))

        return z,weighted_embed,weight

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




def train(model, iterator, optimizer, criterion,labelvector):
    weight=[]
    bias=[]
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            weight.append(m.weight.data)
            bias.append(m.bias.data)
    count=0
    for batch in iterator:

        optimizer.zero_grad()
        text = batch.text
        predictions,cat,_= model(text,labelvector)
       
        pp_log = predictions
        loss = criterion(pp_log, batch.label)
        acc,final= categorical_accuracy(pp_log, batch.label)
        count += len(batch.label)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / count

def evaluate(model, iterator, criterion,texicon,labelvector):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        count=0
        for batch in iterator:
            text= batch.text
            predictions,cat,word_weights = model(text,labelvector)

            prediction_log = predictions

            loss = criterion(prediction_log, batch.label)
            acc,final = categorical_accuracy(prediction_log, batch.label)
            count += len(batch.label)
          
    return epoch_loss / len(iterator), epoch_acc /count,cat_final


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__=='__main__':
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
         
    SEED =1234
  
    data_path = r'data/'


    train_file = 'dataset2_train.csv'
    valid_file = 'dataset2_val.csv'
    test_file = 'dataset2_test.csv'
   
    TEXT, LABEL,train_data, valid_data,test_data, texicon = data_loader(data_path,train_file,valid_file,test_file,SEED)



    N_EPOCHS = 30
    lr = 5e-4
    
    OUTPUT_DIM = len(LABEL.vocab)
    DROPOUT = 0.25
    N_LAYERS = 1
    BIDIRECTIONAL = True
   
    use_cuda = True
    attention_size = 16
    sequence_length = 5000
    BATCH_SIZE = 32

    model = LEAM(batch_size=BATCH_SIZE, output_dim=OUTPUT_DIM,
                  hid_dim=256,nlayers=N_LAYERS,dropout=DROPOUT,
                 ngram=3)

    for name, param in model.named_parameters():
         if name.startswith('bert'):
             param.requires_grad = False

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    train_iterator, valid_iterator,test_iterator = build_dataiterator(TEXT, LABEL,
                                                      train_data,valid_data, test_data,
                                                     device,BATCH_SIZE)


    pathl = r'model/LSTM/bestloss/LSTM-VALLOSS-np-' + str(version) + dataset_name + 'model.pt'


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

        train_loss, train_acc= train(model, train_iterator,
                                                                        optimizer, criterion1,labelvector)

        valid_loss, valid_acc= evaluate(
            model,
            valid_iterator,
            criterion1, texicon,labelvector)



        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < bestl_valid_loss:

        

            torch.save(model.state_dict(), pathl)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
 
    model.load_state_dict(torch.load(pathl))

    test_loss, test_acc= evaluate(
        model, test_iterator, criterion1, texicon,labelvector)
                