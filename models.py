import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution

from transformers import (
        BertModel, BertTokenizer, \
        XLMModel, XLMTokenizer, \
        XLMRobertaForSequenceClassification, XLMRobertaTokenizer)

class MultiGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, dim, pooling):
        super(MultiGCN, self).__init__()
        
        self.dim = dim
        self.pooling = pooling

        # different weights
        self.intras1 = nn.ModuleList([GraphConvolution(nfeat, nhid, dropout, activation = nn.ReLU()) for i in range(dim)])
        self.intras2 = nn.ModuleList([GraphConvolution(nhid*dim, nclass, dropout, activation = nn.ReLU()) for i in range(dim)])


    def forward(self, x, adj, feature_less=False):
        x = torch.stack([self.intras1[i](x,adj[i],feature_less) for i in range(self.dim)]) 
        x = x.permute(1, 0, 2) 
        x = x.reshape(x.size()[0], -1)  
        x = torch.stack([self.intras2[i](x,adj[i]) for i in range(self.dim)]) 

 
        if self.pooling == 'avg':
            return torch.mean(x,0)
        if self.pooling == 'max':
            return torch.max(x,0)[0]
        if self.pooling == 'min':
            return torch.min(x,0)[0] 
class MyBert(nn.Module):
    def __init__(self, bert, word_embedding, hidden_size=768, num_classes=2):
        super(MyBert, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(hidden_size, word_embedding)
        self.fc2 = nn.Linear(word_embedding, num_classes)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, pooled_output = self.bert(x, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        out = self.fc(pooled_output)
        return self.softmax(out)