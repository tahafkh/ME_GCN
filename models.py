import torch
import torch.nn as nn

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
    def __init__(self, bert, word_embedding, num_classes=2):
        super(MyBert, self).__init__()
        self.bert = bert
        self.embedding_dim = bert.config.hidden_size
        self.fc1 = nn.Linear(self.embedding_dim, word_embedding)
        self.fc2 = nn.Linear(word_embedding, num_classes)

    def forward(self, input_ids, attention_mask):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
        out = self.fc1(sequence_output[:, 0, :].view(-1, self.embedding_dim))
        out = self.fc2(out)
        return out