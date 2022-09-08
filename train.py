import numpy as np
from tqdm import tqdm
import time

from math import log
import scipy.sparse as sp

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim

from cli import get_args
from data import prepare_data
from models import MultiGCN
from utils import *

if __name__=='__main__':
    args = get_args()
    train, test = prepare_data(args)

    train_labels = list(train['label'].values)
    test_labels = list(test['label'].values)

    train_sent = list(train['tweet'].values)
    test_sent = list(test['tweet'].values)
    original_sentences = train_sent + test_sent
    train_size = len(train_sent)
    test_size = len(test_sent)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unique_labels=np.unique(train_labels + test_labels)

    num_class = len(unique_labels)
    encoder = LabelEncoder()
    encoder.fit(unique_labels)

    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)

    all_labels = train_labels.tolist() + test_labels.tolist()
    labels = torch.LongTensor(all_labels).to(device)

    tokenize_sentences, word_list, vocab_length = tokenize(args, original_sentences)
    sentences = [' '.join(sentence) for sentence in tokenize_sentences]

    del original_sentences

    #word to id dict
    word_id_map = {word_list[i]: i for i in range(vocab_length)}         
    
    # info dict
    info_dict = {
        'word_list': word_list,
        'tokenize_sentences': tokenize_sentences,
        'sentences': sentences, 
        'all_labels': all_labels,
        'train_size': train_size,
        'num_class': num_class,
    }

    word_emb_dict = get_word_embeddings(args, info_dict)

    doc2vec_npy = get_doc_embeddings(args, info_dict)

    node_size = train_size + vocab_length + test_size
    adj_tensor = []

    ## d2w: tfidf
    tfidf_row = []
    tfidf_col = []
    tfidf_weight = []

    #get each word appears in which document
    word_doc_list = {}
    for word in word_list:
        word_doc_list[word]=[]

    for i in range(len(tokenize_sentences)):
        doc_words = tokenize_sentences[i]
        unique_words = set(doc_words)
        for word in unique_words:
            exsit_list = word_doc_list[word]
            exsit_list.append(i)
            word_doc_list[word] = exsit_list

    #document frequency
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    # term frequency
    doc_word_freq = {}

    for doc_id in range(len(tokenize_sentences)):
        words = tokenize_sentences[doc_id]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(tokenize_sentences)):
        words = tokenize_sentences[i]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row_tmp = i
            else:
                row_tmp = i + vocab_length
            col_tmp = train_size + j
            
            idf = log(1.0 * len(tokenize_sentences) / word_doc_freq[word_list[j]])
            weight_tmp = freq * idf
            doc_word_set.add(word)

            tfidf_row.append(row_tmp)
            tfidf_col.append(col_tmp)
            tfidf_weight.append(weight_tmp)

            tfidf_row.append(col_tmp)
            tfidf_col.append(row_tmp)
            tfidf_weight.append(weight_tmp)


    for i in range(node_size):
        tfidf_row.append(i)
        tfidf_col.append(i)
        tfidf_weight.append(1)

    co_dict = {}
    for sent in tokenize_sentences:
        for i,word1 in enumerate(sent):
            for word2 in sent[i:]:
                co_dict[ordered_word_pair(word_id_map[word1],word_id_map[word2])] = 1

    co_occur_threshold = args['threshold']

    doc_vec_bow = []
    for sent in tokenize_sentences:
        temp = np.zeros((vocab_length))
        for word in sent:
            temp[word_id_map[word]] = 1
        doc_vec_bow.append(temp)

    co_doc_dict = {}
    for i in range(len(doc_vec_bow)-1):
        for j in range(i+1,len(doc_vec_bow)):
            if np.dot(doc_vec_bow[i],doc_vec_bow[j]) >= co_occur_threshold:
                co_doc_dict[(i,j)] = 1


    adj_list = []

    for i in tqdm(range(args['dim'])):
        col = tfidf_col[:]
        row = tfidf_row[:]
        weight = tfidf_weight[:]
        for pair in co_dict:
            ind1, ind2 = pair

            word1 = word_list[ind1]
            word2 = word_list[ind2]
            tmp = np.tanh(1/np.abs(word_emb_dict[word1][i] - word_emb_dict[word2][i]))

            row.append(ind2+train_size)
            col.append(ind1+train_size)
            weight.append(tmp)

            row.append(ind1+train_size)
            col.append(ind2+train_size)
            weight.append(tmp)

        for pair in co_doc_dict:
            ind1, ind2 = pair        
            tmp = np.tanh(1/np.abs(doc2vec_npy[ind1][i] - doc2vec_npy[ind2][i]))

            if ind1>train_size:
                ind1 += vocab_length
            if ind2>train_size:    
                ind2 += vocab_length

            row.append(ind2)
            col.append(ind1)
            weight.append(tmp)

            row.append(ind1)
            col.append(ind2)
            weight.append(tmp)    

        
        adj_tmp = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
        adj_tmp = adj_tmp + adj_tmp.T.multiply(adj_tmp.T > adj_tmp) - adj_tmp.multiply(adj_tmp.T > adj_tmp)
        adj_tmp = normalize_adj(adj_tmp) 
        adj_tmp = sparse_mx_to_torch_sparse_tensor(adj_tmp, device)
        adj_list.append(adj_tmp)


    features = []
    for i in range(train_size):
        features.append(doc2vec_npy[i])

    for word in word_list:
        features.append(word_emb_dict[word])

    for i in range(test_size):
        features.append(doc2vec_npy[train_size+i])

    features = torch.FloatTensor(np.array(features)).to(device)

    # Training
    real_train_size = int((1-args['val_portion'])*train_size)
    val_size = train_size-real_train_size

    idx_train = range(real_train_size)
    idx_val = range(real_train_size,train_size)
    idx_test = range(train_size + vocab_length,node_size)


    final_acc_list = []
    for _ in range(args['runs']):
        model = MultiGCN(nfeat=features.shape[1], nhid=args['hidden_dim'], nclass=num_class, dropout=args['dropout'],
                        dim=args['dim'], pooling=args['pooling']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['decay'])
        criterion = nn.CrossEntropyLoss()

        val_loss = []
        for epoch in range(args['epochs']):

            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj_list)
            loss_train = criterion(output[idx_train], labels[idx_train])
            acc_train = cal_accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()


            model.eval()
            output = model(features, adj_list)

            loss_val = criterion(output[idx_val], labels[idx_val])
            val_loss.append(loss_val.item())
            acc_val = cal_accuracy(output[idx_val], labels[idx_val])
            print(  'Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'acc_val: {:.4f}'.format(acc_val),
                    'time: {:.4f}s'.format(time.time() - t))
            
            if epoch > args['early_stopping'] and np.min(val_loss[-args['early_stopping']:]) > np.min(val_loss[:-args['early_stopping']]) :
                print("Early Stopping...")
                break

        model.eval()
        output = model(features, adj_list)
        loss_test = criterion(output[idx_test], labels[-test_size:])
        acc_test = cal_accuracy(output[idx_test], labels[-test_size:])
        print("Test set results:",
                "loss= {:.4f}".format(loss_test.item()),
                "accuracy= {:.4f}".format(acc_test))

        final_acc_list.append(acc_test)

        print(classification_report(test_labels,torch.argmax(output[idx_test],-1).cpu().tolist(),digits = 4))
