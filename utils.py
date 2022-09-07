import scipy.sparse as sp
import numpy as np
import torch

import nltk
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models import FastText
# from glove import Corpus, Glove

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from transformers import (
        BertModel, BertTokenizer, \
        XLMModel, XLMTokenizer, \
        XLMRobertaModel, XLMRobertaTokenizer)

mybert = None

def ordered_word_pair(a, b):
    if a > b:
        return (b, a)
    else:
        return (a, b)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx, device):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

def cal_accuracy(predictions,labels):
    pred = torch.argmax(predictions,-1).cpu().tolist()
    lab = labels.cpu().tolist()
    cor = 0
    for i in range(len(pred)):
        if pred[i] == lab[i]:
            cor += 1
    return cor/len(pred)

def tokenize(args, original_sentences):
    nltk.download('stopwords')
    if args['data'] == 'en':
        stop_words = set(stopwords.words('english'))

    original_word_freq = {}  # to remove rare words
    for sentence in original_sentences:
        word_list = sentence.split()
        for word in word_list:
            if word in original_word_freq:
                original_word_freq[word] += 1
            else:
                original_word_freq[word] = 1   

    tokenize_sentences = []
    vocab = set()
    for sentence in original_sentences:
        word_list_temp = sentence.split()
        doc_words = []
        for word in word_list_temp:
            if word not in stop_words and original_word_freq[word] >= args['min_frequency']:
                doc_words.append(word)
                vocab.add(word)
        tokenize_sentences.append(doc_words)
    word_list = list(vocab)
    vocab_length = len(word_list)
    return tokenize_sentences, word_list, vocab_length

def get_model_cls(model):
    if model == "bert":
        return "bert-base-multilingual-uncased", BertTokenizer, BertModel
    elif model == "xlm":
        return "xlm-mlm-17-1280", XLMTokenizer, XLMModel
    elif model == "xlmr":
        return "xlm-roberta-base", XLMRobertaTokenizer, XLMRobertaModel
    else:
        raise NameError(f"Unsupported model {model}.")

def get_word2vec_embeddings(args, tokenize_sentences, word_list):
    wv_cbow_model = Word2Vec(sentences=tokenize_sentences, size=args['dim'], window=5, min_count=0, workers=4, sg=0, iter=200)
    word_emb_dict = {word: wv_cbow_model[word].tolist() for word in word_list}
    return word_emb_dict

def get_fasttext_embeddings(args, tokenize_sentences, word_list):
    ft_sg_model = FastText(sentences=tokenize_sentences, size=args['dim'], window=5, min_count=0, workers=4, sg=0, iter = 200)
    word_emb_dict = {word: ft_sg_model[word].tolist() for word in word_list}
    return word_emb_dict

def get_glove_embeddings(args, tokenize_sentences, word_list):
    corpus = Corpus()
    corpus.fit(tokenize_sentences, window=5)
    glove = Glove(no_components=args['dim'], learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=200, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    word_emb_dict = {word: glove.word_vectors[glove.dictionary[word]].tolist() for word in word_list}
    return word_emb_dict

def finetune(args, info_dict, key):
    info = info_dict[key]
    global mybert
    if mybert is None:
        model_name, tokenizer_cls, model_cls = get_model_cls(args['model'])
        tokenizer = tokenizer_cls.from_pretrained(model_name)
        mybert = model_cls.from_pretrained(model_name)
        mybert.to(args['device'])
    tokenized = tokenizer(info, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized["input_ids"].to(args['device'])
    attention_mask = tokenized["attention_mask"].to(args['device'])
    with torch.no_grad():
        last_hidden_states = mybert(input_ids, attention_mask=attention_mask)
    return last_hidden_states[0][:,0,:].cpu().numpy()

def get_bert_embeddings(args, info_dict, key):
    info = info_dict[key]
    global mybert
    if mybert is None:
        finetune(args, info_dict, key)

def get_word_embeddings(args, info_dict):
    tokenize_sentences = info_dict['tokenize_sentences']
    word_list = info_dict['word_list']
    embedding = args['word_embedding']
    if embedding == 'word2vec':
        return get_word2vec_embeddings(args, tokenize_sentences, word_list)
    elif embedding == 'fasttext':
        return get_fasttext_embeddings(args, tokenize_sentences, word_list)
    elif embedding == 'glove':
        return get_glove_embeddings(args, tokenize_sentences, word_list)
    elif embedding == 'bert':
        return get_bert_embeddings(args, info_dict, 'tokenized_sentences')
    else:
        raise NameError(f"Unsupported embedding {embedding}.")

def get_doc2vec_embeddings(args, tokenize_sentences):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenize_sentences)]
    model = Doc2Vec(documents, vector_size=args['dim'], window=5, min_count=1, workers=4, iter=200)

    doc2vec_emb = [model.docvecs[i] for i in range(len(documents))]
    return np.array(doc2vec_emb)

def get_doc_embeddings(args, info_dict):
    tokenize_sentences = info_dict['tokenize_sentences']
    embedding = args['doc_embedding']
    if embedding == 'doc2vec':
        return get_doc2vec_embeddings(args, tokenize_sentences)
    elif embedding == 'bert':
        return get_bert_embeddings(args, info_dict, 'word_list')
    else:
        raise NameError(f"Unsupported embedding {embedding}.")
