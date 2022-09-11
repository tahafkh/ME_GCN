import os
import re
from math import *

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import twitter_samples

import emoji
import wordsegment
from sklearn.model_selection import train_test_split

RAW_DATA_DIRECTORY = 'raw_data'
DATA_DIRECTORY = 'data'
DATASET = 'sample'

def process_tweets(tweets, fa=False):
    # Process tweets
    tweets = emoji2word(tweets)
    tweets = remove_links(tweets)
    tweets = remove_usernames(tweets)
    tweets = replace_rare_words(tweets)
    tweets = remove_replicates(tweets)
    tweets = segment_hashtag(tweets)
    tweets = lower_case(tweets)
    tweets = remove_useless_punctuation(tweets)
    if fa == True:
        tweets = remove_eng(tweets)
        tweets = normalize(tweets)
    tweets = np.array(tweets)
    return tweets

def lower_case(sents):
    for i, sent in enumerate(sents):
        sents[i] = sent.lower()
    return sents

def normalize(sents):
    normalizer = Normalizer()
    for i, sent in enumerate(sents):
      sents[i] = normalizer.normalize(str(sent))
    return sents

def remove_links(sents):
    for i, sent in enumerate(sents):
        sents[i] = re.sub(r'^https?:\/\/.*[\r\n]*', 'http', str(sent), flags=re.MULTILINE)
    return sents

def remove_eng(sents):
    for i, sent in enumerate(sents):
        sents[i] = re.sub('[a-zA-Z0-9]','',str(sent))
    return sents

def remove_usernames(sents):
    for i, sent in enumerate(sents):
        sents[i] = re.sub('@[^\s]+','@USER',str(sent))
    return sents

def emoji2word(sents):
    return [emoji.demojize(str(sent)) for sent in sents]

def remove_useless_punctuation(sents):
    for i, sent in enumerate(sents):
        sent = sent.replace(':', ' ')
        sent = sent.replace('_', ' ')
        sent = sent.replace('...', ' ')
        sent = sent.replace('..', ' ')
        sent = sent.replace('’', '')
        sent = sent.replace('"', '')
        sent = sent.replace(',', '')
        sents[i] = sent
    return sents

def remove_replicates(sents):
    # if there are multiple `@USER` tokens in a tweet, replace it with `@USERS`
    # because some tweets contain so many `@USER` which may cause redundant
    for i, sent in enumerate(sents):
        if sent.find('@USER') != sent.rfind('@USER'):
            sents[i] = sent.replace('@USER ', '')
            sents[i] = '@USERS ' + sents[i]
    return sents

def replace_rare_words(sents):
    rare_words = {
        'URL': 'http'
    }
    for i, sent in enumerate(sents):
        for w in rare_words.keys():
            sents[i] = sent.replace(w, rare_words[w])
    return sents

def segment_hashtag(sents):
    # E.g. '#LunaticLeft' => 'lunatic left'
    for i, sent in enumerate(sents):
        sent_tokens = sent.split(' ')
        for j, t in enumerate(sent_tokens):
            if t.find('#') == 0:
                sent_tokens[j] = ' '.join(wordsegment.segment(t))
        sents[i] = ' '.join(sent_tokens)
    return sents

def read_file(data):
    fa = (data == 'fa')
    if data == 'olid':
        train_name = 'olid-training-v1.0.tsv'
        train = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, train_name), sep='\t', keep_default_na=False)
        train_ids = np.array(train['id'].values)
        train_tweets = np.array(train['tweet'].values)
        train_labels = np.array(train['subtask_a'].values)

        test_name = 'testset-levela.tsv'
        test = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, test_name), sep='\t', keep_default_na=False)
        test_ids = np.array(test['id'].values)
        test_tweets = np.array(test['tweet'].values)
        test_labels = np.array(test['subtask_a'].values)

    elif data == 'germeval':
        train_name = 'germeval2018.training.txt'
        train = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, train_name), sep='\t', keep_default_na=False, header=None)
        train_ids = np.array(range(1,len(train)+1))
        train_tweets = np.array(train[0].values)
        train_labels = np.array(train[1].values)

        test_name = 'germeval2018.test.txt'
        test = pd.read_csv(os.path.join(RAW_DATA_DIRECTORY, test_name), sep='\t', keep_default_na=False, header=None)
        test_ids = np.array(range(1,len(test)+1))
        test_tweets = np.array(test[0].values)
        test_labels = np.array(test[1].values)
    
    elif data == 'nltk':
        nltk.download('twitter_samples')
        pos = twitter_samples.strings('positive_tweets.json')
        pos_labels = ['POS' for i in range(len(pos))]
        neg = twitter_samples.strings('negative_tweets.json')
        neg_labels = ['NEG' for i in range(len(neg))]

        train_tweets = np.array(pos[:int(len(pos)*0.8)] + neg[:int(len(neg)*0.8)])
        train_labels = np.array(pos_labels[:int(len(pos_labels)*0.8)] + neg_labels[:int(len(neg_labels)*0.8)])
        train_ids = np.array(range(1, len(train_tweets)+1))

        test_tweets = np.array(pos[int(len(pos)*0.8):] + neg[int(len(neg)*0.8):])
        test_labels = np.array(pos_labels[int(len(pos_labels)*0.8):] + neg_labels[int(len(neg_labels)*0.8):])
        test_ids = np.array(range(1, len(test_tweets)+1))



    else:
        raise ValueError(f'Data {data} not supported.')

    train_tweets = process_tweets(train_tweets, fa=fa)
    test_tweets = process_tweets(test_tweets, fa=fa)

    train = pd.DataFrame({'id':train_ids, 'tweet':train_tweets, 'label': train_labels})
    test = pd.DataFrame({'id':test_ids, 'tweet':test_tweets, 'label': test_labels})
    return train, test

def sample_data(data, sample_size):
    _, new_data = train_test_split(data, test_size=sample_size, stratify=data['label'], random_state=13)
    return new_data

def prepare_data(args):
    wordsegment.load()
    train, test = read_file(args['dataset'])
    if not args['use_all']:
        train = sample_data(train, args['train_size'])
        test = sample_data(test, args['test_size'])
    return train, test