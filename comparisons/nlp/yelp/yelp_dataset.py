import torch
from torch import nn
import torch.nn.functional as F

import pandas as pd

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer

from sklearn.model_selection import train_test_split

import json

from PAGNN.pagnn import PAGNN

import networkx as nx

from tqdm import tqdm


def map_sentiment(stars_received):
    if stars_received <= 2:
        return 0
    elif stars_received == 3:
        return 1
    return 2


def get_top_data(top_data_df, top_n=5000):
    top_data_df_positive = top_data_df[top_data_df['sentiment'] == 1].head(top_n)
    top_data_df_negative = top_data_df[top_data_df['sentiment'] == -1].head(top_n)
    top_data_df_neutral = top_data_df[top_data_df['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral])
    return top_data_df_small


def split_train_test(top_data_df_small, test_size=0.3, shuffle_state=True):
    X_train, X_test, Y_train, Y_test = train_test_split(top_data_df_small[['business_id', 'cool', 'date', 'funny', 'review_id', 'stars', 'text', 'useful', 'user_id', 'stemmed_tokens']], 
                                                        top_data_df_small['sentiment'], 
                                                        shuffle=shuffle_state,
                                                        test_size=test_size, 
                                                        random_state=15)
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()

    Y_train = Y_train.to_frame()
    Y_train = Y_train.reset_index()
    Y_train = torch.tensor(Y_train['sentiment'])

    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    Y_test = torch.tensor(Y_test['sentiment'])

    return X_train, X_test, Y_train, Y_test


def make_w2v_vector(w2v, sentence, cnn=False, max_sen_len=None, padding_idx=None):
    if cnn:
        X = [padding_idx for i in range(max_sen_len)]
    else:
        X = []

    for i, word in enumerate(sentence):
        if word not in w2v.wv.vocab:
            emb = 0
            if cnn:
                X[i] = emb
            else:
                X.append(emb)
            print(word, 'not in vocab')
        else:
            emb = w2v.wv.vocab[word].index
            if cnn:
                X[i] = emb
            else:
                X.append(emb)

    out = torch.tensor(X, dtype=torch.long)
    if cnn:
        out = out.view(1, -1)

    return out


class CnnTextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, num_filters, embedding_size, window_sizes=(1,2,3,5)):
        super(CnnTextClassifier, self).__init__()
        w2vmodel = gensim.models.KeyedVectors.load('word2vec/word2vec_PAD.model')
        self.w2vmodel = w2vmodel
        weights = w2vmodel.wv
        # With pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights.vectors), padding_idx=w2vmodel.wv.vocab['pad'].index)
        # Without pretrained embeddings
        # self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.convs = nn.ModuleList([
                                   nn.Conv2d(1, num_filters, [window_size, embedding_size], padding=(window_size - 1, 0))
                                   for window_size in window_sizes
        ])

        self.fc = nn.Linear(num_filters * len(window_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x) # [B, T, E]

        # Apply a convolution + max_pool layer for each window size
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)

        # FC
        x = x.view(x.size(0), -1)
        logits = self.fc(x)

        probs = F.softmax(logits, dim = 1)

        return probs


if __name__ == '__main__':
    w2v = Word2Vec.load('word2vec/word2vec.model')
    print(w2v)

    """LOAD AND PREPROCESS DATA"""
    df = pd.read_csv('datasets/yelp_dataset/output_reviews_top.csv')
    df['sentiment'] = [map_sentiment(x) for x in df['stars']]
    df = get_top_data(df, top_n=5000)
    # Tokenize the text column to get the new column 'tokenized_text'
    df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in df['text']] 
    # Get the stemmed_tokens
    porter_stemmer = PorterStemmer()
    df['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in df['tokenized_text'] ]
    # get train / test sets
    X_train, X_test, Y_train, Y_test = split_train_test(df)

    D = w2v.vector_size
    C = 3 # 3 classes = -1, 0, 1 (sentiments)
    pagnn = PAGNN(D + C + 5, D, C, graph_generator=nx.generators.classic.complete_graph)
    struc = pagnn.structure_adj_matrix
    print(pagnn)

    cnn = CnnTextClassifier(vocab_size=len(w2v.wv.vocab), num_classes=C, num_filters=10, embedding_size=D)
    print(cnn)

    max_sen_len = df['stemmed_tokens'].map(len).max()

    use_tqdm = True
    lr = 0.001
    optimizer = torch.optim.Adam(pagnn.parameters(), lr=lr)
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2v.wv.vectors), padding_idx=w2v.wv.vocab['pad'].index)

    cnn_lr = 0.001
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=cnn_lr)
    padding_idx = cnn.w2vmodel.wv.vocab['pad'].index

    for epoch in range(1):
        print('epoch', epoch)
        # TRAIN
        pagnn.train()
        cnn.train()

        total_loss = 0
        cnn_total_loss = 0
        iterator = X_train.iterrows()
        if use_tqdm:
            iterator = tqdm(iterator, total=len(X_train))
        with torch.enable_grad():
            for index, row in iterator:
                optimizer.zero_grad()
                cnn_optimizer.zero_grad()

                # convert review into w2v embedding
                unvec_x = row['stemmed_tokens']
                x = make_w2v_vector(w2v, unvec_x)
                x_cnn = make_w2v_vector(cnn.w2vmodel, unvec_x, cnn=True, max_sen_len=max_sen_len, padding_idx=padding_idx)
                emb = embedding(x)
                t = Y_train[index].unsqueeze(0)

                for vec in emb:
                    vec = vec.unsqueeze(0)
                    struc.load_input_neurons(vec)
                    struc.step()
                y = struc.extract_output_neurons()

                cnn_y = cnn(x_cnn)

                loss = F.cross_entropy(y, t)
                total_loss += loss.item()

                cnn_loss = F.cross_entropy(cnn_y, t)
                cnn_total_loss += loss.item()

                loss.backward()
                optimizer.step()

                cnn_loss.backward()
                cnn_optimizer.step()


        avg_loss = total_loss / len(X_train)
        cnn_avg_loss = cnn_total_loss / len(X_train)
        print('[PAGNN] train loss: %f' % (avg_loss))
        print('[CNN] train loss: %f' % (cnn_avg_loss))

        # EVAL
        pagnn.eval()
        cnn.eval()
        iterator = X_test.iterrows()
        if use_tqdm:
            iterator = tqdm(iterator, total=len(X_test))
        total_correct = 0
        cnn_total_correct = 0
        with torch.no_grad():
            for index, row in iterator:
                # convert review into w2v embedding
                unvec_x = row['stemmed_tokens']
                x = make_w2v_vector(w2v, unvec_x)
                x_cnn = make_w2v_vector(cnn.w2vmodel, unvec_x, cnn=True, max_sen_len=max_sen_len, padding_idx=padding_idx)
                emb = embedding(x)
                t = Y_train[index].unsqueeze(0)

                for vec in emb:
                    vec = vec.unsqueeze(0)
                    struc.load_input_neurons(vec)
                    struc.step()
                y = struc.extract_output_neurons()
                cnn_y = cnn(x_cnn)

                _, pred = torch.max(y.data, 1)
                total_correct += torch.sum(pred == t).item()
                
                _, cnn_pred = torch.max(cnn_y.data, 1)
                cnn_total_correct += torch.sum(cnn_pred == t).item()

        accuracy = total_correct / len(X_test)
        cnn_accuracy = cnn_total_correct / len(X_test)
        print('[PAGNN] test accuracy:', accuracy)
        print('[CNN] test accuracy:', cnn_accuracy)
