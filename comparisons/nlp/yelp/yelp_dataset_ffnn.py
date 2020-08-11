import pandas as pd

import torch

from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer

from sklearn.model_selection import train_test_split


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


def get_top_data(top_data_df, top_n=5000):
    top_data_df_positive = top_data_df[top_data_df['sentiment'] == 2].head(top_n)
    top_data_df_negative = top_data_df[top_data_df['sentiment'] == 0].head(top_n)
    top_data_df_neutral = top_data_df[top_data_df['sentiment'] == 1].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral])
    return top_data_df_small


def map_sentiment(stars_received):
    if stars_received <= 2:
        return 0
    elif stars_received == 3:
        return 1
    return 2


if __name__ == '__main__':

    """LOAD AND PREPROCESS DATA"""
    df = pd.read_csv('datasets/yelp/output_reviews_top.csv')
    df['sentiment'] = [map_sentiment(x) for x in df['stars']]
    df = get_top_data(df, top_n=5000)
    df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in df['text']] 
    porter_stemmer = PorterStemmer()
    df['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in df['tokenized_text'] ]
    X_train, X_test, Y_train, Y_test = split_train_test(df)


