import pandas as pd

import matplotlib.pyplot as plt

from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile

from sklearn.model_selection import train_test_split


GENERATE_OUTPUT_CSV = False 

"""
Function to convert yelp.json into a smaller csv

From: https://towardsdatascience.com/sentiment-classification-using-cnn-in-pytorch-fba3c6840430
"""
if GENERATE_OUTPUT_CSV:
    INPUT_FOLDER = 'datasets/yelp'
    OUTPUT_FOLDER = 'datasets/yelp'
    def load_yelp_orig_data():
        PATH_TO_YELP_REVIEWS = INPUT_FOLDER + '/yelp_academic_dataset_review.json'
        num_bytes = 100000000
    
        # read the entire file into a python array
        with open(PATH_TO_YELP_REVIEWS, 'r') as f:
            data = f.readlines(num_bytes)
    
        data = map(lambda x: x.rstrip(), data)
    
        data_json_str = "[" + ','.join(data) + "]"
    
        data_df = pd.read_json(data_json_str)
        
        data_df.to_csv(OUTPUT_FOLDER + '/output_reviews_top.csv')

    load_yelp_orig_data()

def map_sentiment(stars_received):
    if stars_received <= 2:
        return -1
    elif stars_received == 3:
        return 0
    return 1


def get_top_data(top_data_df, top_n=5000):
    top_data_df_positive = top_data_df[top_data_df['sentiment'] == 1].head(top_n)
    top_data_df_negative = top_data_df[top_data_df['sentiment'] == -1].head(top_n)
    top_data_df_neutral = top_data_df[top_data_df['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative, top_data_df_neutral])
    return top_data_df_small


if __name__ == '__main__':
    df = pd.read_csv('datasets/yelp/output_reviews_top.csv')

    df['sentiment'] = [map_sentiment(x) for x in df['stars']]
    df = get_top_data(df, top_n=5000)

    # take a look at the balance of review sentiments
    # plt.figure()
    # pd.value_counts(df['sentiment']).plot.bar(title="Sentiment distribution in df")
    # plt.xlabel("Sentiment")
    # plt.ylabel("No. of rows in df")
    # plt.show()

    # Tokenize the text column to get the new column 'tokenized_text'
    df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in df['text']] 
    # print(df['tokenized_text'].head(10))

    porter_stemmer = PorterStemmer()
    # Get the stemmed_tokens
    df['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in df['tokenized_text'] ]
    df.to_csv('datasets/yelp/reviews.csv')
    # print(df['stemmed_tokens'].head(10))

    vec_size = 10

    # path = get_tmpfile('word2vec.model')
    temp_df = df['stemmed_tokens']
    model = Word2Vec(temp_df, min_count=1, size=vec_size, workers=3, window=3, sg=1)
    model.save('word2vec/word2vec.model')

    # save padded word2vec
    temp_df = pd.Series(df['stemmed_tokens']).values
    temp_df = list(temp_df)
    temp_df.append(['pad'])
    model = Word2Vec(temp_df, min_count=1, size=vec_size, workers=3, window=3, sg=1)
    model.save('word2vec/word2vec_PAD.model')

    print(model)
