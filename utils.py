import pandas as pd
from tqdm import tqdm

def convert_labels(df, save=None):
    df = pd.read_csv('./data/sentiment_training.csv') # header=0, usecols=[0, 5], names=['score', 'text']

    for i, tweet in enumerate(tqdm(df1['score'])):
        if tweet == 'negative':
            df1['score'][i] = -1
        elif tweet == 'neutral':
            df1['score'][i] = 0
        else:
            df1['score'][i] = 1

     if save != None:
        df.to_csv(save, index=False)
    else:
        return df

def combine_datasets(df1, df2, save=None):
    df1 = pd.read_csv(df1)
    df2 = pd.read_csv(df2)

    df1.append(df2)

    if save != None:
        df1.to_csv(save, index=False)
    else:
        return df1

def unify_header(df, cols=[0, 1], save=None):
    df = pd.read_csv(df, header=0, usecols=cols, names=['score', 'text'])

    if save != None:
        df1.to_csv(save, index=False)
    else:
        return df1