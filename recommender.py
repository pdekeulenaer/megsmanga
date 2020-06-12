import pandas as pd
import numpy as np
from pandas import DataFrame

DATA_PATH = 'data'

def load_data():
    anime = pd.read_csv("{}/anime.csv".format(DATA_PATH))
    ratings = pd.read_csv("{}/rating.csv".format(DATA_PATH))

    df = ratings.merge(anime, on='anime_id')
    df.rename(columns={'rating_x':'user_rating', 'rating_y':'total_avg_rating', 'name':'movie_name'}, inplace=True)

    df['user_has_watched'] = 1
    
    return df, anime, ratings

# Normalize user ratings
# expects dataframe with 1 row per user/anime combination
def normalize_user_ratings(df):
    # remove -1 ratings and replace with average
    df['user_rating'] = df['user_rating'].replace(-1, np.nan)

    # compute means and stadnard deviations
    user_means = df.groupby('user_id')['user_rating'].mean().rename('mean_user_rating')
    user_sdevs = np.sqrt(df.groupby('user_id')['user_rating'].var()).rename('stdev_user_rating')

    # combine and rename
    df = df.merge(user_means, on='user_id').merge(user_sdevs, on='user_id')
    df['user_rating'] = df['user_rating'].fillna(df['mean_user_rating'])

    # compute the normalized rating
    df['normalized_user_rating'] = (df['user_rating'] - df['mean_user_rating']) / (df['stdev_user_rating'])

    # set those with nan (i.e., no variance in rating) to 0
    return df['normalized_user_rating'].fillna(0.0)

def create_user_matrix(df, valuefield='watched'):
    matrix = pd.pivot_table(df, values=valuefield, index='user_id', columns='movie_name')
    return matrix


# process:
# > user based filter
# > content based filter
# > combining different filters into 1
# > serving model for an unknown user
# > configure new filters
# > validation of recommendations

def main():
    df, anime, ratings = load_data()
    print (df.head())

if __name__ == '__main__':
    main()