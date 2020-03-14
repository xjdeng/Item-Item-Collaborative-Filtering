import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def calculate_similarity(data_items):
    """Calculate the column-wise cosine similarity for a sparse
    matrix. Return a new dataframe matrix with similarities.
    """
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    sim = pd.DataFrame(data=similarities, index= data_items.columns, columns= data_items.columns)
    return sim

def preprocess(data):
    if isinstance(data, str):
        data = pd.read_csv(data)
    data_items = data.drop('user', 1)
    magnitude = np.sqrt(np.square(data_items).sum(axis=1))
    data_items = data_items.divide(magnitude, axis='index')
    return calculate_similarity(data_items), data_items

def get_likes(data, data_matrix, data_items, user, n1 = 10, n2 = 20):
    user_index = data[data.user == user].index.tolist()[0]
    data_neighbours = pd.DataFrame(index=data_matrix.columns, columns=range(1,n1 + 1))
    for i in range(0, len(data_matrix.columns)):
        data_neighbours.iloc[i,:n1] = data_matrix.iloc[0:,i].sort_values(ascending=False)[:n1].index
    known_user_likes = data_items.loc[user_index]
    known_user_likes = known_user_likes[known_user_likes >0].index.values
    most_similar_to_likes = data_neighbours.loc[known_user_likes]
    similar_list = most_similar_to_likes.values.tolist()
    similar_list = list(set([item for sublist in similar_list for item in sublist]))
    neighbourhood = data_matrix[similar_list].loc[similar_list]
    user_vector = data_items.ix[user_index].loc[similar_list]
    # Calculate the score.
    score = neighbourhood.dot(user_vector).div(neighbourhood.sum(axis=1))
    
    # Drop the known likes.
    score = score.drop(known_user_likes)
    return score.nlargest(n2)

def run(data, user, n1 = 10, n2 = 20):
    if isinstance(data, str):
        data = pd.read_csv(data)
    data_matrix, data_items = preprocess(data)
    return get_likes(data, data_matrix, data_items, user, n1 ,n2)