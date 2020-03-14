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

class Engine(object):
    
    def __init__(self, data):
        if isinstance(data, str):
            data = pd.read_csv(data)
        self.data = data
        self.data_items = data.drop('user', 1)
        magnitude = np.sqrt(np.square(self.data_items).sum(axis=1))
        self.data_items = self.data_items.divide(magnitude, axis='index')
        self.data_matrix = calculate_similarity(self.data_items)
        
    def get_user_index(self, x):
        strindx = [str(ix) for ix in self.data.user]
        return strindx.index(str(x))
    
    def get_likes(self, user_index, n1 = 10, n2 = 20):
        data_neighbours = pd.DataFrame(index=self.data_matrix.columns, columns=range(1,n1 + 1))
        for i in range(0, len(self.data_matrix.columns)):
            data_neighbours.iloc[i,:n1] = self.data_matrix.iloc[0:,i].sort_values(ascending=False)[:n1].index
        known_user_likes = self.data_items.loc[user_index]
        known_user_likes = known_user_likes[known_user_likes >0].index.values
        most_similar_to_likes = data_neighbours.loc[known_user_likes]
        similar_list = most_similar_to_likes.values.tolist()
        similar_list = list(set([item for sublist in similar_list for item in sublist]))
        neighbourhood = self.data_matrix[similar_list].loc[similar_list]
        user_vector = self.data_items.ix[user_index].loc[similar_list]
        # Calculate the score.
        score = neighbourhood.dot(user_vector).div(neighbourhood.sum(axis=1))
        
        # Drop the known likes.
        score = score.drop(known_user_likes)
        return score.nlargest(n2)
    
def run(data, user, n1 = 10, n2 = 20):
    eng = Engine(data)
    return eng.get_likes(eng.get_user_index(user), n1, n2)