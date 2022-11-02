import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


# settings
FILE = '../z_data/consumers_behavior.csv'
COLS_X = [
    'microwave', 'air_conditioner', 'tublet', 
    'piano', 'bicycle', 'smartphone', 'pc'
]
METRIC = 'euclidean'
METHOD = 'ward'
N_CLUSTERS_RANGE = range(2,6)
N_CLUSTERS = 3


def main():
    # read and cleansing data
    df = read_data()
    df = cleansing(df)
    
    # clustering
    hierarchical_method(df)


def read_data():
    df = pd.read_csv(FILE, index_col=0, header=0, encoding='shift-jis')
    return df


def cleansing(df):
    df[COLS_X] = (df[COLS_X]-df[COLS_X].mean()) / df[COLS_X].std()
    return df    


def hierarchical_method(df):    
    # hierarchy clustering
    result = linkage(df, metric=METRIC, method=METHOD)
    dend = dendrogram(result, labels=df.index)
    plt.show()
 
    
main()