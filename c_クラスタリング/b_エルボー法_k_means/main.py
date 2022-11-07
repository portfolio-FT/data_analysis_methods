import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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
    df_cluster_predict, df_center_of_gravity = k_means(df)
    
    # show result
    for cluster in range(N_CLUSTERS):
        print('')
        print('-------------------------')
        print(f'cluster name : {cluster}')
        for index in df_cluster_predict[df_cluster_predict['cluster_predict']==cluster].index:
            print(index)
        print('-------------------------')
    
    print('')
    print('-------------------------')
    print('center_of_gravity_predict')
    print(df_center_of_gravity)
    print('-------------------------')
    

def read_data():
    df = pd.read_csv(FILE, index_col=0, header=0, encoding='shift-jis')
    return df


def cleansing(df):
    # standardization
    for col in COLS_X:
        df[col] = (df[col]-df[col].mean()) / df[col].std()
    return df    


def k_means(df):
    # get index
    index = df.index
    
    # decision number of clusters by elbow-method
    # create SSE lists
    sse = []    
    for n in N_CLUSTERS_RANGE:
        # create model
        model = KMeans(n_clusters=n, random_state=0)
        model.fit(df[COLS_X])
        sse.append(model.inertia_)        
    plt.plot(N_CLUSTERS_RANGE, sse, marker='o')
    plt.xlabel('number of clusters')
    plt.ylabel('SSE')
    plt.show()
    
    # create k-means model
    model = KMeans(n_clusters=N_CLUSTERS, random_state=0)
    model.fit(df[COLS_X])
    
    # get cluster predict
    cluster_predict = model.predict(df[COLS_X])
    df_cluster_predict = pd.DataFrame(cluster_predict, index=index, columns=['cluster_predict'])
        
    # get center-of-graviy vectors for each cluster
    center_of_gravity_predict = model.cluster_centers_
    df_center_of_gravity = pd.DataFrame(center_of_gravity_predict, columns=COLS_X)
   
    return df_cluster_predict, df_center_of_gravity
    
main()