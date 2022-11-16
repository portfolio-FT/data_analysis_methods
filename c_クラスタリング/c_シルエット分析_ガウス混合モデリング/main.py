import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture


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
    df_cluster_predict, df_center_of_gravity = gaussian_mixture_model(df)
    
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


def gaussian_mixture_model(df):
    # create index
    index = df.index
    
    # decision number of clusters by silhouette analysis
    for n_clusters in N_CLUSTERS_RANGE:
        
        # create Gaussian-Mixture model
        model = GaussianMixture(n_components=n_clusters, random_state=0)
        model.fit(df[COLS_X])
        
        # get cluster labels
        cluster_predict = model.predict(df[COLS_X])
        
        # get silhouette-coefficients
        silhouette_vals = silhouette_samples(df[COLS_X], cluster_predict, metric=METRIC)
        
        # initial settings
        y_lower = 10
        yticks = []
        
        for cluster in range(n_clusters):
            # create silhoutte-values by class
            ith_silhouette_vals = silhouette_vals[cluster_predict==cluster]
            ith_silhouette_vals.sort()
            
            # get size of silhoutte-values
            size_cluster = ith_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster
            
            # plot
            plt.barh(
                range(y_lower, y_upper),
                ith_silhouette_vals,
                height=1.0
            )
            yticks.append((y_lower+y_upper)/2)
            y_lower += len(ith_silhouette_vals)
        
        # plot siohouette coefficient average
        silhouette_avg = silhouette_score(df[COLS_X], cluster_predict)
        plt.axvline(silhouette_avg, color='red', linestyle='--')
        
        # plot settings
        plt.xlabel('silhouette coefficient')
        plt.ylabel('Cluster')
        plt.yticks(yticks, range(1,n_clusters+1))
        plt.show()
        
    # create Gaussian-Mixture model
    model = GaussianMixture(n_components=N_CLUSTERS, random_state=0)
    model.fit(df[COLS_X])
    
    # get cluster predict
    cluster_predict = model.predict(df[COLS_X])
    df_cluster_predict = pd.DataFrame(cluster_predict, index=index, columns=['cluster_predict'])
    
    # get center-of-graviy vectors for each cluster
    center_of_gravity_predict = model.means_
    df_center_of_gravity = pd.DataFrame(center_of_gravity_predict, columns=COLS_X)
    
    return df_cluster_predict, df_center_of_gravity
    

if __name__ == '__main__':
    main()