import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# settings
FILE = '../z_data/financial_indicator.csv'
COL_CLASS = 'condition'
COLS_X = ['FI2', 'FI5', 'FI6', 'FI8', 'FI11', 'FI13']
DIM = len(COLS_X)


def main():
    # read, cleansing and split data
    df = read_data()
    df = cleansing(df)
    df_train, df_test = train_test_split(df, test_size=1/4)
    
    # analysis
    acccuracy, df_accuracy = mahalanobis(df)
    
    # show results
    print('')
    print('--------------------------------------')
    print(f'accuracy score : {round(acccuracy,3)}')
    print('--------------------------------------')
    

def read_data():
    # read data
    df = pd.read_csv(FILE, index_col=0, header=0, encoding='shift-jis')
    return df


def cleansing(df):
    # remove outliers
    for count in range(3):
        for col in COLS_X:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lim_lower = q1 - 1.5*iqr
            lim_upper = q3 + 1.5*iqr
            for i in df.index:
                value = df.at[i,col]
                if value < lim_lower or lim_upper < value:
                    df.at[i,col] = np.nan
        df = df.dropna(subset=COLS_X)
    
    return df


def mahalanobis(df):    
    # get class
    class_list = list( df[COL_CLASS].unique() )
    n_class = len(class_list)
    sample_names = df.index
    
    # divide DataFrame by class
    df_all = df[COLS_X]
    df_list = []
    for n in range(n_class):
        class_name = class_list[n]
        df_class = df[df[COL_CLASS]==class_name][COLS_X]
        df_list.append(df_class)
    
    # get numbers of data for each class
    n_all = len(df_all)
    n_list = []
    for n in range(n_class):
        df_class = df_list[n]
        n = len(df_class)
        n_list.append(n)
    
    # create matrixes
    x_mat_all = np.array(df_all)
    x_mat_list = []
    for n in range(n_class):
        df_class = df_list[n]
        x_mat = np.array(df_class)
        x_mat_list.append(x_mat)
        
    # create mean vectors
    mean_all = np.mean(x_mat_all,axis=0).reshape((DIM,1))
    x_mean_list = []
    for n in range(n_class):
        x_mat = x_mat_list[n]
        x_mean = np.mean(x_mat,axis=0).reshape((DIM,1))
        x_mean_list.append(x_mean)
    
    # create variance-covariance matrixes
    s_list = []
    for n in range(n_class):
        x_mat = x_mat_list[n]
        s = np.cov(x_mat,rowvar=0,bias=0)
        s_list.append(s)
    
    # calculate Mahalanobis's distance for each class
    df_mahala_list = []
    for n in range(n_class):
        class_name = class_list[n]
        mahala_dist_list = []
        x_mean = x_mean_list[n]
        s = s_list[n]
        s_inv = np.linalg.inv(s)
        for n in range(n_all):
            x = x_mat_all[n,:].reshape((DIM,1))
            mahala_dist = np.dot( (x-x_mean).T, s_inv )
            mahala_dist = np.dot( mahala_dist, (x-x_mean) )
            mahala_dist = mahala_dist[0][0]
            mahala_dist_list.append(mahala_dist)
        df_mahala_class = pd.DataFrame(
            mahala_dist_list,
            columns=[class_name], 
            index=sample_names
        )
        df_mahala_list.append(df_mahala_class)
    
    # to DataFrame
    df_mahala = pd.concat( [df for df in df_mahala_list], axis=1 )    
    
    # judge
    df_judge = pd.DataFrame(
        df_mahala[class_list].idxmin(axis=1),
        columns=['judge'],
        index=sample_names
    )
    
    # concat
    df_accuracy = pd.concat([df,df_mahala],axis=1)
    df = pd.concat([df_accuracy,df_judge],axis=1)
    
    # misidentification rate
    for sample in sample_names:
        actual = df.at[sample,COL_CLASS]
        judge = df.at[sample,'judge']
        if actual == judge:
            df.at[sample,'count'] = 1
        else:
            df.at[sample,'count'] = 0
    accuracy = df['count'].sum() / n_all
    
    return accuracy, df_accuracy
    

if __name__ == '__main__':
    main() 