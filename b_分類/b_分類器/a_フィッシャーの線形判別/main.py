import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# settings
FILE = '../z_data/financial_indicator_2classes.csv'
COLS_X = ['FI2', 'FI5', 'FI6', 'FI8', 'FI11', 'FI13']
COL_CLASSNAME = 'condition'
DIM = len(COLS_X)


def main():
    # read, cleansing, split data
    df = read_data()
    df = cleansing(df)
    df_train, df_test = train_test_split(df, test_size=1/4)
    
    # fisher's linear discriminant analysis
    vec_b = analysis(df)
    print('')
    print('--------')
    print('b_vector')
    print(vec_b)
    print('--------')
    
    # accuracy evaluation
    df_pred, accuracy = accuracy_evaluation(df, vec_b)
    print('')
    print('--------')
    print(f'accuracy score : {round(accuracy,3)}')
    print('--------')
    

def read_data():
    # read data
    df = pd.read_csv(FILE, index_col=0, header=0, encoding='shift-jis')
    return df


def cleansing(df):
    # drop outliners
    for count in range(3):
        for col_x in COLS_X:
            q1 = df[col_x].quantile(0.25)
            q3 = df[col_x].quantile(0.75)
            iqr = q3 - q1
            lim_lower = q1 - 1.5*iqr
            lim_upper = q3 + 1.5*iqr
            index = df.index
            for i in index:
                value = df.at[i,col_x]
                if value < lim_lower or lim_upper < value:
                    df.at[i,col_x] = np.nan
        df = df.dropna(how='any')

    # standardize
    for col_x in COLS_X:
        mean = df[col_x].mean()
        std = df[col_x].std()
        df[col_x] = ( df[col_x] - mean ) / std
    
    return df


def analysis(df):    
    # get class names
    classes = df[COL_CLASSNAME].unique()
    class1 = classes[0]
    class2 = classes[1]
    
    # divide dataframe by class
    df_x = df[COLS_X]
    df_class1 = df[df[COL_CLASSNAME]==class1]
    df_class2 = df[df[COL_CLASSNAME]==class2]
    df_class1_x = df_class1[COLS_X]
    df_class2_x = df_class2[COLS_X]
    
    # get numbers of datas
    n = len(df)
    n1 = len(df_class1)
    n2 = len(df_class2)    
    
    # create x matrixes
    x_mat_all = np.array(df_x)
    x_mat_class1 = np.array(df_class1_x)
    x_mat_class2 = np.array(df_class2_x)    
    
    # create mean vectors
    mean_all = np.mean(x_mat_all,axis=0).reshape((1,DIM))
    mean_class1 = np.mean(x_mat_class1,axis=0).reshape((DIM,1))
    mean_class2 = np.mean(x_mat_class2,axis=0).reshape((DIM,1))
    
    # create variance-covariance matrixes
    s1 = np.cov(df_class1_x,rowvar=0,bias=1)
    s2 = np.cov(df_class2_x,rowvar=0,bias=1)
    s = ( n1*s1 + n2*s2 ) / ( n1+n2-2 )
    s_inv = np.linalg.inv(s)
    
    # calculate w vector
    vec_b = np.dot( s_inv, (mean_class1-mean_class2) )
    vec_b = pd.DataFrame(vec_b, index=COLS_X, columns=['vec_b'])
    
    return vec_b


def accuracy_evaluation(df, vec_b):   
    # get class names
    classes = df[COL_CLASSNAME].unique()
    class1 = classes[0]
    class2 = classes[1]
    
    # create numbers of data
    n = len(df)
    
    # create df_x
    df_x = df[COLS_X]
    
    # create x matrix
    x_mat_all = np.array(df_x)    
    
    # create mean vectors
    mean_all = np.mean(x_mat_all,axis=0).reshape((1,DIM))
     
    # calculate  discriminant score
    score = np.dot(x_mat_all,vec_b) - np.dot(mean_all,vec_b)
    df_score = pd.DataFrame(score,columns=['z'],index=df.index)
    df_pred = pd.concat([df,df_score],axis=1)
    
    # create discriminant result dataframe
    index = df_pred.index
    for i in index:
        if df_pred.at[i,'z'] > 0:
            df_pred.at[i,'classname_pred'] = class1
        else:
            df_pred.at[i,'classname_pred'] = class2
    for i in index:
        if df_pred.at[i,COL_CLASSNAME] == df_pred.at[i,'classname_pred']:
            df_pred.at[i,'correct_count'] = 1
        else:
            df_pred.at[i,'correct_count'] = 0
    accuracy = df_pred['correct_count'].sum() / n
    
    return df_pred, accuracy
    
    
main()