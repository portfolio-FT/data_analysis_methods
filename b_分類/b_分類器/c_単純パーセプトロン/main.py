import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# settings
FILE = '../z_data/positive_inspection.csv'
COLS_X = ['inspection1', 'inspection2']
COL_CLASSNAME = 'result'
COL_LABEL = 'label'
CLASS1 = 'positive'
CLASS2 = 'negative'
LABEL_DICT = {CLASS1:1, CLASS2:-1}
COUNT = 300
DIM = len(COLS_X)


def main():
    # read, cleansing and split data
    df = read_data()
    df = cleansing(df)
    df_train, df_test = train_test_split(df, test_size=1/4)
    
    # gradient descent mothod
    vec_b = gradient_descent_method(df_train)
    
    # calculate accuracy
    df_pred, accuracy = accuracy_evaluation(df_test, vec_b)
    
    # show results
    df_vec_b = pd.DataFrame(vec_b, index=['const']+COLS_X, columns=['vec_b'])
    print('')
    print('-------------------------------------')
    print(f'accuracy score : {round(accuracy,3)}')
    print('-------------------------------------')
    print('')
    print('-------------------------------------')
    print('b_vector')
    print(df_vec_b)
    print('-------------------------------------')
    

def read_data():
    # read data
    df = pd.read_csv(FILE, index_col=0, header=0, encoding='shift-jis')
    return df


def cleansing(df):
    # craete correct label
    for i in df.index:
        classname = df.at[i, COL_CLASSNAME]
        df.at[i, COL_LABEL] = LABEL_DICT[classname]

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

    # standardization
    for col in COLS_X:
        df[col] = (df[col]-df[col].mean()) / df[col].std()
    
    return df


def gradient_descent_method(df):
    # get index
    index = df.index
    
    # craete x-matrix and to DataFrame
    mat_x = np.array(df[COLS_X])
    mat_x = np.insert(mat_x,0,1,axis=1)
    cols_mat_x = ['constant'] + COLS_X
    df_mat_x = pd.DataFrame(mat_x, index=index, columns=cols_mat_x)
    
    # initial b-vector
    vec_b = np.zeros((DIM+1,1))
    b_list_list = []
    for w in range(DIM+1):
        b_list = []
        b_list_list.append(b_list)
    
    # gradient boosting
    for count in range(COUNT):  
        # reset g_vector
        g_vec = 0
                  
        # projection
        z = np.dot(mat_x,vec_b)
        df_z = pd.DataFrame(z, index=index, columns=['z'])
        
        # concat
        df_ = pd.concat([df_mat_x, df_z], axis=1)
        df_ = pd.concat([df_, df[COL_LABEL]],axis=1)
        df_['z_*_label'] = df_['z'] * df_[COL_LABEL]
        
        # leave only miss judged
        df_ = df_[df_['z_*_label']<=0]
        
        # create gradient vector
        index_ = df_.index
        for i in index_:
            label = df_.at[i, COL_LABEL]
            x_vec = df_.loc[i, cols_mat_x]
            x_vec = np.array(x_vec).reshape((DIM+1,1))
            g_vec = g_vec - (label * x_vec)
        
        # update b-vector
        alpha = (count+1)**(-1/2)
        vec_b = vec_b -alpha*g_vec
        
        # create b_list
        for i,b_list in enumerate(b_list_list):
            b_list.append(vec_b[i])
        
    # plot gradient boosting
    for i,b_list in enumerate(b_list_list):
        plt.plot(range(COUNT), b_list, label=f'b{i}')
        plt.title('boositng plot')
        plt.xlabel('boosting count')
        plt.ylabel('coefficient')
    plt.legend()
    plt.show()   
    
    
    return vec_b


def accuracy_evaluation(df, vec_b):
    # get index and numbers of data
    index = df.index
    n_data = len(df)
    
    # craete x-matrix
    mat_x = np.array(df[COLS_X])
    mat_x = np.insert(mat_x,0,1,axis=1)
    cols_mat_x = ['constant'] + COLS_X
    df_mat_x = pd.DataFrame(mat_x, index=index, columns=cols_mat_x)
    
    # create (B.T)X
    z = np.dot(mat_x, vec_b)
    df_z = pd.DataFrame(z, index=index, columns=['z'])
    
    # create df_predict
    df_pred = pd.concat([df_mat_x, df_z], axis=1)
    df_pred = pd.concat([df_pred, df[COL_LABEL]],axis=1)
    
    for i in index:
        # create label_predict
        if df_pred.at[i,'z'] >= 0:
            df_pred.at[i,'label_pred'] = 1
        elif df_pred.at[i,'z'] < 0:
            df_pred.at[i,'label_pred'] = -1
    
    # accuracy
    accuracy = (df_pred['label']==df_pred['label_pred']).sum() / n_data
    
    return df_pred, accuracy


if __name__ == '__main__':
    main() 