import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# settings1 columns and dimsensions
COLS_X = ['FI2' ,'FI5', 'FI6', 'FI8', 'FI11', 'FI13']
COL_Y = 'condition'
COL_LABEL = 'label'
LABEL_DICT = {'good':0, 'bad':1}
DIM = len(COLS_X)

# settings3 the others
C = 0.5
FILE = '../z_data/financial_indicator_2classes.csv'


def main():
    # read, cleansing and split data
    df = read_data()
    df = cleansing(df)
    df_train, df_test = train_test_split(df, test_size=1/4)
    
    # create model
    model = craete_svc_model(df)
    
    # create df_vec_b
    vec_b = model.coef_.reshape((DIM,1))
    df_vec_b = pd.DataFrame(vec_b, index=COLS_X, columns=['vec_b'])
    
    # accuracy evaluation
    accuracy, df_accuracy = accuracy_evaluation(df, model)
    print('')
    print('--------')
    print('b_vector')
    print(df_vec_b)
    print('--------')
    print('')
    print('----------------------------')
    print(f'accuracy score : {accuracy}')
    print('----------------------------')
    


def read_data():
    df = pd.read_csv(FILE, encoding='shift-jis', index_col=0, header=0)
    return df


def cleansing(df):
    # drop outliners
    for col_x in COLS_X:
        q1 = df[col_x].quantile(0.25)
        q3 = df[col_x].quantile(0.75)
        iqr = q3 - q1
        lim_upper = q3 + 1.5*iqr
        lim_lower = q1 - 1.5*iqr
        
        df = df[ df[col_x]>lim_lower ]
        df = df[ df[col_x]<lim_upper ]   
    
    # create label
    df[COL_LABEL] = df[COL_Y].replace(LABEL_DICT)
    
    return df
    
    
def craete_svc_model(df):
    x = np.array(df[COLS_X])
    y = np.array(df[COL_LABEL])
    model = svm.LinearSVC(C=C,random_state=0)
    model.fit(x,y)
    
    return model


def accuracy_evaluation(df, model):
    # settings
    index = df.index
    x = np.array(df[COLS_X])
    y = np.array(df[COL_LABEL])
    
    # predict label
    label_predict = model.predict(x)
    
    # calculate accuracy score
    accuracy = accuracy_score(y,label_predict)
    
    # concat DataFrame and label_predict
    srs_y_predict = pd.Series(label_predict, index=index, name='label_predict')    
    df_accuracy = pd.concat([df,srs_y_predict], axis=1)
    
    return accuracy, df_accuracy
    
main() 