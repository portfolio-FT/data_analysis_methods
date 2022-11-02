import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer



# settings
FILE = '../z_data/company_evaluation.csv'
COLS_X = [
    'x1', 'x2', 'x3', 'x4', 'x5', 
    'x6', 'x7', 'x8', 'x9', 'x10', 
]
COL_CLASS = 'evaluation'
DIM = len(COLS_X)
N_FACTOR = 4
ROTATION = 'promax'


def main():
    # read and cleansing data
    df = read_data()
    df, df_st = cleansing(df)
    
    # factor analysis
    df_fa_load, df_fa_score, fig = maximum_likelihood_method(df_st)
    df_fa_load_p_fa = principal_factor_method(df_st)
    

    # show result
    print('')
    print('facor loading matrix------')
    print(df_fa_load)
    
    print('')
    print('facor score------')
    print(df_fa_score)
    
    plt.show()
    

def read_data():
    # read data
    df = pd.read_csv(FILE, index_col=0, header=0, encoding='shift-jis')
    return df


def cleansing(df):
    # astype
    df[COLS_X] = df[COLS_X].astype(float)
    
    # remoce outliners
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
    df_st = pd.DataFrame(columns=COLS_X)
    for col in COLS_X:
        df_st[col] = (df[col] - df[col].mean()) / df[col].std()
    df_st[COL_CLASS] = df[COL_CLASS]
    
    # create index
    index = df.index
    df_st.index = index
    
    return df, df_st


def maximum_likelihood_method(df_st):
    # create df_r_matrix
    df_x_st = df_st[COLS_X]
    df_r = df_st.corr()
    
    # screeplot
    l = np.linalg.eigvals(df_r)
    l_cumusum = np.cumsum(l)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,title='scree plot')
    ax1_2 = ax1.twinx()
    ax1.plot(
        np.arange(1,DIM+1), l, marker='o', label='eig values', color='steelblue'
    )
    ax1_2.plot(
        np.arange(1,DIM+1), l_cumusum, marker='o', label='cumusum cont rate', color='orange'
    )
    ax1.set_xlabel('factor')
    ax1.set_ylabel('eig values')
    handler1,label1 = ax1.get_legend_handles_labels()
    handler2,label2 = ax1_2.get_legend_handles_labels()
    ax1.legend(handler1+handler2, label1+label2)
    
    # create likelihood model
    model = FactorAnalyzer(n_factors=N_FACTOR, rotation=ROTATION)
    model.fit(df_x_st)
    
    # get results
    fa_load_mat = np.round(model.loadings_ ,3)
    fa_score = np.round(model.transform(df_x_st), 3)
    
    # to DataFrame
    factor_label = [f'factor{i}' for i in range(1,N_FACTOR+1)]
    index = df_st.index
    df_fa_load = pd.DataFrame(fa_load_mat, columns=factor_label, index=COLS_X)
    df_fa_score = pd.DataFrame(fa_score, columns=factor_label, index=index)
    df_fa_score[COL_CLASS] = df_st[COL_CLASS]
    
    # plot fa_score
    class_unique = df_fa_score[COL_CLASS].unique()
    ax2 = fig.add_subplot(1,2,2,title='factor score')
    for class_label in class_unique:
        ax = ax2.scatter(
            df_fa_score[df_fa_score[COL_CLASS]==class_label]['factor1'], 
            df_fa_score[df_fa_score[COL_CLASS]==class_label]['factor2'], 
            label=class_label
        )
    ax2.legend()
    ax2.set_xlabel('factor1')
    ax2.set_ylabel('factor2')
    
    return df_fa_load, df_fa_score, fig


def principal_factor_method(df_st):
    # create df_r_matrix
    df_x_st = df_st[COLS_X]
    df_r = df_st.corr()
    
    # get eig vectors and values
    l,v = np.linalg.eig(df_r)
    
    factor_label = [f'factor{i}' for i in range(1,N_FACTOR+1)]
    index = df_st.index
    df_fa_load = pd.DataFrame(columns=factor_label)
    for i in range(N_FACTOR):
        fa_load = np.round(l[i]**0.5 * v[i], 3)
        df_fa_load[f'factor{i+1}'] = fa_load
    df_fa_load.index = COLS_X
    
    return df_fa_load


main()