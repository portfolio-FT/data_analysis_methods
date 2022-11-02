import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# settings
FILE = '../z_data/company_evaluation.csv'
COLS_X = [
    'x1', 'x2', 'x3', 'x4', 'x5', 
    'x6', 'x7', 'x8', 'x9', 'x10', 
]
COL_CLASS = 'evaluation'
DIM = len(COLS_X)


def main():
    # read and cleansing data
    df = read_data()
    df = cleansing(df)
    
    # principal components analysis
    df_l, df_w, df_score, fig = principal_components_analysis(df)
    
    # show results
    print('')
    print('eigen-values')
    print(df_l)
    print('-----------------------------------------')
    
    print('')
    print('eigen-vectors')
    print(df_w)
    print('-----------------------------------------')
    
    print('')
    print('score')
    print(df_score)
    print('-----------------------------------------')
    
    plt.show()
    
    
def read_data():
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
    
    return df


def principal_components_analysis(df):
    # index
    index = df.index
    n = len(df)
    class_unique = df[COL_CLASS].unique()
    
    # create matrixes
    mat = np.array(df)
    mat_x = np.array(df[COLS_X])
    mat_x_st = ( (mat_x-mat_x.mean()) / mat_x.std() )
    
    # create correlation-coefficient matrix
    mat_r = np.corrcoef(mat_x,rowvar=False)
    
    # calculate eig values and vectors
    l,w = np.linalg.eig(mat_r)
    l = np.round(l,3)
    w = np.round(w,3)
    
    # argsort l,w
    id = np.argsort(l)[::-1]
    l = l[id]
    w = w[:,id]
    
    # create contribution-rate and cumusum
    cont_rate = [l_/l.sum() for l_ in l]
    cont_rate = np.round(cont_rate,3)
    cont_rate_cumsum = np.cumsum(cont_rate)
    cont_rate_cumsum = np.round(cont_rate_cumsum,3)
    
    # create df_eig-values
    data_ = {
        'lambda':l, 'cont_rate':cont_rate, 'cont_rate_cumsum':cont_rate_cumsum
    }
    component_label = [f'component{i}' for i in range(1,DIM+1)]
    df_l = pd.DataFrame(data_, index=component_label)
    
    # create df_eig-vectors
    df_w = pd.DataFrame(w, columns=component_label, index=COLS_X)
    
    # create df_score
    df_score_list = []
    for i in range(DIM):
        l_ = l[i]
        w_ = w[:,i]
        score_ = np.dot(w_.T, mat_x_st.T)
        df_score_ = pd.DataFrame(score_, index=index, columns=[f'component{i+1}'])
        df_score_list.append(df_score_)
    df_score = pd.concat(df_score_list, axis=1)
    df_score = pd.concat([df_score, df[COL_CLASS]], axis=1)
    
    # craete fig
    fig = plt.figure()
    
    # plot cumulative contribution-rate
    ax1 = fig.add_subplot(1,2,1,title='cumulative contribution-rate')
    ax1.plot(range(DIM), cont_rate_cumsum, marker='o')
    ax1.set_xlabel('components')
    ax1.set_ylabel('cumulative contribution-rate')
    
    # plot score
    ax2 = fig.add_subplot(1,2,2,title='score plot')
    for class_name in class_unique:
        ax2.scatter(
            df_score[df_score[COL_CLASS]==class_name]['component1'], 
            df_score[df_score[COL_CLASS]==class_name]['component2'],
            label=class_name
        )
    ax2.set_xlabel('score1')
    ax2.set_ylabel('score2')
    ax2.legend()
    
    return df_l, df_w, df_score, fig
       

main()