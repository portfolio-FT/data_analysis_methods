import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# settings
FILE = '../z_data/financial_indicator.csv'
COLS_X = [
    'FI1', 'FI2', 'FI3', 'FI4', 'FI5', 'FI6', 'FI7', 
    'FI8', 'FI9', 'FI10', 'FI11', 'FI12', 'FI13', 'FI14', 
]
COL_CLASS = 'condition'
DIM = len(COLS_X)


def main():
    # read and cleansing data
    df = read_data()
    df = cleansing(df)
    
    # canonical discriminant analysis
    fig, df_w = canonical_discriminant_analysis(df)

    # show result
    print('-------------------------')
    print('canonical axis components')
    print(df_w)
    plt.show()


def read_data():
    df = pd.read_csv(FILE, index_col=0, header=0, encoding='shift-jis')
    return df


def cleansing(df):
    # remove outliers
    for count in range(0):
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
    '''
    for col in COLS_X:
        df[col] = (df[col]-df[col].mean()) / df[col].std()
    '''
    
    return df


def canonical_discriminant_analysis(df):
    # get class names
    class_list = list(df[COL_CLASS].unique())
    n_class = len(class_list)
    
    # create df by class
    df_list = []
    for i in range(n_class):
        df_list.append(df[df[COL_CLASS]==class_list[i]][COLS_X])
    
    # get numbers of data
    n_all_data = len(df)
    n_list = []
    for i in range(n_class):
        n_list.append(len(df_list[i]))
    
    # create martixes
    x_mat_all = np.array(df[COLS_X])
    x_mat_list = []
    for i in range(n_class):        
        x_mat = np.array(df_list[i])
        x_mat_list.append(x_mat)
        
    # create mean vectors
    mean_all = np.mean(x_mat_all,axis=0).reshape((DIM,1))
    mean_list = []
    for i in range(n_class):
        x_mat = x_mat_list[i]
        mean_x = np.mean(x_mat,axis=0).reshape((DIM,1))
        mean_list.append(mean_x)
    
    # create SW
    sw = np.zeros((DIM,DIM))
    for i in range(n_class):
        x_mat = x_mat_list[i]
        n = n_list[i]
        s = np.cov(x_mat,rowvar=0,bias=1)
        s = n * s
        sw += s
    sw_inv = np.linalg.inv(sw)
    
    # create SB
    sb = np.zeros((DIM,DIM))
    for i in range(n_class):
        n = n_list[i]
        mean_x = mean_list[i]
        s = np.dot( (mean_x-mean_all), (mean_x-mean_all).T ) * n
        sb += s
    
    # get eig-values and eig-vectors
    s = np.dot(sw_inv,sb)
    l,w = np.linalg.eig(s)  
    l_abs = np.abs(l)
    l_abs = np.sort(l_abs)[::-1]
    
    # contribution rate
    cont_rate = np.array([ x/np.sum(l_abs) for x in l_abs ])
    cont_rate_cumu = np.cumsum(cont_rate)
    
    # extract 1st and 2nd components
    l1 = l[0]
    l2 = l[1]
    w1 = w[:,0].reshape((DIM,1))
    w2 = w[:,1].reshape((DIM,1))
    w1 = np.round(w1, 5)
    w2 = np.round(w2, 5)
    df_w1 = pd.DataFrame(w1, columns=['w1'], index=COLS_X)
    df_w2 = pd.DataFrame(w2, columns=['w2'], index=COLS_X)
    df_w = pd.concat([df_w1, df_w2], axis=1)
    df_w = df_w.astype(float)
    
    # get score
    score1 = np.dot(x_mat_all,w1)
    score2 = np.dot(x_mat_all,w2)
    
    # to dataframe
    df_score1 = pd.DataFrame(
        score1,columns=['score1'],index=df.index
    )
    df_score2 = pd.DataFrame(
        score2,columns=['score2'],index=df.index
    )
    df_score = pd.concat([df,df_score1,df_score2], axis=1)
    
    # craete fig
    fig = plt.figure()
    
    # cumulative contribution rate
    ax1 = fig.add_subplot(1,2,1,title='cumulative contribution rate')
    ax1.plot( np.arange(1,DIM+1), cont_rate_cumu, marker='o' )
    ax1.set_xlabel('components')
    ax1.set_ylabel('cumulative contribution rate')
    ax1.set_ylim(0,1.2)
    
    # score plot
    ax2 = fig.add_subplot(1,2,2,title='score plot')
    ax2.set_xlabel('score1')
    ax2.set_ylabel('score2')
    for i in range(n_class):
        class_name = class_list[i]
        x = df_score[df_score[COL_CLASS]==class_name]['score1']
        y = df_score[df_score[COL_CLASS]==class_name]['score2']
        ax2.scatter(x,y,label=class_name)
    plt.legend()
    
    return fig, df_w
    
    

main()