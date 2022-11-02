import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# settings
FILE = '../z_data/smoking_questionnaire.csv'
CATEGORY_X = 'position'
CATEGORY_Y = 'smoking'


def main():
    # read data
    df = read_data()
    
    # quantification
    fig = quantification_theory_type3(df)
    
    # fig.show()
    plt.show()  


def read_data():
    df = pd.read_csv(FILE, index_col=0, header=0, encoding='shift-jis')
    return df


def quantification_theory_type3(df):
    # get labels
    labels_x = df.index
    labels_y = df.columns
    
    # create matrixes and dimensions
    m = np.array(df)
    dim_x = m.shape[0]
    dim_y = m.shape[1]
    
    # create diagonal matrixes
    diag_components_x = np.sum(m, axis=1)
    diag_components_y = np.sum(m, axis=0)
    diag_x = np.diag(diag_components_x)
    diag_y = np.diag(diag_components_y)
    
    # prepare for eig
    diag_x_inv = np.linalg.inv(diag_x)
    diag_y_sqrt = diag_y**0.5
    diag_y_sqrt_ = np.linalg.matrix_power(diag_y_sqrt,-1)
    diag_y_sqrt_inv = np.linalg.inv(diag_y_sqrt)
    a = np.dot(diag_y_sqrt_,m.T)
    a = np.dot(a,diag_x_inv)
    a = np.dot(a,m)
    a = np.dot(a,diag_y_sqrt_)
    
    # calculate eigen values and eigen vectors
    l,w = np.linalg.eig(a)
    l = l[1:] **0.5
    l1 = l[0]
    l2 = l[1]
    w = w[:,1:]
    
    # calculate x and y
    y = np.dot( diag_y_sqrt_inv, w )
    y1 = y[:,0].reshape((dim_y,1))
    y2 = y[:,1].reshape((dim_y,1))
    x1 = np.dot( np.dot(diag_x_inv,m), y1 ) / l1
    x2 = np.dot( np.dot(diag_x_inv,m), y2 ) / l2   
           
    # contribution rate
    cont_rate = np.array([ x/np.sum(l) for x in l ])
    cont_rate_cumu = np.cumsum(cont_rate)
    
    # create fig
    fig = plt.figure()
    plt.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.3
    )
    
    # contribution rate
    ax1 = fig.add_subplot(1,2,1,title='cumulative contribution rate')
    ax1.plot( np.arange(1,len(cont_rate)+1), cont_rate_cumu, marker='o' )
    ax1.set_ylim(0,1.2)
    ax1.set_xlabel('components')
    ax1.set_ylabel('cumulative rate')
    
    # score plot
    ax2 = fig.add_subplot(1,2,2,title='score plot')
    ax2.scatter( x1,x2,label=CATEGORY_X )
    ax2.scatter( y1,y2,label=CATEGORY_Y )  
    for i,label in enumerate(labels_x):
        plt.annotate(label, (x1[i],x2[i]))
    for i,label in enumerate(labels_y):
        plt.annotate(label, (y1[i],y2[i]))
    ax2.set_xlabel('score1')
    ax2.set_ylabel('score2')    
    plt.legend()
    
    return fig 
    
    
main()