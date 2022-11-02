import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# settings
FILE = '../z_data/beer_preference_trend.csv'
CATEGORY_LIST = ['area_of_growing', 'detailedness_of_foam', 'bitterness']
N_CATEGORY = len(CATEGORY_LIST)
COL_RESULT = 'preference_result'
METHOD_LIST = ['MONANOVA', 'regression']
B0 = [2,0,1,0,2,1,0]
COUNT = 500
COLOR_DICT = {'MONANOVA':'steelblue', 'regression':'forestgreen'}


def main():
    # read and cleansing data
    df = read_data()
    df_dummy, unique_list, n_unique_list = cleansing(df)
    
    # get values
    df_dummy_x = df_dummy.drop(columns=COL_RESULT)
    df_result = df_dummy[COL_RESULT]
    n_data = len(df_dummy)
    n_params = len(df_dummy_x.columns)
    index = df_dummy.index
    unique_all = df_dummy_x.columns
    
    # MONANOVA-moethod
    s_array, df_b_monanova, df_result_pred_monanova = monanova(
        df_result, 
        df_dummy_x, 
        n_data, 
        n_params, 
        n_unique_list, 
        unique_list, 
        unique_all, 
        index
    )
    
    # Quantification-Thoery-type1
    df_b_regression, df_result_pred_regression, summary = quantification_theory_type1(
        df_dummy_x, 
        df_result
    )
    
    # concat df_b and df_result
    df_b = pd.concat([df_b_monanova, df_b_regression], axis=1)
    df_result = pd.concat(
        [df_result, df_result_pred_monanova, df_result_pred_regression], axis=1
    )
    
    # create df_importance
    df_importance = create_df_importance(unique_list, df_b)
    
    # plot results
    fig = plot_results(
        s_array, 
        index, 
        df_result, 
        n_unique_list,
        unique_list,
        df_b, 
        df_importance
    )  
    
    # print result
    print(summary)
    plt.show()  
    

def read_data():
    # read data
    df = pd.read_csv(FILE, index_col=0, encoding='shift-jis')
    df = df.dropna(how='any')
    return df


def cleansing(df):
    # create df_dummy, df_dummy_x and df_result_ 
    cols_remain = CATEGORY_LIST + [COL_RESULT]
    df_remain = df[cols_remain].dropna(how='any')
    df_dummy = pd.get_dummies(df_remain, prefix='', prefix_sep='')
    
    # create unique_list
    unique_list = []
    for category in CATEGORY_LIST:
        unique = df[category].unique()
        unique = list(unique)
        unique_list.append(unique)
    
    # create n_unique_list
    n_unique_list = []
    for unique in unique_list:
        n_unique = len(unique)
        n_unique_list.append(n_unique)
        
    
    return df_dummy, unique_list, n_unique_list
           

def monanova(
    df_result, 
    df_dummy_x, 
    n_data, 
    n_params, 
    n_unique_list, 
    unique_list, 
    unique_all, 
    index
):   
    # create matrixes and vectors
    z = np.array(df_result).reshape((n_data,1))
    d = np.array(df_dummy_x)
    b = np.array(B0).reshape((n_params,1))
    z_ = np.dot(d,b).reshape((n_data,1))
    z_mean = np.mean(z_)
    v = np.dot( (z-z_).T, (z-z_) )
    w = np.dot( (z_-z_mean).T, (z_-z_mean) )
    s = (v/w)**0.5
    g = -s/w * np.dot( d.T, (z-z_)/(s**2) + (z_-z_mean) )
    
    # gradient descent method
    s_array = np.array(s[0])
    for count in range(1,COUNT):
        alpha = 1/(count**0.5)
        b = b - alpha*g
        '''new params'''
        z_ = np.dot(d,b).reshape((n_data,1))
        z_mean = np.mean(z_)
        v = np.dot( (z-z_).T, (z-z_) )
        w = np.dot( (z_-z_mean).T, (z_-z_mean) )
        s = (v/w)**0.5
        g = -s/w * np.dot( d.T, (z-z_)/(s**2) + (z_-z_mean) )
        s_array = np.append(s_array, s[[0]])
    b = b[:,0]
    
    # to DataFrame
    df_b = pd.DataFrame(b, index=unique_all, columns=['MONANOVA'])
    df_result_pred = pd.DataFrame(z_, index=index, columns=['MONANOVA'])
    
    return s_array, df_b, df_result_pred
       
    
def quantification_theory_type1(df_dummy_x, df_result):    
    # create model
    x = df_dummy_x
    y = df_result
    model = sm.OLS(y, x)
    
    # get results
    result = model.fit()
    summary = result.summary()
    b = result.params
    y_pred = result.predict(x)
    
    # set columns name
    b = pd.DataFrame(b, columns=['regression'])
    y_pred = pd.DataFrame(y_pred, columns=['regression'])
    
    return b, y_pred, summary


def create_df_importance(unique_list, df_b):   
    # create df_width
    df_width_list = []
    for method in ['MONANOVA', 'regression']:
        width_list = []
        for unique in unique_list:
            # calculate width
            min = df_b.loc[unique, method].min()
            max = df_b.loc[unique, method].max()
            width = max - min
            width_list.append(width)
            
        # create df_width
        df = pd.DataFrame(width_list, columns=[method], index=CATEGORY_LIST)
        df_width_list.append(df)
        
    # concat df_width
    df_width = pd.concat(df_width_list, axis=1)
    
    # create df_importance
    df_importance = df_width / df_width.sum()
    
    return df_importance


def plot_results(
    s_array, 
    index, 
    df_result, 
    n_unique_list,
    unique_list,
    df_b, 
    df_importance
):
    # create fig
    fig = plt.figure()
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.5, wspace=0.4
    )
    
    # ax1 stress plot
    ax1 = fig.add_subplot(2,3,1,title='Stress-plot')
    ax1.plot(np.arange(COUNT), s_array, color=COLOR_DICT['MONANOVA'])
    ax1.set_xlabel('count')
    ax1.set_ylabel('stress')
    
    # ax2 result comparison
    ax2 = fig.add_subplot(2,3,2,title='result comparison')
    ax2.plot(
        index, 
        df_result[COL_RESULT], 
        label='result', 
        marker='o', 
        markersize=3, 
        color='darkgray'
    )
    for method in ['MONANOVA', 'regression']:
        ax2.plot(
            index, 
            df_result[method], 
            label=method, 
            marker='o', 
            markersize=3, 
            color=COLOR_DICT[method]
        )
    plt.xticks(rotation=90)
    plt.legend()
    
    # ax3 importance
    align_dict = {'MONANOVA':'edge', 'regression':'center'}
    ax3 = fig.add_subplot(2,3,3,title='importance')
    for method,align in align_dict.items():
        ax3.bar(
            CATEGORY_LIST, 
            df_importance[method], 
            width=0.3, 
            align=align, 
            color=COLOR_DICT[method],
            label=method
        )
        plt.xticks(rotation=40)
        plt.legend()
    
    # ax4 score
    ax4 = fig.add_subplot(2,1,2,title='score')
    for method,color in COLOR_DICT.items():
        for unique in unique_list:
            ax4.plot(
                unique, 
                df_b.loc[unique, method], 
                marker='o', 
                markersize=3, 
                color=color, 
                label=method
            )
        plt.legend() 
    
    return fig
    
    
main()