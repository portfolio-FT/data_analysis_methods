import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from scipy.stats import norm


# settings
FILE = '../z_data/product_property.csv'
COLS_X = ['property1', 'property2']
COL_Y = 'property3'


def main():
    # read, cleansing and split data
    df = read_data()
    df = data_cleansing(df)
    df_train, df_test = train_test_split(df, test_size=1/4)

    # scatter
    plot_distribution(df)

    # for_variables select
    variables_select(df)

    # OLS model
    x, y, y_pred, residual, summary, result, fig = ols_model(df)

    # regression_diagnosis
    fig = regression_diagnosis(y, y_pred, residual, fig)

    # show results
    print(summary)
    plt.show()


def read_data():
    df = pd.read_csv(FILE, encoding='shift-jis', index_col=0, header=0)

    return df


def data_cleansing(df): 
    # settings
    index = df.index
    cols_analysis = COLS_X + [COL_Y]

    # astype(float)
    df[cols_analysis] = df[cols_analysis].astype(float)

    # remove outliners
    for col in cols_analysis:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lim_lower = q1 - 1.5*iqr
        lim_upper = q3 + 1.5*iqr
        for i in index:
            value = df.at[i,col]
            if value < lim_lower or lim_upper < value:
                df.at[i,col] = np.nan
    
    df = df.dropna(subset=cols_analysis)

    # standardization
    for col in cols_analysis:
        df[col] = (df[col]-df[col].mean()) / df[col].std()
    
    return df   


def plot_distribution(df):
    # col_snalysis
    cols_analysis = COLS_X + [COL_Y]

    # scatter
    fig = plt.figure()
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.3, top=0.8, wspace=0.5, hspace=0.3
    )
    for i,col_x in enumerate(COLS_X):
        ax = fig.add_subplot(1,3,i+1)
        ax.scatter(df[col_x], df[COL_Y], s=5)
        ax.set_xlabel(col_x)
        ax.set_ylabel(COL_Y)

    # correlation coefficient matrix
    ax = fig.add_subplot(1,3,3)
    corr = df[cols_analysis].corr(method='pearson')
    ax = sns.heatmap(corr, cmap='Blues', annot=True)
    
    plt.show()


def variables_select(df):
    pass


def ols_model(df):    
    # create model
    index = df.index
    x = df[COLS_X]
    y = df[COL_Y]
    model = sm.OLS(y, sm.add_constant(x))
    result = model.fit()
    
    # get results
    summary = result.summary()
    y_pred = result.predict(sm.add_constant(x))
    residual = y_pred - y
    vec_b = result.params
    coef_const = vec_b['const']
    doef_x1 = vec_b[COLS_X[0]]
    doef_x2 = vec_b[COLS_X[1]]

    # create model figure
    fig = plt.figure()
    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3
    )    
    x1 = np.arange(-3,3,0.1)
    x2 = np.arange(-3,3,0.1)
    x1,x2 = np.meshgrid(x1,x2)
    z = coef_const + doef_x1*x1 + doef_x2*x2
    ax1 = fig.add_subplot(1,2,1, title='distribution and surface-model', projection='3d')
    ax1.scatter(df[COLS_X[0]], df[COLS_X[1]], y)
    ax1.plot_surface(x1,x2,z,alpha=0.3)
    ax1.set_xlabel(f'{COLS_X[0]}')
    ax1.set_ylabel(f'{COLS_X[1]}')
    ax1.set_zlabel(f'y')
    ax1.set_xlim(-3,3)
    ax1.set_ylim(-3,3)
    ax1.set_zlim(-3,3)

    return x, y, y_pred, residual, summary, result, fig


def regression_diagnosis(y, y_pred, residual, fig):    
    # 1.Durbin-Watson ratio
    d_w = durbin_watson(residual)

    # 5.y_pred vs y
    ax2 = fig.add_subplot(2,4,3,title='y_pred vs y')
    ax2.scatter(y, y_pred, s=3)
    ax2.set_xlabel('y st')
    ax2.set_ylabel('y_pred st')

    # 2.Residual plot
    x_ = np.linspace(-3,3,100)
    y_ = x_*0
    
    ax3 = fig.add_subplot(2,4,4,title='Residual plot')
    ax3.scatter(y_pred, residual, s=3)
    ax3.plot(x_, y_, color='darkgray')
    ax3.set_xlabel('y_pred')
    ax3.set_ylabel('residual')
    
    # 3.Q-Q plot
    quantile_list = [ n/100 for n in np.arange(0,100,2) ]
    residual_st = (residual-residual.mean()) / residual.std()
    quantile_residual = residual_st.quantile(q=quantile_list)
    quantile_norm = norm.ppf(quantile_list, 0, 1)
    
    ax4 = fig.add_subplot(2,4,7,title='Q-Q plot')
    ax4.scatter(quantile_norm, quantile_residual, s=3)
    ax4.plot(np.linspace(-3,3,100), np.linspace(-3,3,100), color='darkgray')
    ax4.set_xlabel('Q N(0,1)')
    ax4.set_ylabel('Q Residual-st')

    # 4.histgram of residual
    x_ = np.linspace(-3,3,100)
    ax5 = fig.add_subplot(2,4,8,title='Histgram of Residual-st')
    ax5.hist(residual_st, bins=20, ec='black', label='residual-st')
    ax5.set_xlabel('Residual')
    ax5_2 = ax5.twinx()
    ax5_2.plot(x_, norm.pdf(x_, 0, 1), label=f'PDF N(0,1)', color='darkgray')
    ax5_2.set_ylim(0,0.42)
    handler1,label1 = ax5.get_legend_handles_labels()
    handler2,label2 = ax5_2.get_legend_handles_labels()
    ax5.legend(handler1+handler2, label1+label2)

    return fig


main()