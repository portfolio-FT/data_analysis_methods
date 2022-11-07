import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import pydotplus
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from mlxtend.plotting import plot_decision_regions


# settings
FILE = '../z_data/financial_indicator.csv'
COL_CLASS = 'condition'
COLS_X = ['FI2', 'FI5', 'FI6', 'FI8', 'FI11', 'FI13']
CRITERION = 'gini'
MAX_DEPTH = 3
DIM = len(COLS_X)


def main():
    # read, cleansing and split data
    df = read_data()
    df = cleansing(df)
    df_train, df_test = train_test_split(df, test_size= 1/4)
    
    # create decision tree model
    dt_model(df_train,df_test)    
    random_forest_model(df_train,df_test)  
    gbdt_model(df_train,df_test)   


def read_data():
    df = pd.read_csv(FILE, encoding='shift-jis', index_col=0, header=0)
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


def dt_model(df_train, df_test):
    # create model
    model_dt = DecisionTreeClassifier(max_depth=None, criterion=CRITERION, random_state=0)
    model_dt.fit(df_train[COLS_X], df_train[COL_CLASS])

    # create tree-fig
    dot_data = export_graphviz(
        model_dt,
        out_file=None,
        impurity=False,
        filled=True,
        feature_names=COLS_X,
        class_names=COL_CLASS
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('./tree_figure/decision_tree.png')
    
    # calculate accuracy score    
    accuracy = model_dt.score(df_test[COLS_X],df_test[COL_CLASS])
    print('')
    print('----------------------------------------------')
    print(f'accuracy score for decision tree : {round(accuracy,5)}')
    print('----------------------------------------------')
    
    # plot importance
    plt.title('Importance')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9)
    plt.bar(range(DIM), model_dt.feature_importances_)
    plt.xticks(range(DIM), COLS_X, rotation=90)
    plt.show()


def random_forest_model(df_train,df_test):
    # create model
    model_rf = RandomForestClassifier(
        n_estimators=7,
        max_features=3,
        max_depth=3, 
        criterion=CRITERION, 
        random_state=0
    )
    model_rf.fit(df_train[COLS_X], df_train[COL_CLASS])

    # create tree-fig
    dot_data = export_graphviz(
        model_rf.estimators_[0],
        out_file=None,
        impurity=False,
        filled=True,
        feature_names=COLS_X,
        class_names=COL_CLASS
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('./tree_figure/random_forset.png')
    
    # calculate accuracy score    
    accuracy = model_rf.score(df_test[COLS_X],df_test[COL_CLASS])
    print('')
    print('----------------------------------------------')
    print(f'accuracy score for random forest : {round(accuracy,5)}')
    print('----------------------------------------------')
    
    # plot importance
    plt.title('Importance')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9)
    plt.bar(range(DIM), model_rf.feature_importances_)
    plt.xticks(range(DIM), COLS_X, rotation=90)
    plt.show()
    
   
def gbdt_model(df_train, df_test):
    # create model
    model_gbdt = ensemble.GradientBoostingClassifier(n_estimators=500)
    model_gbdt.fit(df_train[COLS_X], df_train[COL_CLASS])
    
    # export_graphviz
    dot_data = export_graphviz(
        model_gbdt.estimators_[0,0], 
        out_file=None,  
        filled=True, 
        rounded=True,
        special_characters=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('./tree_figure/gradient_bossting_tree.png')
    
    # calculate accuracy score    
    accuracy = model_gbdt.score(df_test[COLS_X],df_test[COL_CLASS])
    print('')
    print('-------------------------------------------------------')
    print(f'accuracy score for gradient boostinf tree : {round(accuracy,5)}')
    print('-------------------------------------------------------')
    
    # plot importance
    plt.title('Importance')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.9)
    plt.bar(range(DIM), model_gbdt.feature_importances_)
    plt.xticks(range(DIM), COLS_X, rotation=90)
    plt.show()
     

main() 