import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.formula.api as smf


# settings1 columns and dimsension
COLS_X = ['FI5', 'FI6', 'FI8', 'FI11', 'FI13']
COL_Y = 'condition'
COL_LABEL = 'label'
LABEL_DICT = {'good':0, 'bad':1}
DIM = len(COLS_X)

# settings2 link function and family
FORMULA = f'''
    {COL_LABEL} ~ 1 + {COLS_X[0]} + {COLS_X[1]} + {COLS_X[2]} + {COLS_X[3]} + {COLS_X[4]}
'''
LINK = sm.families.links.logit()
FAMILY = sm.families.Binomial(link=LINK)

# settings3 the others
NUMS = range(0,11)
LEVEL = 0.6
LEVEL_LIST = [round(num*0.1, 2) for num in NUMS]
FILE = '../z_data/financial_indicator_2classes.csv'

def main():
    # read, cleansing and split data
    df = read_data()
    df = cleansing(df)
    df_train,df_test = train_test_split(df, test_size=1/4)
    
    # create model and get results
    result = log_regression(df)
    b_vec = np.array(result.params).reshape([DIM+1,1])  
    x_mat = np.insert(np.array(df[COLS_X]), 0, 1, axis=1)
    z_hat_vec = np.dot(x_mat, b_vec) 
    prob_hat = np.array(result.predict(df[COLS_X]))
    aic = result.aic
    summary = result.summary()
    
    # append judge to df
    df_accuracy = append_accuracy_label_to_df(df, z_hat_vec, prob_hat) 
    
    # calculate accuracy score
    accuracy_score = calculate_accuracy_score(df_accuracy)
    
    # create fig
    fig = plt.figure()   
    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.3, wspace=0.3
    )
    
    # plot logistic-model and label plot
    fig = plot_logisiotc_model_and_label_plot(df, fig, b_vec, z_hat_vec)
    
    # plot ROC-curve
    fig = plot_roc_curve(df_accuracy, fig)
    
    # plot Recall-Precision curve
    fig = plot_recall_precision_curve(df_accuracy, fig)
    
    # show results
    print(summary)
    print(f'AIC : {aic}')
    print(f'accuracy score : {accuracy_score}')
    plt.show()


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

    # standardization
    for col in COLS_X:
        df[col] = (df[col]-df[col].mean()) / df[col].std()
    
    return df


def log_regression(df):
    # set col_y
    df[COL_LABEL] = df[COL_Y].replace(LABEL_DICT) 
    
    # create model
    logit_model = smf.glm(data=df, formula=FORMULA, family=FAMILY)
    result = logit_model.fit()
    
    return result
 
 
def append_accuracy_label_to_df(df, z_hat_vec, prob_hat):    
    # edit df
    df['z'] = z_hat_vec
    df['probability_hat'] = prob_hat
    df_accuracy = df.sort_values(by='probability_hat')
    df_accuracy = df_accuracy.reset_index(drop=True)
    
    # create accuracy-label for each level
    for level in LEVEL_LIST:
        for row in range(len(df_accuracy)):
            # create predict label
            if df_accuracy.at[row, 'probability_hat'] > level:
                df_accuracy.at[row, f'predict_label_{level}'] = 1
            else:
                df_accuracy.at[row, f'predict_label_{level}'] = 0
            # judge TFPN
            label = df_accuracy.at[row, COL_LABEL]
            predict_label = df_accuracy.at[row, f'predict_label_{level}']
            if label == 0 and predict_label == 0:
                df_accuracy.at[row, f'accuracy_label_{level}'] = 'TN'
            if label == 0 and predict_label == 1:
                df_accuracy.at[row, f'accuracy_label_{level}'] = 'FP'
            if label == 1 and predict_label == 0:
                df_accuracy.at[row, f'accuracy_label_{level}'] = 'FN'
            if label == 1 and predict_label == 1:
                df_accuracy.at[row, f'accuracy_label_{level}'] = 'TP'
    
    return df_accuracy


def calculate_accuracy_score(df_accuracy):
    n_all = len(df_accuracy)
    n_TP = (df_accuracy[f'accuracy_label_{LEVEL}']=='TP').sum()
    n_TN = (df_accuracy[f'accuracy_label_{LEVEL}']=='TN').sum()
    accuracy_score = (n_TP + n_TN) / n_all
    return accuracy_score


def plot_logisiotc_model_and_label_plot(df, fig, b_vec, z_hat_vec):    
    # create logit function
    x = np.linspace(-3,3,1000)
    z = b_vec[0]
    for p in range(1, DIM):
        b = b_vec[p]
        z = z + b*x
    y = 1 / (1 + np.exp(-z))
    
    # ligstic-model and label plot
    ax1 = fig.add_subplot(1,2,1)  
    ax1.plot(z,y, c='darkgray', label='logistic model')
    ax1.scatter(z_hat_vec, df[COL_LABEL], c='steelblue', label='actual')
    ax1.set_title('acrual plot and logistic model')
    ax1.set_xlim(z_hat_vec.min(),z_hat_vec.max())
    ax1.set_xlabel('z')
    ax1.set_ylabel('probability')
    plt.legend()
    
    return fig
      

def plot_roc_curve(df_accuracy, fig):
    # initial settings
    tpr_list = []
    fpr_list = []
    n = len(df_accuracy)
    
    # create TP-rate, FP-rate and append to list 
    for level in LEVEL_LIST:
        tpr = (df_accuracy[f'accuracy_label_{level}']=='TP').sum() / (df_accuracy[COL_LABEL]==1).sum()
        fpr = (df_accuracy[f'accuracy_label_{level}']=='FP').sum() / (df_accuracy[COL_LABEL]==0).sum()
        tpr_list.append(tpr)
        fpr_list.append(fpr)     
    
    # plot   
    ax2 = fig.add_subplot(2,2,2)      
    ax2.plot(fpr_list, tpr_list, marker='o')
    for i,label in enumerate(LEVEL_LIST):
        ax2.annotate(label, (fpr_list[i], tpr_list[i]))
    ax2.set_title('ROC curve')
    ax2.set_xlabel('False Positeve Rate')
    ax2.set_ylabel('True Positeve Rate')
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    
    return fig


def plot_recall_precision_curve(df_accuracy, fig):
    # settings
    n = len(df_accuracy)
    recall_list = []
    precision_list = []
    
    # calculate recall and precision for each level
    for level in LEVEL_LIST:
        n_TP = (df_accuracy[f'accuracy_label_{level}']=='TP').sum()
        n_FP = (df_accuracy[f'accuracy_label_{level}']=='FP').sum()
        n_FN = (df_accuracy[f'accuracy_label_{level}']=='FN').sum()
        recall = n_TP / (n_TP + n_FN)
        precision = n_TP / (n_TP + n_FP)
        recall_list.append(recall)
        precision_list.append(precision)
        
    # plot
    ax3 = fig.add_subplot(2,2,4)
    ax3.plot(recall_list, precision_list, marker='o')
    for i,label in enumerate(LEVEL_LIST):
        ax3.annotate(label, (recall_list[i],precision_list[i]))
    ax3.set_title('Recall-Precision curve')
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_xlim(-0.1, 1.1)
    ax3.set_ylim(-0.1, 1.1)
    
    return fig
            
    
if __name__ == '__main__':
    main()