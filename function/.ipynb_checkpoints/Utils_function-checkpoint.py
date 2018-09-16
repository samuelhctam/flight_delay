import pandas as pd
import numpy as np
pd.set_option("max_columns", 60)
import matplotlib
import matplotlib.pyplot as plt
plt.style.use(u'ggplot')
import statsmodels
import random
import seaborn as sns
random.seed(1)
import pandas.core.series
from scipy.stats import norm


def bin_plot(dat, col,  y="target", method="quantile", b=10, normalized = True, title = "", figsize=(15,6), filters = None, check =False, **kwargs):
    """
    bin plot will do univariate analysis,
    if the column is numerical 3 methods are provided:
        - quantile: plot of the relationship between x and y by bucketing x with quantile, quantile is controlled by b that is supposed to be an integer
        - bin: bucket x according to a list provided by b
        - dtree: will run locally a decision tree to find the most discriminatory split for y using x, the parameters can be passed as optionaml argument
    if the column is categorical then the plot will be per modality

    Parameters:
        - dat pandas dataframe
        - col whether the name of the column or the conresponding pandas series
        - y the name of the column
        - method: can be quantile, bin, or dtree in the case of a numerical pandas series
        - normalized: will look at average of the target in each bucket compare to the total population
        - filters: a boolean pandas series with index matching the pandas dataframe

    """

    import pandas
    if filters is not None:
        df = dat[filters].copy()
    else:
        df = dat.copy()

    if normalized:
        m = df[y].mean()
    else:
        m = 1


    if type(col) == pandas.core.series.Series:
        col = col.name

    if df.dtypes[col] in [np.dtype('float64'),np.dtype('int64')]:
        if method == "quantile":
            try:
                df["bin_"+col] = pd.qcut(df[col], b, labels=None, retbins=False, precision=3)
            except:
                n_rows = df.shape[0]
                std_col = np.std(df[col])/10000
                from scipy.stats import norm
                df["bin_"+col] = pd.qcut(df[col] + norm.rvs(0, std_col, size=n_rows),  b, labels=None, retbins=False, precision=3)
            data = (df.groupby("bin_"+col)[y].mean()/m).reset_index()

            x = "bin_"+col
        elif method == "bin":
            df["bin_"+col]= pd.cut(df[col], b)
            data = (df.groupby("bin_"+col)[y].mean()/m).reset_index()
            x = "bin_"+col
        elif method == "dtree":
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(**kwargs)
            clf.fit(df[[col]], df[y])
            cut_list = list(pd.Series(clf.tree_.threshold).value_counts().index.sort_values())
            df["bin_"+col]=pd.cut(df[col], cut_list)
            data = (df.groupby("bin_"+col)[y].mean()/m).reset_index()
            x = "bin_"+col
        if check:
            print df["bin_"+col].value_counts()
        df = df.drop("bin_"+col, axis =1)
    elif df.dtypes[col] in [np.dtype('O'), np.dtype('bool')]:
        data = (df.groupby(col)[y].mean()/m).reset_index()
        x = col

        if check:
            print df[col].value_counts()


    plt.figure(figsize=figsize)
    sns.barplot(x= x, y = y, data=data)
    plt.title(title);


from sklearn.cross_validation import StratifiedKFold
def cross_validation_scores(model, X, y, K=10, verbose=False):
    scores = pd.DataFrame({"ap_score": [],
                         "roc_auc_score": [],
                         "f1_score": [],
                         "precision": [],
                         "recall": [],
                          "fraction_positive": []})
    Y_score = np.array([])
    Y_test = np.array([])
    if verbose:
        print "---- Cross-validation over {} stratified folds ----".format(K)
    skf = StratifiedKFold(y, 10, random_state =1)
    for (train, test), i in zip(skf, range(K)):
        if verbose:
            sys.stdout.write('{0:d}... '.format(i+1))
        X_train = X.iloc[train, :]
        y_train = y.iloc[train]
        X_test = X.iloc[test, :]
        y_test = y.iloc[test]
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:,1]
        y_pred = model.predict(X_test)

        Y_score = np.concatenate((Y_score, y_score))
        Y_test = np.concatenate((Y_test, y_test))

        scores = scores.append(pd.DataFrame({"ap_score": [average_precision_score(y_test, y_score)],
                                 "roc_auc_score": [roc_auc_score(y_test, y_score)],
                                 "f1_score": [f1_score(y_test, y_pred)],
                                 "precision": [precision_score(y_test, y_pred)],
                                 "recall": [recall_score(y_test, y_pred)],
                                 "fraction_positive": [y_pred.sum()*1./len(y_pred)]}, index=[i]))
    if verbose:
        sys.stdout.write("Done!\n")
    return scores, Y_score, Y_test


def roc(y_test, y_pred_rf):
    from sklearn.metrics import roc_curve
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)


    plt.figure(figsize=(12,8))
    plt.plot(fpr_rf, tpr_rf, label='Random Forest (area = %0.2f)' % roc_auc_score(y_test, y_pred_rf))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for our RF model')
    plt.legend(loc="lower right")
    plt.show()

def display_feat_imp(model, predictors):
    feat_imp = pd.Series(model.feature_importances_, index=predictors)

    return feat_imp.sort_values(ascending=False)


def importance_graph(model,predictors, top = 20):
    importance_list =  display_feat_imp(model, predictors)
    importance_df = importance_list[:top].reset_index()
    plt.subplots(figsize=(20,15))
    sns.barplot(x=0, y="index", data=importance_df)
    #return ax
    
def liftChart(y, y_pred, pctile = .1, plot = False, label = False, Method = "rate",
             path = None, dpi=1000, target_title = "canceled"):
    """Lift chart.

    In binary classification, this function computes the lift
    at each integer multiple of pctile until 1. Namely, this
    computes how accurate the top pctile of y_pred is
    compared to predicting y at random using the same
    proportion of true labels in y.

    Parameters
    ----------
    y : 1d array-like, required
        Ground truth (correct) labels.

    y_pred : 1d array-like, required
        Probability of a true label, as returned by a classifier.

    pctile : 1d array-like, optional (default = .1)
        Percentile at which the lift is being measured.

    colnum : 1d array-like, optional (default = 1)
        Column number in y_pred which contains the probability
        for the target to be true.

    Method : normalised or not, optional (default = 1)
        if normalised then it is adjusted to cumulated ratio

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of lift at each integer multiple of
        pctile where the index is the pctile.
    """

    import numpy as np
    import pandas as pd
    n = int(pctile*len(y))
    sort_idx = np.argsort(-y_pred)
    y_sorted = y[sort_idx]
    baseline = (y.sum()/float(len(y)))
    lift_chart = []
    n_quantile = int(1/pctile)

    if n_quantile > 20:
        grid_b = False
    else:
        grid_b = True

    if Method == "rate":
        for i in range(n_quantile):
            lift_chart.append((y_sorted[(i*n):((i+1)*n)].sum()/float(n))/baseline)
    else:
        for i in range(n_quantile):
            lift_chart.append((y_sorted[(0):((i+1)*n)].sum()/float(n*(i+1)))/baseline)
    df_lift_chart = pd.DataFrame(lift_chart)
    df_lift_chart.index = df_lift_chart.index+1

    if plot:
        lift = df_lift_chart.sort([0], ascending=False)
        lift = pd.DataFrame(lift[0].values)
        lift.index = lift.index+1
        lift = lift.rename(columns = {0:'Baseline : (%s = %.2f%%)' % (target_title, baseline*100)})
        ax = lift.plot(title='Lift Chart',
                       kind='bar',
                       figsize=(15, 10),
                       alpha=0.7,
                       lw='1',
                       grid=grid_b) #or false
        ax.set_title("Lift Chart")
        ax.set_xlabel("Score grouped on " +str(n_quantile)+" quantiles")
        ax.set_ylabel("Lift")
        rects = ax.patches

        if label:
            # Now make some labels
            labels = ["%.2f" % i for i in lift[lift.columns[0]].values]

            for rect, label in zip(rects, labels):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2, height+0.02, label, ha='center', va='bottom')

        if path is not None:
            plt.savefig(path, bbox_inches='tight',  dpi=dpi)

    return df_lift_chart

def freq_table(a):
    Detail_freq = a.loc[:, (a.dtypes == object) | (a.dtypes == long) ].columns.get_values().tolist()
    print(Detail_freq)
    for freq in Detail_freq:
#     if freq != 'insurer_id_enc':
        if (freq != 'POL_NO') & (freq != 'insurer_id_enc'):
#         df =pd.DataFrame(trad_age_si[freq].value_counts(dropna=False).sort_index())
            print(freq)
            df1 = pd.DataFrame(a[freq].value_counts(dropna=False).astype(float).map('{:20,.0f}'.format).sort_index()).rename(columns={freq:'Count'})
            df2 = pd.DataFrame(a[freq].value_counts(normalize = True, dropna=False).map('{:,.2%}'.format).sort_index()).rename(columns={freq:'Percentage'})
            df  = pd.concat([df1, df2], axis = 1) 
            print(df)
