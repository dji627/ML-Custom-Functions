from itertools import cycle

import collections

import math

from helperFunctions import *
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.api.types import is_int64_dtype, is_float_dtype, is_object_dtype, is_datetime64_any_dtype, is_bool_dtype, is_categorical_dtype
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler,label_binarize
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import boxcox, stats

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn.impute import SimpleImputer, KNNImputer

def importFile(file_path, file_name, column_names = None, show_dataFrame = False, **read_csv_kwargs):
    filePath = file_path + file_name
    if column_names != None:
        column_names = stringToList(column_names)
    if (type(column_names) == list): #check to see if column names are provided
        df = pd.read_csv(filePath, column_names, **read_csv_kwargs)
        features = column_names
    else:
        df = pd.read_csv(filePath, **read_csv_kwargs)
    if show_dataFrame != False:
        print(df.head(show_dataFrame))
    return df

def showGraph(df, output=None, feature_selected = 'All',plot_type = None,fig_size_y = 15, fig_size_x = 30,set_yscale = None,
              set_xscale =None, kde = False, show_corrlation = False):
    featuresToApply = selectingFeatures(df, feature_input= feature_selected, output = output)
    if plot_type != None:
        plt.figure(figsize = (fig_size_y, fig_size_x))
        plotCol = min(len(featuresToApply),3)
        for i, feature in enumerate(featuresToApply):
            plt.subplot(len(featuresToApply), plotCol, i + 1)
            if plot_type == 'scatter':
                g = sns.scatterplot(data=df, x=feature, y=output, legend='auto')
            elif plot_type == 'histo':
                g = sns.histplot(data=df, x=feature, hue=output, multiple='stack', kde = kde)
            elif plot_type == 'box':
                g = sns.boxplot(x=df[feature])
            elif plot_type == 'pair':
                g = sns.pairplot(df, hue=output, vars=feature)
            elif plot_type == 'summaryPlot':
                if is_float_dtype(df[feature]) or is_int64_dtype(df[feature]):
                    g = sns.boxplot(data=df, x=feature, hue=output)
                    print('feature is a number')
                elif is_object_dtype(df[feature]):
                    g = sns.countplot(data=df,x=feature, hue=output)
            if set_yscale == 'log':
                g.set_yscale(set_yscale)
        plt.show()
def exploreFeatures(df, feature_selected = 'All', exploration_type = None,fig_size_y = 15, fig_size_x = 30):
    featuresToApply = selectingFeatures(df,feature_input=feature_selected)
    if exploration_type == 'correlation' or 'heatMap:':
        correlationMatrix = df[featuresToApply].corr()
        print (correlationMatrix)
        if exploration_type == 'heatMap':
            plt.figure(figsize=(fig_size_y, fig_size_x))
            sns.heatmap(correlationMatrix,annot = True)
            plt.show()



def showDataInfo(df, features_to_display = 'All', display_data_frame = False, show_rows = 5, show_data_type = True, show_data_details = False):
    featuresToApply = selectingFeatures(df, features_to_display)
    if display_data_frame == True:
        print (df[featuresToApply].head(show_rows))
        print (f'Total number of rows: {df.shape[0]}')
    if show_data_type == True:
        print (df[featuresToApply].dtypes)
    if show_data_details == True:
        for col in featuresToApply:
            print (f'Feature: {col}---------Type: {df[col].dtypes}')
            if df[col].isnull().values.any() == True:
                print(f'{col}: contains {df[col].isna().sum()} missing values (out of {df[col].shape[0]}), needs handling\n')
            if is_int64_dtype(df[col]) == True or is_float_dtype(df[col]) == True:
                max = df[col].max()
                min = df[col].min()
                median = df[col].median()
                mean = df[col].mean()
                mode = df[col].mode()
                std = df[col].std()
                skewness = df[col].skew()
                print(f'{col}: Max:{max}, Min:{min}, Median:{median:.3f}'
                                             f' Mean:{mean:.3f}, std:{std:.3f}, Mode:{mode}, Skewness:{skewness:.3f}')
                print('(-0.5 < skewness < 0.5) -> fairly symmetrical')
                print('(-1 < skewness < -0.5) or (0.5 < skewness < 1) -> moderately skewed')
                print('(skewness < -1) or ( 1 < skewness) -> highly skewed')
            elif is_object_dtype(df[col]) == True:
                uniqueVal, uniqueCount = np.unique(df.loc[:, col], return_counts=True)
                print(f'{col}({len(uniqueVal)} unique values:)')
                for index, (value, count) in enumerate(zip(uniqueVal, uniqueCount)):
                    print (f'{value}: {count}')
                    if index > 9:
                        print ("...more than 10 unique value detected, consider feature type conversioin")
                        break
            elif is_datetime64_any_dtype(df[col]):
                print (f'Oldest Date: {df[col].min()}')
                print (f'Newest Date: {df[col].max()}')
            elif is_bool_dtype(df[col]):
                print (df[col].value_counts())
            elif is_categorical_dtype(df[col]):
                print (df[col].value_counts())
            print('\n')
def preprocessing2(dataframe, handle_missing_values = None, one_hot_encode = None, ordinal_encode = None,
                  apply_log=None, remove_outlier =
                   None,min_max_scaler = None, remove_feature = None,
                  convert_feature = None, convert_to = None):
    print ('Function: preprocessing2 called')
    df = dataframe
    if (apply_log != None):
        featuresToApply = selectingFeatures(df,apply_log)
        for f in featuresToApply:
            df[f]=np.log10(df[f]+1)
        print (f'applied log transformation to: {featuresToApply}')
    if (remove_outlier != None):
        featuresToApply = selectingFeatures(df,remove_outlier)
        for f in featuresToApply:
            zScore = stats.zscore(df[f])
            absZScore = np.abs(zScore)
            filteredEntries = (absZScore < 3).all()
            df[f] = df[filteredEntries]
        print (f'removed outlier for features: {featuresToApply}')
    return df

def preprocessing(dataframe, preprocess_type, feature_selected = 'All', encoder_key = None, convert_to = None,missing_value_handle = None,
                  value_to_remove = None, replace_value = None, imputer = None,  skew_transformation = 'boxCox',**kwargs):
    print ('Function: preprocessing called')
    df = dataframe
    feature_selected = selectingFeatures(dataframe, feature_selected)
    if preprocess_type == 'featToRemove':
        df = df.drop(columns = feature_selected, inplace = False)
    elif preprocess_type == 'oneHotEncode':
        for f in feature_selected:
            df = encode_onehot(df,f)
    elif preprocess_type == 'ordinalEncode':
        if type(encoder_key) is list:
            oEncoder = OrdinalEncoder(categories = [encoder_key])
            df[feature_selected] = oEncoder.fit_transform(df[feature_selected])
        elif type(encoder_key) is dict:
            feature_selected = selectingFeatures(dataframe, feature_selected,string_if_single=True)
            df[feature_selected] = df[feature_selected].map(encoder_key)
    elif preprocess_type == 'minMaxScaler':
        df[feature_selected] = MinMaxScaler().fit_transform(df[feature_selected])
    elif preprocess_type == 'standardization':
        df[feature_selected] = StandardScaler().fit_transform(df[feature_selected])
    elif preprocess_type == 'dataTypeConvert':
        if convert_to == 'integer':
            df[feature_selected] = df[feature_selected].astype(int)
        elif convert_to == 'float':
            df[feature_selected] = df[feature_selected].astype(float)
        elif convert_to == 'string':
            df[feature_selected] = df[feature_selected].astype(str)
        elif convert_to == 'datetime':
            df[feature_selected] = pd.to_datetime(df[feature_selected])
        elif convert_to == 'category':
            df[feature_selected] = df[feature_selected].astype('category')
    elif preprocess_type == 'handleMissingValue':
        if missing_value_handle == 'removeMissingValue':
            df = df.dropna(subset=feature_selected, inplace = False)
        elif missing_value_handle == 'removeSpecificValue':
            df[feature_selected] = df[feature_selected].replace(value_to_remove,np.NaN)
            df = df.dropna(subset=feature_selected, inplace=False)
        elif missing_value_handle == 'replaceValue':
            print('replace values:', replace_value[0], replace_value[1])
            # replacing replace_value[0] wiht replace_value[1]
            df[feature_selected] = df[feature_selected].replace(replace_value[0], replace_value[1])
        elif missing_value_handle == 'fillna':
            df[feature_selected] = df[feature_selected].fillna(replace_value,**kwargs)
        elif missing_value_handle == 'forwardFill':
            df[feature_selected] = df[feature_selected].fillna(axis = 0, method='ffill',inplace = False,**kwargs)
        elif missing_value_handle == 'backwardFill':
            # df[feature_selected] = df[feature_selected].fillna(axis=0, method='bfill',inplace = False, **kwargs)
            df = df.fillna(columns = feature_selected, method= 'bfill', **kwargs)
        if imputer != None:
            # check imputer type validity
            if imputer not in ('mean','median','constant','most_frequent','knn'):
                print('invalid imputer input')
            elif imputer == 'knn':
                knn = KNNImputer(**kwargs)
                df[feature_selected] = knn.fit_transform(df[[feature_selected]])
            else:
                imp = SimpleImputer(strategy = imputer, **kwargs)
                df[feature_selected] = imp.fit_transform(df[[feature_selected]])
    elif preprocess_type == 'handleSkewedData':
        if skew_transformation == 'boxCox':
            for i in feature_selected:
                df[i],lmbda = boxcox(df[i],lmbda=None)
        elif skew_transformation == 'squareRoot':
            df[feature_selected] = df[feature_selected]**(.5)
        elif skew_transformation == 'reciprocal':
            df[feature_selected] = 1/df[feature_selected]
        elif skew_transformation == 'log':
            df[feature_selected] = np.log(df[feature_selected])

    return df

def encode_onehot(df, feature_selected):
    featureSelected = selectingFeatures(df, feature_input= feature_selected, string_if_single= True)
    df2 = pd.get_dummies(df[featureSelected], prefix='', prefix_sep='').groupby(level=0, axis=1).max().add_prefix(featureSelected+' - ')
    df3 = pd.concat([df, df2], axis=1)
    df3 = df3.drop([featureSelected], axis = 1)
    return df3

def trainTestSplit(df,output,test_size = None, train_size = None, random_state = None, shuffle = True, stratify = None):
    from sklearn.model_selection import train_test_split
    x = df.loc[:,df.columns != output].values
    y = df.loc[:,df.columns == output].values.ravel()
    if stratify == True:
        stratify = y
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = test_size, train_size=train_size,random_state=random_state, shuffle=shuffle, stratify=stratify)
    return xTrain, xTest, yTrain, yTest

def fitAndEvaulateModel(xTrain, xTest, yTrain, yTest, model, metricList = None, rocCurve = None, class_to_show = None):

    model = model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    yPred_prob = model.predict_proba(xTest)
    numOfClass = len(model.classes_) #num of output classes
    metricDict = {'accuracy':accuracy_score,
                  'f1':f1_score,
                  'precision':precision_score,
                  'recall':recall_score,
                  'logLoss':log_loss,
                  'rocAuc':roc_auc_score,
                  'confusionMatrix':confusion_matrix,
                  'classificationReport':classification_report,
                  'MAE':mean_absolute_error,
                  'MSE':mean_squared_error,
                  'RMSE':mean_squared_error,
                  'R2':r2_score}
    evalMetric = {}
    metricList = convertToList(metricList) #convert input to list form
    for metric in metricList:
        if metric in ['f1','precision','recall']:
            evalMetric['f1'] = metricDict[metric](yTest, yPred, average=None)
            if numOfClass == 2: #if the output is binary
                evalMetric['f1_binary'] = metricDict[metric](yTest, yPred,average = 'binary')
            else:
                for metric_f1 in ['micro','macro','weighted']:
                    evalMetric['f1_'+metric_f1] = metricDict[metric](yTest, yPred, average = metric_f1)
        elif metric in ['rocAuc']:
            evalMetric['rocAuc'] = metricDict[metric](yTest,yPred_prob, average = None,multi_class = 'ovr')
            if numOfClass == 2: #if the output is binary
                evalMetric['rocAuc_binary'] = metricDict[metric](yTest, yPred_prob ,average = 'binary',multi_class = 'raise')
            else:
                for metric_f1 in ['macro','weighted']:
                    evalMetric['rocAuc_'+metric_f1] = metricDict[metric](yTest, yPred_prob, average = metric_f1,multi_class = 'ovr')
            if rocCurve != None:
                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                # yScore = model.decision_function(xTest)
                yScore = yPred_prob
                yTest = label_binarize(yTest, classes =model.classes_)
                for i, c in enumerate(model.classes_):
                    fpr[c], tpr[c], _ = roc_curve(yTest[:,i], yScore[:,i])
                    roc_auc[c] = auc(fpr[c],tpr[c])
                # Compute micro-average ROC curve and ROC area
                fpr['micro'], tpr['micro'], _ = roc_curve(y_true = yTest.ravel(), y_score=yScore.ravel())
                roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
                if rocCurve == 'micro':
                    plt.figure()
                    lw = 2
                    plt.plot(
                        fpr[class_to_show],
                        tpr[class_to_show],
                        color="darkorange",
                        lw=lw,
                        label= "ROC curve-" + class_to_show +"(area = %0.2f)" % roc_auc[class_to_show],
                    )
                    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Receiver operating characteristic")
                    plt.legend(loc="lower right")
                    plt.show()
                elif rocCurve == 'macro':
                    # First aggregate all false positive rates
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in model.classes_]))

                    # Then interpolate all ROC curves at this points
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in model.classes_:
                        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

                    # Finally average it and compute AUC
                    mean_tpr /= numOfClass

                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                    # Plot all ROC curves
                    plt.figure()
                    plt.plot(
                        fpr["micro"],
                        tpr["micro"],
                        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                        color="deeppink",
                        linestyle=":",
                        linewidth=4,
                    )

                    plt.plot(
                        fpr["macro"],
                        tpr["macro"],
                        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                        color="navy",
                        linestyle=":",
                        linewidth=4,
                    )

                    colors = cycle(["aqua", "darkorange", "cornflowerblue", 'red','green','yellow'])
                    for i, color in zip(model.classes_, colors):
                        plt.plot(
                            fpr[i],
                            tpr[i],
                            color=color,
                            lw=2,
                            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
                        )

                    plt.plot([0, 1], [0, 1], "k--", lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("Some extension of Receiver operating characteristic to multiclass")
                    plt.legend(loc="lower right")
                    plt.show()
        else:
            evalMetric[metric] = metricDict[metric](yTest,yPred)
            if metric == 'confusionMatrix':
                disp = ConfusionMatrixDisplay(confusion_matrix=evalMetric[metric],display_labels=model.classes_)
                disp.plot()
                plt.show()
            if metric == 'classificationReport':
                print (evalMetric[metric])
    return evalMetric

def crossValidate(df, output, model, metricList = None, **stratified_k_fold_args):
    x = df.loc[:, df.columns != output].values
    y = df.loc[:, df.columns == output].values.ravel()
    metricHash = collections.defaultdict(list) #hashtable with a list value

    kf = StratifiedKFold(**stratified_k_fold_args)
    for trainIndex, testIndex in kf.split(x,y): #looping through the train and test set indices for the splits
        evalMetric = fitAndEvaulateModel(x[trainIndex], x[testIndex], y[trainIndex], y[testIndex], model, metricList = metricList)
        for m in evalMetric.keys():
            metricHash[m].append(evalMetric[m])
    # print(f'{model} : {n_splits} fold cross validation result: {np.mean(metricList):.3f}+/-{np.std(metricList):.3f}')
    print(f'{model}: {kf.get_n_splits()} fold cross validation result:')
    for m in metricHash:
        print(f'{m}: {np.mean(metricHash[m]):.3f}+/-{np.std(metricHash[m]):.3f}')

def batchCrossValidate(modelList, *args, **kwargs):
    for model in modelList:
        #crossValidate(df=*args[0], output=*args[1], model=model, n_splits=**kwargs[n_splits], shuffle=**kwargs[shuffle], random_state = **kwargs[random_state], metricList = **kwargs[metricList])
        pass


def gridSearch(df, output, model, param_grid, scoring = None, n_jobs = None, refit = True, cv = 5, verbose = 0,
        pre_dispatch = None, error_score = np.nan, return_train_score = False, show_graph = None, print_results = None):
    x = df.loc[:, df.columns != output].values
    y = df.loc[:, df.columns == output].values.ravel()
    from sklearn.model_selection import GridSearchCV
    gs = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, n_jobs = n_jobs,  refit = refit,
                      cv = cv, verbose = verbose, pre_dispatch = pre_dispatch, error_score = error_score, return_train_score = return_train_score)

    gs.fit(x,y)
    print(gs.scorer_)
    if print_results != None:
        dfResults = pd.DataFrame(gs.cv_results_)
        print (dfResults.to_string())
    if show_graph != None:
        plotGridSearch(dfResults, param_grid = param_grid)
    return gs.best_params_

def plotGridSearch(df_results, param_grid, fig_size_y = 15, fig_size_x = 30):
    paramList = list(param_grid.keys()) #put grid keys to a list
    gridKeyHeaders = ['param_' + param for param in paramList] #convert the parameter to match the column headers cv_results
    print (gridKeyHeaders)
    gridKeyHeaders.reverse()
    print (gridKeyHeaders)
    df_results.sort_values(by = gridKeyHeaders, inplace=True)
    print (df_results.to_string())

    scoresMean = df_results['mean_test_score']
    #plot Grid Search Scores
    if len(paramList) == 1:
        scoresMean = np.array(scoresMean)
        plt.plot(param_grid[paramList[0]], scoresMean)
        plt.title('Grid Search Scores')
        plt.xlabel(paramList[0])
        plt.ylabel('CV Average Score')
        plt.legend()
        plt.show()
    elif len(paramList) == 2:
        # get Test Scores Mean and std for each grid search
        print(f'{param_grid[paramList[0]]} length: {len(param_grid[paramList[0]])}')
        print(f'{param_grid[paramList[1]]} length: {len(param_grid[paramList[1]])}')
        scoresMean = np.array(scoresMean).reshape(len(param_grid[paramList[1]]), len(param_grid[paramList[0]]))
        print(scoresMean)
        for idx, val in enumerate(param_grid[paramList[1]]):
            # print ('idx:',idx, '  val:',val)
            # print ('scoresMean:',scoresMean[idx,:])
            plt.plot(param_grid[paramList[0]], scoresMean[idx,:], '-o', label = paramList[1] + ': ' + str(val))
            plt.title('Grid Search Scores')
            plt.xlabel(paramList[0])
            plt.ylabel('CV Average Score')
            plt.legend()
        plt.show()
    elif len(paramList) == 3:
        dfList = []
        plt.figure(figsize=(fig_size_y, fig_size_x))
        plotCol = min(len(paramList[2]), 2)
        for plotId, param in enumerate(param_grid[paramList[2]]):
            df = df_results[df_results[gridKeyHeaders[0]] == param]# seperating the df_result based on the unique values of paramList[2], gridKeyHeaders is the reverse of paramList
            dfList += [df]
            scoresMean = df['mean_test_score']
            scoresMean = np.array(scoresMean).reshape(len(param_grid[paramList[1]]), len(param_grid[paramList[0]]))
            subplot = plt.subplot(len(paramList[2]), plotCol, plotId + 1)
            for idx, val in enumerate(param_grid[paramList[1]]):
                subplot.plot(param_grid[paramList[0]], scoresMean[idx, :], '-o', label=paramList[1] + ': ' + str(val))
                subplot.set_title(f'Grid Search Scores: {paramList[2]}={param}')
                subplot.set_xlabel(paramList[0])
                subplot.set_ylabel('CV Average Score')
                subplot.legend()
        plt.show()
def creatingSubplots(feature_input, plot_column_size = 2, sharex = False, sharey = False):
    featNum = len(feature_input)
    plot_row_size = int(math.ceil(featNum/plot_column_size))
    fig, ax = plt.subplots(plot_row_size, plot_column_size, sharex = sharex, sharey = sharey)  # subpolt with # of rows and columns defined above
    subplotList = []  # store subplot axis in a linear array
    for i in range(0, plot_row_size):
        for j in range(0, plot_column_size):
            if plot_column_size == 1:
                subplotList += [ax[i]]
            elif plot_row_size == 1:
                subplotList += [ax[j]]
            elif plot_row_size > 1:
                subplotList += [ax[i, j]]
    return subplotList