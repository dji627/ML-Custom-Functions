import math

from helperFunctions import *
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.api.types import is_int64_dtype, is_float_dtype, is_object_dtype
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import boxcox, stats

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

def importFile(file_path, file_name, column_names = None, sep = None, show_dataFrame = False):
    filePath = file_path + file_name
    if column_names != None:
        column_names = stringToList(column_names)
    if (type(column_names) == list): #check to see if column names are provided
        df = pd.read_csv(filePath, sep = sep, names = column_names)
        features = column_names
    else:
        df = pd.read_csv(filePath, sep = sep)
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



def showDataInfo(df, features_to_display = 'All', show_data_frame = True, show_rows = 5, show_data_type = False):
    print ('Function: showDataInfo called')
    featuresToApply = selectingFeatures(df, features_to_display)
    if show_data_frame == True:
        print (df[featuresToApply].head(show_rows))
    if show_data_type == True:
        print (df[featuresToApply].dtypes)
    for col in featuresToApply:
        if df[col].isnull().values.any() == True:
            print(f'{col}: contains missing value, needs handling\n\n')
        elif is_int64_dtype(df[col]) == True or is_float_dtype(df[col]) == True:
            max = df[col].max()
            min = df[col].min()
            median = df[col].median()
            mean = df[col].mean()
            mode = df[col].mode()
            std = df[col].std()
            skewness = df[col].skew()
            print(f'{col}: Max:{max}, Min:{min}, Median:{median:.3f}'
                                         f' Mean:{mean:.3f}, std:{std:.3f}, Mode:{mode}, Skewness:{skewness:.3f}')
        elif is_object_dtype(df[col]) == True:
            uniqueVal, uniqueCount = np.unique(df.loc[:, col], return_counts=True)
            print(f'{col}({len(uniqueVal)} unique values:)')
            for value, count in zip(uniqueVal, uniqueCount):
                print (f'{value}: {count}')
        print('\n')
def preprocessing2(dataframe, handle_missing_values = None, one_hot_encode = None, ordinal_encode = None,
                  apply_log=None, remove_outlier = None,min_max_scaler = None, remove_feature = None,
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
                  value_to_remove = None, replace_value = None, skew_transformation = 'boxCox'):
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
    elif preprocess_type == 'handleMissingValue':
        if missing_value_handle == 'removeMissingValue':
            df.dropna(subset=feature_selected, inplace = True)
        elif missing_value_handle == 'removeSpecificValue':
            df[feature_selected] = df[feature_selected].replace(value_to_remove,np.NaN)
            df.dropna(subset=feature_selected, inplace=True)
        elif missing_value_handle == 'replaceValue':
            print('replace values:', replace_value[0], replace_value[1])
            df[feature_selected] = df[feature_selected].replace(replace_value[0], replace_value[1])
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

def fitAndEvaulateModel(xTrain, xTest, yTrain, yTest, model, metricList = None):
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    numOfClass = len(model.classes_) #num of output classes
    metricDict = {'accuracy':accuracy_score,
                  'f1':f1_score,
                  'precision':precision_score,
                  'recall':recall_score,
                  'logLoss':log_loss,
                  'rocAuc':roc_auc_score,
                  'confusionMatrix':confusion_matrix}
    evalMetric = {}
    metricList = convertToList(metricList) #convert input to list form
    for metric in metricList:
        print(metric)
        if metric in ['f1','precision','recall']:
            if numOfClass == 2: #if the output is binary
                evalMetric['f1_binary'] = metricDict[metric](yTest, yPred,average = 'binary')
            evalMetric['f1'] = metricDict[metric](yTest,yPred, average = None)
            for metric_f1 in ['micro','macro','weighted']:
                evalMetric['f1_'+metric_f1] = metricDict[metric](yTest, yPred, average = metric_f1)
        elif metric in ['rocAuc']:
            yPred_prob = model.predict_proba(xTest)
            print (yPred_prob)
            if numOfClass == 2: #if the output is binary
                evalMetric['f1_binary'] = metricDict[metric](yTest, yPred_prob ,average = 'binary',multi_class = 'ovr')
            evalMetric['f1'] = metricDict[metric](yTest,yPred, average = None,multi_class = 'ovr')
            for metric_f1 in ['micro','macro','weighted']:
                evalMetric['f1_'+metric_f1] = metricDict[metric](yTest, yPred, average = metric_f1,multi_class = 'ovr')
        else:
            print(metric)
            evalMetric[metric] = metricDict[metric](yTest,yPred)
            if metric == 'confusionMatrix':
                disp = ConfusionMatrixDisplay(confusion_matrix=evalMetric[metric],display_labels=model.classes_)
                disp.plot()
                plt.show()
    return evalMetric

def crossValidate(df, output, model, n_splits = 5, shuffle = False, random_state = None, metric = None):
    x = df.loc[:, df.columns != output].values
    y = df.loc[:, df.columns == output].values.ravel()
    metricList = []
    kf = StratifiedKFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)
    for trainIndex, testIndex in kf.split(x,y): #looping through the train and test set indices for the splits
        evalMetric = fitAndEvaulateModel(x[trainIndex], x[testIndex], y[trainIndex], y[testIndex], model, metric = metric)
        metricList += [evalMetric]
    print(f'{model} : {n_splits} fold cross validation result: {np.mean(metricList):.3f}+/-{np.std(metricList):.3f}')

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