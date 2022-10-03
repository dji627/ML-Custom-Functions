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

def showGraph(df, output, feature_selected = 'All',plot_type = None,fig_size_y = 15, fig_size_x = 30,set_yscale = None,
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


    # for metric in metricList:
    #     if metric == 'accuracy':
    #         evalMetric.append(accuracy_score(yTest, yPred))
    #     elif metric == 'f1':
    #         evalMetric.append(f1_score(yTest,yPred))
    #     elif metric == 'precision':
    #         evalMetric.append(precision_score(yTest,yPred))
    #     elif metric == 'recall':
    #         evalMetric.append(recall_score(yTest,yPred))
    #     elif metric == 'logLoss':
    #         evalMetric.append(log_loss(yTest,yPred))
    #     elif metric == 'rocAuc':
    #         evalMetric.append(roc_auc_score(yTest,yPred))
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