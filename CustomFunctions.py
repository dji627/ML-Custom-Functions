from helperFunctions import *
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.api.types import is_int64_dtype, is_float_dtype, is_object_dtype
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
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

def exploratoryAnalysis(df, output, feature_selected = None,plot_type = None,fig_size_y = 15, fig_size_x = 30):
    # if feature_selected != None:
    #     featuresToApply = list(feature_selected)
    featuresToApply = selectingFeatures(df, feature_input= feature_selected, output = output)
    plt.figure(figsize = (fig_size_y, fig_size_x))
    plotCol = min(len(featuresToApply),3)
    for i, feature in enumerate(featuresToApply):
        plt.subplot(len(featuresToApply), plotCol, i + 1)
        if plot_type == 'scatter':
            sns.scatterplot(data=df, x=feature, y=output, legend='auto')
        elif plot_type == 'histo':
            sns.histplot(data=df, x=feature, hue=output, multiple='stack')
        elif plot_type == 'box':
            sns.boxplot(x=df[feature])
        elif plot_type == 'pair':
            sns.pairplot(df, hue=output, vars=feature)
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
            print(f'{col}: Max:{max}, Min:{min}, Median:{median:.3f}'
                                         f' Mean:{mean:.3f}, std:{std:.3f}, Mode:{mode}\n\n')
        elif is_object_dtype(df[col]) == True:
            uniqueVal, uniqueCount = np.unique(df.loc[:, col], return_counts=True)
            print(f'{col}({len(uniqueVal)} unique values:)')
            for value, count in zip(uniqueVal, uniqueCount):
                print (f'{value}: {count}')

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

def preprocessing(dataframe, preprocess_type, feature_selected, encoder_key = None, convert_to = None,missing_value_handle = None,
                  value_to_remove = None, replace_value = None):
    print ('Function: preprocessing2 called')
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
    return df

def encode_onehot(df, feature_selected):
    featureSelected = selectingFeatures(df, feature_input= feature_selected, string_if_single= True)
    df2 = pd.get_dummies(df[featureSelected], prefix='', prefix_sep='').groupby(level=0, axis=1).max().add_prefix(featureSelected+' - ')
    df3 = pd.concat([df, df2], axis=1)
    df3 = df3.drop([featureSelected], axis = 1)
    return df3