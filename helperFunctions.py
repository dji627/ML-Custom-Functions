def stringToList(string, seperator = ','):
    return [x.strip() for x in string.split(seperator)]

# to account for different input format of the targeted features
# string_if_single: whether to make the feature a string or list if there is only 1 feature selected
def selectingFeatures(df, feature_input, output = None, remove_output = False, string_if_single = False, show_features_applied = False):
    featuresToApply = None
    if feature_input == 'All':
        featuresToApply = list(df.columns)
        if remove_output == True:
            featuresToApply.remove(output)
    elif (type(feature_input) == list):
        featuresToApply = feature_input
    elif (type(feature_input) == str):
        featuresToApply = stringToList(feature_input)
    elif type(feature_input) == tuple:
        featuresToApply = list(feature_input)
    if len(featuresToApply) == 1 and string_if_single == True:
        featuresToApply = featuresToApply[0]
    if show_features_applied == True:
        print (f'features selected: {featuresToApply}')
    return featuresToApply

def convertToList(input):
    if type(input) == list:
        return input
    elif type(input) == str:
        return stringToList(input)
    elif type(input) == tuple:
        return list(input)



