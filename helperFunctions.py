def stringToList(string, seperator = ','):
    return [x.strip() for x in string.split(seperator)]

# to account for different input format of the targeted features
# string_if_single: whether to make the feature a string or list if there is only 1 feature selected
def selectingFeatures(df, feature_input, output = None, string_if_single = False):
    featuresToApply = None
    if feature_input == 'All':
        featuresToApply = list(df.columns)
        if output != None:
            featuresToApply.remove(output)
    elif (type(feature_input) == list):
        featuresToApply = feature_input
    elif (type(feature_input) == str):
        featuresToApply = stringToList(feature_input)
    elif type(feature_input) == tuple:
        featuresToApply = list(feature_input)
    if len(featuresToApply) == 1 and string_if_single == True:
        featuresToApply = featuresToApply[0]
    print (f'features selected: {featuresToApply}')
    return featuresToApply

