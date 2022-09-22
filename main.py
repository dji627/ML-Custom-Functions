from CustomFunctions import *
filePath = '../'
fileName = 'predictive_maintenance.csv'

df = importFile(filePath, fileName, show_dataFrame = 10)
features = df.columns.values.tolist()
print(features)

#exploratoryAnalysis(df, output = 'Failure Type', feature_selected=None,plot_type='histo',fig_size_y = 20,fig_size_x = 40)
df2 = preprocessing(df,preprocess_type = 'featToRemove', feature_selected = '\ufeffUDI, Product ID')
#showDataInfo(df2)

featuresToScale = 'Air temperature [K], Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]'
df3 = preprocessing(df2, preprocess_type='minMaxScaler',feature_selected=featuresToScale)
df4 = preprocessing(df3, preprocess_type='ordinalEncode', feature_selected= 'Type', encoder_key = ['H','M','L'])
showDataInfo(df3)