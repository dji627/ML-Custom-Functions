from CustomFunctions import *
filePath = '../'
fileName = 'predictive_maintenance.csv'

df = importFile(filePath, fileName, show_dataFrame = 10)
features = df.columns.values.tolist()
print(features)

#exploratoryAnalysis(df, output = 'Failure Type', feature_selected=None,plot_type='histo',fig_size_y = 20,fig_size_x = 40)
df2 = preprocessing(df,preprocess_type = 'featToRemove', feature_selected = ['\ufeffUDI', 'Product ID'])

featuresToNormalize = 'Air temperature [K],Process temperature [K],Tool wear [min]'
featuresToStandardize = 'Torque [Nm]'

df3 = preprocessing(df2, preprocess_type='minMaxScaler',feature_selected=featuresToNormalize)
df3_1 = preprocessing(df3,preprocess_type='standardization',feature_selected=featuresToStandardize)
df4 = preprocessing(df3_1, preprocess_type='ordinalEncode', feature_selected= ('Type'), encoder_key = {'H':2,'M':1,'L':0})
#df5 = preprocessing(df3, preprocess_type='ordinalEncode', feature_selected= ('Type'), encoder_key = ['H','M','L'])

showDataInfo(df4, 'All', show_data_type = True)

#showGraph(df4,output = 'Failure Type', feature_selected='Rotational speed [rpm],Target', plot_type='histo',set_yscale='log',kde = False)

#exploreFeatures(df4,exploration_type='heatMap')

df5 = preprocessing(df4,feature_selected='Rotational speed [rpm]', preprocess_type = 'handleSkewedData',skew_transformation='boxCox')
#showGraph(df5,output = 'Failure Type', feature_selected='Rotational speed [rpm],Target', plot_type='histo',set_yscale='log',kde = False)
df6 = preprocessing(df5,feature_selected='Rotational speed [rpm]',preprocess_type='minMaxScaler')
showDataInfo(df6)

showGraph(df6,output = 'Failure Type',feature_selected='Rotational speed [rpm],Target', plot_type='histo',set_yscale='log',kde = False)
