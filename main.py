import pandas as pd

from CustomFunctions import *
filePath = '../'
#filePath = '/Users/jidengcheng/Desktop/Machine Learning Data/Data/'
fileName = 'predictive_maintenance.csv'
#fileName = 'mushroom_dataset.csv'

df = importFile(filePath, fileName, show_dataFrame = 10)
features = df.columns.values.tolist()
print(features)

#showGraph(df,output='Failure Type', plot_type='summaryPlot')

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

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

# data = {'feature1': [1,2,3,4,5,6,7,8],'feature2':[100,200,300,400,500,600,700,800],'class':['A','B','C','C','B','A','B','B']}
# df = pd.DataFrame(data)
# df = df6
# print (df)
# x = df.loc[:,df.columns != 'Failure Type'].values
# y = df.loc[:,df.columns == 'Failure Type'].values.ravel()
# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.5, train_size=0.5,random_state=1, shuffle=True, stratify=y)
# cls = DecisionTreeClassifier()
# cls.fit(xTrain,yTrain)
# yPred = cls.predict_proba(xTest)
# score = roc_auc_score(yTest,yPred,average = 'micro',multi_class='ovr')
# print (score)
# exit()
#showGraph(df6,output = 'Failure Type',feature_selected='Rotational speed [rpm],Target', plot_type='histo',set_yscale='log',kde = False)

cls = GaussianNB()
cls = SVC()
cls = MLPClassifier()
cls = DecisionTreeClassifier()
# gridSearch(df6,output='Failure Type', model=cls, param_grid={'criterion':['gini','entropy'],'min_samples_leaf':[2,3],'max_features':['auto','log2','sqrt']},
#            return_train_score=True, show_graph=True,print_results=True)
# cls = RandomForestClassifier()
# cls = KNeighborsClassifier()
a= (stringToList('accuracy,f1'))

#crossValidate(df6,output = 'Failure Type',model = cls,shuffle = True,random_state=0,metricList = 'rocAuc')

xTrain, xTest, yTrain, yTest = trainTestSplit(df6,output='Failure Type', test_size= 0.2, train_size=0.8, stratify = True)
print (fitAndEvaulateModel(xTrain,xTest,yTrain,yTest,model = cls,metricList='rocAuc',rocCurve = 'macro',class_to_show='No Failure'))


