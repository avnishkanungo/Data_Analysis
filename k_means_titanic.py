import matplotlib.pyplot as plt 
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation
import pandas as pd 

data = pd.read_excel('/Users/avnish/Downloads/titanic.xls')
#print data.head()
data.drop(['name','body'],1,inplace=True)
data.convert_objects(convert_numeric = True)
data.fillna(0,inplace = True)
#print data.head()



def handle_nonnumeric_items(data):
	columns = data.columns.values
	for i in columns:
		text_to_num = {}
		def convert_to_int(val):
			return text_to_num[val]

		if data[i].dtype != np.int64 or data[i].dtype != np.float64:
			data_content = data[i].values.tolist()
			unique_content = set(data_content)
			#print unique_content
			x = 0
			for u in unique_content:
				if u not in text_to_num:
					text_to_num[u] = x
					x+=1

			data[i] = list(map(convert_to_int,data[i]))

	return data
	#Remember indentation is a bitch!
	#RETURN ALWAYS OUT OF THE FOR LOOP


data2 = handle_nonnumeric_items(data)
print data2.head()

data2.drop(['sex','boat'],1,inplace = True)
X = np.array(data2.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
print len(X)
y = np.array(data2['survived'].astype(float))
print y

clf = KMeans(n_clusters = 2) 
clf.fit(X)
correct = 0
pred = []
for i in range(len(X)):
	predict = np.array(X[i].astype(float))
	predict = predict.reshape(-1,len(predict))
	prediction = clf.predict(predict)
	print prediction
	# norm_prediction = np.array(prediction.astype(float))
	pred.append(prediction[0])
	pred_array = np.array(prediction.astype(float))
	if prediction[0] == y[i]:
	 	correct+=1
print correct
print pred
count = 0
for i in pred:
	for j in y:
		if i == j:
			count+=1
print count/len(X)

		







