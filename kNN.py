import pandas as pd
import numpy as np 
from sklearn import preprocessing, cross_validation, neighbors
import random
from collections import Counter

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")
num_rows = data.shape[0]
print num_rows

data.columns = ["id","clump_thickness","unif_cell_size","unif_cell_shape","marg_adhesion","single_epth_cell_size","bare_nuclei","bland_chrom","norm_nucleoli","mitosis","class"]
data.replace('?',-99999,inplace = True)
data.drop(["id"],1, inplace = True)
print data.head()

def KNN_Working(data,predict,k):

	if len(data) >= k:
		print ("k is set to value lower than total voting groups")
		return
	else:
		distances = []	
		for group in data:
			for features in data[group]:
				distance = np.linalg.norm(np.array(features)-np.array(predict))
				print distance
				distances.append([distance,group])
				print distances
		votes = [i[1] for i in sorted(distances)[:k]]
		print(Counter(votes).most_common(1))
		vote_result = Counter(votes).most_common(1)[0][0]
		return vote_result


def KNN_Lib(data):
	X = np.array(data.drop(["class"],1))
	y = np.array(data["class"])

	X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2)
	clf = neighbors.KNeighborsClassifier()
	clf.fit(X_train,y_train)

	accuracy = clf.score(X_test,y_test)
	print accuracy

	example = np.array([4,2,2,1,1,4,2,2,5])
	example = example.reshape(1,-1)

	prediction = clf.predict(example)
	print(prediction)

def KNN_Algo(data):
	all_data = data.astype(float).values.tolist()
	#print all_data
	random.shuffle(all_data)
	#print ("##################################################")
	#print all_data
	test_size = 0.2
	train_set = {2:[],4:[]}
	test_set = {2:[],4:[]}
	train_data = all_data[:-int(test_size)*int(len(all_data))]
	test_data = all_data[-int(test_size)*int(len(all_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])
		#print train_set
		#print ("#########################################################")
	for i in test_data:
		test_set[i[-1]].append(i[:-1])
		#print test_set

	correct = 0
	total = 0

	for group in test_set:
		for data in test_set[group]:
			print data
			vote = KNN_Working(train_set, data, k = 5)
			if group == vote:
				correct += 1
				print correct
			total += 1	
	print(correct/total)

KNN_Lib(data)
KNN_Algo(data)

