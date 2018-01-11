import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#Step 1: Get the files
# Read the file from thhe net
dataframe_all = pd.read_csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
# get count of rows
num_rows = dataframe_all.shape[0]

#Step 2: Clean the data
#count th number of missing elements in each column
count_nan = dataframe_all.isnull().sum()
#count not null elements
count_not_nan = count_nan[count_nan==0]
# remove all columns with missing data
dataframe_all = dataframe_all[count_not_nan.keys()]
# remove the first column which has no important info
dataframe_all = dataframe_all.ix[:,7:]
columns = dataframe_all.columns
print columns

#Step 3: Create feature vectors
#choosing the features
x = dataframe_all.ix[:,:-1].values
standard_scalar = StandardScaler()
x_std = standard_scalar.fit_transform(x)

# Step 4: Encoding class labels y
y = dataframe_all.ix[:,-1].values
#encoding the labels
class_labels = np.unique(y)
print class_labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Step5: Creat training and testing sets
x_train,x_test,y_train,y_test = train_test_split(x_std,y,test_size = 0.1, random_state = 0)

# Applying T-NSE
from sklearn.manifold import TSNE

TSNE = TSNE(n_components = 2, random_state = 0)
x_test_2D = TSNE.fit_transform(x_test)
print x_test_2D

markers=('s', 'd', 'o', '^', 'v')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=x_test_2D[y_test==cl,0], y=x_test_2D[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()


