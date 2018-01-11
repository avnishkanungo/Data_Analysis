import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

all_Deaths = pd.read_csv('/Users/avnish/Desktop/character-predictions.csv')

count_nan = all_Deaths.isnull().sum()
count_not_nan = count_nan[count_nan == 0]
num_rows = all_Deaths.shape[1]
print num_rows


all_Deaths = all_Deaths[count_not_nan.keys()]

print all_Deaths.sum()
all_Deaths =all_Deaths.ix[:,7 :]
#all_Deaths =all_Deaths[["actual","pred","alive","plod"]]
columns = all_Deaths.columns
print columns



x = all_Deaths.ix[:,:-1].values
#x = all_Deaths =all_Deaths[["actual","pred","alive","plod"]].values
standard_scalar = StandardScaler()
x_std = standard_scalar.fit_transform(x)

y = all_Deaths[:,-1].values
#y = all_Deaths[["actual","pred","alive","plod"]].values
#encoding the labels
class_labels = np.unique(y)
print class_labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

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
