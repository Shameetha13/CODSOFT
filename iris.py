import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
ds = pd.read_csv("IRIS.csv")
ds.head()
ds.info()
ds.describe()
ds.isnull().sum()
n = len(ds[ds['species'] == 'Iris-versicolor'])
print("No of Versicolor in Dataset:",n)

n1 = len(ds[ds['species'] == 'Iris-virginica'])
print("No of Virginica in Dataset:",n1)

n2 = len(ds[ds['species'] == 'Iris-setosa'])
print("No of Setosa in Dataset:",n2)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Iris-Versicolor', 'Iris-Setosa', 'Iris-Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([ds['sepal_length']])
plt.figure(2)
plt.boxplot([ds['sepal_width']])
plt.show()
ds.hist()
plt.show()
ds.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)
ds.plot(kind ='box',subplots = True, layout =(2,5),sharex = False)
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petal_length',data=ds)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petal_width',data=ds)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='petal_length',data=ds)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='petal_width',data=ds)
sns.pairplot(ds,hue='species');

X = ds['sepal_length'].values.reshape(-1,1)
print(X)

Y = ds['sepal_width'].values.reshape(-1,1)
print(Y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='b')
plt.show()
from sklearn.preprocessing import LabelEncoder

# function for encoding the categorical column
def label_encode_columns(dataframe, columns):
    label_encoder = LabelEncoder()
    for column in columns:
        dataframe[column] = label_encoder.fit_transform(dataframe[column])
    return dataframe

# Example usage for 'Sex' and 'Embarked' columns
columns_to_label_encode = ['species']
ds = label_encode_columns(ds, columns_to_label_encode)

# Display the updated DataFrame
ds.head()
corr_mat = ds.corr()
print(corr_mat)
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(ds.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
train, test = train_test_split(ds, test_size = 0.25)
print(train.shape)
print(test.shape)
train_X = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
train_y = train.species
test_X = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
test_y = test.species
train_X.head()
test_y.head()
test_y.head()
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))
results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.947,0.947,0.947,0.947,0.921]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
