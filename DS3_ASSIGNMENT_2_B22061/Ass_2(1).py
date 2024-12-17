import numpy as np
import pandas as pd

#Problem-I
#a]

f=pd.read_csv("Iris.csv")
#print(f)
Columns=f.columns.tolist()
#print(Columns)
Matrix=f[Columns[0:len(Columns)-1]].values
#print("Matrix of attributes is: ")
#print(Matrix)
Array=np.array(f['Species'])
#print("Array of target attribute is:")
#print(Array)

#b]

X=f.copy()
for i in range(len(Columns)-1):
    q1 = np.percentile(X[Columns[i]], 25)
    q3 = np.percentile(X[Columns[i]], 75)

    
    iqr = q3 - q1

    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Replace outliers with the median
    median = np.median(X[Columns[i]])
    for j in range(len(X[Columns[i]])):
        if (X[Columns[i]][j] <lower_bound) or (X[Columns[i]][j] >upper_bound):
            X[Columns[i]][j]==median
#print('Our X is: ')
#print(X)

#C]
Matrix_of_X=X[Columns[0:len(Columns)-1]].values
#print(Matrix_of_X)
#Step1
mean=np.mean(Matrix_of_X,axis=0)
X_subtract_mean=Matrix_of_X-mean

#Step2:Computing correlation matrix
C=np.dot(X_subtract_mean.T,X_subtract_mean)
#print("Correlation matrix is: ")
#print(C)

#Step3:Eigen analysis of C
Eigenvalues,Eigenvectors=np.linalg.eig(C)
#print("Eigenvalues are: ")
#print(Eigenvalues)
#print("Eigenvalues are: ")
#print(Eigenvectors)

#Step4:Sorting the eigenvalues in descending order
Eigen_index=np.argsort(Eigenvalues)[::-1]
Descend_eigvalues=Eigenvalues[Eigen_index]
Corresponding_eigenvectors=Eigenvectors[:,Eigen_index]
#print('Sorted Eigenvalues are:')
#print(Descend_eigvalues)

#Step5:L=2 and create matrix 2x4 size
L=2
Q=Corresponding_eigenvectors[:,:L].T
#print('Matrix Q is: ')
#print(Q)

#Step6and7:Projecting each sample x  in l directions by performing Y=QX
Y=np.dot(Q,Matrix_of_X.T).T
#print("Reduced data is:")
#print(Y)
a=pd.DataFrame(Y)
####a.to_csv("Reduced Dimension.csv")


#d]
import matplotlib.pyplot as plt
plt.scatter(Y[:,0],Y[:,1],c='b')
#plt.show()
#Now we will superimpose  eigen directions with proper scaling
mean_data=np.mean(Y,axis=0)
print(mean_data)
plt.quiver(mean_data[0],mean_data[1],0,Eigenvectors[0].dot(Eigenvectors[0]),\
           angles='xy',scale_units='xy',scale=0.5,color='red',label='Eigen-Direction')
plt.quiver(mean_data[0],mean_data[1],Eigenvectors[1].dot(Eigenvectors[1]),\
           angles='xy',scale_units='xy',scale=0.5,color='red',label='Eigen-Direction')

plt.title('Scatter Plot of Dimension reduced Data with eigendirections')
plt.show()

#ALGORITHM-2: Reconstruction----
#Step1:Performing X'=YQ
X_recon=np.dot(Y,Q)
print(X_recon)


#f]
rmse_list = []
for col in range(4):  
    rmse = np.sqrt(np.mean((Matrix_of_X[:, col] - X_recon[:, col])**2))
    rmse_list.append(rmse)

# Print RMSE for each attribute
for i, rmse in enumerate(rmse_list):
    print(f"RMSE for attribute {i+1}: {rmse}")

#II]


import sklearn.model_selection 
from sklearn.metrics import confusion_matrix
import seaborn as sns


df2 = pd.read_csv("Iris.csv")
y = df2["Species"]
df = pd.read_csv("Reduced Dimension.csv")
df.drop(df.columns[0],axis = 1, inplace = True)

#Data split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, random_state=104, test_size=0.2, shuffle=True)

x_test.reset_index(drop=True, inplace=True)
x_train.reset_index(drop=True, inplace=True)
y_train2 = y_train.reset_index(drop=True)

predicted = []
for i in range(len(x_test)):
    distance = []
    x = x_test.loc[i]    
    for j in range(len(x_train)):
        y = x_train.loc[j]
        dist = np.linalg.norm(x-y)
        distance.append([dist, y_train2[j]])

    distance.sort(key=lambda x: x[0])
    nearest_neighbors = distance[:5]

    classes = pd.Series([d[1] for d in nearest_neighbors]).value_counts()
    most_common_class = classes.keys()[0]
    predicted.append(most_common_class)

confusion = confusion_matrix(y_test, predicted)
sns.heatmap(confusion, annot=True, cmap='Reds')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()








    