import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix

####Q1####
#i
df_train=pd.read_csv('iris_train.csv')
X_1=df_train[df_train.columns[1:5]].values
y_train=df_train['Species'].values

df_test=pd.read_csv('iris_test.csv')
X_2=df_test[df_test.columns[1:5]].values
y_test=df_test['Species'].values

median=[np.median(X_1[:,i]) for i in range(X_1.shape[1])]

def outlier_correction(X_given):
    X=X_given.copy()
    for i in range(X.shape[1]):
        col=X[:,i]
        q1=np.percentile(col,25)
        q3=np.percentile(col,75)
        IQR=q3-q1
        l_b=q1-1.5*IQR
        u_b=q3+1.5*IQR
        
        for j in range(len(col)):
            if(col[j]<l_b) or (col[j]>u_b):
                col[j]==median[i]
    return X
X_train_otlcrr=outlier_correction(X_1)
X_test_otlcrr=outlier_correction(X_2)

means=np.mean(X_train_otlcrr, axis=0)
X_ms=X_train_otlcrr-means
Corr=np.cov(X_ms.T)
e_val, e_vec=np.linalg.eig(Corr)
sort_index=np.argsort(e_val)[::-1]
sorted_e_val=e_val[sort_index]
sorted_e_vec=e_vec[:,sort_index]
#print(sorted_e_vec)

Q=sorted_e_vec[:,:1].T
X_train=np.dot(Q,X_train_otlcrr.T).T
X_test=np.dot(Q, X_test_otlcrr.T).T
#print(X_train)
#print(X_test)

#ii

def X_stats(X,y):  #calculate mean and variance of each class (the parameters)
    classes=np.unique(y)
    stats={}
    for clas in classes:
        X_cls=X[y==clas]
        stats[clas]=[np.mean(X_cls),np.var(X_cls),len(X_cls)/len(X)]
    return stats

def gaussian(x, mean, var):  #guassion distribution
    coeff=1/np.sqrt(2*np.pi*var)
    exponent=np.exp(-((x-mean)**2)/(2*var))
    return coeff*exponent

def predict(X,stats):
    predictions=[]
    for x in X:
        posteriors={}
        for clas,stat in stats.items():
            prior=stat[2]
            likelihood=gaussian(x, stat[0], stat[1])
            posteriors[clas]=prior*likelihood
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)

stats=X_stats(X_train,y_train)# calculate stats to be applied for predicting on test data

#iii
prediction=predict(X_test,stats)
#print(prediction)

#iv
confusion = confusion_matrix(y_test, prediction)
#print(confusion)

def accuracy(conf_matrix):
    correct_predictions = np.trace(conf_matrix)  # Sum of diagonal elements
    total_predictions = np.sum(conf_matrix)
    return (correct_predictions / total_predictions) * 100
accuracy_1D=accuracy(confusion)
#print(accuracy_1D)


####Q2####
def X_stats_2(X,y):  #calculate mean and variance of each class (the parameters)
    classes=np.unique(y)
    stats={}
    for clas in classes:
        X_cls=X[y==clas]
        stats[clas]=[np.mean(X_cls, axis=0),np.cov(X_cls, rowvar=False),X_cls.shape[0]/X.shape[0]]
    return stats

def gaussian_2(x, mean, cov):  #guassion distribution
    return multivariate_normal.pdf(x,mean=mean, cov=cov)

def predict_2(X,stats):
    predictions=[]
    for x in X:
        posteriors={}
        for clas,stat in stats.items():
            prior=stat[2]
            likelihood=gaussian_2(x, stat[0], stat[1])
            posteriors[clas]=prior*likelihood
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)

#ii
stats_2=X_stats_2(X_train_otlcrr,y_train)# calculate stats to be applied for predicting on test data
prediction_2=predict_2(X_test_otlcrr,stats_2)
#print(prediction_2)

#iii
confusion_2 = confusion_matrix(y_test, prediction)
#print(confusion_2)
accuracy_4D=accuracy(confusion_2)
#print(accuracy_4D)


####Q3####
accuracy_diff = accuracy_4D - accuracy_1D
#print(accuracy_diff)