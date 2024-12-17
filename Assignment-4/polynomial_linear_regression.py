import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

train_data=pd.read_csv("abalone_train.csv")

input_train=train_data["Shell weight"].to_numpy()
target_train=train_data["Rings"].to_numpy()

test_data=pd.read_csv("abalone_test.csv")

input_test=test_data["Shell weight"].to_numpy()
target_test=test_data["Rings"].to_numpy()

#function to calculate vandermode matrix for degree p
def matrix(p, input_data):
    n = len(input_data)
    
    # Initialize the Vandermonde matrix Z with zeros
    Z = np.zeros((n, p + 1))
    

    for i in range(n):
        for j in range(p + 1):
            Z[i][j] = input_data[i] ** j
    
    return Z


#function to caluclate rmse
def rmse(y_predict,y_original):
    rm=np.sqrt((np.sum((y_predict-y_original)**2))/len(y_predict))
    return rm

#function to calculate y predicted for a degree p polynomial
def w_hat_calculate(p,input_data,target):
    n=len(input_data)
    
    #calculating the dagger
    Z=matrix(p,input_data)
    Z_trans=np.transpose(Z)
    temp=np.dot(Z_trans,Z)
    temp=np.linalg.inv(temp)
    temp=np.dot(temp,Z_trans)
    w_hat=np.dot(temp,target)
    return w_hat
   
def predicted(w_hat,Z):
    y_predicted=[]
    n=Z.shape[0]
    for i in range(n):
        y=np.dot(w_hat,np.transpose(Z[i]))
        y_predicted.append(y)
        
    return y_predicted

    
rmse_train=[]
p_values=[2,3,4,5]

for i in p_values:
    w_hat=w_hat_calculate(i, input_train, target_train)
    Z=matrix(i,input_train)
    y_hat=predicted(w_hat,Z)
    rmse_value=rmse(y_hat,target_train)
    mean=np.mean(target_train)
    accuracy=1-(rmse_value/mean)
    print("The prediction accuracy for training data ",i," ",accuracy*100)
    rmse_train.append(rmse_value)

#train_data  
#plotting
plt.scatter(p_values,rmse_train)  
plt.xlabel("p_values")
plt.ylabel('rmse')
plt.title("p_values v/s rmse for train data")
plt.show()


print()
print()
print()



#test_data
rmse_test=[]
p_values=[2,3,4,5]

for i in p_values:
    w_hat=w_hat_calculate(i, input_train, target_train)
    Z=matrix(i,input_test)
    y_hat=predicted(w_hat,Z)
    rmse_value=rmse(y_hat,target_test)
    mean=np.mean(target_test)
    accuracy=1-(rmse_value/mean)
    print("The prediction accuracy for testing data ",i," ",accuracy*100)
    rmse_test.append(rmse_value)

    
#plotting
plt.scatter(p_values,rmse_test)  
plt.xlabel("p_values")
plt.ylabel('rmse')
plt.title("p_values v/s rmse for test data")
plt.show()



#plotting
w_hat_new=w_hat_calculate(5, input_train,target_train)
Z=matrix(5, input_train)
predicted_values=predicted(w_hat_new,Z)

plt.scatter(input_train, target_train, color='blue', label='Data points')  

plt.plot(input_train, predicted_values, color='red', label='Best-fit Curve')  

plt.xlabel("Shell Weight")
plt.ylabel('Rings')
plt.title('Best-Fit Line for Shell_weight vs Rings')
plt.legend()
plt.show()