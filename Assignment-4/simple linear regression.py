import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


df=pd.read_csv("abalone.csv")
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(df,test_size=0.3, random_state=42)
train_data.to_csv("abalone_train.csv",index=False)
test_data.to_csv("abalone_test.csv",index=False)





def pearson_correlation(x,y):
    mean_x=x.mean()
    mean_y=y.mean()
    x=x.to_numpy()
    y=y.to_numpy()
    num=np.sum((x-mean_x)*(y-mean_y))
    A=(x-mean_x)**2
    B=(y-mean_y)**2
    denom=np.sqrt(np.sum(A)*np.sum(B))
    r=num/denom
    return r

corr_coeff=[]
columns=train_data.columns[1:-1]
#print(columns)
for i in range(len(columns)):
    corr_value = pearson_correlation(train_data[columns[i]], train_data["Rings"])
    corr_coeff.append(abs(corr_value))
    
corr_coeff=np.array(corr_coeff)
max_index=np.argmax(corr_coeff)
#print(max_index)
input_variable=train_data[columns[max_index]]
mean_input=input_variable.mean()
 

input_variable=input_variable.to_numpy()
target_variable=train_data["Rings"].to_numpy()
mean_target=np.mean(target_variable)

#calculating w hat
num=np.sum((input_variable-mean_input)*(target_variable-mean_target))
denom=np.sum((input_variable-mean_input)**2)
w_hat=num/denom

print("Slope of best_fit_line",w_hat)


#calculating w_not
w_not=mean_target-w_hat*mean_input
print("intercept of best_fit_line",w_not)


predicted_values=w_hat * input_variable + w_not


#plotting

plt.scatter(input_variable, target_variable, color='blue', label='Data points')  

plt.plot(input_variable, predicted_values, color='red', label='Best-fit line')  

plt.xlabel(columns[max_index])
plt.ylabel('Rings')
plt.title(f'Best-Fit Line for {columns[max_index]} vs Rings')
plt.legend()
plt.show()





#Prediction Accuracy on train data

rmse=np.sqrt((np.sum((predicted_values-target_variable)**2))/len(target_variable))
denom=np.mean(target_variable)
train_accuracy=1-(rmse/denom)

print("The accuracy of training data is",train_accuracy*100)

#Prediction Accuracy on test data
input_variable_test=test_data[columns[max_index]].to_numpy()
target_variable_test=test_data["Rings"].to_numpy()

predicted_values_test=input_variable_test*w_hat+w_not

rmse=np.sqrt((np.sum((predicted_values_test-target_variable_test)**2))/len(target_variable))
denom=np.sum(target_variable_test)/len(target_variable_test)
test_accuracy=1-(rmse/denom)

print("The accuracy of testing data is",test_accuracy*100)



#plotting

plt.scatter(target_variable_test, predicted_values_test, color='blue')  
plt.xlabel("Actual Data(Rings)")
plt.ylabel('Predicted Data("Rings")')
plt.title("Actual Vs Predicted Data")
plt.show()


    
    
    
    