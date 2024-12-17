import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('asianpaint.csv', index_col=0, parse_dates=True)
#print(data)


train_size = int(len(data) * 0.65)
train, test = data[:train_size], data[train_size:]
#print(train)


plt.figure(figsize=(10,6))
plt.plot(train, label='Training Data')
plt.plot(test, label='Test Data')
plt.title('Training and Testing Datasets')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.legend()
plt.show()

train_input=train[:-1]["Open"].to_numpy()
train_output=train[1:]["Open"].to_numpy()
mean_input=np.mean(train_input)
mean_target=np.mean(train_output)


#calculating w hat
num=np.sum((train_input-mean_input)*(train_output-mean_target))
denom=np.sum((train_input-mean_input)**2)
w_hat=num/denom

print("Slope of best_fit_line",w_hat)


#calculating w_not
w_not=mean_target-w_hat*mean_input
print("intercept of best_fit_line",w_not)


test_input=test[:-1]["Open"].to_numpy()
test_output=test[1:]["Open"].to_numpy()

test_predicted=test_input*w_hat+w_not

#plotting
plt.plot(test_output,test_predicted, color='blue')  
plt.xlabel("Actual Data")
plt.ylabel('Predicted Data')
plt.title("Actual Vs Predicted Data")
plt.show()

#RMSE percentage
rmse=np.sqrt((np.sum((test_predicted-test_output)**2))/len(test_predicted))
mean=np.mean(test_output)
rmse_percent=(rmse/mean)*100

print("The rmse_percent of test data is",rmse_percent)

#map
map_percent=np.sum((abs(test_output-test_predicted))/test_output)/len(test_output)*100
print("The map_percent of test data is",map_percent)



