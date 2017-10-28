import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('ex1data2.txt', delimiter = ',')

#Seperate Features and Output

X = data[:, :-1]                #Features without index and ones
y = data[:, -1]                  #Output
num_examples = X.shape[0]        #No. of examples
num_features = X.shape[1]        #No. of features

#Since the features may be on a different scale, we will perform feature normalization on the data.
#Here we compute the mean and standard deviation of all examples feature-wise and then. Then we subtract the mean from data and divide by standard deviation.

mu = np.mean(X, axis = 0, dtype = np.float64)                          #mean of features
sigma = np.std(X, axis = 0, dtype = np.float64)                   #std deviation
X_norm = np.c_[np.ones(num_examples), (X - mu) / sigma]                #normalize features and add bias term

#Now we will perform gradient descent on the data to calculate regression parameters.

param = np.zeros((num_features + 1, 1))                                #initial parameters            
alpha = 0.3                                                         #learning rate
iterations = 400                                                      #no. of iterations     
cost = np.zeros((iterations, 1))

#Begin Gradient Descent

for iter in range(iterations):
	temp_param = np.zeros((num_features + 1, 1))
	for j in range(num_features + 1):
		for i in range(num_examples):
			temp_param[j] = temp_param[j] + (np.dot(X_norm[i, :, np.newaxis].T , param) - y[i]) * X_norm[i, j]
		temp_param[j] = param[j] - (alpha / num_examples) * temp_param[j]
	param = temp_param
	cost[iter] = (1 / 2 * num_examples) * np.sum((np.dot(X_norm, param) - y) ** 2)


#Plot the cost as a function of no. of iterations.

plt.plot(np.arange(len(cost)), cost, color = "b")
plt.xlabel('No. of iterations')                            #Set X-axis label
plt.ylabel('Cost')                                         #Set Y-axis label
plt.show()

#Compute selling price as predicted by linear regression
selling_price = np.squeeze(np.dot(X_norm, param))

#Draw Scatter plot of actual and predicted prices

plt.scatter(selling_price, y, color = "m", marker = "o", s = 30)
plt.show()

#Print the predicted and actual prices

for i in range(num_examples):
	print('House {0:2d}\tActual = {1:.3f}\tPredicted = {2:.3f}\n'.format(i + 1, y[i], selling_price[i]))
