import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import  scipy.optimize as opt

data = pd.read_csv("/home/khaled/Documents/python/Logostic Reggression/Logistic regression with regularization/data.txt"
                   , header=None , names=["Test 1" , "Test 2" , "Accepted"])
# print(f"Data = \n {data.head(10)} \n Discribe \n {data.describe()}")

positive = data[data["Accepted"].isin([1])] 
negative = data[data["Accepted"].isin([0])]

# print(f"\npositive \n {positive}\n negative \n {negative}")
# fig , ax = plt.subplots(figsize=(8,5)) #fig = framework , ax = x , y axiss
# ax.scatter(positive["Test 1"] , positive["Test 2"] , s = 50 , c  = 'b' , label="Accepted")
# ax.scatter(negative["Test 1"]  , negative["Test 2"] , s = 50 , color='r' , marker='x' , label="Not Accepted")
# ax.legend()
# ax.set_xlabel("Test 1 Score")
# ax.set_ylabel("Tets 2 score")
# plt.show();


def sigmoid(z):
    return 1 / (1+np.exp(-z))


def costReg(theta , X , y , lr):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X * theta.T)))
    second = np.multiply((1-y) , np.log(1 - sigmoid(X * theta.T)))
    reg = (lr / (2 * len(X))) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

def gradient(thetav, xv, yv, lr):
    thetav = np.matrix(thetav)
    xv = np.matrix(xv)
    yv = np.matrix(yv)

    parameters = int(thetav.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(xv * thetav.T) - yv

    for i in range(parameters):
        term = np.multiply(error, xv[:, i])
        if i == 0:
            grad[i] = np.sum(term) / len(xv)
        else:
            grad[i] = (np.sum(term) / len(xv)) + ((lr / len(xv)) * thetav[0, i])

    return grad


degree = 5
x1 = data["Test 1"]
x2 = data["Test 2"]
print(f"\n X1 = \n {x1.head(10)} \n X2 = {x2.head(10)} \n")
data.insert(3 , "Ones" , 1) #parameter 1 = index , 2 = name  ,  3 = value

for i in range(1 , degree):
    for j in  range(0 , i):
        data['F' + str(i) + str(j)] = np.power(x1 , i-j)*np.power(x2 , j)

# print(f"\n DAta = {data}")

data.drop("Test 1" , axis=1 , inplace=True)
data.drop("Test 2" , axis=1 , inplace=True)
# print(f"\n Data = {data} \n")




cols = data.shape[1]
print(f"shape \n {cols} \n")
x2 = data.iloc[: , 1:cols]
y2 = data.iloc[: ,0:1]
# print(f"DataX = \n{x2.head(10)} \n DAtay = \n{y2.head(10)} \n")

x2 = np.array(x2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(x2.shape[1])
learungRate = 0.00000001


result = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradient, args=(x2, y2, learungRate))

costAfterOpt = costReg(result[0], x2, y2, learungRate)
print(f"Cost After Optimization = {costAfterOpt}")

def predict(theta, x):
    probability = sigmoid(x * theta.T)
    prob_array = np.array(probability).ravel() 
    return [1 if p >= 0.5 else 0 for p in prob_array]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, x2) # تعديل x لـ x2

correct = [1 if a == b else 0 for (a, b) in zip(predictions, y2.ravel())]

accuracy = (sum(correct) / len(correct)) * 100
print(f"accuracy = {accuracy:.2f}%")

