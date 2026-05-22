import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "/home/khaled/Documents/python/Logostic Reggression/Logistic regression/train.txt"
data = pd.read_csv(path , header=None , names=["Exam 1" , "Exam 2" , "Admitted"])
# print(data.head(10))
# print(f"\n {data.describe()}\n ")







positive = data[data["Admitted"].isin([1])]
negative = data[data["Admitted"].isin([0])]
#separate 0 and 1 
# print(f"\nAdmitted students \n {positive} \n")
# print(f"\nNonAdmitted students \n {negative} \n")
# fig , ax = plt.subplots(figsize=(8,5)) #fig = framework , ax = x , y axiss
"""
positive["Exam 1"] = X axis 
positive["Exam 2"] = Y axis
s=50 =size=50 = size of point
c='b' = color = 'blue' = point color
label = "Admitted" = name of figuer
"""
# ax.scatter(positive["Exam 1"] , positive["Exam 2"] , s = 50 , c  = 'b' , label="Admitted")
# ax.scatter(negative["Exam 1"]  , negative["Exam 2"] , s = 50 , color='r' , marker='x' , label="Not Admitted")
# ax.legend()
# ax.set_xlabel("Exam 1 Score")
# ax.set_ylabel("Exam 2 score")
# plt.show();






def sigmaid(z):
    return 1 / (1 + np.exp(-z))
# nums = np.arange(-10 , 10 , step=1)
# fig2 , ax2 = plt.subplots(figsize=(8,5))
# ax2.plot(nums , sigmaid(nums) , 'r')
# plt.show()







data.insert(0 , "Ones" , 1)
# print(f"\n {data}")
cols = data.shape[1]
x = data.iloc[: , 0:cols-1]
y = data.iloc[: , cols-1 : cols]
# print(f"\n X = \n{x} \n Y = \n {y}")
x =  np.array(x.values)
y =  np.array(y.values)
theta = np.zeros(3)




def cost(thetav  , xv, yv):
    thetav = np.matrix(thetav)
    xv  = np .matrix(xv)
    yv = np.matrix(yv)
    first = np.multiply(-yv , np.log(sigmaid(xv * thetav.T)))
    second = np.multiply((1-yv) , np.log(1-sigmaid(xv*thetav.T)))
    return np.sum(first - second) / len(xv)
thiscost = cost(theta , x , y)
print(f"\n {thiscost} \n")






def gradient(thetav  , xv ,yv):
    thetav = np.matrix(thetav)
    xv = np.matrix(xv)
    yv = np.matrix(yv)

    parameters = int(thetav.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmaid(xv * thetav.T) - yv

    for i in range(parameters):
        term = np.multiply(error,xv[: , i])
        grad[i] = np.sum(term) / len(xv)

    return grad



import scipy.optimize as  opt
result = opt.fmin_tnc(func=cost , x0 = theta , fprime=gradient , args=(x , y))
# print(f"\n result = \n{result} \n")

costAfterOpt = cost(result[0] , x ,y)
# print(f"cost after optimize \n {costAfterOpt}\n")


def predict(theta , x):
    probability = sigmaid(x * theta.T)
    return[1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min , x)
# print(f"new predict = {predictions} \n")
correct = [1 if (a == 1 and b == 1 ) or (b == 0 and b ==0 ) else 0 for (a,b) in zip(predictions , y)]
# print(f"\n {correct}")
accuracy = (sum(map(int , correct)) % len(correct))
# print(f"accuracy = {accuracy}")



