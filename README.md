# Implementation-of-Linear-Regression-Using-Gradient-Descent
## NAME: V MYTHILI(CSE)
## REG NO: 212223040123

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1.Import all the required packages.

2.Display the output values using graphical representation tools as scatter plot and graph.

3.predict the values using predict() function.

4.Display the predicted values and end the program

## Program:

NMAE: V MYTHILI(CSE)
REG NO: 212223040123
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:


![image](https://github.com/user-attachments/assets/615f879e-a6ff-4ff4-8eea-19e34b5751f9)


![image](https://github.com/user-attachments/assets/4e2f6398-ee59-4b3b-a25c-75fe5e30243a)



![image](https://github.com/user-attachments/assets/e1f7be6b-e3a0-4d34-993a-4230efb19d8d)



![image](https://github.com/user-attachments/assets/a5741c81-0421-4bc1-99f3-7d3d0a9cda67)


![image](https://github.com/user-attachments/assets/cbe70dd8-6299-4e46-8053-9a9f85212d19)


![image](https://github.com/user-attachments/assets/8e60571a-30d7-4d20-a819-c1c12da5b99b)




![image](https://github.com/user-attachments/assets/040f3aa6-a1fa-4faa-97b6-fe250c154fef)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
