# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
/*
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHICK RAJ.M
RegisterNumber:  212221040073
```
*/

```
# implement a simple regression model for predicting the marks scored by the students


import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```








 Output:


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/cf7eb0eb-305d-4e13-b2b2-671f3dbb0e87)



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/a3fc88d9-8b53-49ce-b07a-1f843714e465)



![image](https://user-images.githubusercontent.com/128134963/228480373-8445c7b9-b9ce-4716-953c-7edb87d42df4.png)





![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/5aae4940-1e69-47c0-91b8-ab98f77ed94e)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
