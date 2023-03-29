# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHICK RAJ.M
RegisterNumber:  212221040073
*/

# implement a simple regression model for predicting the marks scored by the students

import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/








 Output:


![image](https://user-images.githubusercontent.com/128134963/228480273-53cb6549-4654-4ad5-910f-de5017b4a394.png)


![image](https://user-images.githubusercontent.com/128134963/228480342-6b3609f0-0817-4059-b6cf-bc5385c095b7.png)


![image](https://user-images.githubusercontent.com/128134963/228480373-8445c7b9-b9ce-4716-953c-7edb87d42df4.png)





![image](https://user-images.githubusercontent.com/128134963/228480429-a0108939-aa76-4562-9f7d-4a9a0d4a4261.png)



![image](https://user-images.githubusercontent.com/128134963/228480489-946ec4d6-a298-4664-a84c-22c4429eb074.png)



![image](https://user-images.githubusercontent.com/128134963/228480522-08024e59-869f-404c-80d1-02a28f8dbb1d.png)



![image](https://user-images.githubusercontent.com/128134963/228480562-c0294653-64f5-4252-8914-dadd9558884f.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
