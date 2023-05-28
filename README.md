# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
6.Obtain the graph.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHICK RAJ.M
RegisterNumber:  212221040073
*/
```



```

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
print("RMSE = ",rmse



```






 Output:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127935950/5a006a3c-eef7-4091-9a88-b917abdcf3dd)



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127935950/92ae86df-2cf6-49b4-88e8-2e349c1b0303)


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127935950/363ebd6e-04af-4921-91fc-d767cff8dcf0)



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127935950/0da682c7-7eb1-4230-b01f-99b5893e4d22)


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127935950/d2191303-4bdf-45bb-a025-9a5a2c220150)


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127935950/c4069e74-f883-4d3d-9573-bb8189faef0d)



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127935950/468993a1-e50d-4f10-931e-7c751b28895c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
