# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2.Assign hours to X and scores to Y.
3.Implement training set and test set of the dataframe
4.Plot the required graph both for test data and training data.
5.Find the values of MSE , MAE and RMSE.


## Program:
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PREETHA.S
RegisterNumber:212222230110

# implement a simple regression model for predicting the marks scored by the students
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
#Segregating data to variables
X = df.iloc[:,:-1].values
X 
Y = df.iloc[:,1].values
Y
#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted values
Y_pred
#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_train,regressor.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE =',mse)
mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


## Output:
Dataset
![Screenshot 2023-08-24 103542](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/27b712ea-42a7-4d81-9b48-946e152225e6)

Head value
![Screenshot 2023-08-24 103730](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/1c3a1f43-bca0-4f3f-a01e-0753a7d0a3f6)

Tail value
![Screenshot 2023-08-24 103956](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/f6552697-2f12-4521-82d8-44815f0c3a5d)

Array value of X
![Screenshot 2023-08-24 104129](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/4c2ecf90-82b0-47ea-bece-404e6aeed8c2)

Array value of Y
![Screenshot 2023-08-24 104243](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/a30d084e-80c3-4ce9-b7b8-2c81678dae32)

Values of Y predicted
![Screenshot 2023-08-24 112902](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/f9300620-fa7c-437e-99e1-891826517643)

Predicted value of Y
![Screenshot 2023-08-24 112902](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/1723581d-aed1-4424-ac4b-2975a8b93ff9)

Training set
![Screenshot 2023-08-24 113344](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/6ca5ddea-5894-4983-ab3d-e3c1dec0825f)

Testing set
![Screenshot 2023-08-24 113413](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/ac4131b5-dac3-46e6-9746-2f382750c97f)

MSE, MAE and RMSE
![Screenshot 2023-08-24 113429](https://github.com/Preetha-Senthamilan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390282/815a1e66-bac1-4882-9bc2-eab1700facfd)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
