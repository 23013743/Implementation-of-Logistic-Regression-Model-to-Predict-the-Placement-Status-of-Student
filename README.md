# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
Import the required packages and print the present data.
Print the placement data and salary data.
Find the null and duplicate values.
Using logistic regression find the predicted values of accuracy , confusion matrices.
Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NIRMAL
RegisterNumber:  212223240107

```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv("placement.csv")
print(data)
```
## Output:
![image](https://github.com/user-attachments/assets/e0911405-4bc4-4e07-9e48-dd82e2f60953)
```
data.head()
```
## Output:
![image](https://github.com/user-attachments/assets/88464ba2-173f-4d51-aba8-6fa53f10b53e)
```
data.tail()
```
## Output:
![image](https://github.com/user-attachments/assets/72178cc8-c1f5-41b8-9da8-f03521cd9d89)
```
data.info()
```
## Output:
![image](https://github.com/user-attachments/assets/d806a719-eb49-4841-b021-5d7fae573953)

```
data.drop('sl_no',axis=1,inplace=True)
data.info()
```
## Output:
![image](https://github.com/user-attachments/assets/008d1c8c-32c8-41b7-8740-9e104910acc6)
```
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes
```
## Output:
![image](https://github.com/user-attachments/assets/e490619c-be2e-497a-a44a-50c98f887ce5)
```
data.info()
```
## Output:
![image](https://github.com/user-attachments/assets/6fc5d3fd-e342-44fd-8e0c-62eeaa48ac8a)
```
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
```
```
data.info()
```
## Output:

![image](https://github.com/user-attachments/assets/10d0df7a-1fd9-4e68-bd7c-f2782f2e8c74)
```
data.head()
```
## Output:

![image](https://github.com/user-attachments/assets/57244eee-3ef0-4d9d-829b-16f180e649e1)
```
x=data.iloc[:, :-1].values
x
```
## Output:

![image](https://github.com/user-attachments/assets/28e97a1a-7129-4df4-b750-e31613086495)
```
y=data.iloc[:, -1].values
y
```
## Output:

![image](https://github.com/user-attachments/assets/a0872cc7-de78-435c-9367-7667dbdaf3ae)
```
x.shape
```
## Output:

![image](https://github.com/user-attachments/assets/a14360ab-d403-47de-9152-e4ac4439e723)
```
y.shape
```
## Output:

![image](https://github.com/user-attachments/assets/cb1c631c-0f5e-4f2f-8c05-f9453efb4bdf)
```

```
## Output:

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
