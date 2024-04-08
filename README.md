# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results.
## Program:
```

## Developed by: NIRMAL N
## RegisterNumber: 212223240107

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:
# Placement Data:
![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/76e34352-d9ab-4d37-940e-d125aada36cd)
# Salary Data:

![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/170ae044-059b-4d35-9e55-19931da157ea)
# Checking the null() function:

![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/e47f91e3-32e0-4436-b6ce-7fde27119528)

# Data Duplicate:

![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/c727857b-44d2-4c70-88c5-3fd7dfbbdce6)

# Print Data:

![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/4ef40fa2-fe30-4929-aa65-45ebbaba4f8d)

#Data-Status:
![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/27765dfd-60da-4902-a27d-fd06a5f6715c)

#Y_prediction array:
![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/a716653c-d99b-4c1e-a5b3-b408cb0e0ff3)

# Accuracy value:
![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/5b10d318-562e-4c0b-afd7-d37a953af30c)

# Confusion array:
![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/2e93af08-7283-4022-bbe2-605c3032ac13)

# Classification Report:
![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/fe219d0d-5b58-45bc-a112-c154bab53c92)
# Prediction of LR:
![image](https://github.com/23013743/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/161271714/f851da21-8de9-4d3e-afe7-7bd7221146de)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
