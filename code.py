#importing libraries
import pandas as pd
import streamlit as st

#importing dataset
dataset = pd.read_csv("C:\\Users\\srika\OneDrive\\Desktop\\project2\\obesity.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

#feature scaling
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:, 0])

for i in range (4,6):
  X[:, i] = le.fit_transform(X[:, i])

for i in range (8,10):
  X[:, i] = le.fit_transform(X[:, i])

  X[:, 11] = le.fit_transform(X[:, 11])

for i in range (14,16):
  X[:, i] = le.fit_transform(X[:, i])

#print(X)

#spliting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#print(y_train)
#print(y_test)


# Training the model
from sklearn.svm import SVC
classifier_SVC = SVC(kernel = 'linear', random_state = 0)
classifier_SVC.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier_RFc = RandomForestClassifier(n_estimators = 15, random_state = 0)
classifier_RFc.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
classifier_LogisticReg = LogisticRegression(random_state = 0)
classifier_LogisticReg.fit(X_train,y_train)

from sklearn.neighbors import KNeighborsClassifier ##KNN
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB ##naive_bayes
classifier_naive_bayes = GaussianNB()
classifier_naive_bayes.fit(X_train, y_train)

y_pred = classifier_SVC.predict(X_test)
y_pred1 = classifier_RFc.predict(X_test)
y_pred2 = classifier_LogisticReg.predict(X_test)
y_pred3 = classifier_KNN.predict(X_test)
y_pred4 = classifier_naive_bayes.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
cm1 = confusion_matrix(y_test,y_pred1)
cm2 = confusion_matrix(y_test,y_pred2)
cm3 = confusion_matrix(y_test,y_pred3)
cm4 = confusion_matrix(y_test,y_pred4)
print("confusion_matrix of SVC:")
print(cm)
print("confusion_matrix of RandomForestClassifier:")
print(cm1)
print("confusion_matrix of LogisticRegression:")
print(cm2)
print("confusion_matrix of KNeighborsClassifier:")
print(cm3)
print("confusion_matrix of GaussianNB:")
print(cm4)

print("Accuracy of SVC:",accuracy_score(y_test,y_pred))
print("Accuracy of RandomForestClassifier: ",accuracy_score(y_test,y_pred1))
print("Accuracy of LogisticRegression:",accuracy_score(y_test,y_pred2))
print("Accuracy of KNeighborsClassifier:",accuracy_score(y_test,y_pred3))
print("Accuracy of GaussianNB :",accuracy_score(y_test,y_pred4))



st.header("Predict the your weight class")

ques = st.radio("Please select your gender",("Male","Female"))
if ques =="Male":
  ques=1
else:
  ques=0
ques1 = st.slider("please select your age",min_value=5,max_value=90)

ques2 = st.slider("please select you height in meters",min_value=0.5,max_value=6.0,step=0.1)

ques3 = st.slider("Please select your weigth",max_value=20,min_value=350)

ques4 = st.radio("Family member suffered or suffers from overweight",("Yes","No"))

if ques4 =="Yes":
  ques4 = 0
else:
  ques4 =1

ques5 = st.radio("Do you eat high caloric food frequently",("Yes","No"))

if ques5 =="Yes":
  ques5 = 1
else:
  ques5 =0

ques6 = st.slider("Do you usually eat vegetabless in your meals",min_value=0,max_value=5)

ques7 = st.slider("Number of main meals per day",min_value=0,max_value=7)

ques8 = st.radio("Do you eat any food between meals",("Sometimes","Always","No","Frequently"))

if ques8 == "Sometimes":
  ques8 = 0
elif ques8 == "Always":
  ques8 = 2
elif ques8 =="Frequently":
  ques8 = 1
else:
  ques8 = 3

ques9 = st.radio("Do you smoke",("Yes","No"))

if ques9 =="Yes":
  ques9 = 1
else:
  ques9 =0

ques10 = st.slider("How much water do you drink daily",min_value=0,max_value=5)

ques11 = st.radio("Do you monitor the calories you eat daily",("Yes","No"))

if ques11 =="Yes":
  ques11 = 1
else:
  ques11 =0

ques12 = st.slider("How often do you have physical activity",min_value=0,max_value=5)

ques13 = st.slider("How much time do you use technological devices such as cell phone,videogames and others",min_value=0,max_value=5)


ques14 = st.radio("how often do you drink alcohol",("Sometimes","Always","No","Frequently"))

if ques14 == "Sometimes":
  ques14 = 1
elif ques14 == "Always":
  ques14 = 13
elif ques14 =="Frequently":
  ques14 = 2
else:
  ques14 = 0

ques15 = st.radio("which transportation do you usually use",("Public Transport","Walking","Others","Bike"))

if ques15 == "Public Transport":
  ques15 = 0
elif ques15 == "Walking":
  ques15 = 1
elif ques15 =="Bike":
  ques15 = 3
else:
  ques15 = 2

if st.button("Predict your weight class"):
  pred=classifier_RFc.predict([[int(ques),int(ques1),int(ques2),int(ques3),int(ques4),int(ques5),int(ques6),int(ques7),int(ques8),int(ques9),int(ques10),int(ques11),int(ques12),int(ques13),int(ques14),int(ques15)]])
  if pred == 0:
    st.write("Your weight class is Normal_Weight")
  elif pred ==1:
    st.write("Your weight class is Overweight_Level_I")

  elif pred ==2:
    st.write("Your weight class is Overweight_Level_II")

  elif pred ==3:
    st.write("Your weight class is Obesity")

  else:
    st.write("Your weight class is Insufficient_Weight")