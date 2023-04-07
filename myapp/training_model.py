import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_excel('Dset.xlsx')

# input("Press enter to view information about the dataset")
# learn the dataset
# print(df.info())

# # View the dataset

# input("Press enter to view the dataset")

# print(df.to_string())

# # Cleaning the dataset

# # fill the empty cells
# input("Press enter to fill empty values with 11")

df['GRADE'].fillna(11,inplace=True)

# Delete duplicates

# input("Press enter to check for duplicates")
# print(df.duplicated().to_string())

# # Training Models

# # registering classifiers

dtc = DecisionTreeClassifier()
lrg = LogisticRegression(solver='lbfgs',max_iter=10000)
svm_classifier = svm.SVC(kernel='linear')
rfc = RandomForestClassifier(n_estimators=100)

#  Split dataset into training set and test set

x = df.drop(columns=['COMBINATION','OBSERVATION'])
y = df['OBSERVATION']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2)

# # Training the models
input("Press enter to train the models")
dtc.fit(x_train,y_train)
lrg.fit(x_train,y_train)
svm_classifier.fit(x_train,y_train)
rfc.fit(x_train,y_train)

# create predictions (Test)

dtc_prediction = dtc.predict(x_test)
lrg_prediction = lrg.predict(x_test)
svm_prediction = svm_classifier.predict(x_test)
rfc_prediction = rfc.predict(x_test)


# determining the most accurate model

dtc_accuracy = accuracy_score(dtc_prediction,y_test)
lrg_accuracy = accuracy_score(lrg_prediction,y_test)
svm_accuracy = accuracy_score(svm_prediction,y_test)
rfc_accuracy = accuracy_score(rfc_prediction,y_test)

print("dtc",dtc_accuracy*100)
print("lrg",lrg_accuracy*100)
print("svm",svm_accuracy*100)
print("rfc",rfc_accuracy*100)

joblib.dump(rfc,'../recomendation-system.joblib')


# model = joblib.load('../recomendation-system.joblib')

# prediction = model.predict([[3,67]])

# print(prediction)
