'''project id-#CC69856
project id-predicting employee attrition
internship domain-data science intern
project level-intermediate level
assigned by-CodeClause Internship
assigned to-Mevania Alexander
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#loading the dataset
file_path='C:/Users/mevan/Desktop/codeclause/CodeClauseInternship/project_prediction/hr_data.csv'
data=pd.read_csv(file_path)

#previewing the data
print("Columns in dataset:",data.columns)
print("\nMissing values per column:\n",data.isnull().sum())

#handling missing data
data['last_new_job'].fillna(data['last_new_job'].mode()[0],inplace=True)

#encoding
label_encoder=LabelEncoder()
data['last_new_job']=label_encoder.fit_transform(data['last_new_job'])

#defining target variable
target_column='training_hours'
if target_column not in data.columns:
    raise ValueError(f"The dataset must have a '{target_column}' column indicating employee attrition.")

X=data[['last_new_job']]
y=data[target_column]

#splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#training the dataset
model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

#predicting
y_pred=model.predict(X_test)

#evaluating the model
print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))

#saving the model
model_path='employee_attrition_last_new_job_model.pkl'
joblib.dump(model,model_path)
print(f"\nModel saved as '{model_path}'")
loaded_model=joblib.load(model_path)
new_data=pd.DataFrame({'last_new_job':['1']}) 
new_data['last_new_job']=label_encoder.transform(new_data['last_new_job'])
attrition_prediction=loaded_model.predict(new_data)
print("\nAttrition Prediction for New Data:\n",attrition_prediction)

