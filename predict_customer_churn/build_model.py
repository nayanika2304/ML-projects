import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("churn.csv")

#sns.countplot(x='Churn', data=df)
#plt.show()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

numRetained = df[df.Churn == 'No'].shape[0]
numChurned = df[df.Churn == 'Yes'].shape[0]

# print the percentage of customers that stayed
print(numRetained/(numRetained + numChurned) * 100,'% of customers stayed in the company')
# peint the percentage of customers that left
print(numChurned/(numRetained + numChurned) * 100, '% of customers left with the company')

#sns.countplot(x ='gender', hue='Churn', data=df)
sns.countplot(x='InternetService', hue='Churn', data=df)
#plt.show()

numericFeatures = ['tenure', 'MonthlyCharges']
fig, ax = plt.subplots(1,2, figsize=(28, 8))
for i, feature in enumerate(numericFeatures):
    no_churn_data = df[df.Churn == "No"][feature].values
    churn_data = df[df.Churn == "Yes"][feature].values
    ax[i].hist(no_churn_data, bins=20, color='blue', alpha=0.5, label='No Churn')
    ax[i].hist(churn_data, bins=20, color='orange', alpha=0.5, label='Churn')
    ax[i].set_title(feature)
    ax[i].legend()

cleanDF = df.drop('customerID', axis=1)

# Handle missing values - drop rows with NaN
cleanDF = cleanDF.dropna()

#Convert all the non-numeric columns to numeric
for column in cleanDF.columns:
  if cleanDF[column].dtype == np.number:
    continue
  cleanDF[column] = LabelEncoder().fit_transform(cleanDF[column])

print(cleanDF.dtypes)

#Scaled the data
x = cleanDF.drop('Churn', axis=1)
y = cleanDF['Churn']
x = StandardScaler().fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=42)

model = LogisticRegression()
# Train the model
model.fit(xtrain, ytrain)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

predictions = model.predict(xtest)

# print the predictions
print(predictions)

print(classification_report(ytest, predictions))