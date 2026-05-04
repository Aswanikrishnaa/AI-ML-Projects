import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# simple dataset
data = {
    'YearsExperience': [1,2,3,4,5,6,7,8,9,10],
    'Salary': [20000,25000,30000,35000,40000,50000,55000,60000,65000,70000]
}

df = pd.DataFrame(data)

X = df[['YearsExperience']]
y = df['Salary']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "model.pkl")
print("Model trained successfully")