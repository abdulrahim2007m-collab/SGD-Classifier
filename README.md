# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Standardization: The features are scaled so they all have a similar range. This helps the optimizer converge much faster and prevents one feature from dominating the others.
2. Data Splitting: The data is divided into a Training Set (to teach the model) and a Test Set (to verify if the model actually learned or just memorized).
3. Stochastic Gradient Descent (SGD): Instead of looking at the entire dataset at once, the model updates its internal weights by looking at one random data point at a time. This makes it very efficient for large datasets.
4. Weight Optimization: During the "fit" phase, the model iteratively adjusts its parameters to minimize the difference between its predictions and the actual labels (the loss).
5. Evaluation: The model's final performance is measured by comparing its predictions on the test set against the real answers, resulting in an Accuracy score.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Abdul Rahim M
RegisterNumber:  212225040007
*/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SGDClassifier(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:

<img width="493" height="228" alt="image" src="https://github.com/user-attachments/assets/48c13bc4-be4f-4ac3-918e-fcd819b921c2" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
