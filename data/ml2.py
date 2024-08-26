from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier with k=3

knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the trainning data

knn.fit(X_train, y_train)

# Predict the labels for the test set

y_pred = knn.predict(X_test)

# Check the accuracy of the KNN model

accuracy_knn = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy_knn:.2f}")
