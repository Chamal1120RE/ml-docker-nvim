from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = load_iris()

# Inspecting the data
print("Type of 'data':", type(data))
print("'data' keys:", data.keys())
print("'data'.data shape:", data.data.shape)
print("'data'.target shape:", data.target.shape)

# Step 2: Split the data into features and target
X = data.data  # Features
y = data.target  # Target labels

# Inspecting features and target
print("\nFeatures (X) sample:\n", X[:5])
print("Target (y) sample:", y[:5])
print("Type of X:", type(X))
print("Type of y:", type(y))

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inspecting the split data
print("\nTraining features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Training labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)

# Step 4: Initialize and train the Decision Tree classifier
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = clf.predict(X_test)

# Inspecting predictions
print("\nPredicted labels sample:", y_pred[:5])
print("Actual labels sample:", y_test[:5])

# Step 6: Check accuracy
accuracy = clf.score(X_test, y_test)
print("\nAccuracy of the model:", accuracy)

# Access the Gini index for each node
gini_impurities = clf.tree_.impurity

# Display the Gini index for each node
print("Gini Index (Impurity) for each node in the decision tree:")
print(gini_impurities)

plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
#plt.show()

plt.savefig('/data/decision_tree.png')  # Save to /data directory

print("Plot saved as /data/decision_tree.png")
