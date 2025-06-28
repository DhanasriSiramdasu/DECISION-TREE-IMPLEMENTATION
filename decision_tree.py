import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train decision tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("Decision Tree Classifier Results")
print("-------------------------------")
print(f"Accuracy: {accuracy:.2f}")
print("\nFeature Importance:")
for feature, importance in zip(iris.feature_names, dt.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Visualize the tree
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree_plot.png")
plt.close()

# Sample output for submission
print("\nSample Submission Output:")
print("Test Instance | Predicted Class")
for i, pred in enumerate(y_pred[:5]):  # Show first 5 predictions
    print(f"Instance {i+1} | {iris.target_names[pred]}")
