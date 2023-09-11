from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target variable (iris species)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(sns.load_dataset("iris"), hue="species")
plt.show()

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors as needed.

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


new_data = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your own values
predicted_species = model.predict(new_data)
print(f"Predicted species: {iris.target_names[predicted_species][0]}")
