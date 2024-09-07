#Made By Murtuza Mahudawala
# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the Iris dataset
iris = load_iris()

# Create a DataFrame for easier analysis
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target (species) to the DataFrame
data['species'] = iris.target

# Step 2: Exploratory Data Analysis (EDA)
# Display the first few rows of the data
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Plot pairplot to visualize relationships
sns.pairplot(data, hue='species')
plt.show()

# Step 3: Split the data into features (X) and target (y)
X = data.drop('species', axis=1)
y = data['species']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model building (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=3)

# Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# Classification Report
print('Classification Report:\n', classification_report(y_test, y_pred))

# Step 8: Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 9: Save the trained model
import joblib
joblib.dump(model, 'iris_knn_model.pkl')

print("Model saved as 'iris_knn_model.pkl'")
