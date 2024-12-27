import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load and preprocess the dataset
data = pd.read_csv("Cleaned_Students_Performance.csv")

# Ensure 'average_score' column exists
if 'average_score' not in data.columns:
    raise KeyError("Column 'average_score' not found!")

# Create the target variable
def categorize_performance(score):
    if score < 50:
        return 'Low'
    elif 50 <= score < 75:
        return 'Medium'
    else:
        return 'High'

data['performance_level'] = data['average_score'].apply(categorize_performance)

# Prepare the features (X) and target (y)
X = data.drop(columns=['performance_level'])
X = pd.get_dummies(X, drop_first=True)
y = data['performance_level']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a list of models
models = {
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=4000),
    "SVM": SVC(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Neural Network": MLPClassifier(random_state=42, max_iter=2000)
}

# Train and evaluate models
results = {}
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)*100
    results[name] = accuracy

    print(f"{name} Performance:")
    print("Accuracy Score:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 50)

    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# Print model comparison
print("\nModel Comparison:")
for name, acc in results.items():
    print(f"{name}: Accuracy = {acc:.4f}")

print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save the best model and features
model_data = {
    "model": best_model,
    "features": list(X_train.columns)
}

with open("model.pkl", "wb") as file:
    pickle.dump(model_data, file)

print("Best model saved successfully.")
