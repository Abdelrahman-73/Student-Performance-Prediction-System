import pandas as pd
from flask import Flask, request, render_template, send_file
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Create Flask app
flask_app = Flask(__name__)

# Load the model and feature list
with open("model.pkl", "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    expected_features = model_data["features"]


@flask_app.route("/")
def home():
    return render_template("index.html")


@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values
        input_data = {key: request.form[key] for key in request.form.keys()}
        input_df = pd.DataFrame([input_data])

        # Cast numerical fields to integers
        numeric_fields = ["math_score", "reading_score", "writing_score"]
        input_df[numeric_fields] = input_df[numeric_fields].astype(int)

        # Add calculated fields
        input_df['total_score'] = (
                input_df['math_score'] + input_df['reading_score'] + input_df['writing_score']
        )
        input_df['average_score'] = input_df['total_score'] / 3

        # Perform one-hot encoding and align features
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

        # Make prediction
        prediction = model.predict(input_df)
        performance_level = prediction[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Performance Level: {performance_level}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error during prediction: {e}"
        )


# Visualization routes
@flask_app.route("/graph-gender")
def graph_gender():
    data = pd.read_csv("Cleaned_Students_Performance.csv")

    # Aggregate data by gender
    avg_scores_by_gender = data.groupby('gender')['average_score'].mean().reset_index()

    # Map gender values for better readability
    avg_scores_by_gender['gender'] = avg_scores_by_gender['gender'].map({0: 'Female', 1: 'Male'})

    # Create the bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_scores_by_gender, x='gender', y='average_score', palette='coolwarm')
    plt.title("Average Score by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Average Score")

    # Save and return the plot
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


@flask_app.route("/graph-education")
def graph_education():
    data = pd.read_csv("Cleaned_Students_Performance.csv")

    # Create a boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='parental_level_of_education', y='average_score', palette='viridis')
    plt.title("Score Distribution by Parental Education Level")
    plt.xticks(rotation=45)
    plt.xlabel("Parental Education Level")
    plt.ylabel("Average Score")

    # Save and return the plot
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


@flask_app.route("/graph-correlation")
def graph_correlation():
    data = pd.read_csv("Cleaned_Students_Performance.csv")

    # Select relevant columns
    scores = data[['math_score', 'reading_score', 'writing_score']]

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(scores.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap of Scores")

    # Save and return the plot
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


@flask_app.route("/graph-ethnicity")
def graph_ethnicity():
    data = pd.read_csv("Cleaned_Students_Performance.csv")

    # Aggregate data by race/ethnicity
    avg_scores_by_ethnicity = data.groupby('race_ethnicity')['average_score'].mean().reset_index()

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_scores_by_ethnicity, x='race_ethnicity', y='average_score', palette='cubehelix')
    plt.title("Average Score by Race/Ethnicity")
    plt.xlabel("Race/Ethnicity")
    plt.ylabel("Average Score")

    # Save and return the plot
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


@flask_app.route("/graph-test-preparation")
def graph_test_preparation():
    data = pd.read_csv("Cleaned_Students_Performance.csv")

    # Create a violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x='test_preparation_course', y='average_score', palette='muted')
    plt.title("Score Distribution by Test Preparation")
    plt.xlabel("Test Preparation Course (0 = None, 1 = Completed)")
    plt.ylabel("Average Score")

    # Save and return the plot
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')


if __name__ == "__main__":
    flask_app.run(debug=True)

