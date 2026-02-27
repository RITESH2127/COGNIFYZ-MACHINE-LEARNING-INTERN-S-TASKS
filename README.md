

Machine Learning Internship - Cognifyz Technologies
Duration: February

This repository contains the Machine Learning tasks and projects completed during my internship at Cognifyz Technologies. The projects primarily focus on applying data preprocessing, recommendation systems, regression, and classification techniques to a real-world restaurant dataset.

📂 Project Structure
Dataset.csv: The primary restaurant dataset used across all tasks.

ML_TASK2.py: A Content-Based Restaurant Recommendation System.

ML_TASK3.py: A Regression model to predict restaurant ratings.

ML_TASK4.py: A Classification model to predict a restaurant's primary cuisine.

M2.png, M3.png, M4.png: Output screenshots and visual results for Tasks 2, 3, and 4 respectively.

🚀 Tasks Overview
Task 2: Restaurant Recommendation System (ML_TASK2.py)
Built a recommendation engine that suggests restaurants based on user preferences for Cuisine and Price Range.

Technique Used: Content-Based Filtering using Cosine Similarity.

Workflow: The model encodes categorical features (Cuisines, Price range) using LabelEncoder and computes the similarity scores to return the top 5 closest restaurant matches for the user's input.

Task 3: Rating Prediction Model (ML_TASK3.py)
Developed a machine learning model to predict the Aggregate Rating of a restaurant based on various features like location, cost, and availability of online booking/delivery.

Algorithm Used: RandomForestRegressor

Features Used: Longitude, Latitude, Average Cost for two, Has Table booking, Has Online delivery, Price range, Votes, etc.

Evaluation Metrics: Mean Squared Error (MSE) and R-squared (R 
2
 ) Score.

Task 4: Cuisine Classification Model (ML_TASK4.py)
Created a classification model to predict the Primary Cuisine of a restaurant based on its attributes.

Algorithm Used: RandomForestClassifier (with balanced class weights).

Workflow: Extracted the primary cuisine from a comma-separated list, filtered the dataset to only include the Top 10 most popular cuisines, and trained the classifier on geographical and operational features.

Evaluation Metrics: Accuracy Score and Classification Report (Precision, Recall, F1-Score).

🛠️ Technologies & Libraries Used
Python 3.x

Pandas: Data manipulation and cleaning.

NumPy: Numerical operations.

Scikit-Learn: Machine learning algorithms (RandomForestRegressor, RandomForestClassifier), preprocessing (LabelEncoder), and evaluation metrics.

⚙️ How to Run
Clone this repository to your local machine:

Bash
git clone <your-repository-url>
Ensure you have the required libraries installed:

Bash
pip install pandas numpy scikit-learn
Run the Python scripts individually from your terminal:

Bash
python ML_TASK2.py
python ML_TASK3.py
python ML_TASK4.py
