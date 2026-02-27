import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("Loading dataset...")
    df = pd.read_csv("Dataset.csv")

    print("Preprocessing data for classification...")
    df = df.dropna(subset=['Cuisines'])
    df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: str(x).split(',')[0].strip())

    top_10_cuisines = df['Primary Cuisine'].value_counts().nlargest(10).index
    print(f"\nFiltering for the Top 10 primary cuisines:\n{list(top_10_cuisines)}")
    
    df_filtered = df[df['Primary Cuisine'].isin(top_10_cuisines)]

    print(f"Data reduced from {len(df)} to {len(df_filtered)} rows representing the top 10 cuisines.")

    features_to_use = [
        'Country Code', 
        'Longitude', 
        'Latitude', 
        'Average Cost for two', 
        'Has Table booking', 
        'Has Online delivery', 
        'Price range', 
        'Aggregate rating', 
        'Votes'
    ]

    X = df_filtered[features_to_use].copy()
    y = df_filtered['Primary Cuisine']

    categorical_cols = ['Has Table booking', 'Has Online delivery']
    le_features = LabelEncoder()
    for col in categorical_cols:
         X[col] = le_features.fit_transform(X[col])

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Predicting and Evaluating...\n")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- Overall Accuracy ---\n{accuracy * 100:.2f}%\n")

    print("--- Detailed Classification Report ---")
    report = classification_report(y_test, y_pred, target_names=le_target.classes_)
    print(report)

if __name__ == '__main__':
    main()