import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("Loading dataset...")
    df = pd.read_csv("Dataset.csv")

    print("Preprocessing data...")
    df = df.dropna(subset=['Cuisines'])

    features_to_use = [
        'Country Code',
        'Longitude',
        'Latitude',
        'Average Cost for two',
        'Has Table booking',
        'Has Online delivery',
        'Is delivering now',
        'Switch to order menu',
        'Price range',
        'Votes'
    ]

    X = df[features_to_use].copy()
    y = df['Aggregate rating']

    categorical_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
    le = LabelEncoder()
    for col in categorical_cols:
         X[col] = le.fit_transform(X[col])

    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Predicting and Evaluatng...")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

    print("\n--- Feature Importances ---")
    importances = model.feature_importances_
    features = X.columns
    
    indices = np.argsort(importances)[::-1]
    
    for i in range(len(indices)):
        print(f"{i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

if __name__ == '__main__':
    main()
