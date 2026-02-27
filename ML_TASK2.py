import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv("Dataset.csv")

data.columns = data.columns.str.strip()

print(data.columns)

data = data.dropna()

le_cuisine = LabelEncoder()
le_price = LabelEncoder()

data['Cuisines'] = le_cuisine.fit_transform(data['Cuisines'])
data['Price range'] = le_price.fit_transform(data['Price range'])

features = data[['Cuisines', 'Price range']]

similarity = cosine_similarity(features)

def recommend(cuisine, price):
    try:
        cuisine_val = le_cuisine.transform([cuisine])[0]
    except Exception:
        classes = list(le_cuisine.classes_)
        matches = [c for c in classes if cuisine.lower() in c.lower()]
        if matches:
            cuisine_val = le_cuisine.transform([matches[0]])[0]
        else:
            print("Cuisine not found. Examples:", classes[:10])
            return

    try:
        price_val = le_price.transform([price])[0]
    except Exception:
        try:
            price_num = int(price)
            price_val = le_price.transform([price_num])[0]
        except Exception:
            try:
                classes_price = np.array(list(le_price.classes_)).astype(float)
                low, med, high = int(np.min(classes_price)), int(np.median(classes_price)), int(np.max(classes_price))
                word = str(price).strip().lower()
                mapping = {'low': low, 'medium': med, 'med': med, 'high': high}
                if word in mapping:
                    price_val = le_price.transform([mapping[word]])[0]
                else:
                    print("Price not recognized. Available examples:", list(le_price.classes_))
                    return
            except Exception:
                print("Unable to interpret price input. Available examples:", list(le_price.classes_))
                return

    user_input = [[cuisine_val, price_val]]

    sim_scores = cosine_similarity(user_input, features)

    scores = list(enumerate(sim_scores[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top = scores[:5]

    for idx, score in top:
        row = data.iloc[idx]
        rest_name = row.get('Restaurant Name', 'Unknown')
        try:
            cuisine_orig = le_cuisine.inverse_transform([int(row['Cuisines'])])[0]
        except Exception:
            cuisine_orig = row.get('Cuisines', 'Unknown')
        try:
            price_orig = le_price.inverse_transform([int(row['Price range'])])[0]
        except Exception:
            price_orig = row.get('Price range', 'Unknown')
        print(f"{rest_name} - {cuisine_orig} - {price_orig} (score: {score:.4f})")

if __name__ == '__main__':
    recommend('North Indian', 3)
    print(data.columns)