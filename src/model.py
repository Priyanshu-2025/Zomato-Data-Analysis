import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(df: pd.DataFrame):
    """Train ML model to predict ratings"""
    # Features (you can adjust based on dataset)
    X = df[['cost', 'votes']]  
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, preds))
    print("R2 Score:", r2_score(y_test, preds))

    joblib.dump(model, "models/zomato_rating_model.pkl")
    print("✅ Model saved at models/zomato_rating_model.pkl")
