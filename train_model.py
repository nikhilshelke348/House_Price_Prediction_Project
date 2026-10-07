import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_and_evaluate():
    data_path = 'housing_data.csv'
    if not os.path.exists(data_path):
        from generate_data import generate_housing_dataset
        df = generate_housing_dataset()
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded dataset shape: {df.shape}")
    
    # Feature columns and Target variable
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Preprocessing setup
    numeric_features = ['Square_Feet', 'Bedrooms', 'Bathrooms', 'Age_Years', 'Garage_Cars', 'Overall_Condition']
    categorical_features = ['Location']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_features)
        ]
    )

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionary of models to train and compare
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    best_pipeline = None

    print("\n--- Training and Evaluating ML Models (Prices in INR Rs.) ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results[name] = {
            'R2': round(r2, 4),
            'MAE': round(mae, 2),
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2)
        }
        
        print(f"\nModel: {name}")
        print(f"  R^2 Score: {results[name]['R2']}")
        print(f"  MAE      : Rs. {results[name]['MAE']:,.2f}")
        print(f"  RMSE     : Rs. {results[name]['RMSE']:,.2f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n*** Best Model Selected: {best_model_name} with R^2 = {best_r2:.4f} ***")

    # Extract Feature Names after Encoding
    ohe_categories = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_categories)

    # Feature Importance
    regressor = best_pipeline.named_steps['regressor']
    feature_importance = None
    if hasattr(regressor, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': regressor.feature_importances_
        }).sort_values(by='Importance', ascending=False).to_dict(orient='records')
    elif hasattr(regressor, 'coef_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(regressor.coef_)
        }).sort_values(by='Importance', ascending=False).to_dict(orient='records')

    # Save artifact pipeline & metadata dictionary
    model_bundle = {
        'model_pipeline': best_pipeline,
        'best_model_name': best_model_name,
        'metrics': results,
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }

    model_file = 'house_price_model.pkl'
    joblib.dump(model_bundle, model_file)
    print(f"Model bundle successfully saved to '{model_file}'.")

if __name__ == '__main__':
    train_and_evaluate()
