import pandas as pd
import numpy as np
import os

def generate_housing_dataset(n_samples=1500, random_seed=42):
    np.random.seed(random_seed)
    
    # Generate feature distributions
    sqft = np.random.normal(loc=2000, scale=700, size=n_samples).astype(int)
    sqft = np.clip(sqft, 600, 5500)
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.08, 0.25, 0.42, 0.18, 0.05, 0.02])
    bathrooms = np.clip((bedrooms * 0.75 + np.random.choice([-0.5, 0, 0.5, 1.0], size=n_samples)), 1.0, 5.0)
    
    age_years = np.random.randint(0, 45, size=n_samples)
    garage_cars = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.20, 0.40, 0.32, 0.08])
    
    locations = np.random.choice(['Suburbs', 'Downtown', 'Rural', 'Prime Coastal'], size=n_samples, p=[0.40, 0.25, 0.20, 0.15])
    
    overall_condition = np.random.choice(range(1, 11), size=n_samples, p=[0.02, 0.03, 0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.03, 0.02])
    
    # Location multipliers & bonuses in Indian Rupees (INR ₹)
    loc_multipliers = {
        'Rural': 0.85,
        'Suburbs': 1.15,
        'Downtown': 1.65,
        'Prime Coastal': 2.20
    }
    
    loc_base_bonus = {
        'Rural': 0,
        'Suburbs': 1200000,
        'Downtown': 4500000,
        'Prime Coastal': 8500000
    }
    
    # Price formula in Indian Rupees (₹)
    # Base rate ~ ₹4,500/sqft adjusted by location
    price = (
        (sqft * 4800 * np.array([loc_multipliers[loc] for loc in locations])) +
        (bedrooms * 550000) +
        (bathrooms * 450000) +
        (garage_cars * 350000) +
        (overall_condition * 300000) -
        (age_years * 45000) +
        np.array([loc_base_bonus[loc] for loc in locations])
    )
    
    # Market noise variance (₹3,00,000 std dev)
    noise = np.random.normal(loc=0, scale=350000, size=n_samples)
    price = price + noise
    
    # Ensure realistic positive price rounded to nearest 50,000
    price = np.round(np.clip(price, 2500000, 95000000), -4)
    
    df = pd.DataFrame({
        'Square_Feet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age_Years': age_years,
        'Garage_Cars': garage_cars,
        'Location': locations,
        'Overall_Condition': overall_condition,
        'Price': price
    })
    
    output_path = 'housing_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created in Rupees (INR ₹) and saved to {output_path}.")
    return df

if __name__ == '__main__':
    generate_housing_dataset()
