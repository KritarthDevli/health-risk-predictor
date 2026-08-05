import pandas as pd
import json
import os

# Align dataset output path with app.py expectation
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

tasks = [
    {
        "csv_file": "diabetes_data.csv",
        "target": "Diabetes",
        "features": ["Sex", "Age", "Smoker", "PhysActivity", "HighBP", "HighBMI"], 
        "output_json": "diabetes_probabilities.json"
    },
    {
        "csv_file": "hypertension_data.csv",
        "target": "target", 
        "features": ["sex", "age", "fbs", "high_bp", "high_bmi"], 
        "output_json": "hypertension_probabilities.json"
    },
    {
        "csv_file": "stroke_data.csv",
        "target": "stroke", 
        "features": ["sex", "age", "smoking_status", "high_bp", "high_bmi"], 
        "output_json": "stroke_probabilities.json"
    }
]

for task in tasks:
    csv_file = task["csv_file"]
    target = task["target"]
    features = task["features"]
    output_json_path = os.path.join(DATASET_DIR, task["output_json"])
    
    if not os.path.exists(csv_file):
        print(f"Skipping {csv_file}: File not found in current directory.")
        continue

    print(f"Processing {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Feature Engineering for Blood Pressure & BMI if raw columns are present
    if 'systolic' in df.columns and 'diastolic' in df.columns and 'high_bp' not in df.columns:
        df['high_bp'] = ((df['systolic'] >= 130) | (df['diastolic'] >= 80)).astype(int)
    if 'bmi' in df.columns and 'high_bmi' not in df.columns:
        df['high_bmi'] = (df['bmi'] >= 25.0).astype(int)

    total_count = len(df)
    disease_counts = df[target].value_counts()
    has_disease_count = disease_counts.get(1, 0)
    no_disease_count = disease_counts.get(0, 0)
    
    priors = {
        "has_disease": float(has_disease_count / total_count),
        "no_disease": float(no_disease_count / total_count)
    }
    
    conditionals = {}
    for col in features:
        if col not in df.columns:
            continue

        standard_name = col.lower()
        if standard_name == 'fbs': standard_name = 'high_sugar'
        if standard_name == 'smoking_status': standard_name = 'smoker'
        if standard_name == 'highbp': standard_name = 'high_bp'
        if standard_name == 'highbmi': standard_name = 'high_bmi'
        
        conditionals[standard_name] = {}
        unique_traits = df[col].unique()
        
        for trait_value in unique_traits:
            key_name = str(trait_value)
            if key_name in ['Smokes', 'formerly smoked']: key_name = '1'
            if key_name == 'never smoked': key_name = '0'
            
            # Laplace Smoothing: (+1 / +K) prevents zero-probability collapse
            has_d_and_trait = len(df[(df[target] == 1) & (df[col] == trait_value)])
            no_d_and_trait = len(df[(df[target] == 0) & (df[col] == trait_value)])

            p_given_has = (has_d_and_trait + 1) / (has_disease_count + len(unique_traits)) if has_disease_count > 0 else 0.5
            p_given_no = (no_d_and_trait + 1) / (no_disease_count + len(unique_traits)) if no_disease_count > 0 else 0.5

            conditionals[standard_name][key_name] = {
                "given_has_disease": float(p_given_has),
                "given_no_disease": float(p_given_no)
            }
                
    model_data = {
        "target_disease": csv_file.split('_')[0].capitalize(),
        "priors": priors,
        "conditionals": conditionals
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(model_data, f, indent=4)

print("\nUpdated probability models created inside the /dataset directory.")
