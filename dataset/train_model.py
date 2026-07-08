import pandas as pd
import json

# Define our 3 distinct datasets, their targets, and relevant lifestyle features
tasks = [
    {
        "csv_file": "diabetes_data.csv",
        "target": "Diabetes",
        "features": ["Sex", "Age", "Smoker", "PhysActivity"], 
        "output_json": "diabetes_probabilities.json"
    },
    {
        "csv_file": "hypertension_data.csv",
        "target": "target", 
        "features": ["sex", "age", "fbs"], # 'fbs' is Fasting Blood Sugar in this dataset
        "output_json": "hypertension_probabilities.json"
    },
    {
        "csv_file": "stroke_data.csv",
        "target": "stroke", 
        "features": ["sex", "age", "smoking_status"], 
        "output_json": "stroke_probabilities.json"
    }
]

for task in tasks:
    csv_file = task["csv_file"]
    target = task["target"]
    features = task["features"]
    output_json = task["output_json"]
    
    print(f"Processing {csv_file}...")
    df = pd.read_csv(csv_file)
    
    total_count = len(df)
    disease_counts = df[target].value_counts()
    has_disease_count = disease_counts.get(1, 0)
    no_disease_count = disease_counts.get(0, 0)
    
    priors = {
        "has_disease": has_disease_count / total_count,
        "no_disease": no_disease_count / total_count
    }
    
    conditionals = {}
    for col in features:
        # Standardize matching key names across all datasets for our web front-end
        standard_name = col.lower()
        if standard_name == 'fbs': standard_name = 'high_sugar'
        if standard_name == 'smoking_status': standard_name = 'smoker'
        
        conditionals[standard_name] = {}
        
        for trait_value in df[col].unique():
            key_name = str(trait_value)
            # Map different string variants from the stroke data to binary strings
            if key_name in ['Smokes', 'formerly smoked']: key_name = '1'
            if key_name == 'never smoked': key_name = '0'
            
            conditionals[standard_name][key_name] = {}
            
            # P(Feature | Has Disease)
            if has_disease_count > 0:
                has_d_and_trait = len(df[(df[target] == 1) & (df[col] == trait_value)])
                conditionals[standard_name][key_name]["given_has_disease"] = has_d_and_trait / has_disease_count
            else:
                conditionals[standard_name][key_name]["given_has_disease"] = 0
            
            # P(Feature | No Disease)
            if no_disease_count > 0:
                no_d_and_trait = len(df[(df[target] == 0) & (df[col] == trait_value)])
                conditionals[standard_name][key_name]["given_no_disease"] = no_d_and_trait / no_disease_count
            else:
                conditionals[standard_name][key_name]["given_no_disease"] = 0
                
    model_data = {
        "target_disease": csv_file.split('_')[0].capitalize(),
        "priors": priors,
        "conditionals": conditionals
    }
    
    with open(output_json, 'w') as f:
        json.dump(model_data, f, indent=4)

print("\nAll done! Updated probability models successfully created.")