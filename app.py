from flask import Flask, render_template, request
import json
import os

app = Flask(__name__)

# Locate the dataset folder where the pre-computed JSON models live
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

# Comprehensive Knowledge Base containing symptoms and precautions for our 3 diseases
DISEASE_KNOWLEDGE_BASE = {
    'Diabetes': {
        'symptoms': [
            "Increased thirst, frequent urination, and constant dry mouth.",
            "Chronic fatigue, lethargy, and unexplained weight changes.",
            "Blurred vision and slow-healing cuts or sores."
        ],
        'precautions': [
            "Monitor daily sugar, glucose, and carbohydrate intake carefully.",
            "Incorporate at least 30 minutes of moderate cardio exercise into your daily routine.",
            "Schedule an HbA1c screening with a general practitioner to evaluate long-term trends."
        ]
    },
    'Hypertension': {
        'symptoms': [
            "Severe morning headaches, neck pain, and occasional dizziness.",
            "Irregular heart rhythms (palpitations) or chest tightness.",
            "Visual disturbances, buzzing noise in ears, or shortness of breath."
        ],
        'precautions': [
            "Significantly reduce dietary sodium (salt) intake across your daily meals.",
            "Incorporate regular stress-management practices such as meditation or light evening walking.",
            "Measure and track your blood pressure weekly using a validated home blood pressure monitor."
        ]
    },
    'Stroke': {
        'symptoms': [
            "Sudden numbness or weakness in the face, arm, or leg (especially on one side of the body).",
            "Sudden confusion, slurred speech, or difficulty understanding verbal statements.",
            "Sudden trouble seeing in one or both eyes, loss of balance, or severe unexplained headache."
        ],
        'precautions': [
            "Avoid nicotine exposure (smoking/vaping) entirely and closely limit alcohol consumption.",
            "Maintain a nutrient-dense dietary tracking plan rich in leafy greens, fiber, and healthy fats.",
            "Consult a clinical physician regarding routine lipid panel checks and arterial blood flow metrics."
        ]
    }
}

def calculate_naive_bayes(user_inputs):
    results = []
    files = {
        'Diabetes': 'diabetes_probabilities.json',
        'Hypertension': 'hypertension_probabilities.json',
        'Stroke': 'stroke_probabilities.json'
    }
    
    for disease, filename in files.items():
        filepath = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(filepath): 
            continue
            
        with open(filepath, 'r') as f:
            model = json.load(f)
            
        # Extract initial prior rates derived from the training dataset
        prior_has = model['priors']['has_disease']
        prior_no = model['priors']['no_disease']
        
        # Start our running multiplication loops with the priors
        p_features_given_has = prior_has
        p_features_given_no = prior_no
        
        # Multiply independent conditional feature scores
        for feature_name, user_value in user_inputs.items():
            if feature_name in model['conditionals']:
                feature_rules = model['conditionals'][feature_name]
                if str(user_value) in feature_rules:
                    p_features_given_has *= feature_rules[str(user_value)]['given_has_disease']
                    p_features_given_no *= feature_rules[str(user_value)]['given_no_disease']
        
        # Execute the normalization phase to get a proper score out of 100%
        total_evidence = p_features_given_has + p_features_given_no
        final_probability = (p_features_given_has / total_evidence) * 100 if total_evidence > 0 else 0.0
            
        results.append({
            'disease': disease,
            'probability': round(final_probability, 2),
            'symptoms': DISEASE_KNOWLEDGE_BASE[disease]['symptoms'],
            'precautions': DISEASE_KNOWLEDGE_BASE[disease]['precautions']
        })
        
    # Sort descending based on probability risk assessment
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

@app.route('/', methods=['GET', 'POST'])
def index():
    error_message = None
    risk_results = None
    
    if request.method == 'POST':
        try:
            # 1. Capture continuous values from form fields and catch conversion issues
            try:
                systolic = int(request.form.get('systolic', 0))
                diastolic = int(request.form.get('diastolic', 0))
                bmi = float(request.form.get('bmi', 0))
            except ValueError:
                raise ValueError("Format error! Please ensure you enter pure numeric digits for blood pressure and BMI fields.")
            
            # 2. Strict Input Physiological Range Verification Checks
            if systolic < 50 or systolic > 250 or diastolic < 30 or diastolic > 150:
                raise ValueError("Please check your entry. Blood pressure readings entered are out of realistic physiological bounds.")
            if bmi < 10 or bmi > 60:
                raise ValueError("Please check your entry. BMI must be a realistic numeric calculation between 10 and 60.")
            
            # 3. Discretize numerical inputs into binary categories matching database expectations
            high_bp_flag = "1" if (systolic >= 130 or diastolic >= 80) else "0"
            high_bmi_flag = "1" if bmi >= 25.0 else "0"
            
            user_inputs = {
                'sex': request.form.get('sex'),
                'age': request.form.get('age'),
                'smoker': request.form.get('smoker'),
                'physactivity': request.form.get('physactivity'),
                'high_sugar': request.form.get('high_sugar'),
                'high_bp': high_bp_flag,
                'high_bmi': high_bmi_flag
            }
            
            # Run the data mapping logic engine
            risk_results = calculate_naive_bayes(user_inputs)
            
        except ValueError as e:
            # Catch mistakes and build an exception pipeline back to front-end layout blocks
            error_message = str(e)
            
        return render_template('index.html', results=risk_results, error=error_message)
        
    return render_template('index.html', results=None, error=None)

if __name__ == '__main__':
    app.run(debug=True)