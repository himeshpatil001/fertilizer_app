from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Check files in directory
print("🔍 Current directory:", os.getcwd())
print("🔍 Files in directory:", [f for f in os.listdir('.') if f.endswith('.pkl')])

# Load the trained model
try:
    with open('fertilizer_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
    print(f"✅ Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Try multiple methods to load scaler
scaler = None
scaler_loading_methods = [
    ("Standard pickle", lambda: pickle.load(open('scaler.pkl', 'rb'))),
    ("Joblib", lambda: joblib.load('scaler.pkl')),
    ("Pickle with latin1", lambda: pickle.load(open('scaler.pkl', 'rb'), encoding='latin1')),
    ("Pickle with bytes", lambda: pickle.load(open('scaler.pkl', 'rb'), encoding='bytes')),
]

for method_name, loader in scaler_loading_methods:
    try:
        loaded_object = loader()
        print(f"✅ Scaler loaded successfully using {method_name}!")
        print(f"✅ Scaler type: {type(loaded_object)}")

        # Check if it's a proper scaler object
        if hasattr(loaded_object, 'transform') and hasattr(loaded_object, 'fit'):
            scaler = loaded_object
            print("✅ Loaded object is a proper scaler!")
            break
        elif isinstance(loaded_object, np.ndarray):
            print("⚠️ Loaded object is a numpy array, not a scaler - will create fallback")
            scaler = None
            break
        else:
            print(f"⚠️ Loaded object is {type(loaded_object)} - not a usable scaler")
            continue

    except Exception as e:
        print(f"❌ {method_name} failed: {e}")

# If scaler still can't be loaded, create a reasonable fallback based on typical agricultural data
if scaler is None:
    print("⚠️ Creating fallback scaler based on typical agricultural data ranges...")
    scaler = StandardScaler()

    # Typical ranges for agricultural data (based on common datasets)
    # Order: Temperature, Moisture, Rainfall, PH, Nitrogen, Phosphorous, Potassium, Carbon, Crop
    typical_agricultural_data = np.array([
        [15.0, 20.0, 20.0, 4.5, 10.0, 5.0, 10.0, 0.5, 0],  # Minimum typical values
        [20.0, 40.0, 50.0, 5.5, 30.0, 25.0, 35.0, 1.5, 10],  # Low values
        [25.0, 60.0, 100.0, 6.5, 50.0, 50.0, 50.0, 2.5, 15],  # Medium values
        [30.0, 80.0, 200.0, 7.5, 80.0, 80.0, 80.0, 3.5, 20],  # High values
        [35.0, 90.0, 250.0, 8.0, 100.0, 100.0, 100.0, 4.0, 25],  # Higher values
        [40.0, 100.0, 300.0, 8.5, 140.0, 120.0, 130.0, 5.0, 30],  # Maximum typical values
    ])

    scaler.fit(typical_agricultural_data)
    print("✅ Fallback scaler created with typical agricultural data ranges")

    # Save the fallback scaler for future use
    try:
        joblib.dump(scaler, 'scaler_fallback.pkl')
        print("✅ Fallback scaler saved as 'scaler_fallback.pkl'")
    except Exception as e:
        print(f"⚠️ Could not save fallback scaler: {e}")

# Fertilizer mappings
FERTILIZER_MAPPING = {
    0: 'Balanced NPK Fertilizer',
    1: 'Compost',
    2: 'DAP',
    3: 'General Purpose Fertilizer',
    4: 'Gypsum',
    5: 'Lime',
    6: 'Muriate of Potash',
    7: 'Organic Fertilizer',
    8: 'Urea',
    9: 'Water Retaining Fertilizer'
}

# Crop mappings
CROP_MAPPING = {
    'Adzuki Beans': 0, 'Black gram': 1, 'Chickpea': 2, 'Coconut': 3, 'Coffee': 4,
    'Cotton': 5, 'Ground Nut': 6, 'Jute': 7, 'Kidney Beans': 8, 'Lentil': 9,
    'Moth Beans': 10, 'Mung Bean': 11, 'Peas': 12, 'Pigeon Peas': 13, 'Rubber': 14,
    'Sugarcane': 15, 'Tea': 16, 'Tobacco': 17, 'apple': 18, 'banana': 19,
    'grapes': 20, 'maize': 21, 'mango': 22, 'millet': 23, 'muskmelon': 24,
    'orange': 25, 'papaya': 26, 'pomegranate': 27, 'rice': 28, 'watermelon': 29, 'wheat': 30
}


def preprocess_data(data):
    """Preprocess input data to match model training format"""
    try:
        # Create a DataFrame with all expected columns
        expected_columns = ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen',
                            'Phosphorous', 'Potassium', 'Carbon', 'Crop', 'Acidic_Soil',
                            'Alkaline_Soil', 'Loamy_Soil', 'Neutral_Soil', 'Peaty_Soil']

        # Initialize DataFrame with zeros
        df = pd.DataFrame(0, index=[0], columns=expected_columns)

        # Fill numerical features
        numerical_features = ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen',
                              'Phosphorous', 'Potassium', 'Carbon']
        for feature in numerical_features:
            df[feature] = float(data[feature])

        # Handle crop encoding
        crop_name = data['Crop']
        if crop_name in CROP_MAPPING:
            df['Crop'] = CROP_MAPPING[crop_name]
        else:
            raise ValueError(f"Unknown crop: {crop_name}")

        # Handle soil type one-hot encoding
        soil_type = data['Soil']
        soil_columns = ['Acidic_Soil', 'Alkaline_Soil', 'Loamy_Soil', 'Neutral_Soil', 'Peaty_Soil']

        # Reset all soil columns to 0
        for col in soil_columns:
            df[col] = 0

        # Set the selected soil type to 1
        if soil_type in soil_columns:
            df[soil_type] = 1
        else:
            raise ValueError(f"Unknown soil type: {soil_type}")

        print(f"🔸 Before scaling: {df.iloc[0].to_dict()}")

        # Apply scaling if scaler is available
        if scaler is not None:
            try:
                # Scale only the numerical columns (including Crop but not one-hot encoded soil features)
                numerical_cols = ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen',
                                  'Phosphorous', 'Potassium', 'Carbon', 'Crop']

                # Create a copy for scaling
                numerical_data = df[numerical_cols].values.reshape(1, -1)
                scaled_numerical_data = scaler.transform(numerical_data)

                # Update the dataframe with scaled values
                for i, col in enumerate(numerical_cols):
                    df[col] = scaled_numerical_data[0][i]

                print("✅ Applied scaling to numerical features")
                print(f"🔸 After scaling (numerical only): {df[numerical_cols].iloc[0].to_dict()}")
            except Exception as scale_error:
                print(f"❌ Scaling error: {scale_error}")
                print("⚠️ Continuing with raw values")
                # You might want to add basic normalization here as a backup

        else:
            print("⚠️ No scaler available - using raw values")

        print(f"🔹 Final data for prediction:\n{df.iloc[0].to_dict()}")
        return df

    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        raise


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get JSON data
        data = request.get_json()
        print(f"🔹 Received data: {data}")

        # Validate input data
        required_fields = ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen',
                           'Phosphorous', 'Potassium', 'Carbon', 'Crop', 'Soil']
        for field in required_fields:
            if field not in data or data[field] == '':
                return jsonify({'error': f'Missing or empty field: {field}'}), 400

        # Preprocess data
        processed_data = preprocess_data(data)

        # Make prediction with probabilities for debugging
        prediction = model.predict(processed_data)[0]

        # Get prediction probabilities to see what's happening
        try:
            probabilities = model.predict_proba(processed_data)[0]
            print(f"🎯 All prediction probabilities:")

            # Create a list of (fertilizer_name, probability) pairs
            prob_list = []
            for i, prob in enumerate(probabilities):
                fertilizer_name = FERTILIZER_MAPPING.get(i, f"Unknown_{i}")
                prob_list.append((fertilizer_name, prob))
                if prob > 0.01:  # Only print probabilities > 1%
                    print(f"   {fertilizer_name}: {prob:.3f} ({prob * 100:.1f}%)")

            # Sort by probability and show top 3
            prob_list.sort(key=lambda x: x[1], reverse=True)
            print(f"🏆 Top 3 predictions:")
            for i, (name, prob) in enumerate(prob_list[:3]):
                print(f"   {i + 1}. {name}: {prob:.3f} ({prob * 100:.1f}%)")

        except Exception as prob_error:
            print(f"🎯 Probabilities error: {prob_error}")

        fertilizer_name = FERTILIZER_MAPPING.get(prediction, f"Unknown Fertilizer (Code: {prediction})")

        print(f"✅ Prediction successful!")
        print(f"📊 Raw prediction: {prediction}")
        print(f"🌱 Recommended fertilizer: {fertilizer_name}")
        print("-" * 50)

        return jsonify({
            'prediction': fertilizer_name,
            'prediction_code': int(prediction),
            'success': True
        })

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 400


@app.route('/debug')
def debug_info():
    """Debug endpoint to see model and scaler information"""
    try:
        info = {
            'model_loaded': model is not None,
            'model_type': str(type(model)) if model else None,
            'scaler_loaded': scaler is not None,
            'scaler_type': str(type(scaler)) if scaler else None,
            'files_in_directory': [f for f in os.listdir('.') if f.endswith('.pkl')],
            'fertilizer_mapping': FERTILIZER_MAPPING,
            'crop_mapping': CROP_MAPPING,
            'working_directory': os.getcwd()
        }

        if scaler is not None:
            if hasattr(scaler, 'mean_'):
                info['scaler_mean'] = scaler.mean_.tolist()
            if hasattr(scaler, 'scale_'):
                info['scaler_scale'] = scaler.scale_.tolist()
            if hasattr(scaler, 'feature_names_in_'):
                info['scaler_feature_names'] = scaler.feature_names_in_.tolist()

        return jsonify(info)

    except Exception as e:
        return jsonify({'error': str(e), 'debug_failed': True}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 Fertilizer Recommendation System Starting")
    print("=" * 50)
    print(f"📂 Working Directory: {os.getcwd()}")
    print(f"🤖 Model Status: {'✅ Loaded' if model else '❌ Failed'}")
    print(f"⚖️ Scaler Status: {'✅ Loaded' if scaler else '❌ Failed'}")
    print("🌐 Visit http://127.0.0.1:5000 to use the application")
    print("🔧 Visit http://127.0.0.1:5000/debug for debugging info")
    print("=" * 50)
    app.run(debug=True)