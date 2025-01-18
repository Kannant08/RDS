from flask import Flask, request, jsonify, render_template
import pandas as pd
from datetime import datetime
import holidays
from openai import AzureOpenAI
from evalml.automl import AutoMLSearch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import json
import numpy as np
import os

app = Flask(__name__)

# Securely fetch API credentials from environment variables
api = os.getenv("AZURE_API_KEY", "8aa00b22342c40f5882e116427e55dd7")
endpoint = os.getenv("AZURE_ENDPOINT", "https://feat1.openai.azure.com/")

chat_client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_deployment="heybuddy",
    api_key=api,
    api_version="2024-02-01"
)

def add_calendar_features(data, date_col):
    holiday_list = holidays.country_holidays('IN')
    data[date_col] = pd.to_datetime(data[date_col])
    data['month'] = data[date_col].dt.month
    data['day_of_week'] = data[date_col].dt.dayofweek
    data['week_of_year'] = data[date_col].dt.isocalendar().week
    data['holiday'] = data[date_col].apply(lambda x: 1 if x in holiday_list else 0)
    data = data.drop(date_col,axis = 1)
    return data

def preprocessing_if_date_col(data,date_col,target):
    if date_col in data.columns:
        data = data.drop([col for col in data.columns if col not in [date_col, target]], axis=1)
    return data

def train_random_forest(data,target):
    X = data.drop(columns = [target],axis = 1) 
    y = data[target] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline = RandomForestRegressor()
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def train_evalml(features, target):
    automl = AutoMLSearch(X_train=features, y_train=target, problem_type="regression", max_batches=1)
    automl.search()
    best_pipeline = automl.best_pipeline
    return best_pipeline

def prediction_format(input_date):
    holiday_list = holidays.country_holidays('IN')
    input_date = pd.to_datetime(input_date)
    df = pd.DataFrame([input_date], columns=['Date'])

    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['holiday'] = df['Date'].apply(lambda x: 1 if x in holiday_list else 0)
    df = df.drop('Date',axis = 1)  
    return df             

    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    input_date_0 = request.form.get('start_date')
    input_date_1 = request.form.get('end_date')

    if not input_date_0 or not input_date_1:
        return jsonify({"error": "Start date and end date are required."}), 400
    
    try:
        start_date = pd.to_datetime(input_date_0)
        end_date = pd.to_datetime(input_date_1)
        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date.")
    except Exception as e:
        return jsonify({"error": f"Invalid date format: {str(e)}"}), 400


    if file:
        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file type."}), 400

        prompt = f"""You are a data scientist and working on demand forecasting of sales and bills for products. Extract the features as 'features', target column as 'target' for the product which has to be stocked in quantity, and date column as 'date' (if no date column leave the list empty). Output in JSON: {list(data.columns)}.
                      Note that the output features, target and date columns should be given in form of list only."""

        try:
            response = chat_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1,
                model="feature"
            )
            
            print("Raw response from Azure OpenAI:")
            #print(response.choices[0].message.content)
            

            response_content = response.choices[0].message.content
            cleaned_content = response_content.replace('```json\n', '').replace('\n```', '').strip()
            print(cleaned_content)


            try:
                parsed_response = json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                return jsonify({"error": f"JSON decoding failed: {str(e)}"}), 400

            feature_cols = parsed_response.get("features", [])
            target_col = parsed_response.get("target", [])
            date_col = parsed_response.get("date", [])

            if not feature_cols or not target_col:
                raise ValueError("Features or target column extraction failed.")
        except Exception as e:
            return jsonify({"error": f"Failed to parse AI response: {str(e)}"}), 400
        
        features = data[feature_cols]
        target = data[target_col]
        
        if date_col:
            data = preprocessing_if_date_col(data, date_col[0],target_col[0])
            data = add_calendar_features(data, date_col[0])
            model = train_random_forest(data,target_col[0])

            date_range = pd.date_range(start=start_date, end=end_date)
            print(date_range)
            predictions = []
            for date in date_range:
                prediction_data = prediction_format(date)
                predictions.extend(model.predict(prediction_data).tolist())
            mean_prediction = np.mean(predictions)
            print(mean_prediction)

            return jsonify({
                     "message": f"Predicted mean {target_col[0]} between {input_date_0} and {input_date_1}",
                     "value": mean_prediction})

   
if __name__ == '__main__':
    app.run(debug=True)
    



        
    
        



