from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for model state
model_state = {
    'data_uploaded': False,
    'data_preprocessed': False,
    'model_trained': False,
    'dataset': None,
    'preprocessed_data': None,
    'model': None,
    'scaler': None,
    'feature_columns': None,
    'target_column': None,
    'metrics': None,
    'forecast_results': None
}

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_state': model_state
    }), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files allowed'}), 400
        
        df = pd.read_csv(file)
        model_state['dataset'] = df
        model_state['data_uploaded'] = True
        
        return jsonify({
            'success': True,
            'message': f'File uploaded successfully with {len(df)} rows',
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'preview': df.head(5).to_dict()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    try:
        if not model_state['data_uploaded']:
            return jsonify({'error': 'No data uploaded'}), 400
        
        df = model_state['dataset'].copy()
        data = request.json
        
        target_column = data.get('target_column', df.columns[-1])
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Remove outliers using IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
        
        # Normalize numeric columns
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        model_state['preprocessed_data'] = df
        model_state['data_preprocessed'] = True
        model_state['target_column'] = target_column
        model_state['scaler'] = scaler
        
        return jsonify({
            'success': True,
            'message': 'Data preprocessed successfully',
            'rows_removed': initial_rows - len(df),
            'final_shape': df.shape,
            'preview': df.head(5).to_dict()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        if not model_state['data_preprocessed']:
            return jsonify({'error': 'Data not preprocessed'}), 400
        
        df = model_state['preprocessed_data']
        target_col = model_state['target_column']
        data = request.json
        
        model_type = data.get('model_type', 'random_forest')
        test_size = data.get('test_size', 0.2)
        
        # Prepare features and target
        if target_col not in df.columns:
            return jsonify({'error': f'Target column {target_col} not found'}), 400
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        model_state['feature_columns'] = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        if model_type == 'linear_regression':
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        model_state['model'] = model
        model_state['model_trained'] = True
        model_state['metrics'] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'model_type': model_type,
            'metrics': model_state['metrics'],
            'test_samples': len(X_test)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        if not model_state['model_trained']:
            return jsonify({'error': 'Model not trained'}), 400
        
        data = request.json
        new_data = pd.DataFrame(data.get('data', []))
        
        if new_data.empty:
            return jsonify({'error': 'No data provided for forecast'}), 400
        
        predictions = model_state['model'].predict(new_data)
        
        model_state['forecast_results'] = {
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'predictions': predictions.tolist(),
            'count': len(predictions)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def analytics():
    try:
        if not model_state['metrics']:
            return jsonify({'error': 'No trained model'}), 400
        
        return jsonify({
            'metrics': model_state['metrics'],
            'features': model_state['feature_columns'],
            'target': model_state['target_column'],
            'data_shape': model_state['preprocessed_data'].shape if model_state['preprocessed_data'] is not None else None
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-forecast', methods=['GET'])
def export_forecast():
    try:
        if not model_state['forecast_results']:
            return jsonify({'error': 'No forecast results'}), 400
        
        df = pd.DataFrame({
            'prediction': model_state['forecast_results']['predictions']
        })
        
        filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join('/tmp', filename)
        df.to_csv(filepath, index=False)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'rows': len(df)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    return jsonify({
        'data_uploaded': model_state['data_uploaded'],
        'data_preprocessed': model_state['data_preprocessed'],
        'model_trained': model_state['model_trained'],
        'metrics': model_state['metrics'],
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000