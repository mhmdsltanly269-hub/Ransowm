# ============================================================
# RIDS - Ransomware Intelligent Detection System
# Master's Thesis Project - Academic Version
# ============================================================

import os
import sys
import json
import logging
import traceback
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# استيراد الإعدادات
from config import (
    BASE_DIR, MODELS_DIR, TEMPLATES_DIR, STATIC_DIR,
    UPLOAD_DIR, SAVED_PREDICTIONS_DIR, LOGS_DIR,
    MODEL_FILES, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH,
    SECRET_KEY, DEBUG, PORT, PROJECT_NAME, STUDENT_NAME,
    check_models, print_banner, LOG_FILE, LOG_FORMAT, LOG_LEVEL
)

# ============================================================
# إعداد التسجيل الأكاديمي
# ============================================================
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(getattr(logging, LOG_LEVEL))

formatter = logging.Formatter(LOG_FORMAT)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
logger.addHandler(console)

print_banner()

# ============================================================
# تهيئة تطبيق Flask
# ============================================================
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# ============================================================
# تحميل النماذج
# ============================================================
models_loaded = False
xgb_model = None
scaler = None
feature_names = []
feature_importance_df = None

def load_models():
    global models_loaded, xgb_model, scaler, feature_names, feature_importance_df
    try:
        missing = check_models()
        if missing:
            logger.error(f"Missing model files: {missing}")
            return False
        
        with open(MODEL_FILES["feature_names"], 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"✅ Loaded {len(feature_names)} features")
        
        scaler = joblib.load(MODEL_FILES["scaler"])
        logger.info("✅ Loaded StandardScaler")
        
        xgb_model = joblib.load(MODEL_FILES["xgboost"])
        logger.info("✅ Loaded XGBoost model")
        
        if MODEL_FILES["feature_importance"].exists():
            feature_importance_df = pd.read_csv(MODEL_FILES["feature_importance"])
            logger.info("✅ Loaded feature_importance.csv")
        
        models_loaded = True
        logger.info("=" * 60)
        logger.info("✅ ALL MODELS LOADED SUCCESSFULLY!")
        logger.info(f"   Features: {len(feature_names)}")
        logger.info("=" * 60)
        return True
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

load_models()

# ============================================================
# دوال المعالجة
# ============================================================
def preprocess_input(data):
    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return None, "Unsupported data format"
        
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            return None, f"Missing {len(missing)} features"
        
        df_ordered = df[feature_names].copy()
        df_ordered = df_ordered.apply(pd.to_numeric, errors='coerce').fillna(0)
        scaled = scaler.transform(df_ordered)
        return scaled, df_ordered
    except Exception as e:
        return None, str(e)

def predict_ransomware(scaled_data):
    try:
        proba = xgb_model.predict_proba(scaled_data)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            'predictions': pred.tolist(),
            'probabilities': proba.tolist(),
            'model_used': 'XGBoost (50 features)'
        }
    except Exception as e:
        raise

# ============================================================
# API Endpoints
# ============================================================
@app.route('/')
def index():
    return render_template('index.html',
        project_name=PROJECT_NAME,
        student_name=STUDENT_NAME,
        models_loaded=models_loaded,
        feature_count=len(feature_names)
    )

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'features_count': len(feature_names),
        'student': STUDENT_NAME,
        'project': PROJECT_NAME,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        scaled, _ = preprocess_input(data)
        if scaled is None:
            return jsonify({'error': 'Invalid features'}), 400
        results = predict_ransomware(scaled)
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'student': STUDENT_NAME,
            'project': PROJECT_NAME,
            'results': results
        })
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        filename = secure_filename(file.filename)
        filepath = UPLOAD_DIR / filename
        file.save(filepath)
        # محاكاة استخراج الميزات (للأغراض الأكاديمية)
        import random
        features = {f: random.uniform(0.1, 0.9) for f in feature_names}
        scaled, _ = preprocess_input(features)
        if scaled is None:
            return jsonify({'error': 'Feature extraction failed'}), 400
        results = predict_ransomware(scaled)
        return jsonify({
            'status': 'success',
            'filename': filename,
            'prediction': results['predictions'][0],
            'label': 'Ransomware' if results['predictions'][0] == 1 else 'Benign',
            'probability': results['probabilities'][0],
            'student': STUDENT_NAME,
            'project': PROJECT_NAME
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)