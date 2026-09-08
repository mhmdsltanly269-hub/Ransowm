# ============================================================
# RIDS - Ransomware Intelligent Detection System
# Master Application - Full Academic Version
# ============================================================

import os
import sys
import json
import csv
import io
import uuid
import shutil
import logging
import traceback
from datetime import datetime
from io import StringIO, BytesIO
from pathlib import Path
import hashlib

import pandas as pd
import numpy as np
import joblib
from flask import (
    Flask, request, jsonify, render_template, 
    send_from_directory, redirect, url_for, flash, session
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.exceptions import RequestEntityTooLarge

# ============================================================
# استيراد الإعدادات
# ============================================================

from config import (
    BASE_DIR, MODELS_DIR, UPLOAD_DIR, SAVED_PREDICTIONS_DIR,
    LOGS_DIR, MODEL_FILES, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH,
    AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME,
    AZURE_STORAGE_ACCOUNT_NAME, DEBUG, PORT, SECRET_KEY,
    LOG_LEVEL, LOG_FILE, LOG_FORMAT, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    PROJECT_NAME, STUDENT_NAME, check_models, print_banner
)

# ============================================================
# إعداد التسجيل المتقدم (Logging)
# ============================================================

from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(getattr(logging, LOG_LEVEL))

formatter = logging.Formatter(LOG_FORMAT)

file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ============================================================
# تهيئة تطبيق Flask
# ============================================================

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app, resources={r"/*": {"origins": "*"}})

# عرض معلومات البداية
print_banner()
logger.info("🚀 Starting RIDS - Ransomware Intelligent Detection System")
logger.info(f"👩‍🎓 Student: {STUDENT_NAME}")

# ============================================================
# تحميل النماذج والمكونات
# ============================================================

models_loaded = False
xgb_model = None
scaler = None
feature_names = []
feature_importance_df = None
model_error = None

def load_models():
    global models_loaded, xgb_model, scaler, feature_names, feature_importance_df, model_error
    try:
        missing = check_models()
        if missing:
            error_msg = f"Missing model files: {missing}"
            logger.error(f"❌ {error_msg}")
            model_error = error_msg
            return False
        
        # تحميل أسماء الميزات
        with open(MODEL_FILES["feature_names"], 'r', encoding='utf-8') as f:
            feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"✅ Loaded {len(feature_names)} features")
        
        # تحميل المقياس
        scaler = joblib.load(MODEL_FILES["scaler"])
        logger.info("✅ Loaded StandardScaler")
        
        # تحميل النموذج
        xgb_model = joblib.load(MODEL_FILES["xgboost"])
        logger.info("✅ Loaded XGBoost model")
        
        # تحميل أهمية الميزات
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
        model_error = str(e)
        logger.error(f"❌ Failed to load models: {e}")
        logger.error(traceback.format_exc())
        return False

load_models()

# ============================================================
# إعداد التخزين (Azure Blob + محلي)
# ============================================================

blob_service_client = None
use_azure_storage = False

try:
    if AZURE_STORAGE_CONNECTION_STRING:
        from azure.storage.blob import BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        use_azure_storage = True
        logger.info("✅ Connected to Azure Blob Storage (Connection String)")
    elif AZURE_STORAGE_ACCOUNT_NAME:
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(
            account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/",
            credential=credential
        )
        use_azure_storage = True
        logger.info("✅ Connected to Azure Blob Storage (Managed Identity)")
    else:
        logger.info("ℹ️ Azure Storage not configured. Using local storage only.")
except ImportError:
    logger.warning("⚠️ Azure SDK not installed. Using local storage only.")
except Exception as e:
    logger.error(f"❌ Azure Blob connection failed: {e}")
    logger.info("ℹ️ Using local storage only.")

def save_to_azure_blob(data, blob_name_prefix="prediction"):
    if not use_azure_storage or blob_service_client is None:
        return False
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        blob_name = f"{blob_name_prefix}_{timestamp}.json"
        json_data = json.dumps(data, default=str, ensure_ascii=False)
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_STORAGE_CONTAINER_NAME, 
            blob=blob_name
        )
        blob_client.upload_blob(json_data, overwrite=True)
        logger.info(f"✅ Saved to Azure Blob: {blob_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Azure Blob save failed: {e}")
        return False

def save_to_local_file(data, file_prefix="prediction"):
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{file_prefix}_{timestamp}.json"
        filepath = SAVED_PREDICTIONS_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=str, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved locally: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Local save failed: {e}")
        return False

def save_prediction_result(sample_data, prediction, probability, model_used, user_id=None, file_path=None):
    record = {
        'timestamp': datetime.now().isoformat(),
        'student': STUDENT_NAME,
        'project': PROJECT_NAME,
        'prediction': int(prediction),
        'label': 'Ransomware' if prediction == 1 else 'Benign',
        'probability': float(probability),
        'model_used': model_used,
        'user_id': user_id or 'anonymous',
        'file_path': file_path,
        'sample_hash': hashlib.md5(str(sample_data).encode()).hexdigest()[:16],
        'sample_data': sample_data
    }
    azure_success = save_to_azure_blob(record, "prediction")
    local_success = save_to_local_file(record, "prediction")
    return azure_success or local_success

# ============================================================
# دوال المعالجة الأساسية
# ============================================================

def preprocess_input(data):
    try:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return None, "Unsupported data format. Expected dict, list, or DataFrame."
        
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            return None, f"Missing {len(missing)} features: {missing[:5]}..."
        
        df_ordered = df[feature_names].copy()
        df_ordered = df_ordered.apply(pd.to_numeric, errors='coerce')
        df_ordered = df_ordered.fillna(0)
        scaled_data = scaler.transform(df_ordered)
        return scaled_data, df_ordered
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
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
        logger.error(f"Prediction error: {e}")
        raise

def get_shap_explanation(scaled_data, index=0):
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(scaled_data)
        sample_shap = shap_values[index]
        sorted_features = sorted(
            zip(feature_names, sample_shap),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return {
            'base_value': float(explainer.expected_value),
            'top_features': [
                {'feature': f, 'shap_value': float(v)}
                for f, v in sorted_features[:10]
            ]
        }
    except ImportError:
        return None
    except Exception as e:
        logger.error(f"SHAP error: {e}")
        return None

def get_lime_explanation(scaled_data, index=0):
    try:
        import lime
        import lime.lime_tabular
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=scaled_data,
            feature_names=feature_names,
            class_names=['Benign', 'Ransomware'],
            mode='classification',
            verbose=False
        )
        exp = explainer.explain_instance(
            scaled_data[index],
            xgb_model.predict_proba,
            num_features=10
        )
        return [{'feature': f, 'weight': float(w)} for f, w in exp.as_list()]
    except ImportError:
        return None
    except Exception as e:
        logger.error(f"LIME error: {e}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features_from_file(filepath):
    """استخراج الميزات من ملف (نموذج أكاديمي)"""
    logger.warning(f"⚠️ Extracting simulated features from {filepath}")
    np.random.seed(hash(filepath) % 2**32)
    features = np.random.randn(50) * 0.5 + 0.5
    features = np.clip(features, 0, 1)
    return {feature: float(value) for feature, value in zip(feature_names, features)}

# ============================================================
# API Endpoints
# ============================================================

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html',
        project_name=PROJECT_NAME,
        student_name=STUDENT_NAME,
        models_loaded=models_loaded,
        feature_count=len(feature_names),
        storage_configured=use_azure_storage,
        model_error=model_error
    )

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'features_count': len(feature_names),
        'storage_configured': use_azure_storage,
        'student': STUDENT_NAME,
        'project': PROJECT_NAME,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    importance = None
    if feature_importance_df is not None:
        importance = feature_importance_df.head(20).to_dict('records')
    return jsonify({
        'model_type': 'XGBoost',
        'features_count': len(feature_names),
        'features_list': feature_names[:10],
        'feature_importance': importance,
        'status': 'ready' if models_loaded else 'not_loaded',
        'student': STUDENT_NAME,
        'project': PROJECT_NAME
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Please check server logs.'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        scaled_data, df_ordered = preprocess_input(data)
        if scaled_data is None:
            return jsonify({'error': df_ordered}), 400
        
        results = predict_ransomware(scaled_data)
        
        user_id = request.headers.get('X-User-Id', request.remote_addr)
        for i in range(len(scaled_data)):
            save_prediction_result(
                sample_data=data,
                prediction=results['predictions'][i],
                probability=results['probabilities'][i],
                model_used=results['model_used'],
                user_id=user_id
            )
        
        explanations = []
        if data.get('explain', False):
            for i in range(min(3, len(scaled_data))):
                shap_exp = get_shap_explanation(scaled_data, i)
                lime_exp = get_lime_explanation(scaled_data, i)
                explanations.append({'shap': shap_exp, 'lime': lime_exp})
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'student': STUDENT_NAME,
            'project': PROJECT_NAME,
            'samples_count': len(scaled_data),
            'results': results,
            'explanations': explanations if explanations else None,
            'model_info': {
                'type': 'XGBoost',
                'features_used': len(feature_names)
            }
        }
        
        logger.info(f"✅ Prediction completed: {len(scaled_data)} samples")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    
    try:
        data = request.get_json()
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided'}), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({'error': 'Samples must be a list'}), 400
        
        results = []
        for i, sample in enumerate(samples):
            try:
                scaled_data, _ = preprocess_input(sample)
                if scaled_data is not None:
                    pred_result = predict_ransomware(scaled_data)
                    results.append({
                        'sample_index': i,
                        'prediction': pred_result['predictions'][0],
                        'probability': pred_result['probabilities'][0],
                        'status': 'success'
                    })
                else:
                    results.append({
                        'sample_index': i,
                        'status': 'error',
                        'error': 'Invalid features'
                    })
            except Exception as e:
                results.append({
                    'sample_index': i,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'student': STUDENT_NAME,
            'project': PROJECT_NAME,
            'total_samples': len(samples),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """استقبال ملف وتحليله"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        filename = secure_filename(file.filename)
        filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
        file.save(filepath)
        logger.info(f"📁 File saved: {filepath}")
        
        # استخراج الميزات من الملف
        features = extract_features_from_file(str(filepath))
        
        # التنبؤ
        scaled_data, _ = preprocess_input(features)
        if scaled_data is None:
            return jsonify({'error': 'Feature extraction failed'}), 400
        
        results = predict_ransomware(scaled_data)
        
        # حفظ النتيجة
        user_id = request.headers.get('X-User-Id', request.remote_addr)
        save_prediction_result(
            sample_data=features,
            prediction=results['predictions'][0],
            probability=results['probabilities'][0],
            model_used=results['model_used'],
            user_id=user_id,
            file_path=str(filepath)
        )
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'student': STUDENT_NAME,
            'project': PROJECT_NAME,
            'filename': filename,
            'file_size': filepath.stat().st_size,
            'prediction': results['predictions'][0],
            'label': 'Ransomware' if results['predictions'][0] == 1 else 'Benign',
            'probability': results['probabilities'][0],
            'model_used': results['model_used']
        })
        
    except RequestEntityTooLarge:
        return jsonify({'error': f'File too large. Max size: {MAX_CONTENT_LENGTH // (1024*1024)} MB'}), 413
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download_logs', methods=['GET'])
def download_logs():
    """تحميل سجلات النظام"""
    try:
        if LOG_FILE.exists():
            return send_from_directory(str(LOGS_DIR), "system.log", as_attachment=True)
        return jsonify({'error': 'Log file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions_history', methods=['GET'])
def predictions_history():
    """عرض تاريخ التنبؤات المحفوظة محلياً"""
    try:
        files = sorted(SAVED_PREDICTIONS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        history = []
        for f in files[:50]:  # آخر 50 تنبؤ
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    history.append({
                        'file': f.name,
                        'timestamp': data.get('timestamp', ''),
                        'label': data.get('label', ''),
                        'probability': data.get('probability', 0),
                        'prediction': data.get('prediction', 0)
                    })
            except:
                continue
        return jsonify({'status': 'success', 'history': history, 'count': len(history)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# تشغيل التطبيق
# ============================================================

if __name__ == '__main__':
    logger.info(f"🚀 Running RIDS on port {PORT}")
    logger.info(f"👩‍🎓 Student: {STUDENT_NAME}")
    logger.info("🌐 Open http://localhost:{}".format(PORT))
    
    if not models_loaded:
        logger.warning("⚠️ Models not loaded! Please check the models directory.")
    
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)