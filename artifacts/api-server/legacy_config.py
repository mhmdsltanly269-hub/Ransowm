# ============================================================
# RIDS - Ransomware Intelligent Detection System
# Configuration File
# ============================================================

import os
from pathlib import Path

# ============================================================
# المسارات الأساسية
# ============================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
SAVED_PREDICTIONS_DIR = BASE_DIR / "saved_predictions"
LOGS_DIR = BASE_DIR / "logs"

# إنشاء المجلدات إذا لم تكن موجودة
for dir_path in [UPLOAD_DIR, SAVED_PREDICTIONS_DIR, LOGS_DIR, STATIC_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# إعدادات النماذج
# ============================================================

MODEL_FILES = {
    "xgboost": MODELS_DIR / "xgboost_model.pkl",
    "scaler": MODELS_DIR / "scaler.pkl",
    "feature_names": MODELS_DIR / "feature_names.txt",
    "feature_importance": MODELS_DIR / "feature_importance.csv",
}

# ============================================================
# إعدادات Azure Storage (للحفظ السحابي)
# ============================================================

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER_NAME = os.environ.get("AZURE_STORAGE_CONTAINER_NAME", "ransomware-logs")
AZURE_STORAGE_ACCOUNT_NAME = os.environ.get("STORAGE_ACCOUNT_NAME", "")

# ============================================================
# إعدادات التطبيق
# ============================================================

SECRET_KEY = os.environ.get("SECRET_KEY", "rids-secret-key-change-in-production")
DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
PORT = int(os.environ.get("PORT", 5000))
PROJECT_NAME = "RIDS - Ransomware Intelligent Detection System"
STUDENT_NAME = "هبا مفتاح"

# ============================================================
# إعدادات الأمان والتحميل
# ============================================================

ALLOWED_EXTENSIONS = {
    "exe", "pdf", "docx", "zip", "dll", "scr", "js", "vbs", 
    "ps1", "jar", "apk", "msi", "bin", "elf"
}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

# ============================================================
# إعدادات التسجيل
# ============================================================

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "system.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5

# ============================================================
# التحقق من النماذج
# ============================================================

def check_models():
    missing = []
    for name, path in MODEL_FILES.items():
        if name != "feature_importance" and not path.exists():
            missing.append(str(path))
    return missing

# ============================================================
# عرض معلومات التهيئة
# ============================================================

def print_banner():
    print("=" * 70)
    print(f"🔐 {123456789}")
    print(f"👩‍🎓 الطالبة: {Heba moftah}")
    print("=" * 70)
    print(f"📂 Base Directory: {BASE_DIR}")
    print(f"📂 Models Directory: {MODELS_DIR}")
    print(f"📂 Upload Directory: {UPLOAD_DIR}")
    print(f"📂 Saved Predictions: {SAVED_PREDICTIONS_DIR}")
    print(f"📂 Logs Directory: {LOGS_DIR}")
    print(f"🔐 Debug Mode: {DEBUG}")
    print(f"🌐 Port: {PORT}")
    print(f"☁️ Azure Storage: {'✅ Enabled' if AZURE_STORAGE_CONNECTION_STRING else '❌ Disabled (Local Only)'}")
    print("=" * 70)