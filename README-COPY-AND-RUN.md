# RIDS — Final System Copy

This package is the final Ransomware Intelligent Detection System version.
It includes:

- English LTR security dashboard with the student name: Hiba Muftah
- JSON/JSONL, CSV/TSV, XLSX/XLS, LOG/TXT, BIN/EXE/DLL/ELF and related file analysis
- Real XGBoost model and StandardScaler assets
- Exact 50-feature order in `artifacts/api-server/models/feature_names.txt`
- SHAP/LIME-style explanations and top-feature contribution charts
- JSON and CSV prediction report downloads
- Local JSON prediction history with feature and extraction metadata

## Copy into another Replit workspace

1. Download and extract this ZIP at the workspace root.
2. Keep the folder structure exactly as extracted.
3. Install JavaScript packages:

```bash
pnpm install
```

4. Install the Python packages from `pyproject.toml` using the workspace Python setup:

```bash
uv sync
```

If `uv` is not available, install the listed packages with Python 3.10+:

```bash
python -m pip install joblib numpy pandas scikit-learn xgboost xlrd xlwt
```

## Run the system

Start the API server:

```bash
PORT=8080 pnpm --filter @workspace/api-server run dev
```

Start the web interface in a second shell:

```bash
pnpm --filter @workspace/rids-ransomware-detection run dev
```

The API uses the included model files automatically. In Replit, the existing artifact/workflow configuration can be used after the files are copied.

## Important model compatibility rule

Do not rename, replace, reorder, or independently mix these files:

- `artifacts/api-server/models/xgboost_model.pkl`
- `artifacts/api-server/models/scaler.pkl`
- `artifacts/api-server/models/feature_names.txt`

They are a matched set and must remain paired for valid inference.

## Runtime data

Prediction history is created automatically in:

```text
artifacts/api-server/data/predictions/
```

The directory is intentionally empty in this delivery so the copied system starts with a clean history.
