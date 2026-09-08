# RIDS — Ransomware Intelligent Detection System

An English LTR web interface for analyzing ransomware behavior with the supplied 50-feature XGBoost model, including real raw-file feature extraction.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the inference API server
- `pnpm --filter @workspace/rids-ransomware-detection run dev` — run the original RIDS web interface
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/rids-ransomware-detection/index.html` — RIDS interface with the English light theme
- `artifacts/api-server/src/routes/rids.ts` — RIDS API endpoints, upload handling, and local history
- `artifacts/api-server/src/rids-engine.py` — feature extraction and model inference process
- `artifacts/api-server/models/` — supplied XGBoost model, scaler, and exact feature order
- `lib/api-spec/openapi.yaml` — source of truth for generated API contracts

## Architecture decisions

- The original HTML/CSS/JavaScript behavior is kept intact; the presentation is English/LTR with a calm light security palette.
- Python is used for inference so the supplied pickle artifacts are loaded by the same ML ecosystem they were trained with.
- Structured raw behavior exports use exact feature IDs and common `event_id/count`-style columns; binary samples use deterministic byte statistics instead of random placeholders.
- Prediction history remains local JSON, matching the original project's local-storage behavior without requiring an external integration.

## Product

- Predict from the original 50-feature JSON input.
- Upload supported raw behavior exports or binary files and extract features before prediction.
- Accept JSON, JSONL, CSV, TSV, XLSX, XLS, LOG, TXT, and common executable/raw sample extensions.
- View model health, feature importance, prediction history, and SHAP/LIME-style explanations.
- View post-prediction confidence and feature-contribution charts, then download a JSON or CSV report.
- Prediction history files retain the input features, extraction metadata, and explanation payload for auditability.

## User preferences

- Preserve the supplied interface behavior and model contract; visual changes should remain readable and research-oriented.

## Gotchas

- The inference engine must receive the exact feature order from `models/feature_names.txt`; do not reorder the model vector.
- The model artifacts are loaded in the Python runtime available through `.pythonlibs`.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
