import { Router, type IRouter, type Request } from "express";
import multer from "multer";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, readdir, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  GetRidsHistoryResponse,
  GetRidsHealthResponse,
  GetRidsModelInfoResponse,
  PredictRansomwareBatchBody,
  PredictRansomwareBody,
  PredictRansomwareResponse,
} from "@workspace/api-zod";

const router: IRouter = Router();
const projectRoot = path.resolve(fileURLToPath(import.meta.url), "../../../..");
const uploadsDir = path.join(projectRoot, "artifacts/api-server/uploads");
const historyDir = path.join(projectRoot, "artifacts/api-server/data/predictions");
const modelsDir = path.join(projectRoot, "artifacts/api-server/models");
const enginePath = path.join(projectRoot, "artifacts/api-server/src/rids-engine.py");
const pythonPath = path.join(projectRoot, ".pythonlibs/bin/python");
const pythonExecutable = existsSync(pythonPath) ? pythonPath : "python3";
const featureNamesPath = path.join(modelsDir, "feature_names.txt");
const studentName = "هبا مفتاح";
const projectName = "RIDS - Ransomware Intelligent Detection System";

const upload = multer({
  dest: uploadsDir,
  limits: { fileSize: 50 * 1024 * 1024 },
});

type JsonObject = Record<string, unknown>;

async function runEngine(payload: JsonObject): Promise<JsonObject> {
  await mkdir(uploadsDir, { recursive: true });
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable, [enginePath], {
      cwd: projectRoot,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `Inference engine exited with code ${code}`));
        return;
      }
      try {
        const parsed = JSON.parse(stdout) as JsonObject;
        if (parsed.error) {
          reject(new Error(String(parsed.error)));
          return;
        }
        resolve(parsed);
      } catch {
        reject(new Error(stderr || "Inference engine returned invalid JSON"));
      }
    });
    child.stdin.end(JSON.stringify(payload));
  });
}

async function getFeatureNames(): Promise<string[]> {
  const contents = await readFile(featureNamesPath, "utf8");
  return contents.split(/\r?\n/).map((name) => name.trim()).filter(Boolean);
}

async function appendHistory(record: JsonObject) {
  await mkdir(historyDir, { recursive: true });
  const filename = `prediction_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
  await writeFile(path.join(historyDir, filename), JSON.stringify(record, null, 2), "utf8");
}

function normalizePredictionResponse(
  engine: JsonObject,
  samplesCount: number,
  explanations: JsonObject[] | null = null,
) {
  return PredictRansomwareResponse.parse({
    status: "success",
    timestamp: new Date().toISOString(),
    student: studentName,
    project: projectName,
    samples_count: samplesCount,
    results: {
      predictions: engine.predictions,
      probabilities: engine.probabilities,
      model_used: engine.model_used,
    },
    explanations,
    model_info: { type: "XGBoost", features_used: 50 },
  });
}

router.get("/rids/health", async (_req, res) => {
  const features = await getFeatureNames();
  res.json(
    GetRidsHealthResponse.parse({
      status: features.length === 50 ? "healthy" : "unhealthy",
      models_loaded: features.length === 50,
      features_count: features.length,
      storage_configured: false,
      student: studentName,
      project: projectName,
      timestamp: new Date().toISOString(),
    }),
  );
});

router.get("/rids/model-info", async (_req, res) => {
  const features = await getFeatureNames();
  const metadata = await runEngine({ action: "metadata" });
  res.json(
    GetRidsModelInfoResponse.parse({
      model_type: "XGBoost",
      features_count: features.length,
      features_list: features.slice(0, 10),
      feature_importance: metadata.feature_importance,
      status: "ready",
      student: studentName,
      project: projectName,
    }),
  );
});

router.post("/rids/predict", async (req, res) => {
  const parsed = PredictRansomwareBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid feature payload", details: parsed.error.flatten() });
    return;
  }
  const body = parsed.data;
  const samples = body.samples ?? (body.sample ? [body.sample] : []);
  if (!samples.length) {
    res.status(400).json({ error: "Provide sample or samples" });
    return;
  }
  try {
    const engine = await runEngine({ action: "predict", samples, explain: Boolean(body.explain) });
    const explanation = engine.explanation
      ? [{ shap: engine.explanation, lime: (engine.explanation as JsonObject).lime }]
      : null;
    const response = normalizePredictionResponse(engine, samples.length, explanation);
    for (let index = 0; index < samples.length; index += 1) {
      await appendHistory({
        timestamp: response.timestamp,
        source: "json-input",
        label: Number((engine.predictions as number[])[index]) === 1 ? "Ransomware" : "Benign",
        prediction: (engine.predictions as number[])[index],
        probability: (engine.probabilities as number[])[index],
        model_used: engine.model_used,
        features: samples[index],
        explanation: explanation?.[index] ?? null,
      });
    }
    res.json(response);
  } catch (error) {
    res.status(500).json({ error: error instanceof Error ? error.message : "Prediction failed" });
  }
});

router.post("/rids/predict-batch", async (req, res) => {
  const parsed = PredictRansomwareBatchBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid batch payload", details: parsed.error.flatten() });
    return;
  }
  try {
    const engine = await runEngine({ action: "predict", samples: parsed.data.samples });
    const predictions = engine.predictions as number[];
    const probabilities = engine.probabilities as number[];
    const results = predictions.map((prediction, index) => ({
      sample_index: index,
      prediction,
      probability: probabilities[index],
      status: "success",
    }));
    res.json({
      status: "success",
      timestamp: new Date().toISOString(),
      student: studentName,
      project: projectName,
      total_samples: results.length,
      results,
    });
  } catch (error) {
    res.status(500).json({ error: error instanceof Error ? error.message : "Batch prediction failed" });
  }
});

router.post("/rids/upload", upload.single("file"), async (req: Request, res) => {
  if (!req.file) {
    res.status(400).json({ error: "No file uploaded" });
    return;
  }
  try {
    const allowedExtensions = new Set([
      "exe", "pdf", "docx", "zip", "dll", "scr", "js", "vbs", "ps1", "jar", "apk", "msi", "bin", "elf",
      "csv", "json", "jsonl", "tsv", "xlsx", "xls", "log", "txt",
    ]);
    const originalName = req.file.originalname;
    const extension = path.extname(originalName).slice(1).toLowerCase();
    if (!allowedExtensions.has(extension)) {
      await unlink(req.file.path);
      res.status(400).json({ error: `File type not allowed. Allowed: ${Array.from(allowedExtensions).join(", ")}` });
      return;
    }
    const engine = await runEngine({
      action: "upload",
      path: req.file.path,
      original_name: originalName,
      explain: true,
    });
    const prediction = (engine.predictions as number[])[0];
    const probability = (engine.probabilities as number[])[0];
    const record = {
      timestamp: new Date().toISOString(),
      label: prediction === 1 ? "Ransomware" : "Benign",
      prediction,
      probability,
      filename: originalName,
      file_size: req.file.size,
      source: "file-upload",
      model_used: engine.model_used,
      features: engine.features,
      extraction: engine.extraction,
      explanation: engine.explanation ?? null,
    };
    await appendHistory(record);
    res.json({
      status: "success",
      ...record,
      model_used: engine.model_used,
      features: engine.features,
      extraction: engine.extraction,
      explanation: engine.explanation ?? null,
      student: studentName,
      project: projectName,
    });
  } catch (error) {
    res.status(500).json({ error: error instanceof Error ? error.message : "File analysis failed" });
  } finally {
    await unlink(req.file.path).catch(() => undefined);
  }
});

router.get("/rids/history", async (_req, res) => {
  await mkdir(historyDir, { recursive: true });
  const filenames = (await readdir(historyDir)).filter((filename) => filename.endsWith(".json")).sort().reverse();
  const history: JsonObject[] = [];
  for (const filename of filenames.slice(0, 50)) {
    try {
      history.push(JSON.parse(await readFile(path.join(historyDir, filename), "utf8")) as JsonObject);
    } catch {
      // Ignore a partially written record.
    }
  }
  res.json(GetRidsHistoryResponse.parse({ status: "success", history, count: history.length }));
});

export default router;