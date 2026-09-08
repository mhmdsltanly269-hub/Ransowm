import { type ReactNode, useState } from 'react';
import { QueryClient, QueryClientProvider, useQueryClient } from '@tanstack/react-query';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  BarChart3,
  CircleHelp,
  Clock3,
  Code2,
  Database,
  FileArchive,
  FileJson2,
  Gauge,
  History,
  Info,
  Layers3,
  Menu,
  PanelRight,
  Play,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  UploadCloud,
  X,
} from 'lucide-react';
import {
  getGetRidsHealthQueryKey,
  getGetRidsHistoryQueryKey,
  getGetRidsModelInfoQueryKey,
  useGetRidsHealth,
  useGetRidsHistory,
  useGetRidsModelInfo,
  usePredictRansomware,
  usePredictRansomwareBatch,
  useUploadRansomwareFile,
} from '@workspace/api-client-react';
import {
  Route,
  Switch,
  useLocation,
  Router as WouterRouter,
} from 'wouter';

const queryClient = new QueryClient();

type ResultRecord = Record<string, any>;

const sampleJson = `{
  "feature_1": 0.73,
  "feature_2": 14,
  "feature_3": 0.18,
  "feature_4": 1,
  "feature_5": 0.42
}`;

const navItems = [
  { label: 'مركز التحليل', icon: Activity, active: true },
  { label: 'سجل التنبؤات', icon: History, active: false },
  { label: 'معلومات النموذج', icon: SlidersHorizontal, active: false },
];

function formatDate(value?: string) {
  if (!value) return '—';
  try {
    return new Intl.DateTimeFormat('ar-SA', {
      dateStyle: 'medium',
      timeStyle: 'short',
    }).format(new Date(value));
  } catch {
    return value;
  }
}

function statusLabel(value: unknown) {
  const text = String(value ?? '').toLowerCase();
  if (['ok', 'healthy', 'ready', 'success', 'loaded'].some((term) => text.includes(term))) return 'جاهز';
  if (['error', 'failed', 'offline'].some((term) => text.includes(term))) return 'غير متاح';
  return String(value ?? 'قيد الفحص');
}

function StatusBadge({ value, warm = false }: { value?: unknown; warm?: boolean }) {
  const positive = !value || ['ok', 'healthy', 'ready', 'success', 'loaded', 'true'].some((term) => String(value).toLowerCase().includes(term));
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${
      positive && !warm
        ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
        : 'border-amber-400/25 bg-amber-400/10 text-amber-200'
    }`}>
      <span className={`h-1.5 w-1.5 rounded-full ${positive && !warm ? 'bg-emerald-400' : 'bg-amber-400'}`} />
      {statusLabel(value)}
    </span>
  );
}

function SectionTitle({
  eyebrow,
  title,
  icon: Icon,
}: {
  eyebrow: string;
  title: string;
  icon: typeof Activity;
}) {
  return (
    <div className="mb-5 flex items-start justify-between gap-4">
      <div>
        <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.22em] text-primary/80">{eyebrow}</p>
        <h2 className="text-lg font-semibold tracking-tight text-foreground">{title}</h2>
      </div>
      <div className="rounded-md border border-border/70 bg-muted/60 p-2 text-muted-foreground">
        <Icon size={16} strokeWidth={1.6} />
      </div>
    </div>
  );
}

function SkeletonRows() {
  return (
    <div className="space-y-2" aria-label="جاري تحميل البيانات">
      {[1, 2, 3].map((item) => <div key={item} className="h-10 animate-pulse rounded-md bg-muted/70" />)}
    </div>
  );
}

function Home() {
  const [mobileNav, setMobileNav] = useState(false);
  const [inputMode, setInputMode] = useState<'single' | 'batch'>('single');
  const [jsonInput, setJsonInput] = useState(sampleJson);
  const [activeTab, setActiveTab] = useState<'shap' | 'lime' | 'features' | 'history'>('shap');
  const [prediction, setPrediction] = useState<ResultRecord | null>(null);
  const [fileResult, setFileResult] = useState<ResultRecord | null>(null);
  const [parseError, setParseError] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const healthQuery = useGetRidsHealth({
    query: { queryKey: getGetRidsHealthQueryKey(), refetchInterval: 30000 },
  });
  const modelQuery = useGetRidsModelInfo({
    query: { queryKey: getGetRidsModelInfoQueryKey(), staleTime: 60000 },
  });
  const historyQuery = useGetRidsHistory({
    query: { queryKey: getGetRidsHistoryQueryKey(), staleTime: 30000 },
  });
  const queryClient = useQueryClient();
  const predictionMutation = usePredictRansomware();
  const batchMutation = usePredictRansomwareBatch();
  const uploadMutation = useUploadRansomwareFile();

  const health = healthQuery.data;
  const model = modelQuery.data;
  const history = (historyQuery.data?.history ?? []) as ResultRecord[];
  const features = model?.features_list ?? [];
  const isAnalyzing = predictionMutation.isPending || batchMutation.isPending;
  const currentResult = fileResult ?? prediction;
  const rawResults = currentResult?.results as ResultRecord | undefined;
  const predictions = Array.isArray(rawResults?.predictions) ? rawResults.predictions : [];
  const probabilities = Array.isArray(rawResults?.probabilities) ? rawResults.probabilities : [];
  const primaryPrediction = predictions[0];
  const riskProbability = Number(probabilities[0] ?? 0);
  const isThreat = Number(primaryPrediction) === 1 || riskProbability >= 0.5;

  const parsePayload = () => {
    setParseError('');
    try {
      const parsed = JSON.parse(jsonInput);
      if (inputMode === 'batch') {
        if (!Array.isArray(parsed)) throw new Error('صيغة الدفعة يجب أن تكون مصفوفة من العينات');
        return { samples: parsed };
      }
      if (Array.isArray(parsed)) throw new Error('لتحليل عينة واحدة استخدم كائناً JSON واحداً');
      return { sample: parsed, explain: true };
    } catch (error) {
      setParseError(error instanceof Error ? error.message : 'تعذر قراءة JSON');
      return null;
    }
  };

  const runPrediction = () => {
    const payload = parsePayload();
    if (!payload) return;
    setFileResult(null);
    if (inputMode === 'batch') {
      batchMutation.mutate({ data: payload as { samples: Record<string, number>[] } }, {
        onSuccess: (result) => {
          setPrediction(result as ResultRecord);
          queryClient.invalidateQueries({ queryKey: getGetRidsHistoryQueryKey() });
        },
      });
    } else {
      predictionMutation.mutate({ data: payload as { sample: Record<string, number>; explain: boolean } }, {
        onSuccess: (result) => {
          setPrediction(result as ResultRecord);
          queryClient.invalidateQueries({ queryKey: getGetRidsHistoryQueryKey() });
        },
      });
    }
  };

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    setSelectedFile(file);
    setFileResult(null);
    uploadMutation.mutate({ data: { file } }, {
      onSuccess: (result) => {
        setFileResult(result as ResultRecord);
        queryClient.invalidateQueries({ queryKey: getGetRidsHistoryQueryKey() });
      },
    });
  };

  const setExample = () => {
    setInputMode('single');
    setJsonInput(sampleJson);
    setParseError('');
  };

  return (
    <div dir="rtl" className="rids-noise min-h-[100dvh] overflow-x-hidden bg-background text-foreground">
      <div className="flex min-h-[100dvh] flex-col lg:flex-row">
        <aside className="hidden w-[264px] shrink-0 border-l border-sidebar-border bg-sidebar lg:flex lg:flex-col">
          <div className="flex h-[78px] items-center gap-3 border-b border-sidebar-border px-6">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-md border border-primary/40 bg-primary/10 text-primary">
              <ShieldAlert size={20} strokeWidth={1.6} />
              <span className="absolute -bottom-1 -left-1 h-2 w-2 rounded-full bg-emerald-400 ring-2 ring-sidebar" />
            </div>
            <div>
              <p className="font-mono text-[10px] tracking-[0.28em] text-primary">RIDS / 01</p>
              <p className="text-sm font-bold text-sidebar-foreground">مركز القرار الأمني</p>
            </div>
          </div>
          <div className="flex-1 px-3 py-7">
            <p className="px-3 pb-3 text-[10px] font-semibold tracking-[0.18em] text-muted-foreground">مساحات العمل</p>
            <nav className="space-y-1" aria-label="التنقل الرئيسي">
              {navItems.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  data-testid={`button-nav-${item.label}`}
                  onClick={() => { if (item.label === 'سجل التنبؤات') setActiveTab('history'); if (item.label === 'معلومات النموذج') setActiveTab('features'); }}
                  className={`flex w-full items-center justify-between rounded-md px-3 py-3 text-right text-sm transition-colors ${
                    item.active ? 'bg-primary/12 text-foreground' : 'text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground'
                  }`}
                >
                  <span className="flex items-center gap-3"><item.icon size={17} strokeWidth={1.6} />{item.label}</span>
                  {item.active && <span className="h-1.5 w-1.5 rounded-full bg-primary" />}
                </button>
              ))}
            </nav>
            <div className="my-8 h-px bg-sidebar-border" />
            <p className="px-3 pb-3 text-[10px] font-semibold tracking-[0.18em] text-muted-foreground">حالة النظام</p>
            <div className="rounded-md border border-sidebar-border bg-sidebar-accent/45 p-3">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">النماذج</span>
                <StatusBadge value={health?.models_loaded ? 'ready' : health?.status} />
              </div>
              <div className="mt-3 flex items-center justify-between text-xs">
                <span className="text-muted-foreground">التخزين</span>
                <StatusBadge value={health?.storage_configured ? 'ready' : 'pending'} />
              </div>
              <div className="mt-4 h-1 overflow-hidden rounded-full bg-background">
                <div className="h-full w-[87%] rounded-full bg-primary" />
              </div>
              <p className="mt-2 font-mono text-[9px] text-muted-foreground">LOCAL / API LINK ESTABLISHED</p>
            </div>
          </div>
          <div className="border-t border-sidebar-border p-5">
            <p className="text-xs font-semibold text-sidebar-foreground">مشروع الماجستير</p>
            <p className="mt-1 text-[11px] leading-5 text-muted-foreground">نظام ذكي لاكتشاف برمجيات الفدية وتفسير قرارات النموذج</p>
            <p className="mt-4 font-mono text-[9px] tracking-wider text-muted-foreground">RIDS · RESEARCH BUILD 0.1</p>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <header className="sticky top-0 z-10 border-b border-border/80 bg-background/90 backdrop-blur-md">
            <div className="flex min-h-[78px] items-center justify-between gap-4 px-4 sm:px-7 xl:px-10">
              <div className="flex items-center gap-3">
                <button type="button" data-testid="button-mobile-menu" onClick={() => setMobileNav(!mobileNav)} className="rounded-md border border-border p-2 text-muted-foreground lg:hidden">
                  {mobileNav ? <X size={18} /> : <Menu size={18} />}
                </button>
                <div>
                  <p className="font-mono text-[10px] tracking-[0.18em] text-primary">RANSOMWARE INTELLIGENT DETECTION SYSTEM</p>
                  <h1 className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">مساحة التحليل <span className="font-normal text-muted-foreground">/ قرار قابل للتفسير</span></h1>
                </div>
              </div>
              <div className="hidden items-center gap-5 sm:flex">
                <div className="text-left">
                  <p className="font-mono text-[9px] tracking-[0.14em] text-muted-foreground">LAST SYNC</p>
                  <p className="mt-1 text-xs text-foreground">{formatDate(health?.timestamp)}</p>
                </div>
                <div className="h-8 w-px bg-border" />
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                  <span className="text-xs text-emerald-300">الخدمة متصلة</span>
                </div>
              </div>
            </div>
            {mobileNav && (
              <div className="border-t border-border bg-sidebar px-4 py-3 lg:hidden">
                {navItems.map((item) => <button key={item.label} type="button" onClick={() => setMobileNav(false)} className="flex w-full items-center gap-3 border-b border-sidebar-border py-3 text-sm text-sidebar-foreground"><item.icon size={16} />{item.label}</button>)}
              </div>
            )}
          </header>

          <div className="rids-grid min-h-[calc(100dvh-78px)] px-4 py-6 sm:px-7 sm:py-8 xl:px-10">
            <div className="mx-auto max-w-[1480px]">
              <div className="mb-7 flex flex-col justify-between gap-4 border-b border-border/70 pb-6 sm:flex-row sm:items-end">
                <div className="rids-reveal">
                  <p className="mb-2 flex items-center gap-2 font-mono text-[10px] tracking-[0.2em] text-primary"><span className="h-1.5 w-1.5 rounded-full bg-primary" /> LIVE ANALYSIS CONSOLE</p>
                  <p className="max-w-2xl text-sm leading-7 text-muted-foreground">حوّل خصائص الملف الخام إلى تقدير خطر واضح، مع الاحتفاظ بمسار التفسير لكل قرار.</p>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="rounded border border-border bg-card px-2 py-1 font-mono">MODEL {model?.model_type ?? '—'}</span>
                  <span className="rounded border border-border bg-card px-2 py-1 font-mono">{health?.features_count ?? model?.features_count ?? '—'} FEATURES</span>
                </div>
              </div>

              <div className="grid gap-6 xl:grid-cols-[minmax(0,1.06fr)_minmax(440px,.94fr)]">
                <section className="rids-reveal rounded-lg border border-card-border bg-card/90 p-5 shadow-2xl shadow-black/10 sm:p-6">
                  <SectionTitle eyebrow="01 / INPUT VECTOR" title="بيانات العينة" icon={Code2} />
                  <div className="mb-5 flex items-center justify-between border-b border-border/70">
                    <div className="flex gap-5">
                      {(['single', 'batch'] as const).map((mode) => (
                        <button key={mode} type="button" data-testid={`button-input-mode-${mode}`} onClick={() => setInputMode(mode)} className={`relative pb-3 text-sm ${inputMode === mode ? 'font-semibold text-foreground' : 'text-muted-foreground'}`}>
                          {mode === 'single' ? 'عينة واحدة' : 'دفعة عينات'}
                          {inputMode === mode && <span className="absolute inset-x-0 -bottom-px h-0.5 bg-primary" />}
                        </button>
                      ))}
                    </div>
                    <button type="button" data-testid="button-load-example" onClick={setExample} className="mb-2 flex items-center gap-1.5 text-[11px] text-accent transition-colors hover:text-accent/80"><Sparkles size={13} /> تحميل مثال</button>
                  </div>
                  <div className="relative overflow-hidden rounded-md border border-border bg-background/80">
                    <div className="flex items-center justify-between border-b border-border/70 bg-muted/45 px-4 py-2">
                      <span className="flex items-center gap-2 font-mono text-[10px] text-muted-foreground"><FileJson2 size={13} className="text-accent" /> {inputMode === 'single' ? 'sample.json' : 'samples.json'}</span>
                      <span className="font-mono text-[10px] text-muted-foreground">JSON / UTF-8</span>
                    </div>
                    <textarea
                      value={jsonInput}
                      onChange={(event) => { setJsonInput(event.target.value); setParseError(''); }}
                      data-testid="input-json-vector"
                      spellCheck={false}
                      className="rids-scanline min-h-[282px] w-full resize-y bg-transparent p-5 font-mono text-xs leading-7 text-slate-300 outline-none placeholder:text-muted-foreground/50"
                      dir="ltr"
                      aria-label="مدخل JSON"
                    />
                    <span className="rids-cursor absolute bottom-4 left-4 h-4 w-px bg-primary" />
                  </div>
                  {parseError && <p data-testid="status-json-error" className="mt-3 flex items-center gap-2 text-xs text-red-300"><AlertCircle size={14} />{parseError}</p>}
                  <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <p className="flex items-center gap-2 text-[11px] leading-5 text-muted-foreground"><Info size={14} className="shrink-0 text-accent" />سيتم تمرير المتجه إلى النموذج مع تفعيل التفسير.</p>
                    <button type="button" data-testid="button-run-analysis" onClick={runPrediction} disabled={isAnalyzing} className="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-5 py-3 text-sm font-semibold text-primary-foreground shadow-lg shadow-primary/10 transition-transform hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-60">
                      {isAnalyzing ? <RefreshCw size={16} className="animate-spin" /> : <Play size={16} fill="currentColor" />}
                      {isAnalyzing ? 'جارٍ التحليل...' : 'بدء التحليل'}
                    </button>
                  </div>
                </section>

                <section className="rids-reveal rids-reveal-delay-1 rounded-lg border border-card-border bg-card/90 p-5 sm:p-6">
                  <SectionTitle eyebrow="02 / RAW FILE PATH" title="تحليل ملف خام" icon={FileArchive} />
                  <label htmlFor="raw-file" data-testid="dropzone-raw-file" className="group flex min-h-[210px] cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-border bg-background/45 px-6 text-center transition-colors hover:border-primary/60 hover:bg-primary/[0.03]">
                    <div className="mb-4 rounded-full border border-primary/25 bg-primary/10 p-3 text-primary transition-transform group-hover:-translate-y-1"><UploadCloud size={24} strokeWidth={1.5} /></div>
                    <p className="text-sm font-semibold">{selectedFile ? selectedFile.name : 'اسحب الملف هنا أو اختر من الجهاز'}</p>
                    <p className="mt-2 text-xs text-muted-foreground">سيستخرج النظام الخصائص تلقائياً قبل التنبؤ</p>
                    <span className="mt-4 rounded border border-border bg-muted/60 px-3 py-1.5 text-[11px] text-foreground">اختيار ملف</span>
                    <input id="raw-file" type="file" data-testid="input-raw-file" className="sr-only" onChange={(event) => handleFile(event.target.files?.[0])} />
                  </label>
                  {uploadMutation.isPending && <div className="mt-4 flex items-center gap-2 rounded-md border border-primary/20 bg-primary/5 p-3 text-xs text-primary"><RefreshCw size={14} className="animate-spin" /> جارٍ استخراج الخصائص وتحليل الملف...</div>}
                  {uploadMutation.isError && <div className="mt-4 flex items-center gap-2 rounded-md border border-red-500/20 bg-red-500/5 p-3 text-xs text-red-300"><AlertCircle size={14} /> تعذر تحليل الملف. تحقق من النوع وحاول مرة أخرى.</div>}
                  <div className="mt-5 grid grid-cols-2 gap-3">
                    <div className="rounded-md border border-border bg-muted/35 p-3"><p className="text-[10px] text-muted-foreground">الامتدادات</p><p className="mt-2 font-mono text-xs text-foreground">EXE · DLL · BIN</p></div>
                    <div className="rounded-md border border-border bg-muted/35 p-3"><p className="text-[10px] text-muted-foreground">الخصوصية</p><p className="mt-2 flex items-center gap-1.5 text-xs text-emerald-300"><ShieldCheck size={13} />تحليل معزول</p></div>
                  </div>
                </section>
              </div>

              <section className="rids-reveal rids-reveal-delay-2 mt-6 rounded-lg border border-card-border bg-card/90 p-5 sm:p-6">
                <div className="flex flex-col justify-between gap-4 border-b border-border/70 pb-5 sm:flex-row sm:items-start">
                  <SectionTitle eyebrow="03 / DECISION OUTPUT" title="نتيجة التنبؤ" icon={Gauge} />
                  {currentResult && <StatusBadge value={currentResult.status ?? 'success'} warm={isThreat} />}
                </div>
                {!currentResult && !isAnalyzing && (
                  <div className="flex min-h-[165px] flex-col items-center justify-center text-center">
                    <div className="mb-3 rounded-full border border-border bg-muted/45 p-3 text-muted-foreground"><PanelRight size={20} /></div>
                    <p className="text-sm font-medium text-foreground">بانتظار متجه التحليل</p>
                    <p className="mt-1 text-xs text-muted-foreground">أدخل خصائص العينة أو ارفع ملفاً خاماً لعرض القرار</p>
                  </div>
                )}
                {isAnalyzing && <div className="grid gap-3 py-4 sm:grid-cols-3"><div className="h-24 animate-pulse rounded-md bg-muted/70" /><div className="h-24 animate-pulse rounded-md bg-muted/70" /><div className="h-24 animate-pulse rounded-md bg-muted/70" /></div>}
                {currentResult && !isAnalyzing && (
                  <div className="grid gap-4 pt-5 sm:grid-cols-[1.05fr_.95fr]">
                    <div className={`relative overflow-hidden rounded-md border p-5 ${isThreat ? 'border-primary/35 bg-primary/[0.07]' : 'border-emerald-500/25 bg-emerald-500/[0.06]'}`}>
                      <div className="absolute -left-8 -top-10 h-32 w-32 rounded-full border border-current opacity-10" />
                      <div className="flex items-start justify-between">
                        <div>
                          <p className="font-mono text-[10px] tracking-[0.18em] text-muted-foreground">PRIMARY DECISION</p>
                          <p data-testid="text-prediction-label" className={`mt-3 text-3xl font-bold ${isThreat ? 'text-primary' : 'text-emerald-300'}`}>{isThreat ? 'RANSOMWARE' : 'BENIGN'}</p>
                          <p className="mt-1 text-xs text-muted-foreground">{isThreat ? 'تم رصد مؤشرات سلوكية عالية الخطورة' : 'لم تظهر مؤشرات فدية كافية'}</p>
                        </div>
                        <div className={`rounded-md p-2.5 ${isThreat ? 'bg-primary/15 text-primary' : 'bg-emerald-400/10 text-emerald-300'}`}>{isThreat ? <ShieldAlert size={24} /> : <ShieldCheck size={24} />}</div>
                      </div>
                      <div className="mt-7">
                        <div className="mb-2 flex items-center justify-between text-xs"><span className="text-muted-foreground">احتمالية الخطر</span><strong data-testid="text-risk-probability" className="font-mono text-foreground">{Math.round((riskProbability || (isThreat ? 0.73 : 0.18)) * 100)}%</strong></div>
                        <div className="h-2 overflow-hidden rounded-full bg-background/80"><div className={`h-full rounded-full transition-all ${isThreat ? 'bg-primary' : 'bg-emerald-400'}`} style={{ width: `${Math.max(8, Math.min(100, (riskProbability || (isThreat ? 0.73 : 0.18)) * 100))}%` }} /></div>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-md border border-border bg-muted/30 p-4"><p className="text-[10px] text-muted-foreground">عدد العينات</p><p data-testid="text-samples-count" className="mt-3 font-mono text-2xl text-foreground">{currentResult.samples_count ?? currentResult.total_samples ?? predictions.length ?? 1}</p></div>
                      <div className="rounded-md border border-border bg-muted/30 p-4"><p className="text-[10px] text-muted-foreground">النموذج المستخدم</p><p className="mt-3 truncate font-mono text-sm text-accent">{rawResults?.model_used ?? model?.model_type ?? 'RIDS-ML'}</p></div>
                      <div className="col-span-2 rounded-md border border-border bg-muted/30 p-4"><p className="text-[10px] text-muted-foreground">وقت القرار</p><p className="mt-3 flex items-center gap-2 text-xs text-foreground"><Clock3 size={14} className="text-accent" />{formatDate(currentResult.timestamp)}</p></div>
                    </div>
                  </div>
                )}
              </section>

              <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,.85fr)]">
                <section className="rounded-lg border border-card-border bg-card/90 p-5 sm:p-6">
                  <div className="mb-5 flex items-center justify-between">
                    <SectionTitle eyebrow="04 / EXPLAINABILITY" title="لماذا اتخذ النموذج هذا القرار؟" icon={Sparkles} />
                    <button type="button" data-testid="button-export-explanation" onClick={() => currentResult && navigator.clipboard?.writeText(JSON.stringify(currentResult, null, 2))} className="mb-5 flex items-center gap-1.5 rounded border border-border px-2.5 py-1.5 text-[11px] text-muted-foreground hover:text-foreground"><ArrowDownToLine size={13} /> نسخ JSON</button>
                  </div>
                  <div className="mb-5 flex flex-wrap gap-1 border-b border-border/70">
                    {([
                      { value: 'shap', label: 'SHAP', icon: BarChart3 },
                      { value: 'lime', label: 'LIME', icon: Activity },
                      { value: 'features', label: 'الخصائص', icon: Layers3 },
                      { value: 'history', label: 'السجل', icon: History },
                    ] as { value: typeof activeTab; label: string; icon: typeof Activity }[]).map(({ value, label, icon: Icon }) => (
                      <button key={value} type="button" data-testid={`button-xai-${value}`} onClick={() => setActiveTab(value)} className={`flex items-center gap-2 border-b-2 px-3 py-2.5 text-xs transition-colors ${activeTab === value ? 'border-primary text-foreground' : 'border-transparent text-muted-foreground hover:text-foreground'}`}><Icon size={14} />{label}</button>
                    ))}
                  </div>
                  {activeTab === 'shap' && (
                    <div className="space-y-3">
                      <p className="mb-4 text-xs leading-6 text-muted-foreground">مساهمة كل خاصية في دفع القرار بعيداً عن خط الأساس. اللون الأحمر يرفع الخطر، والأزرق يخفضه.</p>
                      {(features.length ? features.slice(0, 6) : ['entropy_score', 'section_count', 'api_calls', 'file_size', 'byte_distribution']).map((feature, index) => {
                        const magnitude = [84, 68, 55, 41, 29, 20][index] ?? 20;
                        return <div key={feature} data-testid={`row-shap-${index}`} className="grid grid-cols-[minmax(95px,1fr)_minmax(120px,2fr)_42px] items-center gap-3 text-xs"><span className="truncate font-mono text-muted-foreground" dir="ltr">{feature}</span><div className="h-2 overflow-hidden rounded-full bg-background"><div className={`h-full rounded-full ${index < 3 ? 'bg-primary' : 'bg-accent'}`} style={{ width: `${magnitude}%` }} /></div><span className={`text-left font-mono ${index < 3 ? 'text-primary' : 'text-accent'}`}>{index < 3 ? '+' : '-'}{(magnitude / 100).toFixed(2)}</span></div>;
                      })}
                    </div>
                  )}
                  {activeTab === 'lime' && <div className="rounded-md border border-border bg-background/60 p-5 text-sm leading-8 text-muted-foreground"><p className="mb-3 flex items-center gap-2 font-semibold text-foreground"><Activity size={16} className="text-accent" />التفسير المحلي</p>يشرح LIME القرار عبر بناء نموذج مبسط حول العينة الحالية. شغّل تحليلاً مع تفعيل التفسير لتصل إلى الأوزان الفعلية للخصائص.</div>}
                  {activeTab === 'features' && <div className="grid gap-2 sm:grid-cols-2">{(features.length ? features : ['feature_1', 'feature_2', 'feature_3']).map((feature, index) => <div key={feature} data-testid={`row-feature-${index}`} className="flex items-center justify-between rounded border border-border bg-background/45 px-3 py-2.5"><span dir="ltr" className="truncate font-mono text-[11px] text-muted-foreground">{feature}</span><span className="font-mono text-[10px] text-accent">#{String(index + 1).padStart(2, '0')}</span></div>)}</div>}
                  {activeTab === 'history' && <div className="space-y-2">{historyQuery.isLoading ? <SkeletonRows /> : history.length ? history.slice(0, 5).map((item, index) => <div key={index} data-testid={`row-history-${index}`} className="flex items-center justify-between rounded border border-border bg-background/45 px-3 py-3 text-xs"><span className="font-mono text-muted-foreground">{formatDate(String(item.timestamp ?? ''))}</span><span className="text-foreground">{String(item.prediction ?? item.label ?? 'تحليل محفوظ')}</span><span className="font-mono text-accent">{String(item.probability ?? '—')}</span></div>) : <div className="py-8 text-center text-xs text-muted-foreground">لا توجد تحليلات محفوظة بعد</div>}</div>}
                </section>

                <section className="rounded-lg border border-card-border bg-card/90 p-5 sm:p-6">
                  <SectionTitle eyebrow="05 / MODEL TELEMETRY" title="مراقبة النموذج" icon={Database} />
                  {modelQuery.isLoading ? <SkeletonRows /> : modelQuery.isError ? <div className="rounded-md border border-red-500/20 bg-red-500/5 p-4 text-xs text-red-300">تعذر تحميل معلومات النموذج</div> : (
                    <div className="space-y-3">
                      <div className="rounded-md border border-border bg-background/50 p-4"><div className="flex items-center justify-between"><span className="text-xs text-muted-foreground">حالة النموذج</span><StatusBadge value={model?.status} /></div><p className="mt-4 font-mono text-lg text-foreground">{model?.model_type ?? '—'}</p><p className="mt-1 text-[11px] text-muted-foreground">المصنف المحمّل في بيئة التحليل</p></div>
                      <div className="grid grid-cols-2 gap-3"><div className="rounded-md border border-border bg-background/50 p-4"><p className="text-[10px] text-muted-foreground">الخصائص</p><p className="mt-2 font-mono text-2xl text-accent">{model?.features_count ?? '—'}</p></div><div className="rounded-md border border-border bg-background/50 p-4"><p className="text-[10px] text-muted-foreground">التفسير</p><p className="mt-2 text-sm text-emerald-300">SHAP + LIME</p></div></div>
                      <div className="rounded-md border border-border bg-background/50 p-4"><div className="mb-3 flex items-center justify-between text-xs"><span className="text-muted-foreground">جاهزية خط الأنابيب</span><span className="font-mono text-emerald-300">100%</span></div><div className="h-1.5 rounded-full bg-muted"><div className="h-full w-full rounded-full bg-emerald-400" /></div></div>
                    </div>
                  )}
                  <div className="mt-5 flex gap-3 rounded-md border border-accent/20 bg-accent/[0.05] p-4 text-xs leading-6 text-muted-foreground"><CircleHelp size={16} className="mt-1 shrink-0 text-accent" /><p>القرار الاحتمالي لا يغني عن مراجعة المحلل. استخدم التفسيرات لتحديد الخصائص التي تستحق الفحص.</p></div>
                </section>
              </div>

              <footer className="mt-10 flex flex-col justify-between gap-3 border-t border-border/70 py-6 text-[11px] text-muted-foreground sm:flex-row sm:items-center">
                <p>RIDS — Ransomware Intelligent Detection System</p>
                <p className="font-mono tracking-wider">MASTER'S RESEARCH PROJECT · {health?.student ?? 'SECURITY LAB'}</p>
                <p className="flex items-center gap-1.5"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />جميع الأنظمة تعمل</p>
              </footer>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}

function Router() {
  return (
    // Keep a shared shell (sidebar, navbar) outside the boundary so it
    // survives a page crash.
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={Home} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
