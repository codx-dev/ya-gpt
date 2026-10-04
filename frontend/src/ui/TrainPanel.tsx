import { useEffect, useRef, useState } from 'react';
import type { Backend, ModelConfig, TrainingJob, TrainingProgress, TrainingResult } from '../model/types';
import { presets } from '../model/validation';
import type { Preset } from '../model/validation';
import { Field, NumberField, SeedField, Validation } from './shared';
import { runtime } from '../runtime';
import { trainingErrors } from '../runtime/training-bridge';
import { authorError } from '../runtime/serialization-bridge';

export function TrainPanel({ backend, available, blocked, onUseModel }: { backend: Backend; available: boolean; blocked: boolean; onUseModel(file: File): Promise<void> }) {
  const [preset, setPreset] = useState<Preset | 'custom'>('tiny');
  const [config, setConfig] = useState<ModelConfig>(presets.tiny);
  const [batchSize, setBatchSize] = useState(8);
  const [iterations, setIterations] = useState(1000);
  const [learningRate, setLearningRate] = useState(0.0003);
  const [weightDecay, setWeightDecay] = useState(0.01);
  const [seed, setSeed] = useState('');
  const [reportLoss, setReportLoss] = useState(false);
  const [datasetMode, setDatasetMode] = useState<'paste' | 'file'>('paste');
  const [text, setText] = useState('');
  const [fileText, setFileText] = useState('');
  const [datasetName, setDatasetName] = useState('');
  const [datasetError, setDatasetError] = useState('');
  const [reading, setReading] = useState(false);
  const readId = useRef(0);
  const job = useRef<TrainingJob | undefined>(undefined);
  const mounted = useRef(true);
  const [running, setRunning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [author, setAuthor] = useState('');
  const [exporting, setExporting] = useState<'download' | 'use'>();
  const [exportError, setExportError] = useState('');
  const exportActive = useRef(false);
  const prepared = useRef<{ original: File; author: string; file: File } | undefined>(undefined);
  const [progress, setProgress] = useState<TrainingProgress>();
  const [trained, setTrained] = useState<{ result: TrainingResult; file: File }>();
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; job.current?.stop(); ++readId.current; };
  }, []);

  const dataset = datasetMode === 'paste' ? text : fileText;
  const options = { backend, model: config, batch_size: batchSize, iterations,
    learning_rate: learningRate, weight_decay: weightDecay, seed, report_loss_per_step: reportLoss };
  const errors = trainingErrors(options, dataset);
  const invalidAuthor = authorError(author);

  async function start() {
    if (job.current || exportActive.current || blocked || !available || reading || errors.length) return;
    setRunning(true); setStopping(false); setError(''); setNotice('');
    setProgress({ phase: 'initializing', completed: 0, total: iterations, elapsedMs: 0 });
    try {
      const next = runtime.train(dataset, options, (value) => { if (mounted.current) setProgress(value); });
      job.current = next;
      const result = await next.result;
      const file = new File([result.bytes], `ya-gpt-${result.completed}-steps.bin`, { type: 'application/octet-stream' });
      const metadata = await runtime.inspectModel(file);
      if (mounted.current) {
        setTrained({ result, file });
        setAuthor(metadata.header.authors); setExportError(''); prepared.current = undefined;
        setNotice(result.stopped ? `Stopped after ${result.completed} completed steps. Your model is ready.` : 'Training complete. Your model is ready.');
      }
    } catch (cause) {
      if (mounted.current) {
        if (cause instanceof Error && cause.name === 'AbortError') setNotice('Stopped before the first completed step. No new model was created.');
        else setError(cause instanceof Error ? cause.message : String(cause));
      }
    } finally {
      job.current = undefined;
      if (mounted.current) { setRunning(false); setStopping(false); }
    }
  }

  async function exportModel(action: 'download' | 'use') {
    if (!trained || running || blocked || exportActive.current || invalidAuthor) return;
    exportActive.current = true; setExporting(action); setExportError('');
    try {
      const cached = prepared.current;
      let file: File;
      if (cached?.original === trained.file && cached.author === author) file = cached.file;
      else {
        const bytes = await runtime.serializeModel(trained.file, author);
        if (!mounted.current) return;
        file = new File([bytes], trained.file.name, { type: trained.file.type });
        prepared.current = { original: trained.file, author, file };
      }
      if (action === 'use') await onUseModel(file);
      else {
        const url = URL.createObjectURL(file);
        try {
          const link = document.createElement('a'); link.href = url; link.download = file.name;
          document.body.append(link); link.click(); link.remove();
        } finally { setTimeout(() => URL.revokeObjectURL(url), 1000); }
      }
    } catch (cause) {
      if (mounted.current) setExportError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      exportActive.current = false;
      if (mounted.current) setExporting(undefined);
    }
  }

  function applyPreset(value: Preset) {
    const { batch_size, iterations, ...model } = presets[value];
    setPreset(value); setConfig(model); setBatchSize(batch_size); setIterations(iterations);
  }
  function updateConfig(key: keyof ModelConfig, value: number) {
    setPreset('custom'); setConfig((config) => ({ ...config, [key]: value }));
  }
  async function readDataset(file?: File) {
    if (!file) return;
    const id = ++readId.current;
    setReading(true); setFileText(''); setDatasetError(''); setDatasetName(file.name);
    try {
      const content = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer());
      if (id === readId.current) setFileText(content);
    } catch {
      if (id === readId.current) setDatasetError('Unable to read this dataset. Select a UTF-8 text file.');
    } finally { if (id === readId.current) setReading(false); }
  }

  return <>
    <div className="panel-heading"><span className="eyebrow">02 / TRAIN</span></div>
    <fieldset className="training-fields" disabled={running}>
    <div className="section-label"><h3>Training data</h3><span>90% train / 10% validation</span></div>
    <div className="segmented" aria-label="Dataset source">
      <button aria-pressed={datasetMode === 'paste'} onClick={() => setDatasetMode('paste')}>Paste text</button>
      <button aria-pressed={datasetMode === 'file'} onClick={() => setDatasetMode('file')}>Text file</button>
    </div>
    {datasetMode === 'paste' ? <Field label="Training text"><textarea rows={5} placeholder="Paste the text your model will learn from…" value={text} onChange={(event) => setText(event.target.value)} /></Field>
      : <Field label="Dataset file" hint={reading ? 'Reading dataset…' : datasetName || 'Choose a UTF-8 .txt file'}>
        <span className="dataset-file-picker"><span aria-hidden="true">Choose file</span>
          <input aria-label="Dataset file" type="file" accept=".txt,text/plain" onChange={(event) => { void readDataset(event.target.files?.[0]); event.target.value = ''; }} />
        </span>
      </Field>}
    {datasetError && datasetMode === 'file' && <p role="alert" className="error">{datasetError}</p>}
    <div className="section-label"><h3>Model architecture</h3><span>Preset values are editable</span></div>
    <Field label="Preset"><select value={preset} onChange={(event) => applyPreset(event.target.value as Preset)}>
      {Object.keys(presets).map((name) => <option key={name} value={name}>{name[0].toUpperCase() + name.slice(1)}</option>)}
      {preset === 'custom' && <option value="custom" disabled>Custom</option>}
    </select></Field>
    <div className="field-grid architecture-grid">
      <NumberField label="Context length" value={config.block_size} onChange={(v) => updateConfig('block_size', v)} />
      <NumberField label="Embedding width" value={config.n_embd} onChange={(v) => updateConfig('n_embd', v)} />
      <NumberField label="Attention heads" value={config.n_head} onChange={(v) => updateConfig('n_head', v)} />
      <NumberField label="Transformer layers" value={config.n_layer} onChange={(v) => updateConfig('n_layer', v)} />
      <NumberField label="Dropout" value={config.dropout} onChange={(v) => updateConfig('dropout', v)} min={0} max={0.999999} step={0.01} />
    </div>
    <div className="section-label"><h3>Optimization</h3><span>AdamW</span></div>
    <div className="field-grid architecture-grid">
      <NumberField label="Batch size" value={batchSize} onChange={(v) => { setPreset('custom'); setBatchSize(v); }} />
      <NumberField label="Iterations" value={iterations} onChange={(v) => { setPreset('custom'); setIterations(v); }} />
      <NumberField label="Learning rate" value={learningRate} onChange={setLearningRate} min={0} max={1} step={0.0001} />
      <NumberField label="Weight decay" value={weightDecay} onChange={setWeightDecay} min={0} max={1} step={0.001} />
      <SeedField value={seed} onChange={setSeed} />
    </div>
    <label className="checkbox"><input type="checkbox" checked={reportLoss} onChange={(event) => setReportLoss(event.target.checked)} />Report loss after every step <small>Adds overhead</small></label>
    <Validation errors={errors} />
    </fieldset>
    {!available && <p className="helper">Training requires WebGPU and web workers in a supported browser.</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {notice && <p className="helper" role="status">{notice}</p>}
    <div className="actions">
      <button className="primary" disabled={!available || blocked || running || !!exporting || reading || errors.length > 0} onClick={() => void start()}>Start training <span aria-hidden="true">↗</span></button>
      {running && <button disabled={stopping || progress?.phase === 'exporting' || progress?.phase === 'complete'} onClick={() => { setStopping(true); job.current?.stop(); }}>{stopping ? 'Stopping…' : 'Stop training'}</button>}
    </div>
    {progress && <div className="training-progress" aria-busy={running}>
      <p role="status">{error ? 'Training failed' : stopping ? 'Stopping after the current step, then evaluating and exporting…' : notice || ({ initializing: 'Initializing training…', training: 'Training…', evaluating: 'Evaluating loss…', exporting: 'Exporting model…', complete: 'Complete' }[progress.phase])}</p>
      <progress aria-label="Training progress" value={progress.completed} max={progress.total} />
      <p className="helper">{progress.completed} / {progress.total} steps · {(progress.elapsedMs / 1000).toFixed(1)} seconds</p>
      <dl className="training-losses"><div><dt>Train loss</dt><dd>{progress.trainLoss?.toFixed(4) ?? '—'}</dd></div><div><dt>Validation loss</dt><dd>{progress.validationLoss?.toFixed(4) ?? '—'}</dd></div></dl>
    </div>}
    {trained && <div className="trained-model" aria-busy={!!exporting}>
      <p className="helper">Available model: {trained.file.name} · seed {trained.result.seed}. Results remain available until you close or reload this page.</p>
      <Field label="Author" hint="Leave empty to use the crate authors.">
        <input aria-label="Author" type="text" value={author} disabled={running || !!exporting}
          onChange={(event) => { setAuthor(event.target.value); setExportError(''); }} />
      </Field>
      {(invalidAuthor || exportError) && <p className="error" role="alert">{invalidAuthor || exportError}</p>}
      <div className="actions">
        <button disabled={blocked || running || !!exporting || !!invalidAuthor} onClick={() => void exportModel('download')}>{exporting === 'download' ? 'Preparing model…' : 'Download trained model'}</button>
        <button disabled={blocked || running || !!exporting || !!invalidAuthor} onClick={() => void exportModel('use')}>{exporting === 'use' ? 'Loading model…' : 'Use model'}</button>
      </div>
      {!running && trained.result.validationLoss > 1.88 && <p className="training-warning" role="status">
        <strong>Warning:</strong> This model’s validation loss is {trained.result.validationLoss.toFixed(4)} (above 1.88). It may produce gibberish text.
      </p>}
    </div>}
  </>;
}
