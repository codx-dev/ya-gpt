import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import exampleUrl from '../../assets/model-shakespeare-small-30000-bs32-nembd32-head8-layer4.bin?url';
import { ModelLoader } from './model/loader';
import type { LoadedModel } from './model/loader';
import type { Backend } from './model/types';
import { detectBackends, runtime } from './runtime';
import type { BackendSupport } from './runtime';
import { downloadError, downloadModel, supportsTemporaryStorage, TemporaryStorage } from './storage';
import type { DownloadProgress } from './storage';
import { GeneratePanel } from './ui/GeneratePanel';
import { TrainPanel } from './ui/TrainPanel';
import { InspectPanel } from './ui/InspectPanel';
import { Field, formatBytes } from './ui/shared';

const tabs = ['Generate', 'Train', 'Inspect'] as const;
const exampleName = 'model-shakespeare-small-30000-bs32-nembd32-head8-layer4.bin';

export default function App() {
  const [tab, setTab] = useState<typeof tabs[number]>('Generate');
  const [model, setModel] = useState<LoadedModel>();
  const [error, setError] = useState('');
  const [warning, setWarning] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress>();
  const [url, setUrl] = useState('');
  const [source, setSource] = useState<'file' | 'url'>('file');
  const [backend, setBackend] = useState<Backend>('webgpu');
  const [support, setSupport] = useState<BackendSupport>();
  const [trainingAvailable, setTrainingAvailable] = useState(false);
  const computing = useSyncExternalStore(runtime.subscribe, () => runtime.busy);
  const [wasmStatus, setWasmStatus] = useState('Loading WebAssembly…');
  const [storage] = useState(() => new TemporaryStorage());
  const [loader] = useState(() => new ModelLoader(runtime.inspectModel, (error) => setWarning(`Temporary file cleanup failed: ${downloadError(error)} Reload to retry abandoned-session cleanup.`)));
  const request = useRef(0);
  const canDownload = supportsTemporaryStorage();

  useEffect(() => {
    let alive = true;
    if (supportsTemporaryStorage()) void storage.prepare().catch((error: unknown) => {
      if (alive) setWarning(`Temporary storage: ${downloadError(error)}`);
    });
    void runtime.initialize().then((capabilities) => { if (alive) { setWasmStatus(capabilities.generate ? 'WebGPU runtime ready' : capabilities.error || 'Generation unavailable'); setTrainingAvailable(capabilities.train); } })
      .catch((error: unknown) => { if (alive) setWasmStatus(`WASM failed: ${downloadError(error)}`); });
    void detectBackends().then((capabilities) => {
      if (!alive) return;
      setSupport(capabilities); setBackend('webgpu');
    });
    // Closed-tab leftovers are cleaned under locks on the next startup.
    return () => { alive = false; loader.cancel(); void loader.clear().then(() => storage.dispose()).catch(() => {}); };
  }, [loader, storage]);

  async function load(acquire: Parameters<ModelLoader['load']>[0], name: string, source: string) {
    const id = ++request.current;
    setBusy(true); setError(''); setProgress(undefined);
    try {
      const next = await loader.load(acquire, name, source);
      if (id === request.current && next) { setModel(next); setTab('Inspect'); }
    } catch (error) { if (id === request.current) setError(downloadError(error)); }
    finally { if (id === request.current) { setBusy(false); setProgress(undefined); } }
  }

  function fromUrl(address: string, name: string, source: string) {
    const id = request.current + 1;
    void load((signal) => downloadModel(address, storage, signal, (value) => {
      if (id === request.current && !signal.aborted) setProgress(value);
    }), name, source);
  }

  function cancel() { ++request.current; loader.cancel(); setBusy(false); setProgress(undefined); }
  async function clear() {
    cancel(); setModel(undefined); setError('');
    try { await loader.clear(); } catch (error) { setWarning(downloadError(error)); }
  }

  async function useTrainedModel(file: File) {
    const id = ++request.current;
    setBusy(true); setError(''); setProgress(undefined);
    try {
      const next = await loader.load(async () => ({ file }), file.name, 'Trained in browser');
      if (id === request.current && next) { setModel(next); setTab('Generate'); }
    } catch (error) { if (id === request.current) { setError(downloadError(error)); throw error; } }
    finally { if (id === request.current) setBusy(false); }
  }

  return <div className="app-shell">
    <main>
      <div className="workspace">
        <aside aria-label="Model and backend" className="sidebar">
          <section className="card model-card"><div className="card-heading"><h2>Model</h2><span className="tag">.BIN</span></div><p className="helper">Load a serialized YA-GPT model.</p>
            <p className="helper"><a href="https://github.com/codx-dev/ya-gpt/tree/main/assets" target="_blank" rel="noopener noreferrer" aria-label="Browse pretrained models (opens in a new tab)">Browse pretrained models <span aria-hidden="true">↗</span></a></p>
            <div className="segmented"><button aria-pressed={source === 'file'} onClick={() => setSource('file')}>Local file</button><button aria-pressed={source === 'url'} onClick={() => setSource('url')}>From URL</button></div>
            {source === 'file' ? <label className="file-picker"><span className="file-icon" aria-hidden="true">↥</span><strong>{model ? model.name : 'Choose a model file'}</strong><small>{model ? 'Replace model' : 'Binary model from your disk'}</small><input aria-label="Model file" type="file" accept=".bin,application/octet-stream" onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void load(async () => ({ file }), file.name, 'Local file');
              event.target.value = '';
            }} /></label> : <form className="url-form" onSubmit={(event) => {
              event.preventDefault();
              try {
                const parsed = new URL(url);
                const name = decodeURIComponent(parsed.pathname.split('/').pop() || 'downloaded-model.bin');
                fromUrl(parsed.href, name, parsed.origin);
              } catch { setError('Enter a valid HTTP or HTTPS model URL.'); }
            }}><Field label="Model URL"><input type="url" required placeholder="https://…/model.bin" value={url} onChange={(event) => setUrl(event.target.value)} /></Field><button className="full-width" disabled={!canDownload}>Download model</button>
              <p className="helper">Saved to temporary browser storage. The server must allow CORS.</p></form>}
            <div className="divider"><span>OR START WITH AN EXAMPLE</span></div>
            <button className="example-button" disabled={!canDownload} onClick={() => fromUrl(new URL(exampleUrl, location.href).href, exampleName, 'Bundled example')}><span><strong>Load Shakespeare example</strong><small>55.8K parameters · 545.6 KB</small></span><span aria-hidden="true">↗</span></button>
            {!canDownload && <p className="error">Temporary disk storage is unavailable. Use HTTPS or localhost in a browser with OPFS and Web Locks. Local files remain available.</p>}
            {busy && <div className="download-progress" role="status"><div><span>{progress ? `Downloaded ${formatBytes(progress.received)}${progress.total ? ` / ${formatBytes(progress.total)}` : ''}` : 'Reading model…'}</span><button className="text-button" onClick={cancel}>Cancel</button></div><progress aria-label="Model loading progress" value={progress?.total ? progress.received : undefined} max={progress?.total ?? 1} /></div>}
            {error && <p role="alert" className="error">{error}</p>}
            {warning && <p role="alert" className="error">{warning}</p>}
            {model && <div className="selected-model"><small>{formatBytes(model.file.size)} · {model.source}</small><button className="text-button" onClick={() => void clear()}>Unload model</button></div>}
          </section>
          <section className="card backend-card"><div className="card-heading"><h2>Compute</h2><span className="status-dot" /></div><Field label="Backend"><select value={backend} onChange={(event) => setBackend(event.target.value as Backend)}>
            <option value="webgpu" disabled={!support?.webgpu}>WebGPU {!support ? '(checking…)' : support.webgpu ? '(ready)' : '(unavailable)'}</option>
            <option value="simd" disabled>CPU · WASM SIMD (not implemented)</option>
          </select></Field>
            <p className="helper"><a href="https://webgpureport.org/" target="_blank" rel="noopener noreferrer" aria-label="Troubleshoot WebGPU (opens in a new tab)">Troubleshoot WebGPU <span aria-hidden="true">↗</span></a></p>
            <div className="runtime-status" role="status"><span aria-hidden="true">◈</span>{wasmStatus}</div></section>
          <p className="sidebar-note">Built for experimentation.<br />No account. No uploads. No server compute.</p>
        </aside>
        <section className="card workbench" aria-label="Model workspace">
          <div className="workbench-header">
            <div className="tabs" role="tablist" aria-label="Model operations">{tabs.map((name, index) => <button key={name} id={`tab-${name}`} role="tab" aria-selected={tab === name} aria-controls={`panel-${name}`} tabIndex={tab === name ? 0 : -1} onClick={() => setTab(name)} onKeyDown={(event) => {
              const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : undefined;
              if (next !== undefined) { event.preventDefault(); setTab(tabs[next]); document.getElementById(`tab-${tabs[next]}`)?.focus(); }
            }}><span className="tab-number">0{index + 1}</span>{name}</button>)}</div>
            <a className="github-link" href="https://github.com/codx-dev/ya-gpt" target="_blank" rel="noopener noreferrer" aria-label="View ya-gpt on GitHub (opens in a new tab)" title="View ya-gpt on GitHub">
              <svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" focusable="false">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.65 7.65 0 0 1 2-.27c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
              </svg>
            </a>
          </div>
          <div id="panel-Generate" role="tabpanel" aria-labelledby="tab-Generate" hidden={tab !== 'Generate'} className="panel"><GeneratePanel backend={backend} model={model?.file} available={!!support?.webgpu} blocked={computing} /></div>
          <div id="panel-Train" role="tabpanel" aria-labelledby="tab-Train" hidden={tab !== 'Train'} className="panel"><TrainPanel backend={backend} available={trainingAvailable} blocked={computing} onUseModel={useTrainedModel} /></div>
          <div id="panel-Inspect" role="tabpanel" aria-labelledby="tab-Inspect" hidden={tab !== 'Inspect'} className="panel"><InspectPanel model={model} /></div>
        </section>
      </div>
    </main>
  </div>;
}
