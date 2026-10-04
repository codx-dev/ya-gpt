import { useEffect, useRef, useState } from 'react';
import type { Backend } from '../model/types';
import { generateErrors } from '../model/validation';
import { Field, NumberField, SeedField, Validation } from './shared';
import { runtime } from '../runtime';

export function GeneratePanel({ backend, model, available, blocked = false }: { backend: Backend; model?: File; available: boolean; blocked?: boolean }) {
  const [prompt, setPrompt] = useState('ROMEO:');
  const [numTokens, setNumTokens] = useState(250);
  const [seed, setSeed] = useState('');
  const [busy, setBusy] = useState(false);
  const [output, setOutput] = useState('');
  const [generated, setGenerated] = useState(0);
  const [error, setError] = useState('');
  const request = useRef(0);
  const running = useRef(false);
  useEffect(() => {
    ++request.current; setOutput(''); setGenerated(0); setError('');
    return () => { ++request.current; };
  }, [model]);
  const errors = generateErrors({ backend, prompt, num_tokens: numTokens, seed });

  async function generate() {
    if (blocked || running.current) return;
    if (!model) {
      setError('No model selected. Click “Load Shakespeare example” in the sidebar to get started.');
      return;
    }
    if (!available || backend !== 'webgpu' || errors.length) return;
    const id = ++request.current;
    running.current = true; setBusy(true); setError('');
    try {
      const text = await runtime.generate(model, { backend, prompt, num_tokens: numTokens, seed });
      if (id === request.current) { setOutput(text); setGenerated(numTokens); }
    } catch (cause) {
      if (id === request.current) setError(cause instanceof Error ? cause.message : String(cause));
    } finally { running.current = false; setBusy(false); }
  }

  return <>
    <div className="panel-heading"><span className="eyebrow">01 / GENERATE</span></div>
    <div className="generation-grid">
      <div className="form-stack">
        <Field label="Prompt"><textarea rows={8} value={prompt} onChange={(event) => setPrompt(event.target.value)} spellCheck={false} /></Field>
        <div className="field-grid"><NumberField label="New tokens" value={numTokens} onChange={setNumTokens} /><SeedField value={seed} onChange={setSeed} /></div>
        <Validation errors={errors} />
        {!model && !error && <p className="helper">Select a model from the sidebar to prepare a run.</p>}
        {!available && <p className="helper">Generation requires an available WebGPU runtime.</p>}
        {error && <p role="alert" className="error">{error}</p>}
        <button className="primary" disabled={busy || blocked || (!!model && (!available || backend !== 'webgpu' || errors.length > 0))} onClick={() => void generate()}>{busy ? 'Generating…' : 'Generate text'} {busy ? <span className="loading-spinner" aria-hidden="true" /> : <span aria-hidden="true">↗</span>}</button>
      </div>
      <div className="output-panel"><div className="output-title"><span>OUTPUT</span><span>{generated || '—'} tokens</span></div>
        <div role="status">{busy && <div className="generation-status"><span className="loading-spinner" aria-hidden="true" />Generating text…</div>}</div>
        <div className="output-content" aria-busy={busy}>
          {output ? <pre className="generated-output" aria-label="Generated text">{output}</pre> : <div className="empty-output"><span className="text-cursor" aria-hidden="true">▌</span><p>{busy ? 'Generating your continuation…' : 'The next word starts here.'}</p><small>Generated text will appear here.</small></div>}
        </div>
      </div>
    </div>
  </>;
}
