import type { ReactNode } from 'react';

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <label className="field"><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>;
}

export function NumberField({ label, value, onChange, min = 1, max, step = 1 }: {
  label: string; value: number; onChange(value: number): void; min?: number; max?: number; step?: number;
}) {
  return <Field label={label}><input type="number" min={min} max={max} step={step}
    value={Number.isNaN(value) ? '' : value} onChange={(event) => onChange(event.currentTarget.valueAsNumber)} /></Field>;
}

export function SeedField({ value, onChange }: { value: string; onChange(value: string): void }) {
  return <Field label="Random seed" hint="Optional · leave empty for a random seed">
    <input inputMode="numeric" placeholder="Random" value={value} onChange={(event) => onChange(event.target.value)} />
  </Field>;
}

export function Validation({ errors }: { errors: string[] }) {
  return errors.length > 0 && <ul className="validation" aria-live="polite">{errors.map((error) => <li key={error}>{error}</li>)}</ul>;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
