import type { LoadedModel } from '../model/loader';
import { formatBytes } from './shared';

export function InspectPanel({ model }: { model?: LoadedModel }) {
  if (!model) return <>
    <div className="panel-heading"><span className="eyebrow">03 / INSPECT</span></div>
    <div className="empty-inspection"><span aria-hidden="true">{ '{ }' }</span><h3>No model selected</h3><p>Choose a local binary, download from a URL, or load the Shakespeare example to inspect its metadata.</p></div>
  </>;

  const { header, config, vocab_size } = model.metadata;
  return <>
    <div className="panel-heading"><span className="eyebrow">03 / INSPECT</span><span className="tag success">METADATA READ</span></div>
    <p className="section-description">{model.name}</p>
    <div className="metrics">
      <div><span>PARAMETERS</span><strong>{header.params.toLocaleString('en-US')}</strong></div>
      <div><span>TRAIN LOSS</span><strong>{header.train_loss.toFixed(4)}</strong></div>
      <div><span>VALIDATION LOSS</span><strong>{header.validation_loss.toFixed(4)}</strong></div>
    </div>
    <div className="inspection-grid">
      <div><h3>Architecture</h3><dl>
        <div><dt>Context length</dt><dd>{config.block_size}</dd></div>
        <div><dt>Embedding width</dt><dd>{config.n_embd}</dd></div>
        <div><dt>Attention heads</dt><dd>{config.n_head}</dd></div>
        <div><dt>Transformer layers</dt><dd>{config.n_layer}</dd></div>
        <div><dt>Dropout</dt><dd>{Number(config.dropout.toFixed(6))}</dd></div>
        <div><dt>Vocabulary</dt><dd>{vocab_size} characters</dd></div>
      </dl></div>
      <div><h3>Model file</h3><dl>
        <div><dt>Name</dt><dd>{header.name}</dd></div>
        <div><dt>Format version</dt><dd>{header.major}</dd></div>
        <div><dt>Precision</dt><dd>{header.fp}</dd></div>
        <div><dt>File size</dt><dd>{formatBytes(model.file.size)}</dd></div>
        <div><dt>Source</dt><dd>{model.source}</dd></div>
        <div><dt>Authors</dt><dd>{header.authors}</dd></div>
      </dl></div>
    </div>
    <p className="helper">Metadata inspection reads the binary header and configuration. Tensor data has not been validated or loaded into a compute backend.</p>
    <details className="json-details"><summary>View metadata as JSON</summary><pre>{JSON.stringify(model.metadata, null, 2)}</pre></details>
  </>;
}
