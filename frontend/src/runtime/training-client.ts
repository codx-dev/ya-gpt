import type { TrainOptions, TrainingJob, TrainingProgress, TrainingResult } from '../model/types';
import type { TrainingReply, TrainingRequest } from './training-worker';

export function startTrainingWorker(text: string, options: TrainOptions, report?: (value: TrainingProgress) => void): TrainingJob {
  const worker = new Worker(new URL('./training-worker.ts', import.meta.url), { type: 'module' });
  let settled = false;
  const result = new Promise<TrainingResult>((resolve, reject) => {
    const finish = () => { settled = true; worker.terminate(); };
    worker.onmessage = ({ data }: MessageEvent<TrainingReply>) => {
      if (settled) return;
      if (data.type === 'progress') {
        try { report?.(data.progress); }
        catch (error) { finish(); reject(error); }
        return;
      }
      finish();
      if (data.type === 'result') resolve(data.result);
      else { const error = new Error(data.message); error.name = data.name; reject(error); }
    };
    worker.onerror = (event) => { finish(); reject(new Error(event.message || 'Training worker failed.')); };
    worker.onmessageerror = () => { finish(); reject(new Error('Unable to read training worker response.')); };
    try { worker.postMessage({ type: 'start', text, options } satisfies TrainingRequest); }
    catch (error) { finish(); reject(error); }
  });
  return { result, stop() { if (!settled) worker.postMessage({ type: 'stop' } satisfies TrainingRequest); } };
}
