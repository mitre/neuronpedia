import { CLTGraph } from './graph-types';

export async function computeGraphScoresInWorker(
  graph: CLTGraph,
  pinnedIds: string[] = [],
): Promise<{ replacementScore: number; completenessScore: number }> {
  const { nodes, links } = graph;

  // Check if Worker is available
  if (typeof Worker === 'undefined') {
    throw new Error('Web Workers are not supported in this environment');
  }

  // ensure we have multiple cores
  if (navigator.hardwareConcurrency <= 1) {
    throw new Error('This browser does not support multiple cores');
  }

  const worker = new Worker(new URL('./score.worker.ts', import.meta.url), { type: 'module' });

  return new Promise((resolve, reject) => {
    function handleMessage(ev: MessageEvent) {
      const data = ev.data as any;
      // @ts-ignore
      worker.removeEventListener('message', handleMessage as EventListener);
      // eslint-disable-next-line @typescript-eslint/no-use-before-define
      worker.removeEventListener('error', handleError as EventListener);
      worker.terminate();
      if (data?.error) {
        reject(new Error(data.error));
        return;
      }
      resolve({
        replacementScore: data.replacementScore || 0,
        completenessScore: data.completenessScore || 0,
      });
    }
    function handleError(err: ErrorEvent) {
      // @ts-ignore
      worker.removeEventListener('message', handleMessage as EventListener);
      // @ts-ignore
      worker.removeEventListener('error', handleError as EventListener);
      worker.terminate();
      reject(err.error || err);
    }
    worker.addEventListener('message', handleMessage as EventListener);
    worker.addEventListener('error', handleError as EventListener);
    try {
      worker.postMessage({ requestId: 1, graph: { nodes, links }, pinnedIds });
    } catch (err) {
      // @ts-ignore
      worker.removeEventListener('message', handleMessage as EventListener);
      worker.removeEventListener('error', handleError as EventListener);
      worker.terminate();
      reject(err);
    }
  });
}
