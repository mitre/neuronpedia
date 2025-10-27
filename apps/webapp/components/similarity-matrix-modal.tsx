'use client';

import { useGlobalContext } from '@/components/provider/global-provider';
import { replaceHtmlAnomalies } from '@/lib/utils/activations';
import * as Dialog from '@radix-ui/react-dialog';
import { useEffect, useState } from 'react';

export default function SimilarityMatrixModal() {
  const { similarityMatrixFeature, similarityMatrixText, similarityMatrixModalOpen, setSimilarityMatrixModalOpen } =
    useGlobalContext();

  // State for tokens and similarity matrix
  const [tokens, setTokens] = useState<string[]>([]);
  const [similarityMatrix, setSimilarityMatrix] = useState<number[][]>([]);
  const [selectedToken, setSelectedToken] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch similarity matrix from server when modal opens
  useEffect(() => {
    if (!similarityMatrixModalOpen) return;
    const run = async () => {
      try {
        setLoading(true);
        setError(null);
        setSelectedToken(null);
        const modelId = similarityMatrixFeature?.modelId || 'gemma-2-2b';
        const sourceId = (similarityMatrixFeature?.layer as string) || '12-temporal-res';
        const indexRaw = (similarityMatrixFeature?.index as unknown) ?? 0;
        const index = typeof indexRaw === 'string' ? parseInt(indexRaw, 10) : (indexRaw as number);
        const text =
          similarityMatrixText && similarityMatrixText.trim().length > 0
            ? similarityMatrixText
            : 'The cat sat on the mat';

        const resp = await fetch('/api/similarity-matrix-pred', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ modelId, sourceId, index, text }),
        });
        if (!resp.ok) {
          const detail = await resp.text();
          throw new Error(detail || 'Request failed');
        }
        const data = await resp.json();
        const newTokens: string[] = data.tokens || [];
        const matrix: number[][] = data.similarity_matrix || [];
        setTokens(newTokens);
        setSimilarityMatrix(matrix);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    };
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [similarityMatrixModalOpen, similarityMatrixFeature, similarityMatrixText]);

  // Magma color scheme interpolation
  const getMagmaColor = (value: number) => {
    // Magma colormap approximation: dark purple -> pink -> orange -> yellow
    const colors = [
      { pos: 0.0, r: 0, g: 0, b: 4 },
      { pos: 0.25, r: 60, g: 15, b: 80 },
      { pos: 0.5, r: 180, g: 50, b: 120 },
      { pos: 0.75, r: 250, g: 140, b: 70 },
      { pos: 1.0, r: 252, g: 253, b: 191 },
    ];

    let lower = colors[0];
    let upper = colors[colors.length - 1];

    for (let i = 0; i < colors.length - 1; i += 1) {
      if (value >= colors[i].pos && value <= colors[i + 1].pos) {
        lower = colors[i];
        upper = colors[i + 1];
        break;
      }
    }

    const range = upper.pos - lower.pos;
    const rangePct = range === 0 ? 0 : (value - lower.pos) / range;

    const r = Math.round(lower.r + (upper.r - lower.r) * rangePct);
    const g = Math.round(lower.g + (upper.g - lower.g) * rangePct);
    const b = Math.round(lower.b + (upper.b - lower.b) * rangePct);

    return `rgb(${r}, ${g}, ${b})`;
  };

  const handleTokenClick = (index: number) => {
    setSelectedToken(selectedToken === index ? null : index);
  };

  // Dynamic cell size based on number of tokens to maintain consistent grid size
  const maxGridSize = 400; // Maximum size for the heatmap grid
  const cellSize = Math.max(20, Math.min(70, maxGridSize / tokens.length));
  const labelSize = 30;
  const padding = 40;

  return (
    <Dialog.Root open={similarityMatrixModalOpen}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-slate-600/20" />
        <Dialog.Content
          onPointerDownOutside={() => {
            setSimilarityMatrixModalOpen(false);
          }}
          className="fixed left-[50%] top-[50%] z-50 flex h-[100vh] max-h-[100vh] w-[100vw] max-w-[100%] translate-x-[-50%] translate-y-[-50%] flex-col overflow-y-scroll bg-slate-50 shadow-xl focus:outline-none sm:top-[50%] sm:h-[90vh] sm:max-h-[90vh] sm:w-[95vw] sm:max-w-[95%] sm:rounded-md"
        >
          <div className="sticky top-0 z-20 w-full flex-col items-center border-slate-300">
            <div className="mb-0 flex w-full flex-row items-start justify-between gap-x-4 rounded-t-md border-b bg-white px-2 pb-2 pt-2 sm:px-4">
              <button
                type="button"
                className="flex flex-row items-center justify-center gap-x-1 rounded-full bg-slate-300 px-3 py-1.5 text-[11px] text-slate-600 hover:bg-sky-700 hover:text-white focus:outline-none"
                aria-label="Close"
                onClick={() => {
                  setSimilarityMatrixModalOpen(false);
                }}
              >
                Done
              </button>
              <Dialog.Title className="absolute left-1/2 top-[55%] -translate-x-1/2 -translate-y-1/2 text-center text-xs font-medium text-slate-700">
                Similarity Matrix
              </Dialog.Title>
            </div>
          </div>
          <div className="flex flex-1 flex-col overflow-x-auto overflow-y-auto bg-slate-50 p-0">
            <div className="flex w-full flex-1 flex-col rounded-lg bg-white p-0 shadow-lg">
              {loading && <div className="mb-6 pt-6 text-center text-sm text-gray-600">Loading similarity matrix…</div>}
              {error && <div className="mb-6 rounded bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>}

              {!loading && !error && (
                <div className="flex flex-1 items-center justify-center overflow-x-auto overflow-y-auto p-4 pt-12">
                  <div className="inline-block">
                    {/* Token boxes */}
                    {/* <div className="mb-6 mt-4 flex justify-start gap-1">
                      {tokens.map((token, i) => (
                        <div
                          key={i}
                          role="button"
                          tabIndex={0}
                          onClick={() => handleTokenClick(i)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              handleTokenClick(i);
                            }
                          }}
                          className="cursor-pointer rounded bg-blue-100 px-2 py-1 text-center text-xs font-medium transition-all hover:bg-blue-200"
                          style={{
                            border: selectedToken === i ? '1px solid #3b82f6' : '1px solid #93c5fd',
                          }}
                        >
                          {token}
                        </div>
                      ))}
                    </div> */}

                    {/* Heatmap with axis labels */}
                    <div className="mb-8 flex">
                      {/* Y-axis labels */}
                      <div className="flex flex-col justify-start" style={{ marginTop: `42px` }}>
                        {tokens.map((token, i) => (
                          <div
                            key={`y-label-${i}`}
                            className="flex items-center justify-end pr-2 text-[11px] text-gray-600"
                            style={{ height: `${cellSize}px` }}
                          >
                            {replaceHtmlAnomalies(token)}
                          </div>
                        ))}
                      </div>

                      {/* Heatmap container */}
                      <div>
                        {/* X-axis labels */}
                        <div className="flex pb-2 pl-0" style={{ marginBottom: '4px' }}>
                          {tokens.map((token, i) => (
                            <div
                              key={`x-label-${i}`}
                              className="flex items-end justify-center text-start text-[10px] text-gray-600"
                              style={{ width: `${cellSize}px`, height: `${labelSize}px` }}
                            >
                              <span
                                className="inline-block -rotate-90 transform text-center"
                                style={{ width: `${labelSize}px` }}
                              >
                                {replaceHtmlAnomalies(token)}
                              </span>
                            </div>
                          ))}
                        </div>

                        {/* Similarity matrix heatmap */}
                        <div className="inline-block border-2 border-gray-300">
                          {similarityMatrix.map((row, i) => (
                            <div key={i} className="flex">
                              {row.map((value, j) => {
                                const isInSelectedRow = selectedToken !== null && i === selectedToken;
                                const isInSelectedCol = selectedToken !== null && j === selectedToken;
                                const isFirstInRow = j === 0;
                                const isLastInRow = j === tokens.length - 1;
                                const isFirstInCol = i === 0;
                                const isLastInCol = i === tokens.length - 1;

                                const borderStyle = '1px solid rgba(255,255,255,0.3)';
                                let borderTop = borderStyle;
                                let borderBottom = borderStyle;
                                let borderLeft = borderStyle;
                                let borderRight = borderStyle;

                                const highlightBorder = '5px solid #3b82f6';

                                if (isInSelectedRow) {
                                  borderTop = highlightBorder;
                                  borderBottom = highlightBorder;
                                  if (isFirstInRow) borderLeft = highlightBorder;
                                  if (isLastInRow) borderRight = highlightBorder;
                                }

                                if (isInSelectedCol) {
                                  borderLeft = highlightBorder;
                                  borderRight = highlightBorder;
                                  if (isFirstInCol) borderTop = highlightBorder;
                                  if (isLastInCol) borderBottom = highlightBorder;
                                }

                                return (
                                  <div
                                    key={`${i}-${j}`}
                                    role="button"
                                    tabIndex={0}
                                    onClick={() => handleTokenClick(Math.min(i, j))}
                                    onKeyDown={(e) => {
                                      if (e.key === 'Enter' || e.key === ' ') {
                                        handleTokenClick(Math.min(i, j));
                                      }
                                    }}
                                    className="flex cursor-pointer items-center justify-center font-mono text-[9.5px] transition-all"
                                    style={{
                                      width: `${cellSize}px`,
                                      height: `${cellSize}px`,
                                      backgroundColor: getMagmaColor(value),
                                      color: value > 0.5 ? 'black' : 'white',
                                      borderTop,
                                      borderBottom,
                                      borderLeft,
                                      borderRight,
                                    }}
                                  >
                                    {/* {value.toFixed(2)} */}
                                  </div>
                                );
                              })}
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Colorbar */}
                      <div className="ml-6 flex hidden flex-col" style={{ marginTop: `${padding}px` }}>
                        <div className="flex">
                          <div className="flex flex-col border-2 border-gray-300">
                            {Array.from({ length: 50 }).map((_, i) => {
                              const value = 1 - i / 49; // Go from 1.0 to 0.0
                              return (
                                <div
                                  key={i}
                                  style={{
                                    width: '30px',
                                    height: `${(cellSize * tokens.length) / 50}px`,
                                    backgroundColor: getMagmaColor(value),
                                  }}
                                />
                              );
                            })}
                          </div>
                          <div className="ml-2 flex flex-col justify-between text-xs text-gray-700">
                            <span>1.0</span>
                            <span>0.75</span>
                            <span>0.5</span>
                            <span>0.25</span>
                            <span>0.0</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    {/* 
              <div className="mt-6 text-center text-sm text-gray-600">
                {selectedToken !== null
                  ? `Selected: Token "${tokens[selectedToken]}" (position ${selectedToken})`
                  : 'Click a token or any cell to highlight its row and column'}
              </div> */}
                  </div>
                </div>
              )}
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
