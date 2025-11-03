'use client';

import { Button } from '@/components/shadcn/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/shadcn/card';
import { SourceWithPartialRelations } from '@/prisma/generated/zod';
import { Grid } from 'lucide-react';
import { useState } from 'react';
import { useGlobalContext } from '../provider/global-provider';

export default function SourceSimilarityMatrixPane({ source }: { source: SourceWithPartialRelations }) {
  const { setSimilarityMatrix } = useGlobalContext();
  const [customText, setCustomText] = useState<string>('');

  return (
    <div className="flex w-full flex-col items-center">
      <Card className="mt-0 w-full max-w-screen-lg bg-white">
        <CardHeader className="w-full pb-3 pt-5">
          <div className="flex w-full flex-row items-center justify-between">
            <CardTitle>Similarity Matrix</CardTitle>
          </div>
        </CardHeader>
        <CardContent className="flex flex-row gap-2 pt-2">
          <div className="flex w-full gap-2">
            <textarea
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  setSimilarityMatrix(source, customText);
                }
              }}
              placeholder="Enter some text to generate a similarity matrix, or just click 'Generate' to use a demo text."
              className="flex-1 rounded border border-slate-300 px-3 py-2 text-[13px] leading-normal text-slate-800 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
              rows={4}
            />
          </div>
          <Button
            className="flex h-auto flex-col gap-y-0.5 rounded-md bg-sky-700 px-3 text-[10.5px] font-medium text-white hover:bg-sky-800 hover:text-white"
            variant="outline"
            onClick={() => {
              setSimilarityMatrix(source, customText);
            }}
          >
            <Grid className="h-4 w-4" /> Generate
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
