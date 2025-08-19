import { prisma } from '@/lib/db';
import { env } from '@/lib/env';
import { redirect } from 'next/navigation';

export default async function Page() {
  if (!env.DEFAULT_STEER_MODEL) {
    const model = await prisma.model.findFirst({
      where: {
        visibility: 'PUBLIC',
      },
    });
    redirect(`/${model?.id}/steer`);
  }
  redirect(`/${env.DEFAULT_STEER_MODEL}/steer`);
}
