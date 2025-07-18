import { config } from 'dotenv';
import { z } from 'zod/v4';

// If it's not undefined, then it's a one click deploy. It doesn't matter what the value itself is.
// Also, if it's one-click-deploy on Vercel, we always use the demo environment variables.
export const SITE_NAME_VERCEL_DEPLOY = process.env.NEXT_PUBLIC_SITE_NAME_VERCEL_DEPLOY;
export const IS_ONE_CLICK_VERCEL_DEPLOY = SITE_NAME_VERCEL_DEPLOY !== undefined;
if (SITE_NAME_VERCEL_DEPLOY) {
  // @ts-ignore
  if (typeof EdgeRuntime !== 'string') {
    config({ path: '.env.demo', override: true });
  }
}

// Define the environment schema
const envSchema = z.object({
  // Domain of your main site
  NEXT_PUBLIC_URL: z.string().default(''),

  // Auth will redirect to this domain
  NEXTAUTH_URL: z.string().default(''),

  // Secret for hashing auth tokens (Used by NextAuth, not our code directly)
  // NEXTAUTH_SECRET: z.string().default(''),

  // Database (Used by Prisma, not our code directly)
  // POSTGRES_PRISMA_URL: z.string().default(''),
  // POSTGRES_URL_NON_POOLING: z.string().default(''),

  // Feature Flags
  ENABLE_RATE_LIMITER: z.stringbool().default(false),
  ENABLE_VERCEL_ANALYTICS: z.stringbool().default(false),
  NEXT_PUBLIC_ENABLE_SIGNIN: z.stringbool().default(false),

  // Default Values
  NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS: z.email().default('johnny@neuronpedia.org'),
  NEXT_PUBLIC_DEFAULT_RELEASE_NAME: z.string().default(''),
  NEXT_PUBLIC_DEFAULT_MODELID: z.string().default(''),
  NEXT_PUBLIC_DEFAULT_SOURCESET: z.string().default(''),
  NEXT_PUBLIC_DEFAULT_SOURCE: z.string().default(''),
  NEXT_PUBLIC_DEFAULT_STEER_MODEL: z.string().default(''),
  NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS: z
    .string()
    .default('')
    .transform((v) => (v ? v.split(',').map((m) => m.trim()) : [])),

  // Default User IDs
  // The fallback values are users in the seeded database.
  DEFAULT_CREATOR_USER_ID: z.string().default('clkht01d40000jv08hvalcvly'),
  INFERENCE_ACTIVATION_USER_ID: z.string().default('cljgamm90000076zdchicy6zj'),
  PUBLIC_ACTIVATIONS_USER_IDS: z
    .string()
    .default('')
    .transform((v) =>
      v ? v.split(',').map((id) => id.trim()) : ['cljj57d3c000076ei38vwnv35', 'clkht01d40000jv08hvalcvly'],
    ),

  // Email Sending Providers
  // AWS SES: more reliable (used if both AWS and Resend are defined)
  AWS_ACCESS_KEY_ID: z.string().default(''),
  AWS_SECRET_ACCESS_KEY: z.string().default(''),
  // Resend.com: easier to setup
  RESEND_EMAIL_API_KEY: z.string().default(''),

  // External Services
  // AI API Keys (Mostly for auto-interp for whitelisted accounts)
  OPENAI_API_KEY: z.string().default(''),
  GEMINI_API_KEY: z.string().default(''),
  ANTHROPIC_API_KEY: z.string().default(''),
  OPENROUTER_API_KEY: z.string().default(''),
  AZURE_OPENAI_API_KEY: z.string().default(''),

  // Azure-specific provider config parameters
  AZURE_OPENAI_ENDPOINT: z.url().or(z.literal('')).default(''),
  AZURE_API_VERSION: z.string().default('2024-02-01'),
  OPENAI_DEPLOYMENT_NAME: z.string().default('gpt-4o-mini'),

  // Provider selection flags
  USE_AZURE_OPENAI: z.stringbool().default(false),
  USE_OPENROUTER: z.stringbool().default(false),
  USE_OPENAI: z.stringbool().default(true),

  // Sentry (Crash Reporting - Used by Sentry, not by us directly)
  // SENTRY_DSN: z.string().default(''),
  // SENTRY_AUTH_TOKEN: z.string().default(''),

  // Rate Limiting - Redis (Used by Upstash, not by us directly)
  // KV_URL: z.string().default(''),
  // KV_REST_API_URL: z.string().default(''),
  // KV_REST_API_TOKEN: z.string().default(''),
  // KV_REST_API_READ_ONLY_TOKEN: z.string().default(''),

  // Support Servers
  // Inference Server
  USE_LOCALHOST_INFERENCE: z.stringbool().default(false),
  INFERENCE_SERVER_SECRET: z.string().default(''),
  NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH: z
    .string()
    .transform((v) => (v ? parseInt(v, 10) : 1024))
    .default(1024),

  // Autointerp Server
  USE_LOCALHOST_AUTOINTERP: z.stringbool().default(false),
  AUTOINTERP_SERVER: z.string().default(''),
  AUTOINTERP_SERVER_SECRET: z.string().default(''),

  // Graph Server
  // Three possible states: Localhost Graph, Remote Graph, and Runpod Graph
  // USE_LOCALHOST_GRAPH and USE_RUNPOD_GRAPH cannot both be true.
  USE_LOCALHOST_GRAPH: z.stringbool().default(false),
  GRAPH_SERVER: z.string().default(''),
  GRAPH_SERVER_SECRET: z.string().default(''),

  // Runpod Graph
  USE_RUNPOD_GRAPH: z.stringbool().default(false),
  GRAPH_RUNPOD_SECRET: z.string().default(''),
  GRAPH_RUNPOD_SERVER: z.string().default(''),

  // Authentication Methods
  // Apple
  APPLE_CLIENT_ID: z.string().default(''),
  APPLE_CLIENT_SECRET: z.string().default(''),
  // GitHub
  GITHUB_ID: z.string().default(''),
  GITHUB_SECRET: z.string().default(''),
  // Google
  GOOGLE_CLIENT_ID: z.string().default(''),
  GOOGLE_CLIENT_SECRET: z.string().default(''),

  // Authentication Refresh - for updating the Apple Client Secret every 6 months (using ./node scripts/apple-gen-secret.js and .secret.apple.p8)
  // APPLE_KEY_ID: z.string().default(''),
  // APPLE_TEAM_ID: z.string().default(''),

  // Misc
  NODE_ENV: z.string().default(''),
  IS_DOCKER_COMPOSE: z.stringbool().default(false),
  NEXT_PUBLIC_DEMO_MODE: z.stringbool().default(false),
  GRAPH_ADMIN_BROWSE_KEY: z.string().default(''),
  HIGHER_LIMIT_API_TOKENS: z
    .string()
    .default('')
    .transform((v) => (v ? v.split(',').map((t) => t.trim()) : [])),
});

// Parse and validate environment variables
const env = envSchema.parse(process.env);

// Domain of your main site
export const NEXT_PUBLIC_URL = env.NEXT_PUBLIC_URL;

// Auth will redirect to this domain
export const NEXTAUTH_URL = env.NEXTAUTH_URL;

// Secret for hashing auth tokens (Used by NextAuth, not our code directly)
// export const NEXTAUTH_SECRET = env.NEXTAUTH_SECRET;

// Database (Used by Prisma, not our code directly)
// export const POSTGRES_PRISMA_URL = env.POSTGRES_PRISMA_URL;
// export const POSTGRES_URL_NON_POOLING = env.POSTGRES_URL_NON_POOLING;

// Feature Flags
export const ENABLE_RATE_LIMITER = env.ENABLE_RATE_LIMITER;
export const ENABLE_VERCEL_ANALYTICS = env.ENABLE_VERCEL_ANALYTICS;
export const NEXT_PUBLIC_ENABLE_SIGNIN = env.NEXT_PUBLIC_ENABLE_SIGNIN && !IS_ONE_CLICK_VERCEL_DEPLOY;

// Default Values
export const CONTACT_EMAIL_ADDRESS = env.NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS;
export const DEFAULT_RELEASE_NAME = env.NEXT_PUBLIC_DEFAULT_RELEASE_NAME;
export const DEFAULT_MODELID = env.NEXT_PUBLIC_DEFAULT_MODELID;
export const DEFAULT_SOURCESET = env.NEXT_PUBLIC_DEFAULT_SOURCESET;
export const DEFAULT_SOURCE = env.NEXT_PUBLIC_DEFAULT_SOURCE;
export const DEFAULT_STEER_MODEL = env.NEXT_PUBLIC_DEFAULT_STEER_MODEL;
export const STEER_FORCE_ALLOW_INSTRUCT_MODELS = env.NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS;

// Default User IDs
// The fallback values are users in the seeded database.
export const DEFAULT_CREATOR_USER_ID = env.DEFAULT_CREATOR_USER_ID;
export const INFERENCE_ACTIVATION_USER_ID_DO_NOT_INCLUDE_IN_PUBLIC_ACTIVATIONS = env.INFERENCE_ACTIVATION_USER_ID;
export const PUBLIC_ACTIVATIONS_USER_IDS = env.PUBLIC_ACTIVATIONS_USER_IDS;

// Email
// For email sending providers, choose EITHER AWS SES or Resend.com.
// Resend is easier to set up, but AWS is more reliable.
// If both are defined, AWS will be used.
// AWS
export const AWS_ACCESS_KEY_ID = env.AWS_ACCESS_KEY_ID;
export const AWS_SECRET_ACCESS_KEY = env.AWS_SECRET_ACCESS_KEY;
// Resend
export const RESEND_EMAIL_API_KEY = env.RESEND_EMAIL_API_KEY;

// External Services
// AI API Keys (Mostly for auto-interp for whitelisted accounts)
export const OPENAI_API_KEY = env.OPENAI_API_KEY;
// if (!process.env.OPENAI_API_KEY) {
//   console.warn(
//     'OPENAI_API_KEY is not set. Search Explanations will not work. Set the key in the file neuronpedia/apps/webapp/.env',
//   );
// }
export const GEMINI_API_KEY = env.GEMINI_API_KEY;
export const ANTHROPIC_API_KEY = env.ANTHROPIC_API_KEY;
export const OPENROUTER_API_KEY = env.OPENROUTER_API_KEY;

export const AZURE_OPENAI_API_KEY = env.AZURE_OPENAI_API_KEY;
export const AZURE_OPENAI_ENDPOINT = env.AZURE_OPENAI_ENDPOINT;
export const AZURE_API_VERSION = env.AZURE_API_VERSION;

export const OPENAI_DEPLOYMENT_NAME = env.OPENAI_DEPLOYMENT_NAME;

// Provider selection flags
export const USE_AZURE_OPENAI = env.USE_AZURE_OPENAI;
export const USE_OPENROUTER = env.USE_OPENROUTER;
export const USE_OPENAI = env.USE_OPENAI;

// Sentry (Crash Reporting - Used by Sentry, not by us directly)
// export const SENTRY_DSN = env.SENTRY_DSN;
// export const SENTRY_AUTH_TOKEN = env.SENTRY_AUTH_TOKEN;

// Rate Limiting - Redis (Used by Upstash, not by us directly)
// export const KV_URL = env.KV_URL;
// export const KV_REST_API_URL = env.KV_REST_API_URL;
// export const KV_REST_API_TOKEN = env.KV_REST_API_TOKEN;
// export const KV_REST_API_READ_ONLY_TOKEN = env.KV_REST_API_READ_ONLY_TOKEN;

// Support Servers
// Inference Server
export const USE_LOCALHOST_INFERENCE = env.USE_LOCALHOST_INFERENCE;
export const INFERENCE_SERVER_SECRET = env.INFERENCE_SERVER_SECRET;

export const NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH = env.NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH;

// Autointerp Server
export const USE_LOCALHOST_AUTOINTERP = env.USE_LOCALHOST_AUTOINTERP;
export const AUTOINTERP_SERVER = env.AUTOINTERP_SERVER;
export const AUTOINTERP_SERVER_SECRET = env.AUTOINTERP_SERVER_SECRET;

// Graph Server
// Three possible states: Localhost Graph, Remote Graph, and Runpod Graph
// USE_LOCALHOST_GRAPH and USE_RUNPOD_GRAPH cannot both be true.
export const USE_LOCALHOST_GRAPH = env.USE_LOCALHOST_GRAPH;
export const GRAPH_SERVER = env.GRAPH_SERVER;
export const GRAPH_SERVER_SECRET = env.GRAPH_SERVER_SECRET;

// Runpod Graph
export const USE_RUNPOD_GRAPH = env.USE_RUNPOD_GRAPH;
if (USE_RUNPOD_GRAPH && USE_LOCALHOST_GRAPH) {
  throw new Error('USE_LOCALHOST_GRAPH and USE_RUNPOD_GRAPH cannot both be true.');
}
export const GRAPH_RUNPOD_SECRET = env.GRAPH_RUNPOD_SECRET;
export const GRAPH_RUNPOD_SERVER = env.GRAPH_RUNPOD_SERVER;

// Authentication Methods
// Apple
export const APPLE_CLIENT_ID = env.APPLE_CLIENT_ID;
export const APPLE_CLIENT_SECRET = env.APPLE_CLIENT_SECRET;
// GitHub
export const GITHUB_ID = env.GITHUB_ID;
export const GITHUB_SECRET = env.GITHUB_SECRET;
// Google
export const GOOGLE_CLIENT_ID = env.GOOGLE_CLIENT_ID;
export const GOOGLE_CLIENT_SECRET = env.GOOGLE_CLIENT_SECRET;

// Authentication Refresh - for updating the Apple Client Secret every 6 months (using ./node scripts/apple-gen-secret.js and .secret.apple.p8)
// export const APPLE_KEY_ID = env.APPLE_KEY_ID;
// export const APPLE_TEAM_ID = env.APPLE_TEAM_ID;

export const IS_LOCALHOST = env.NEXT_PUBLIC_URL === 'http://localhost:3000';
export const IS_ACTUALLY_NEURONPEDIA_ORG =
  env.NEXT_PUBLIC_URL === 'https://neuronpedia.org' || env.NEXT_PUBLIC_URL === 'https://www.neuronpedia.org';

// Misc
export const NODE_ENV = env.NODE_ENV;
export const IS_DOCKER_COMPOSE = env.IS_DOCKER_COMPOSE;
export const DEMO_MODE = env.NEXT_PUBLIC_DEMO_MODE || IS_ONE_CLICK_VERCEL_DEPLOY;
export const ASSET_BASE_URL = 'https://neuronpedia.s3.us-east-1.amazonaws.com/site-assets';
export const GRAPH_ADMIN_BROWSE_KEY = env.GRAPH_ADMIN_BROWSE_KEY;

export const API_KEY_HEADER_NAME = 'x-api-key';
export const HIGHER_LIMIT_API_TOKENS = env.HIGHER_LIMIT_API_TOKENS;
