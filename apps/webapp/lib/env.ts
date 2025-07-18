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

// Create schema factory to ensure fresh instance
const createEnvSchema = () => {
  return z
    .object({
      // Domain of your main site
      NEXT_PUBLIC_URL: z.string().default(''),

      // Auth will redirect to this domain
      NEXTAUTH_URL: z.string().default(''),

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
      DEFAULT_CREATOR_USER_ID: z.string().default('clkht01d40000jv08hvalcvly'),
      INFERENCE_ACTIVATION_USER_ID: z.string().default('cljgamm90000076zdchicy6zj'),
      PUBLIC_ACTIVATIONS_USER_IDS: z
        .string()
        .default('')
        .transform((v) =>
          v ? v.split(',').map((id) => id.trim()) : ['cljj57d3c000076ei38vwnv35', 'clkht01d40000jv08hvalcvly'],
        ),

      // Email Sending Providers
      USE_AWS_SES: z.stringbool().default(false),
      USE_RESEND: z.stringbool().default(false),
      USE_SMTP: z.stringbool().default(false),

      // AWS SES: more reliable (used if both AWS and Resend are defined)
      AWS_ACCESS_KEY_ID: z.string().default(''),
      AWS_SECRET_ACCESS_KEY: z.string().default(''),
      // Resend.com: easier to setup
      RESEND_EMAIL_API_KEY: z.string().default(''),
      // SMTP
      SMTP_SERVER_HOST: z.string().default(''),
      SMTP_SERVER_PORT: z.coerce.number().min(1).max(65535).default(25),
      SMTP_SERVER_FROM: z.email().default('noreply@neuronpedia.org'),

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

      // Misc
      NODE_ENV: z.string().default(''),
      IS_DOCKER_COMPOSE: z.stringbool().default(false),
      NEXT_PUBLIC_DEMO_MODE: z.stringbool().default(false),
      GRAPH_ADMIN_BROWSE_KEY: z.string().default(''),
      HIGHER_LIMIT_API_TOKENS: z
        .string()
        .default('')
        .transform((v) => (v ? v.split(',').map((t) => t.trim()) : [])),
    })
    .check((ctx) => {
      const {
        USE_AZURE_OPENAI,
        USE_OPENROUTER,
        USE_OPENAI,
        OPENAI_API_KEY,
        OPENROUTER_API_KEY,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        OPENAI_DEPLOYMENT_NAME,
      } = ctx.value;

      // Only one provider check
      const trueCount = [USE_AZURE_OPENAI, USE_OPENROUTER, USE_OPENAI].filter(Boolean).length;
      if (trueCount !== 1) {
        ctx.issues.push({
          code: 'custom',
          message: 'Exactly one provider must be enabled',
          path: ['envFile'],
          input: {
            USE_AZURE_OPENAI,
            USE_OPENROUTER,
            USE_OPENAI,
          },
        });
      }

      // OpenAI configuration check
      if (USE_OPENAI && !OPENAI_API_KEY) {
        console.error('START :: AN ERROR OCCURRED!');
        ctx.issues.push({
          code: 'custom',
          message: 'OPENAI_API_KEY is required when USE_OPENAI is enabled',
          path: ['OPENAI_API_KEY'],
          input: { USE_OPENAI, OPENAI_API_KEY },
        });
        console.error('END :: AN ERROR OCCURRED!');
      }

      // OpenRouter configuration check
      if (USE_OPENROUTER && !OPENROUTER_API_KEY) {
        ctx.issues.push({
          code: 'custom',
          message: 'OPENROUTER_API_KEY is required when USE_OPENROUTER is enabled',
          path: ['OPENROUTER_API_KEY'],
          input: { USE_OPENROUTER, OPENROUTER_API_KEY },
        });
      }

      // Azure OpenAI configuration check
      if (USE_AZURE_OPENAI) {
        if (!AZURE_OPENAI_API_KEY) {
          ctx.issues.push({
            code: 'custom',
            message: 'AZURE_OPENAI_API_KEY is required when USE_AZURE_OPENAI is enabled',
            path: ['AZURE_OPENAI_API_KEY'],
            input: { USE_AZURE_OPENAI, AZURE_OPENAI_API_KEY },
          });
        }
        if (!AZURE_OPENAI_ENDPOINT) {
          ctx.issues.push({
            code: 'custom',
            message: 'AZURE_OPENAI_ENDPOINT is required when USE_AZURE_OPENAI is enabled',
            path: ['AZURE_OPENAI_ENDPOINT'],
            input: { USE_AZURE_OPENAI, AZURE_OPENAI_ENDPOINT },
          });
        }
        if (!OPENAI_DEPLOYMENT_NAME) {
          ctx.issues.push({
            code: 'custom',
            message: 'OPENAI_DEPLOYMENT_NAME is required when USE_AZURE_OPENAI is enabled',
            path: ['OPENAI_DEPLOYMENT_NAME'],
            input: { USE_AZURE_OPENAI, OPENAI_DEPLOYMENT_NAME },
          });
        }
      }
    })
    .check((ctx) => {
      const {
        USE_AWS_SES,
        USE_RESEND,
        USE_SMTP,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        RESEND_EMAIL_API_KEY,
        SMTP_SERVER_HOST,
        SMTP_SERVER_PORT,
        SMTP_SERVER_FROM,
      } = ctx.value;

      // Only one email provider check
      const trueCount = [USE_AWS_SES, USE_RESEND, USE_SMTP].filter(Boolean).length;
      if (trueCount > 1) {
        ctx.issues.push({
          code: 'custom',
          message: 'Only one email provider can be enabled at a time',
          path: [],
          input: {
            USE_AWS_SES,
            USE_RESEND,
            USE_SMTP,
          },
        });
      }

      // AWS SES configuration check
      if (USE_AWS_SES) {
        if (!AWS_ACCESS_KEY_ID) {
          ctx.issues.push({
            code: 'custom',
            message: 'AWS_ACCESS_KEY_ID is required when USE_AWS_SES is enabled',
            path: ['AWS_ACCESS_KEY_ID'],
            input: { USE_AWS_SES, AWS_ACCESS_KEY_ID },
          });
        }
        if (!AWS_SECRET_ACCESS_KEY) {
          ctx.issues.push({
            code: 'custom',
            message: 'AWS_SECRET_ACCESS_KEY is required when USE_AWS_SES is enabled',
            path: ['AWS_SECRET_ACCESS_KEY'],
            input: { USE_AWS_SES, AWS_SECRET_ACCESS_KEY },
          });
        }
      }

      // Resend configuration check
      if (USE_RESEND && !RESEND_EMAIL_API_KEY) {
        ctx.issues.push({
          code: 'custom',
          message: 'RESEND_EMAIL_API_KEY is required when USE_RESEND is enabled',
          path: ['RESEND_EMAIL_API_KEY'],
          input: { USE_RESEND, RESEND_EMAIL_API_KEY },
        });
      }

      // SMTP configuration check
      if (USE_SMTP) {
        if (!SMTP_SERVER_HOST) {
          ctx.issues.push({
            code: 'custom',
            message: 'SMTP_SERVER_HOST is required when USE_SMTP is enabled',
            path: ['SMTP_SERVER_HOST'],
            input: { USE_SMTP, SMTP_SERVER_HOST },
          });
        }
        if (!SMTP_SERVER_PORT) {
          ctx.issues.push({
            code: 'custom',
            message: 'SMTP_SERVER_PORT is required when USE_SMTP is enabled',
            path: ['SMTP_SERVER_PORT'],
            input: { USE_SMTP, SMTP_SERVER_PORT },
          });
        }
        if (!SMTP_SERVER_FROM) {
          ctx.issues.push({
            code: 'custom',
            message: 'SMTP_SERVER_FROM is required when USE_SMTP is enabled',
            path: ['SMTP_SERVER_FROM'],
            input: { USE_SMTP, SMTP_SERVER_FROM },
          });
        }
      }
    })
    .check((ctx) => {
      const { USE_RUNPOD_GRAPH, USE_LOCALHOST_GRAPH } = ctx.value;

      // Runtime validation
      if (USE_RUNPOD_GRAPH && USE_LOCALHOST_GRAPH) {
        ctx.issues.push({
          code: 'custom',
          message: 'USE_LOCALHOST_GRAPH and USE_RUNPOD_GRAPH cannot both be true.',
          path: ['USE_LOCALHOST_GRAPH', 'USE_RUNPOD_GRAPH'],
          input: { USE_LOCALHOST_GRAPH, USE_RUNPOD_GRAPH },
        });
      }
    });
};

// Parse environment variables with a fresh schema instance
const parseEnvironment = () => {
  try {
    // Create a fresh schema instance for this parse
    const schema = createEnvSchema();

    // Use safeParse to avoid the context contamination issue
    const result = schema.safeParse(process.env);

    if (!result.success) {
      console.error('Environment validation failed:', z.prettifyError(result.error));
      throw result.error;
    }

    console.log('Parsed environment variables successfully.');
    return result.data;
  } catch (error) {
    console.error('Failed to parse environment:', error);
    throw error;
  }
};

// Parse and validate environment variables
const env = parseEnvironment();

// Destructure all environment variables at once
export const {
  // Domain and Auth
  NEXT_PUBLIC_URL,
  NEXTAUTH_URL,

  // Feature Flags
  ENABLE_RATE_LIMITER,
  ENABLE_VERCEL_ANALYTICS,
  NEXT_PUBLIC_ENABLE_SIGNIN: NEXT_PUBLIC_ENABLE_SIGNIN_RAW,

  // Default Values
  NEXT_PUBLIC_CONTACT_EMAIL_ADDRESS: CONTACT_EMAIL_ADDRESS,
  NEXT_PUBLIC_DEFAULT_RELEASE_NAME: DEFAULT_RELEASE_NAME,
  NEXT_PUBLIC_DEFAULT_MODELID: DEFAULT_MODELID,
  NEXT_PUBLIC_DEFAULT_SOURCESET: DEFAULT_SOURCESET,
  NEXT_PUBLIC_DEFAULT_SOURCE: DEFAULT_SOURCE,
  NEXT_PUBLIC_DEFAULT_STEER_MODEL: DEFAULT_STEER_MODEL,
  NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS: STEER_FORCE_ALLOW_INSTRUCT_MODELS,

  // Default User IDs
  DEFAULT_CREATOR_USER_ID,
  INFERENCE_ACTIVATION_USER_ID: INFERENCE_ACTIVATION_USER_ID_DO_NOT_INCLUDE_IN_PUBLIC_ACTIVATIONS,
  PUBLIC_ACTIVATIONS_USER_IDS,

  // Email Provider selection flags
  USE_AWS_SES,
  USE_RESEND,
  USE_SMTP,

  // AWS
  AWS_ACCESS_KEY_ID,
  AWS_SECRET_ACCESS_KEY,

  // Resend
  RESEND_EMAIL_API_KEY,

  // SMTP
  SMTP_SERVER_HOST,
  SMTP_SERVER_PORT,
  SMTP_SERVER_FROM,

  // AI API Keys
  OPENAI_API_KEY,
  GEMINI_API_KEY,
  ANTHROPIC_API_KEY,
  OPENROUTER_API_KEY,

  // Azure OpenAI
  AZURE_OPENAI_API_KEY,
  AZURE_OPENAI_ENDPOINT,
  AZURE_API_VERSION,
  OPENAI_DEPLOYMENT_NAME,

  // Provider selection flags
  USE_AZURE_OPENAI,
  USE_OPENROUTER,
  USE_OPENAI,

  // Support Servers - Inference
  USE_LOCALHOST_INFERENCE,
  INFERENCE_SERVER_SECRET,
  NEXT_PUBLIC_SEARCH_TOPK_MAX_CHAR_LENGTH,

  // Support Servers - Autointerp
  USE_LOCALHOST_AUTOINTERP,
  AUTOINTERP_SERVER,
  AUTOINTERP_SERVER_SECRET,

  // Support Servers - Graph
  USE_LOCALHOST_GRAPH,
  GRAPH_SERVER,
  GRAPH_SERVER_SECRET,
  USE_RUNPOD_GRAPH,
  GRAPH_RUNPOD_SECRET,
  GRAPH_RUNPOD_SERVER,

  // Authentication - Apple
  APPLE_CLIENT_ID,
  APPLE_CLIENT_SECRET,

  // Authentication - GitHub
  GITHUB_ID,
  GITHUB_SECRET,

  // Authentication - Google
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,

  // Misc
  NODE_ENV,
  IS_DOCKER_COMPOSE,
  NEXT_PUBLIC_DEMO_MODE,
  GRAPH_ADMIN_BROWSE_KEY,
  HIGHER_LIMIT_API_TOKENS,
} = env;

// Computed values that depend on other variables
export const NEXT_PUBLIC_ENABLE_SIGNIN = NEXT_PUBLIC_ENABLE_SIGNIN_RAW && !IS_ONE_CLICK_VERCEL_DEPLOY;

// Derived values
export const IS_LOCALHOST = NEXT_PUBLIC_URL === 'http://localhost:3000';
export const IS_ACTUALLY_NEURONPEDIA_ORG =
  NEXT_PUBLIC_URL === 'https://neuronpedia.org' || NEXT_PUBLIC_URL === 'https://www.neuronpedia.org';
export const DEMO_MODE = NEXT_PUBLIC_DEMO_MODE || IS_ONE_CLICK_VERCEL_DEPLOY;

// Constants
export const ASSET_BASE_URL = 'https://neuronpedia.s3.us-east-1.amazonaws.com/site-assets';
export const API_KEY_HEADER_NAME = 'x-api-key';
