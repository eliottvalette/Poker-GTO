This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## Supabase (static browser runtime)

Copy `.env.example` to `.env.local` and provide the project's publishable key.
The local environment file is ignored by Git. Configure the same two `NEXT_PUBLIC_`
variables in the build environment; static exports embed them at build time.
Never put a secret or service-role key in a `NEXT_PUBLIC_` variable.

Import `createClient` from `@/utils/supabase/client` inside a browser effect or event
handler. It returns one shared client, persists authenticated sessions in browser
storage and refreshes them automatically. Missing configuration and server-side
calls throw explicit errors. Check and surface Supabase's returned `error` for every
request.

The application keeps `output: "export"`. There is no request-time Next.js server:
`next/headers` cookies, server session helpers and middleware are therefore not
installed. The requested `@supabase/ssr` package is available but unused in this
static runtime; browser operations use `@supabase/supabase-js` directly. See the
[Next.js static export restrictions](https://nextjs.org/docs/app/guides/static-exports#unsupported-features)
and [Supabase client options](https://supabase.com/docs/reference/javascript/initializing).

Policy delivery uses Supabase Storage, with public ONNX/JSON artifacts and private
server-only publication credentials. `NEXT_PUBLIC_POLICY_SOURCE=supabase` selects
remote policies; `local` explicitly selects the existing static catalog. There is
no silent fallback between sources. The UI and worker poll remote versions and
validate model hashes. See [the live pipeline runbook](../docs/LIVE_PIPELINE.md).
`vercel.json` supplies public build settings for the existing Vercel Git deployment.
No secret/service-role key belongs in that file or in the browser environment.

## Dependency security maintenance (2026-10-09)

Next.js and its ESLint configuration are aligned at 15.5.27. The lockfile updates
patched transitive dependencies, including tar, sharp, minimatch and source-map-js.
Two targeted overrides retain fixes that would otherwise resolve to vulnerable
versions: Next.js's PostCSS uses 8.5.29; brace-expansion 1.x uses 1.1.21. Remove an
override only after its parent dependency resolves a patched version itself.

`npm ci` reproduces this dependency tree. `npm audit --omit=dev` reports zero
vulnerabilities. The complete audit still reports five high-severity package entries
from one unpatched advisory, [GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm):
`eslint-config-next -> @next/eslint-plugin-next -> fast-glob -> micromatch -> braces`.
All entries are development dependencies. This is not a clean full audit; no alert
has been suppressed. The advisory lists no patched braces release. Do not pass
untrusted glob patterns to lint tooling. Recheck upstream before updating; npm's
suggested forced downgrade to eslint-config-next 14 is not applied.

Validation includes the static production build, ESLint, and the browser regression
suite. The suite's dynamic action-module import uses an explicit `.js` extension as
required by its Node16 module resolution.
