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

This prepares client connectivity only. It does not create a `todos` table, replace
the poker homepage, provision Storage buckets/policies, upload training exports or
switch the UI's policy source. Remote policy publication still requires an explicit
Storage contract and server-only upload credentials; the browser publishable key
is not an administrative upload credential. Optional third-party agent skills were
not installed.
