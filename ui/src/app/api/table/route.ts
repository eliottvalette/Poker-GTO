import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const origin = request.headers.get("origin");
    if (origin !== null && (new URL(origin).host !== request.headers.get("host")
        || new URL(origin).protocol !== new URL(request.url).protocol)) {
      return NextResponse.json({ error: "Cross-origin table commands are excluded" }, { status: 403 });
    }
    if (!request.headers.get("content-type")?.startsWith("application/json")) {
      return NextResponse.json({ error: "Table commands require application/json" }, { status: 400 });
    }
    const { operation, ...payload } = await request.json();
    if (!["new", "action", "next", "close"].includes(operation)) {
      return NextResponse.json({ error: `Unknown table operation ${operation}` }, { status: 400 });
    }
    const configured = process.env.POKER_ENGINE_URL ?? "http://127.0.0.1:8765";
    const url = new URL(configured);
    if (url.hostname !== "127.0.0.1" || url.protocol !== "http:") {
      throw new Error(`POKER_ENGINE_URL must be a local HTTP engine URL, got ${configured}`);
    }
    const response = await fetch(`${url.origin}/table/${operation}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), cache: "no-store", signal: AbortSignal.timeout(10000),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return NextResponse.json({ error: `Poker engine unavailable: ${reason}. Restart the application with npm run dev or npm run start; these commands start both services.` }, { status: 502 });
  }
}
