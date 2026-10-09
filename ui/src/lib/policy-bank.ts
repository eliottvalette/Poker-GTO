import { fetchPolicyCatalog, type PolicyCatalog } from "./policy-catalog";
import { loadPublishedPolicies, type LoadedAveragePolicy } from "./onnx-policy";

type Entry = { model: LoadedAveragePolicy; users: number; retired: boolean; released: boolean };

/** Stable query handles; retired inference sessions survive only their in-flight queries. */
export class PolicyBank {
  private current: Record<number, Entry> = {};
  private fingerprint = "";
  private pending: Promise<boolean> | null = null;
  private closed = false;
  readonly models: Record<number, LoadedAveragePolicy> = {};

  constructor(private catalog = fetchPolicyCatalog,
    private load: (catalog: PolicyCatalog) => Promise<Record<number, LoadedAveragePolicy>> = loadPublishedPolicies) {}

  private async retire(entry: Entry) {
    entry.retired = true;
    if (!entry.users && !entry.released) {
      entry.released = true;
      await entry.model.release();
    }
  }

  refresh(): Promise<boolean> {
    if (this.closed) return Promise.reject(new Error("Policy bank is closed"));
    if (!this.pending) this.pending = this.update().finally(() => { this.pending = null; });
    return this.pending;
  }

  private async update(): Promise<boolean> {
    const catalog = await this.catalog();
    const fingerprint = JSON.stringify(catalog);
    if (fingerprint === this.fingerprint) return false;
    const loaded = await this.load(catalog);
    if (this.closed) { await Promise.all(Object.values(loaded).map(model => model.release())); return false; }
    const previous = this.current;
    this.current = Object.fromEntries(Object.entries(loaded).map(([count, model]) => [count, { model, users: 0, retired: false, released: false }]));
    for (const count of Object.keys(loaded).map(Number)) {
      if (this.models[count]) continue;
      const bank = this;
      this.models[count] = {
        get manifest() {
          const entry = bank.current[count];
          if (!entry) throw new Error(`Policy unavailable for ${count} players`);
          return entry.model.manifest;
        },
        async query(observation) {
          const entry = bank.current[count];
          if (bank.closed || !entry) throw new Error(`Policy unavailable for ${count} players`);
          entry.users++;
          try { return await entry.model.query(observation); }
          finally { entry.users--; if (entry.retired) await bank.retire(entry); }
        },
        async release() { throw new Error("Shared policies are owned by PolicyBank"); },
      };
    }
    for (const count of Object.keys(this.models).map(Number)) if (!loaded[count]) delete this.models[count];
    this.fingerprint = fingerprint;
    await Promise.all(Object.values(previous).map(entry => this.retire(entry)));
    return true;
  }

  async close() {
    this.closed = true;
    try { await this.pending; }
    finally {
      await Promise.all(Object.values(this.current).map(entry => this.retire(entry)));
      this.current = {};
    }
  }
}
