import { fetchPolicyCatalog, type PolicyCatalog } from "./policy-catalog";
import { loadPublishedPolicies, type LoadedAveragePolicy } from "./onnx-policy";

type Entry = { model: LoadedAveragePolicy; users: number; retired: boolean; released: boolean };

/** Stable query handles; retired inference sessions survive only their in-flight queries. */
export class PolicyBank {
  private current: Record<number, Entry> = {};
  private fingerprint = "";
  private lastCatalog: PolicyCatalog | undefined;
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
    const changed = Object.fromEntries(Object.entries(catalog.exports).filter(([track, entry]) =>
      JSON.stringify(entry) !== JSON.stringify(this.lastCatalog?.exports[track])));
    const loaded = await this.load({ ...catalog, exports: changed });
    if (this.closed) { await Promise.all(Object.values(loaded).map(model => model.release())); return false; }
    const previous = this.current;
    this.current = { ...previous, ...Object.fromEntries(Object.entries(loaded).map(([count, model]) => [count, { model, users: 0, retired: false, released: false }])) };
    for (const count of Object.keys(this.current).map(Number)) {
      // Track removals must not retain an invisible model indefinitely.
      if (!((count === 2 ? "hu" : "3max") in catalog.exports)) delete this.current[count];
    }
    for (const count of Object.keys(this.current).map(Number)) {
      if (this.models[count]) continue;
      const getEntry = () => this.current[count];
      const query: LoadedAveragePolicy["query"] = async observation => {
        const entry = getEntry();
        if (this.closed || !entry) throw new Error(`Policy unavailable for ${count} players`);
        entry.users++;
        try { return await entry.model.query(observation); }
        finally { entry.users--; if (entry.retired) await this.retire(entry); }
      };
      this.models[count] = {
        get manifest() {
          const entry = getEntry();
          if (!entry) throw new Error(`Policy unavailable for ${count} players`);
          return entry.model.manifest;
        },
        query,
        async release() { throw new Error("Shared policies are owned by PolicyBank"); },
      };
    }
    for (const count of Object.keys(this.models).map(Number)) if (!this.current[count]) delete this.models[count];
    this.fingerprint = fingerprint;
    this.lastCatalog = catalog;
    await Promise.all(Object.entries(previous).filter(([count, entry]) => this.current[Number(count)] !== entry).map(([, entry]) => this.retire(entry)));
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
