import { Media, FeedRow, SearchModel, Filters } from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "https://nexus-neural-search.onrender.com";

// The API sleeps on Render's free tier and takes ~40s to boot. A bare fetch()
// never times out, so a cold start used to look like a dead button: the spinner
// span forever with no signal. We give normal calls a short budget, retry once
// on a wake-up, and let the second attempt run long enough to cover a cold boot.
const WARM_TIMEOUT = 12_000;
const COLD_TIMEOUT = 75_000;

// Lets the UI say "waking the neural core" instead of showing a stuck spinner.
let wakingListener: ((waking: boolean) => void) | null = null;
export function onWaking(fn: ((waking: boolean) => void) | null) {
  wakingListener = fn;
}
function setWaking(v: boolean) {
  wakingListener?.(v);
}

function authHeaders(token?: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** fetch + timeout + one retry, so a sleeping backend wakes instead of failing. */
async function req(path: string, init: RequestInit = {}): Promise<Response> {
  let lastErr: unknown;
  for (let attempt = 0; attempt < 2; attempt++) {
    const cold = attempt > 0;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), cold ? COLD_TIMEOUT : WARM_TIMEOUT);
    if (cold) setWaking(true);
    try {
      return await fetch(`${API_URL}${path}`, { ...init, signal: ctrl.signal });
    } catch (e) {
      lastErr = e;
    } finally {
      clearTimeout(timer);
      if (cold) setWaking(false);
    }
  }
  throw new Error(
    lastErr instanceof Error && lastErr.name === "AbortError" ? "TIMEOUT" : "OFFLINE"
  );
}

async function jsonOrThrow(res: Response) {
  if (res.status === 401) throw new Error("UNAUTHORIZED");
  // 503 = the search index itself is down, which is NOT "no results found".
  if (res.status === 503) throw new Error("INDEX_DOWN");
  if (!res.ok) {
    let detail = "Request failed";
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  /** true only when the API is up AND the search index actually has data. */
  async health(): Promise<boolean> {
    try {
      return (await req("/")).ok;
    } catch {
      return false;
    }
  },

  async login(username: string, password: string): Promise<string> {
    const form = new FormData();
    form.append("username", username);
    form.append("password", password);
    const res = await req("/login", { method: "POST", body: form });
    if (!res.ok) {
      // A 401 here means bad credentials (not a session-expiry), so show the real reason.
      let detail = "Invalid credentials";
      try {
        detail = (await res.json()).detail || detail;
      } catch {}
      throw new Error(detail);
    }
    return (await res.json()).access_token as string;
  },

  async signup(username: string, email: string, password: string): Promise<void> {
    const res = await req("/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password }),
    });
    await jsonOrThrow(res);
  },

  async search(
    text: string,
    opts: { model: SearchModel; token?: string | null; filters?: Filters; top_k?: number }
  ): Promise<Media[]> {
    const endpoint = opts.token ? "/recommend/personalized" : "/recommend";
    const res = await req(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(opts.token) },
      body: JSON.stringify({
        text,
        top_k: opts.top_k ?? 12,
        model: opts.model,
        ...(opts.filters || {}),
      }),
    });
    return jsonOrThrow(res);
  },

  async similar(id: string): Promise<Media[]> {
    const res = await req("/similar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    return jsonOrThrow(res);
  },

  async detail(id: string): Promise<Media> {
    const res = await req(`/title/${encodeURIComponent(id)}`);
    return jsonOrThrow(res);
  },

  async discover(): Promise<FeedRow[]> {
    const res = await req("/discover");
    return jsonOrThrow(res);
  },

  async feed(token: string): Promise<FeedRow[]> {
    const res = await req("/feed", { headers: authHeaders(token) });
    return jsonOrThrow(res);
  },

  async wishlist(token: string): Promise<Media[]> {
    const res = await req("/wishlist", { headers: authHeaders(token) });
    return jsonOrThrow(res);
  },

  async toggleWishlist(id: string, add: boolean, token: string): Promise<void> {
    const res = await req(`/wishlist/${add ? "add" : "remove"}/${encodeURIComponent(id)}`, {
      method: add ? "POST" : "DELETE",
      headers: authHeaders(token),
    });
    await jsonOrThrow(res);
  },

  async interact(media_id: string, kind: "view" | "like" | "dismiss", token: string): Promise<void> {
    try {
      await req("/interactions", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders(token) },
        body: JSON.stringify({ media_id, kind }),
      });
    } catch {
      /* best-effort */
    }
  },

  async searchHistory(token: string): Promise<{ query: string; model: string; at: string }[]> {
    const res = await req("/history/search", { headers: authHeaders(token) });
    return jsonOrThrow(res);
  },
};
