import { Media, FeedRow, SearchModel, Filters } from "./types";

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "https://nexus-neural-search.onrender.com";

function authHeaders(token?: string | null): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function jsonOrThrow(res: Response) {
  if (res.status === 401) throw new Error("UNAUTHORIZED");
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
  async health(): Promise<boolean> {
    try {
      const r = await fetch(`${API_URL}/`);
      return r.ok;
    } catch {
      return false;
    }
  },

  async login(username: string, password: string): Promise<string> {
    const form = new FormData();
    form.append("username", username);
    form.append("password", password);
    const res = await fetch(`${API_URL}/login`, { method: "POST", body: form });
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
    const res = await fetch(`${API_URL}/signup`, {
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
    const res = await fetch(`${API_URL}${endpoint}`, {
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
    const res = await fetch(`${API_URL}/similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    return jsonOrThrow(res);
  },

  async detail(id: string): Promise<Media> {
    const res = await fetch(`${API_URL}/title/${encodeURIComponent(id)}`);
    return jsonOrThrow(res);
  },

  async discover(): Promise<FeedRow[]> {
    const res = await fetch(`${API_URL}/discover`);
    return jsonOrThrow(res);
  },

  async feed(token: string): Promise<FeedRow[]> {
    const res = await fetch(`${API_URL}/feed`, { headers: authHeaders(token) });
    return jsonOrThrow(res);
  },

  async wishlist(token: string): Promise<Media[]> {
    const res = await fetch(`${API_URL}/wishlist`, { headers: authHeaders(token) });
    return jsonOrThrow(res);
  },

  async toggleWishlist(id: string, add: boolean, token: string): Promise<void> {
    const res = await fetch(`${API_URL}/wishlist/${add ? "add" : "remove"}/${encodeURIComponent(id)}`, {
      method: add ? "POST" : "DELETE",
      headers: authHeaders(token),
    });
    await jsonOrThrow(res);
  },

  async interact(media_id: string, kind: "view" | "like" | "dismiss", token: string): Promise<void> {
    try {
      await fetch(`${API_URL}/interactions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders(token) },
        body: JSON.stringify({ media_id, kind }),
      });
    } catch {
      /* best-effort */
    }
  },

  async searchHistory(token: string): Promise<{ query: string; model: string; at: string }[]> {
    const res = await fetch(`${API_URL}/history/search`, { headers: authHeaders(token) });
    return jsonOrThrow(res);
  },
};
