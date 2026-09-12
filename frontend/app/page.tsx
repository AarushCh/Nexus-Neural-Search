"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Media, FeedRow, Filters } from "@/lib/types";
import { useStore } from "@/lib/store";
import { MenuButton, Sidebar, ModelSelector, ThemeToggle } from "@/components/Chrome";
import AuthModal from "@/components/AuthModal";
import DetailModal from "@/components/DetailModal";
import Feed from "@/components/Feed";
import Card from "@/components/Card";
import { randomPrompt } from "@/lib/prompts";

const CATS = [
  { id: "ALL", label: "ALL" },
  { id: "MOVIE", label: "MOVIES" },
  { id: "TV", label: "TV" },
  { id: "ANIME", label: "ANIME" },
  { id: "DOCUMENTARY", label: "DOCS" },
];

type View = "home" | "results" | "wishlist" | "similar";
// Why a result grid is empty. "No matches" and "the backend is down" used to
// render identically as NO PATTERNS FOUND.
type Problem = null | "none" | "index" | "timeout" | "offline";

export default function Home() {
  const { token, model, online, waking, checkHealth, logout, toast, toastMsg } = useStore();

  const [view, setView] = useState<View>("home");
  const [feedRows, setFeedRows] = useState<FeedRow[]>([]);
  const [results, setResults] = useState<Media[]>([]);
  const [loading, setLoading] = useState(false);
  const [problem, setProblem] = useState<Problem>(null);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("ALL");
  const [minRating, setMinRating] = useState(0);
  const [sortByRating, setSortByRating] = useState(false);
  const [similarTitle, setSimilarTitle] = useState("");
  const [detail, setDetail] = useState<Media | null>(null);
  const [sidebar, setSidebar] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [history, setHistory] = useState<{ query: string; at: string }[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  // The results the user left behind when opening a similar-items view, so
  // "RETURN TO SEARCH" goes back to them instead of dumping them at home.
  const priorResults = useRef<Media[] | null>(null);
  const lastQuery = useRef("");
  const urlBootstrap = useRef(false);

  const loadFeed = useCallback(async () => {
    setHistory([]);
    try {
      const rows = token ? await api.feed(token) : await api.discover();
      setFeedRows(rows);
    } catch {
      try {
        setFeedRows(await api.discover());
      } catch {}
    }
  }, [token]);

  // Home feed loads on mount and on auth change — unless a search/similar view
  // was restored from the URL (so returning here from a detail page keeps it).
  useEffect(() => {
    if (urlBootstrap.current || view !== "home") return;
    loadFeed();
  }, [loadFeed, view]);

  useEffect(() => {
    document.body.classList.toggle("menu-open", sidebar);
  }, [sidebar]);

  const buildFilters = (cat: string, rating: number): Filters => ({
    category: cat === "ALL" ? null : cat,
    min_rating: rating || null,
  });

  const runSearch = useCallback(
    async (q: string, cat = category, rating = minRating) => {
      if (!q.trim()) return;
      lastQuery.current = q;
      setQuery(q);
      setView("results");
      setLoading(true);
      setProblem(null);
      setSidebar(false);
      // Persist the search in the URL so opening a detail page and pressing
      // Back restores this exact results view instead of dropping to home.
      const p = new URLSearchParams({ q });
      if (cat && cat !== "ALL") p.set("cat", cat);
      if (rating) p.set("rating", String(rating));
      window.history.replaceState({}, "", `/?${p.toString()}`);
      const filters = buildFilters(cat, rating);
      try {
        let data: Media[];
        try {
          data = await api.search(q, { model, token, filters });
        } catch (e) {
          // An expired JWT used to dead-end here with an empty grid. A search
          // doesn't need auth, so drop the stale token and just run it anonymously.
          if ((e as Error).message !== "UNAUTHORIZED") throw e;
          logout();
          toast("Session expired — searching as guest");
          data = await api.search(q, { model, token: null, filters });
        }
        setResults(data);
        priorResults.current = data;
        setProblem(data.length ? null : "none");
      } catch (e) {
        const msg = (e as Error).message;
        setResults([]);
        priorResults.current = null;
        setProblem(msg === "INDEX_DOWN" ? "index" : msg === "TIMEOUT" ? "timeout" : "offline");
        checkHealth();
      } finally {
        setLoading(false);
      }
    },
    [category, minRating, model, token, toast, logout, checkHealth]
  );

  const openSimilar = useCallback(async (m: Media) => {
    setDetail(null);
    setView("similar");
    setSimilarTitle(m.title);
    setLoading(true);
    setProblem(null);
    try {
      const data = await api.similar(String(m.id));
      setResults(data);
      setProblem(data.length ? null : "none");
    } catch (e) {
      const msg = (e as Error).message;
      setResults([]);
      setProblem(msg === "INDEX_DOWN" ? "index" : msg === "TIMEOUT" ? "timeout" : "offline");
    } finally {
      setLoading(false);
    }
  }, []);

  // On first mount, restore the view encoded in the URL: a saved search (?q=)
  // — so Back from a detail page lands on the same results — or an
  // "Explore Similar" deep-link (?similar=<id>) from the standalone title page.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sid = params.get("similar");
    const q = params.get("q");
    if (sid) {
      urlBootstrap.current = true;
      window.history.replaceState({}, "", "/");
      api.detail(sid).then((m) => openSimilar(m)).catch(() => {});
    } else if (q) {
      urlBootstrap.current = true;
      const cat = params.get("cat") || "ALL";
      const rating = Number(params.get("rating") || 0);
      setCategory(cat);
      setMinRating(rating);
      if (inputRef.current) inputRef.current.value = q;
      runSearch(q, cat, rating);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const openWishlist = useCallback(async () => {
    setSidebar(false);
    if (!token) return setAuthOpen(true);
    setView("wishlist");
    setLoading(true);
    try {
      setResults(await api.wishlist(token));
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, [token]);

  const openHistory = useCallback(async () => {
    setSidebar(false);
    if (!token) return setAuthOpen(true);
    try {
      const h = await api.searchHistory(token);
      setHistory(h.map((x) => ({ query: x.query, at: new Date(x.at).toLocaleString() })));
      setView("home");
      setFeedRows([]);
    } catch {}
  }, [token]);

  const onNav = (v: "search" | "wishlist" | "history" | "about") => {
    if (v === "search") {
      urlBootstrap.current = false;
      window.history.replaceState({}, "", "/");
      setView("home");
      setSidebar(false);
      loadFeed();
    } else if (v === "wishlist") openWishlist();
    else if (v === "history") openHistory();
    else if (v === "about") {
      setAboutOpen(true);
      setSidebar(false);
    }
  };

  const changeCategory = (cat: string) => {
    setCategory(cat);
    if (view === "results") runSearch(lastQuery.current, cat, minRating);
  };
  const changeRating = (r: number) => {
    setMinRating(r);
    if (view === "results") runSearch(lastQuery.current, category, r);
  };

  let shown = results;
  if (sortByRating)
    shown = [...results].sort((a, b) => (parseFloat(String(b.rating)) || 0) - (parseFloat(String(a.rating)) || 0));

  const showFilters = view === "results";
  const showBack = view === "similar" || view === "wishlist";

  return (
    <>
      <MenuButton onClick={() => setSidebar((s) => !s)} open={sidebar} />
      <Sidebar open={sidebar} onClose={() => setSidebar(false)} onNav={onNav} view={view} onLogin={() => { setAuthOpen(true); setSidebar(false); }} />
      <ModelSelector />
      <ThemeToggle />
      <AuthModal open={authOpen} onClose={() => setAuthOpen(false)} />
      <DetailModal media={detail} onClose={() => setDetail(null)} onSimilar={openSimilar} />

      <div
        className={`main-container ${sidebar ? "menu-open" : ""}`}
        // Tapping the dimmed page closes the expanded rail. Only while it is
        // open, so this never swallows a normal click.
        onClick={sidebar ? () => setSidebar(false) : undefined}
      >
        <div className="hero">
          <h1 className="cyber-glitch" data-text="NEXUS">NEXUS</h1>
          <div className="subtitle">
            Multimodal Intelligence Engine v10 · {waking ? "WAKING…" : online ? "ONLINE" : "OFFLINE"}
          </div>
        </div>

        <div className="search-container">
          <div className="input-wrapper">
            <input
              ref={inputRef}
              id="search-input"
              placeholder="Describe your mood…"
              autoComplete="off"
              defaultValue={query}
              onKeyDown={(e) => e.key === "Enter" && runSearch((e.target as HTMLInputElement).value)}
            />
            <button className="icon-btn dice-btn" title="Random" onClick={() => {
              const q = randomPrompt();
              if (inputRef.current) inputRef.current.value = q;
              runSearch(q);
            }}>🎲</button>
            <button className="icon-btn search-btn" title="Search" onClick={() => runSearch(inputRef.current?.value || "")}>➜</button>
          </div>
        </div>

        {showFilters && (
          <div id="filter-bar">
            {CATS.map((c) => (
              <button key={c.id} className={`filter-btn ${category === c.id ? "active" : ""}`} onClick={() => changeCategory(c.id)}>
                {c.label}
              </button>
            ))}
            <div style={{ width: 1, background: "var(--border-color)", height: 20, margin: "0 6px" }} />
            <select className="filter-select" value={minRating} onChange={(e) => changeRating(Number(e.target.value))}>
              <option value={0}>ANY RATING</option>
              <option value={7}>7+</option>
              <option value={8}>8+</option>
              <option value={9}>9+</option>
            </select>
            <button className={`filter-btn ${sortByRating ? "active" : ""}`} onClick={() => setSortByRating((s) => !s)}>
              SORT: {sortByRating ? "RATING" : "RELEVANCE"}
            </button>
          </div>
        )}

        {showBack && (
          <button
            className="back-btn"
            onClick={() => {
              urlBootstrap.current = false;
              const back = view === "similar" && priorResults.current?.length ? priorResults.current : null;
              if (back) {
                const p = new URLSearchParams({ q: lastQuery.current });
                if (category !== "ALL") p.set("cat", category);
                if (minRating) p.set("rating", String(minRating));
                window.history.replaceState({}, "", `/?${p.toString()}`);
                setSimilarTitle("");
                setResults(back);
                setProblem(null);
                setView("results");
              } else {
                window.history.replaceState({}, "", "/");
                setView("home");
                loadFeed();
              }
            }}
          >
            {view === "similar" && priorResults.current?.length ? "← RETURN TO SEARCH" : "← RETURN HOME"}
          </button>
        )}

        {similarTitle && view === "similar" && (
          <div className="subtitle" style={{ marginBottom: 20 }}>SIMILAR TO {similarTitle.toUpperCase()}</div>
        )}

        {view === "home" && history.length > 0 && (
          <div style={{ width: "100%", maxWidth: 1000, margin: "0 auto 30px" }}>
            <div className="feed-row-title" style={{ padding: "0 40px" }}>Recent Searches</div>
            {history.map((h, i) => (
              <div className="history-row" key={i}>
                <div className="hist-time">{h.at}</div>
                <div className="hist-query">{h.query}</div>
                <button className="hist-btn" onClick={() => runSearch(h.query)}>RELOAD</button>
              </div>
            ))}
          </div>
        )}
        {view === "home" && history.length === 0 && <Feed rows={feedRows} onOpen={setDetail} />}

        {view !== "home" && (
          <div className="results-grid">
            {loading ? (
              <h2 style={{ gridColumn: "1/-1", textAlign: "center", color: "var(--neon-blue)" }}>
                {waking
                  ? "WAKING THE NEURAL CORE… THIS TAKES ~40s AFTER IDLE"
                  : view === "similar"
                  ? "VECTOR TRIANGULATION…"
                  : "NEURAL SCAN IN PROGRESS…"}
              </h2>
            ) : shown.length === 0 ? (
              <h3 style={{ gridColumn: "1/-1", textAlign: "center" }}>
                {view === "wishlist"
                  ? "YOUR WISHLIST IS EMPTY"
                  : problem === "index"
                  ? "SEARCH INDEX OFFLINE — THE CATALOGUE IS REBUILDING"
                  : problem === "timeout"
                  ? "CORE STILL WAKING — RUN THAT SEARCH AGAIN"
                  : problem === "offline"
                  ? "CANNOT REACH THE NEURAL CORE — CHECK YOUR CONNECTION"
                  : "NO PATTERNS FOUND"}
              </h3>
            ) : (
              shown.map((item) => (
                <Card key={String(item.id)} item={item} onOpen={setDetail} onSimilar={openSimilar} showSimilar={view !== "similar"} />
              ))
            )}
          </div>
        )}
      </div>

      <div className={`modal ${aboutOpen ? "" : "hidden"}`} onClick={() => setAboutOpen(false)}>
        <div className="modal-box about-box" onClick={(e) => e.stopPropagation()}>
          <button className="modal-close" onClick={() => setAboutOpen(false)}>✕</button>
          <h2 className="modal-title">System Overview</h2>
          <div className="about-content">
            <div className="about-section">
              <h3>Intelligence Cores</h3>
              <div className="model-card">
                <div className="mc-header"><span style={{ color: "var(--neon-blue)" }}>FreeMe Neural</span> <span className="badge">HYBRID</span></div>
                <p>Local hybrid retrieval — dense embeddings + BM25 keyword fusion, reranked by a cross-encoder. Understands vibes <strong>and</strong> exact names.</p>
              </div>
              <div className="model-card">
                <div className="mc-header"><span style={{ color: "#ff0055" }}>NVIDIA Nemotron</span> <span className="badge">RAG</span></div>
                <p>Grounded reasoning that reorders and explains <strong>real</strong> results — never hallucinated titles.</p>
              </div>
            </div>
            <div className="about-section">
              <h3>System Features</h3>
              <ul className="feature-list">
                <li><strong>Neural Search:</strong> describe a mood or type an exact title.</li>
                <li><strong>For You:</strong> recommendations that learn from your wishlist & history.</li>
                <li><strong>Detail View:</strong> trailers, cast, and where to watch.</li>
                <li><strong>Adaptive UI:</strong> reactive neural background, light/dark themes.</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {toastMsg && <div className="toast">{toastMsg}</div>}
    </>
  );
}
