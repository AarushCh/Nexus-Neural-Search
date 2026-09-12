"use client";

import { useEffect, useState } from "react";
import { useStore } from "@/lib/store";
import { SearchModel } from "@/lib/types";

const MODELS: { id: SearchModel; title: string; desc: string }[] = [
  { id: "internal", title: "FreeMe Neural", desc: "Local hybrid embedding + rerank." },
  { id: "api", title: "NVIDIA: Nemotron", desc: "Grounded reasoning over real results." },
];

export function MenuButton({ onClick, open }: { onClick: () => void; open: boolean }) {
  return (
    <button id="menu-btn" className="menu-btn" onClick={onClick} aria-label="Menu" data-open={open}>
      <div className="bar" />
      <div className="bar" />
      <div className="bar" />
    </button>
  );
}

/* Inline SVGs rather than an icon package: five glyphs is not worth a dependency,
   and `currentColor` lets them inherit the active/hover states for free. */
const Icon = {
  home: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 10.5 12 3l9 7.5" /><path d="M5 9.5V21h14V9.5" /><path d="M9.5 21v-6h5v6" />
    </svg>
  ),
  heart: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.8 5.6a5 5 0 0 0-7.1 0L12 7.3l-1.7-1.7a5 5 0 1 0-7.1 7.1l8.8 8.8 8.8-8.8a5 5 0 0 0 0-7.1z" />
    </svg>
  ),
  history: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /><path d="M12 7.5V12l3.5 2" />
    </svg>
  ),
  info: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" /><path d="M12 11v5" /><path d="M12 7.6h.01" />
    </svg>
  ),
  login: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" /><path d="M10 17l5-5-5-5" /><path d="M15 12H3" />
    </svg>
  ),
  logout: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h4" /><path d="M16 17l5-5-5-5" /><path d="M21 12H9" />
    </svg>
  ),
};

function NavItem({
  icon, label, active, onClick,
}: { icon: React.ReactNode; label: string; active?: boolean; onClick: () => void }) {
  return (
    <button
      className={`nav-item ${active ? "active" : ""}`}
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      <span className="nav-icon">{icon}</span>
      <span className="nav-label">{label}</span>
    </button>
  );
}

export function Sidebar({
  open,
  onClose,
  onNav,
  onLogin,
  view = "search",
}: {
  open: boolean;
  onClose: () => void;
  onNav: (view: "search" | "wishlist" | "history" | "about") => void;
  onLogin: () => void;
  view?: string;
}) {
  const { user, token, online, waking, logout } = useStore();
  const go = (v: "search" | "wishlist" | "history" | "about") => () => {
    onNav(v);
    onClose();
  };
  return (
    <nav id="sidebar" className={`sidebar ${open ? "open" : ""}`} aria-label="Main">
      <div className="hud-header">
        <span className="hud-title">System HUD</span>
      </div>

      <div className="status-box" title={waking ? "Waking" : online ? "Online" : "Offline"}>
        {/* Same 24px slot the nav icons use, so the dot lines up with them
            in the collapsed rail instead of sitting a few pixels to the left. */}
        <span className="nav-icon">
          <span
            className="status-indicator"
            style={{ background: waking ? "#ffb800" : online ? "#00ff9d" : "#ff0055" }}
          />
        </span>
        <span className="nav-label">{waking ? "WAKING" : online ? "ONLINE" : "OFFLINE"}</span>
      </div>

      <div className="nav-menu">
        <NavItem icon={Icon.home} label="Home" active={view === "search" || view === "home"} onClick={go("search")} />
        <NavItem icon={Icon.heart} label="Wishlist" active={view === "wishlist"} onClick={go("wishlist")} />
        <NavItem icon={Icon.history} label="History" active={view === "history"} onClick={go("history")} />
        <NavItem icon={Icon.info} label="About" onClick={go("about")} />
        {token ? (
          <NavItem icon={Icon.logout} label="Logout" onClick={() => { logout(); onClose(); }} />
        ) : (
          <NavItem icon={Icon.login} label="Login" onClick={onLogin} />
        )}
      </div>

      <div className="hud-footer">
        <span className="nav-label">USER: <span>{user || "GUEST"}</span></span>
      </div>
    </nav>
  );
}

export function ModelSelector() {
  const { model, setModel } = useStore();
  const [open, setOpen] = useState(false);
  const current = MODELS.find((m) => m.id === model) || MODELS[1];
  return (
    <div className="model-selector">
      <div className="selected-model" onClick={() => setOpen((o) => !o)}>
        <span>{current.title.toUpperCase()}</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </div>
      <div className={`model-menu ${open ? "open" : ""}`}>
        {MODELS.map((m) => (
          <div
            key={m.id}
            className={`model-option ${m.id === model ? "active" : ""}`}
            onClick={() => {
              setModel(m.id);
              setOpen(false);
            }}
          >
            <div className="opt-title">{m.title}</div>
            <div className="opt-desc">{m.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ThemeToggle() {
  const [light, setLight] = useState(false);
  useEffect(() => {
    setLight(localStorage.getItem("theme") === "light");
  }, []);
  const onChange = (checked: boolean) => {
    setLight(checked);
    document.documentElement.classList.toggle("light-mode", checked);
    localStorage.setItem("theme", checked ? "light" : "dark");
  };
  return (
    <div className="theme-wrapper">
      <label className="switch" title="Toggle Theme">
        <input type="checkbox" checked={light} onChange={(e) => onChange(e.target.checked)} />
        <span className="slider">
          <svg className="icon sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </svg>
          <svg className="icon moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
          </svg>
        </span>
      </label>
    </div>
  );
}
