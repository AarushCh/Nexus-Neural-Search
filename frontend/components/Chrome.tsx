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

export function Sidebar({
  open,
  onClose,
  onNav,
  onLogin,
}: {
  open: boolean;
  onClose: () => void;
  onNav: (view: "search" | "wishlist" | "history" | "about") => void;
  onLogin: () => void;
}) {
  const { user, token, online, logout } = useStore();
  return (
    <div id="sidebar" className={`sidebar ${open ? "open" : ""}`}>
      <div className="hud-header">SYSTEM HUD</div>
      <div className="status-box">
        <div className="status-indicator" style={{ background: online ? "#00ff9d" : "#ff0055" }} />
        <span>{online ? "ONLINE" : "OFFLINE"}</span>
      </div>
      <div className="nav-menu">
        <div className="nav-item active" onClick={() => onNav("search")}>NEURAL SEARCH</div>
        <div className="nav-item" onClick={() => onNav("wishlist")}>♥ WISHLIST</div>
        <div className="nav-item" onClick={() => onNav("history")}>↺ HISTORY</div>
        <div className="nav-item" onClick={() => onNav("about")}>ⓘ ABOUT</div>
        {token ? (
          <div className="nav-item" onClick={() => { logout(); onClose(); }}>LOGOUT</div>
        ) : (
          <div className="nav-item" onClick={onLogin}>LOGIN</div>
        )}
      </div>
      <div className="hud-footer">USER: <span>{user || "GUEST"}</span></div>
    </div>
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
