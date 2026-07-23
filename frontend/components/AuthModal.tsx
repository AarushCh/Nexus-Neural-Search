"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { useStore } from "@/lib/store";

export default function AuthModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { login } = useStore();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [show, setShow] = useState(false);
  const [f, setF] = useState({ username: "", email: "", password: "", confirm: "" });

  const set = (k: string, v: string) => setF((p) => ({ ...p, [k]: v }));

  const doLogin = async () => {
    if (!f.username || !f.password) return setError("MISSING CREDENTIALS");
    setBusy(true);
    setError("");
    try {
      const token = await api.login(f.username, f.password);
      await login(token, f.username);
      onClose();
    } catch (e) {
      setError("LOGIN FAILED: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const doSignup = async () => {
    if (f.password !== f.confirm) return setError("PASSWORDS DO NOT MATCH");
    setBusy(true);
    setError("");
    try {
      await api.signup(f.username, f.email, f.password);
      setMode("login");
      setError("ACCOUNT CREATED — PLEASE LOGIN");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={`modal ${open ? "" : "hidden"}`} onClick={onClose}>
      <div className="modal-box auth-box" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          ✕
        </button>
        <h2 className="modal-title">{mode === "login" ? "ACCESS LOGIN" : "NEW IDENTITY"}</h2>
        <div className="auth-forms">
          <input placeholder="Username" value={f.username} onChange={(e) => set("username", e.target.value)} />
          {mode === "signup" && (
            <input placeholder="Email Address" value={f.email} onChange={(e) => set("email", e.target.value)} />
          )}
          <div className="password-wrap">
            <input
              type={show ? "text" : "password"}
              placeholder="Password"
              value={f.password}
              onChange={(e) => set("password", e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && mode === "login" && doLogin()}
            />
            <span className="toggle-pass" onClick={() => setShow((s) => !s)}>
              {show ? "HIDE" : "SHOW"}
            </span>
          </div>
          {mode === "signup" && (
            <div className="password-wrap">
              <input
                type={show ? "text" : "password"}
                placeholder="Confirm Password"
                value={f.confirm}
                onChange={(e) => set("confirm", e.target.value)}
              />
            </div>
          )}
          <button className="primary" disabled={busy} onClick={mode === "login" ? doLogin : doSignup}>
            {busy ? "…" : mode === "login" ? "AUTHENTICATE" : "REGISTER"}
          </button>
          <button
            className="secondary"
            onClick={() => {
              setMode(mode === "login" ? "signup" : "login");
              setError("");
            }}
          >
            {mode === "login" ? "CREATE ID" : "BACK"}
          </button>
        </div>
        {error && <p id="auth-error">{error}</p>}
      </div>
    </div>
  );
}
