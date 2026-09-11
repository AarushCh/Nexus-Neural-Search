"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api, onWaking } from "./api";
import { Media, SearchModel } from "./types";

interface Store {
  token: string | null;
  user: string | null;
  online: boolean;
  waking: boolean;
  checkHealth: () => void;
  wishlistIds: Set<string>;
  model: SearchModel;
  setModel: (m: SearchModel) => void;
  login: (token: string, user: string) => Promise<void>;
  logout: () => void;
  toggleWishlist: (m: Media) => Promise<boolean>;
  isSaved: (id: string) => boolean;
  toast: (msg: string) => void;
  toastMsg: string | null;
}

const Ctx = createContext<Store | null>(null);

export function StoreProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<string | null>(null);
  const [online, setOnline] = useState(false);
  const [waking, setWaking] = useState(false);
  const [wishlistIds, setWishlistIds] = useState<Set<string>>(new Set());
  const [model, setModelState] = useState<SearchModel>("api");
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const checking = useRef(false);

  const toast = useCallback((msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 2200);
  }, []);

  const refreshWishlist = useCallback(async (tok: string) => {
    try {
      const items = await api.wishlist(tok);
      setWishlistIds(new Set(items.map((i) => String(i.id))));
    } catch (e) {
      if ((e as Error).message === "UNAUTHORIZED") logoutRaw();
    }
  }, []);

  const logoutRaw = () => {
    localStorage.removeItem("nexus_token");
    localStorage.removeItem("nexus_user");
    setToken(null);
    setUser(null);
    setWishlistIds(new Set());
  };

  const checkHealth = useCallback(() => {
    if (checking.current) return;
    checking.current = true;
    api
      .health()
      .then(setOnline)
      .finally(() => {
        checking.current = false;
      });
  }, []);

  useEffect(() => {
    const t = localStorage.getItem("nexus_token");
    const u = localStorage.getItem("nexus_user");
    const m = localStorage.getItem("nexus_model") as SearchModel | null;
    if (m) setModelState(m);
    if (t) {
      setToken(t);
      setUser(u);
      refreshWishlist(t);
    }
    checkHealth();
    onWaking(setWaking);
    return () => onWaking(null);
  }, [refreshWishlist, checkHealth]);

  // The health probe used to run exactly once, at mount. If the API was asleep
  // right then, the badge read OFFLINE until a manual reload — which is what you
  // came back to after being AFK. Re-probe whenever the tab regains attention,
  // and keep retrying on a slow loop while we believe we're offline.
  useEffect(() => {
    const onWake = () => {
      if (document.visibilityState === "visible") checkHealth();
    };
    document.addEventListener("visibilitychange", onWake);
    window.addEventListener("focus", onWake);
    window.addEventListener("online", checkHealth);
    const id = online ? undefined : setInterval(checkHealth, 20_000);
    return () => {
      document.removeEventListener("visibilitychange", onWake);
      window.removeEventListener("focus", onWake);
      window.removeEventListener("online", checkHealth);
      if (id) clearInterval(id);
    };
  }, [checkHealth, online]);

  const login = useCallback(
    async (tok: string, u: string) => {
      localStorage.setItem("nexus_token", tok);
      localStorage.setItem("nexus_user", u);
      setToken(tok);
      setUser(u);
      await refreshWishlist(tok);
    },
    [refreshWishlist]
  );

  const logout = useCallback(() => logoutRaw(), []);

  const setModel = useCallback((m: SearchModel) => {
    setModelState(m);
    localStorage.setItem("nexus_model", m);
  }, []);

  const toggleWishlist = useCallback(
    async (m: Media): Promise<boolean> => {
      if (!token) return false;
      const id = String(m.id);
      const currentlySaved = wishlistIds.has(id);
      const next = new Set(wishlistIds);
      if (currentlySaved) next.delete(id);
      else next.add(id);
      setWishlistIds(next);
      try {
        await api.toggleWishlist(id, !currentlySaved, token);
        if (!currentlySaved) api.interact(id, "like", token);
      } catch {
        setWishlistIds(wishlistIds); // revert
      }
      return !currentlySaved;
    },
    [token, wishlistIds]
  );

  const isSaved = useCallback((id: string) => wishlistIds.has(String(id)), [wishlistIds]);

  const value = useMemo<Store>(
    () => ({
      token,
      user,
      online,
      waking,
      checkHealth,
      wishlistIds,
      model,
      setModel,
      login,
      logout,
      toggleWishlist,
      isSaved,
      toast,
      toastMsg,
    }),
    [token, user, online, waking, checkHealth, wishlistIds, model, setModel, login, logout,
     toggleWishlist, isSaved, toast, toastMsg]
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useStore must be used within StoreProvider");
  return ctx;
}
