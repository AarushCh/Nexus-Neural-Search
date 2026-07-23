// Poster URL resolver — mirrors the old frontend's wsrv.nl proxy + dead-CDN fallback.
export function posterUrl(image: string | undefined, title: string): string {
  const placeholder = `https://placehold.co/300x450/111/FFF?text=${encodeURIComponent(title || "N/A")}`;
  const raw = (image || "").trim();
  const dead = raw.includes("cdn-dena.com") || raw.includes("myanimelist.cdn-dena");
  if (!raw || dead || raw.includes("null") || raw.length < 5) return placeholder;
  if (raw.includes("pollinations") || raw.includes("bing.net")) return raw;
  return `https://wsrv.nl/?url=${encodeURIComponent(raw)}&w=400&output=webp&default=${encodeURIComponent(placeholder)}`;
}

export function typeClass(type?: string): string {
  const t = (type || "").toUpperCase();
  if (t.includes("ANIME")) return "type-anime";
  if (t.includes("DOC")) return "type-doc";
  if (t.includes("TV")) return "type-tv";
  return "type-movie";
}
