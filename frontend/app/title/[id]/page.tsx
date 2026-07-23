"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Media } from "@/lib/types";
import { api } from "@/lib/api";
import { posterUrl, typeClass } from "@/lib/img";
import { useStore } from "@/lib/store";

export default function TitlePage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { token, isSaved, toggleWishlist, toast } = useStore();
  const [m, setM] = useState<Media | null>(null);
  const [loading, setLoading] = useState(true);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!id) return;
    setLoading(true);
    if (token) api.interact(String(id), "view", token);
    api
      .detail(String(id))
      .then((d) => {
        setM(d);
        setSaved(isSaved(d.id));
      })
      .catch(() => setM(null))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  if (loading) return <div className="detail-page-state">Loading…</div>;
  if (!m) return <div className="detail-page-state">Title not found. <Link href="/">← Home</Link></div>;

  const ratingVal = parseFloat(String(m.rating));
  const rating = !isNaN(ratingVal) && ratingVal > 0 ? `★ ${ratingVal.toFixed(1)}` : "";
  const type = (m.type || "MOVIE").toUpperCase();

  const onHeart = async () => {
    if (!token) return toast("Login to save titles");
    setSaved(await toggleWishlist(m));
  };

  return (
    <div className="detail-page">
      <div className="detail-page-nav">
        <button className="detail-btn" onClick={() => router.back()}>← Back</button>
        <Link className="detail-btn" href="/">Home</Link>
      </div>

      <div className="detail-box detail-page-box">
        {m.trailer_key ? (
          <div className="detail-trailer">
            <iframe
              src={`https://www.youtube.com/embed/${m.trailer_key}?rel=0`}
              title="Trailer"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          </div>
        ) : (
          <div
            className="detail-backdrop"
            style={{ backgroundImage: `url(${m.backdrop || posterUrl(m.image, m.title)})` }}
          />
        )}

        <div className={`detail-body ${m.trailer_key ? "with-trailer" : ""}`}>
          <div className="detail-head">
            <img className="detail-poster" src={posterUrl(m.image, m.title)} alt={m.title} />
            <div>
              <div className="detail-title">{m.title}</div>
              <div className="detail-meta">
                <span className={`type-badge ${typeClass(m.type)}`}>{type}</span>
                {rating && <span className="rating-badge">{rating}</span>}
                {m.year && <span className="badge">{m.year}</span>}
                {m.runtime ? <span className="badge">{m.runtime} min</span> : null}
              </div>
            </div>
          </div>

          <div className="detail-actions">
            <button className={`detail-btn ${saved ? "save" : ""}`} onClick={onHeart}>
              {saved ? "♥ SAVED" : "♡ SAVE"}
            </button>
            <Link className="detail-btn primary-btn" href={`/?similar=${encodeURIComponent(String(m.id))}`}>
              EXPLORE SIMILAR
            </Link>
          </div>

          <p className="detail-overview">{m.description || "No description available."}</p>

          {m.providers && m.providers.length > 0 && (
            <>
              <div className="detail-section-h">Where to Watch</div>
              <div className="providers">
                {m.providers.slice(0, 8).map((p) => (
                  <div className="provider-chip" key={p.name}>
                    {p.logo && <img src={p.logo} alt={p.name} />}
                    <span>{p.name}</span>
                  </div>
                ))}
              </div>
            </>
          )}

          {m.cast && m.cast.length > 0 && (
            <>
              <div className="detail-section-h">Cast</div>
              <div className="cast-track">
                {m.cast.map((c) => (
                  <div className="cast-card" key={c.name}>
                    <img
                      className="cast-photo"
                      src={c.profile || `https://placehold.co/96x96/111/FFF?text=${encodeURIComponent(c.name?.[0] || "?")}`}
                      alt={c.name}
                    />
                    <div className="cast-name">{c.name}</div>
                    {c.character && <div className="cast-char">{c.character}</div>}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
