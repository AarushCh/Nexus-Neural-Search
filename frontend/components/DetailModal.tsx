"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Media } from "@/lib/types";
import { api } from "@/lib/api";
import { posterUrl, typeClass } from "@/lib/img";
import { useStore } from "@/lib/store";

export default function DetailModal({
  media,
  onClose,
  onSimilar,
}: {
  media: Media | null;
  onClose: () => void;
  onSimilar: (m: Media) => void;
}) {
  const { token, isSaved, toggleWishlist, toast } = useStore();
  const [full, setFull] = useState<Media | null>(null);
  const [loading, setLoading] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!media) return;
    setFull(null);
    setSaved(isSaved(media.id));
    setLoading(true);
    if (token) api.interact(String(media.id), "view", token);
    api
      .detail(String(media.id))
      .then((d) => setFull({ ...media, ...d }))
      .catch(() => setFull(media))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [media]);

  if (!media) return null;
  const m = full || media;
  const ratingVal = parseFloat(String(m.rating));
  const rating = !isNaN(ratingVal) && ratingVal > 0 ? `★ ${ratingVal.toFixed(1)}` : "";
  const type = (m.type || "MOVIE").toUpperCase();

  const onHeart = async () => {
    if (!token) return toast("Login to save titles");
    setSaved(await toggleWishlist(m));
  };

  return (
    <div className="modal" onClick={onClose}>
      <div className="modal-box detail-box" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose} style={{ zIndex: 10 }}>
          ✕
        </button>

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
            <button className="detail-btn primary-btn" onClick={() => onSimilar(m)}>
              EXPLORE SIMILAR
            </button>
            <Link className="detail-btn" href={`/title/${encodeURIComponent(String(m.id))}`}>
              ↗ OPEN PAGE
            </Link>
          </div>

          <p className="detail-overview">
            {loading ? "Loading details…" : m.description || "No description available."}
          </p>

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
