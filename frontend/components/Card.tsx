"use client";

import { useState } from "react";
import { Media } from "@/lib/types";
import { posterUrl, typeClass } from "@/lib/img";
import { useStore } from "@/lib/store";

export default function Card({
  item,
  onOpen,
  onSimilar,
  showSimilar = true,
}: {
  item: Media;
  onOpen: (m: Media) => void;
  onSimilar?: (m: Media) => void;
  showSimilar?: boolean;
}) {
  const { isSaved, toggleWishlist, token, toast } = useStore();
  const [saved, setSaved] = useState(isSaved(item.id));
  const [src, setSrc] = useState(posterUrl(item.image, item.title));

  const ratingVal = parseFloat(String(item.rating));
  const rating = !isNaN(ratingVal) && ratingVal > 0 ? `★ ${ratingVal.toFixed(1)}` : "";
  const type = (item.type || "MOVIE").toUpperCase();

  const onHeart = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!token) return toast("Login to save titles");
    const nowSaved = await toggleWishlist(item);
    setSaved(nowSaved);
  };

  return (
    <div className="card">
      <div className="card-media-wrapper" onClick={() => onOpen(item)}>
        <img
          src={src}
          loading="lazy"
          alt={item.title}
          onLoad={(e) => e.currentTarget.classList.add("loaded")}
          onError={() => setSrc(`https://placehold.co/300x450/111/FFF?text=${encodeURIComponent(item.title)}`)}
        />
        <div className="match-bar-track">
          <span className="match-label-base label-cyan">{item.score || 85}% MATCH</span>
        </div>
      </div>
      <div className="card-content">
        <div className="badge-row">
          <span className={`type-badge ${typeClass(item.type)}`}>{type}</span>
          {rating && <span className="rating-badge">{rating}</span>}
          <button onClick={onHeart} className="wishlist-btn" style={{ color: saved ? "#ff0055" : "#888" }}>
            {saved ? "♥" : "♡"}
          </button>
        </div>
        <h3 onClick={() => onOpen(item)} style={{ cursor: "pointer" }}>
          {item.title}
        </h3>
        <p>{item.description || "No data."}</p>
        {showSimilar && onSimilar && (
          <button className="similar-btn" onClick={() => onSimilar(item)}>
            EXPLORE SIMILAR
          </button>
        )}
      </div>
    </div>
  );
}
