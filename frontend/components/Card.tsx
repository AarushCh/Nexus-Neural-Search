"use client";

import { useState } from "react";
import { Media } from "@/lib/types";
import { posterUrl, typeClass } from "@/lib/img";
import { useStore } from "@/lib/store";

const KIND_LABEL: Record<string, string> = {
  "type-movie": "Movie",
  "type-tv": "TV",
  "type-anime": "Anime",
  // Short on purpose: it is a pill on a ~150px poster.
  "type-doc": "Doc",
};

/* A result tile, in the order people actually decide: poster to recognise it,
   title and year to be sure which one it is, a two-line synopsis to see why it
   matched, then the two things you can do without opening it.

   The whole card opens the title. That is the block-link pattern — the title's
   button stretches an ::after over the card — rather than wrapping everything
   in a link, which makes a screen reader announce the entire card as one
   control name. The action buttons are lifted above the stretched layer. */
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
  // Read from the store, not seeded local state: the wishlist arrives after
  // login, and a card rendered before it kept an empty heart for a saved title.
  const saved = isSaved(item.id);
  const [src, setSrc] = useState(posterUrl(item.image, item.title));

  const rating = parseFloat(String(item.rating));
  const year = parseInt(String(item.year), 10);
  const kind = typeClass(item.type);

  const onHeart = async () => {
    if (!token) return toast("Login to save titles");
    await toggleWishlist(item);
  };

  return (
    <article className="card">
      <div className="card-poster">
        {/* Decorative: the title is right below, so alt text would only make a
            screen reader say it twice. */}
        <img
          src={src}
          loading="lazy"
          alt=""
          onError={() => setSrc(`https://placehold.co/300x450/111/FFF?text=${encodeURIComponent(item.title)}`)}
        />
        {/* Only where a real relevance score exists; feed, wishlist and random
            cards carry none. */}
        {typeof item.score === "number" && item.score > 0 && (
          <span className="card-match">
            {item.score}%<small> match</small>
          </span>
        )}
        {/* On the poster rather than in the meta line: at two tiles per row
            "MOVIE · 2008 · ★ 7.6" does not fit ~126px, and a wrapped meta line
            knocks the synopsis out of line with its neighbour. */}
        <span className="card-kind" data-kind={kind}>
          {KIND_LABEL[kind]}
        </span>
      </div>

      <div className="card-body">
        <h3 className="card-title">
          <button className="card-open" onClick={() => onOpen(item)}>
            {item.title}
          </button>
        </h3>

        <p className="card-meta">
          {year > 0 && <span>{year}</span>}
          {rating > 0 && (
            <span className="card-rating">
              <span aria-hidden="true">★</span> {rating.toFixed(1)}
              <span className="sr-only"> out of 10</span>
            </span>
          )}
        </p>

        {item.description && <p className="card-desc">{item.description}</p>}

        <div className="card-actions">
          {showSimilar && onSimilar && (
            <button
              className="card-similar"
              onClick={() => onSimilar(item)}
              aria-label={`Similar to ${item.title}`}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <circle cx="9" cy="9" r="5.5" />
                <circle cx="15" cy="15" r="5.5" />
              </svg>
              Similar
            </button>
          )}
          <button
            className="card-save"
            onClick={onHeart}
            aria-pressed={saved}
            aria-label={`Save ${item.title} to wishlist`}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M20.8 5.6a5 5 0 0 0-7.1 0L12 7.3l-1.7-1.7a5 5 0 1 0-7.1 7.1l8.8 8.8 8.8-8.8a5 5 0 0 0 0-7.1z" />
            </svg>
          </button>
        </div>
      </div>
    </article>
  );
}
