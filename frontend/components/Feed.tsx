"use client";

import { useState } from "react";
import { FeedRow, Media } from "@/lib/types";
import { posterUrl } from "@/lib/img";

const PER_PAGE = 6;

function FeedPoster({ item, onOpen }: { item: Media; onOpen: (m: Media) => void }) {
  const [src, setSrc] = useState(posterUrl(item.image, item.title));
  return (
    <div className="feed-card" onClick={() => onOpen(item)}>
      <div className="feed-poster">
        <img
          src={src}
          loading="lazy"
          alt={item.title}
          onError={() => setSrc(`https://placehold.co/300x450/111/FFF?text=${encodeURIComponent(item.title)}`)}
        />
      </div>
      <div className="feed-card-title">{item.title}</div>
    </div>
  );
}

function Row({ row, onOpen, perPage = PER_PAGE }: { row: FeedRow; onOpen: (m: Media) => void; perPage?: number }) {
  const [page, setPage] = useState(0);
  const [dir, setDir] = useState(1);
  const pages = Math.max(1, Math.ceil(row.items.length / perPage));
  const start = page * perPage;
  const view = row.items.slice(start, start + perPage);

  const go = (d: number) => {
    setDir(d);
    setPage((p) => Math.min(pages - 1, Math.max(0, p + d)));
  };

  return (
    <div className="feed-row">
      <div className="feed-row-title">{row.title}</div>
      <div className="feed-carousel">
        <button className="feed-nav" disabled={page === 0} onClick={() => go(-1)} aria-label="Previous">
          ‹
        </button>
        <div className="feed-viewport">
          <div
            className={`feed-page ${dir > 0 ? "slide-from-right" : "slide-from-left"}`}
            key={page}
          >
            {view.map((it) => (
              <FeedPoster key={String(it.id)} item={it} onOpen={onOpen} />
            ))}
          </div>
        </div>
        <button
          className="feed-nav"
          disabled={page >= pages - 1}
          onClick={() => go(1)}
          aria-label="Next"
        >
          ›
        </button>
      </div>
    </div>
  );
}

export default function Feed({ rows, onOpen }: { rows: FeedRow[]; onOpen: (m: Media) => void }) {
  return (
    <div className="feed">
      {rows.map((row) => (
        // Recently Viewed shows more at once (up to 12 → two rows in the grid).
        <Row key={row.title} row={row} onOpen={onOpen} perPage={row.title === "Recently Viewed" ? 12 : PER_PAGE} />
      ))}
    </div>
  );
}
