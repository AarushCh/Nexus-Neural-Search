export interface Media {
  id: string;
  title: string;
  description?: string;
  image?: string;
  type?: string;
  category?: string;
  rating?: number | string;
  year?: string | number;
  score?: number;
  // detail enrichment (from /title/:id)
  tmdb_id?: number | null;
  trailer_key?: string | null;
  backdrop?: string | null;
  runtime?: number | null;
  cast?: { name: string; character?: string; profile?: string | null }[];
  providers?: { name: string; logo?: string | null }[];
}

export interface FeedRow {
  title: string;
  items: Media[];
}

export type SearchModel = "internal" | "api";

export interface Filters {
  category?: string | null;
  min_rating?: number | null;
  year_min?: number | null;
  year_max?: number | null;
}
