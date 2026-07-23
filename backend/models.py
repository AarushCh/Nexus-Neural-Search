from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float
from sqlalchemy.orm import relationship
from datetime import datetime

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    wishlist = relationship("WishlistItem", back_populates="user", cascade="all, delete-orphan")
    searches = relationship("SearchHistory", back_populates="user", cascade="all, delete-orphan")
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")


class WishlistItem(Base):
    __tablename__ = "wishlist_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # Qdrant point ids are uuid5 strings, so this must be a String (not Integer).
    media_id = Column(String, nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="wishlist")


class SearchHistory(Base):
    """Every authenticated search, so history survives across devices/sessions."""
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    query = Column(Text, nullable=False)
    model = Column(String, default="internal")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="searches")


class Interaction(Base):
    """
    A user's signal on a title. `kind` is one of:
      view    -> opened its detail (weak positive)
      like    -> explicit thumbs up (strong positive)
      dismiss -> "not interested" (negative)
    These feed the history-driven "For You" recommender.
    """
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    media_id = Column(String, nullable=False, index=True)
    kind = Column(String, nullable=False, default="view")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="interactions")


class MediaDetail(Base):
    """
    Cache of TMDB enrichment (trailer / cast / streaming providers) per media_id,
    so the detail view doesn't hit TMDB on every open.
    """
    __tablename__ = "media_details"

    media_id = Column(String, primary_key=True, index=True)
    tmdb_id = Column(Integer, nullable=True)
    poster = Column(String, nullable=True)         # corrected TMDB poster (w500)
    trailer_key = Column(String, nullable=True)   # YouTube key
    cast_json = Column(Text, nullable=True)        # JSON list of {name, character, profile}
    providers_json = Column(Text, nullable=True)   # JSON list of {name, logo}
    backdrop = Column(String, nullable=True)
    runtime = Column(Integer, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
