import os
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.database import DATABASE_URL, SessionLocal
from backend.models import User

# ---------------- CONFIG ----------------

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # A missing key in production would make every token forgeable, so refuse to
    # boot there. Local SQLite dev gets a throwaway key instead of a hard stop.
    if not DATABASE_URL.startswith("sqlite"):
        raise RuntimeError(
            "SECRET_KEY is not set. Generate one with:\n"
            '  python -c "import secrets; print(secrets.token_hex(32))"'
        )
    SECRET_KEY = "dev-only-insecure-secret-change-me"

ALGORITHM = "HS256"

# 30 days. A 60-minute token meant that leaving the tab open over lunch and then
# searching sent an expired JWT to /recommend/personalized, which 401'd and left
# the UI showing an empty result grid with no explanation.
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24 * 30))

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ---------------- PASSWORD UTILS ----------------

def hash_password(password: str) -> str:
    # bcrypt limit = 72 bytes → safe truncate
    password = password.encode("utf-8")[:72]
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    plain = plain.encode("utf-8")[:72]
    return pwd_context.verify(plain, hashed)

# ---------------- JWT ----------------

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ---------------- DB DEP ----------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------- AUTH DEP ----------------

def get_current_user_db(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception

    return user

# ---------------- LOGIN ----------------

def login_user(form_data, db: Session):
    user = db.query(User).filter(
        User.username == form_data.username
    ).first()

    if not user or not verify_password(
        form_data.password,
        user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
