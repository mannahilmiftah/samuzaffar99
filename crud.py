from .database import SessionLocal
from .model import User

def create_user(db, name: str):
    db_user = User(name=name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def update_user(db, user_id: int, name: str):
    db_user = get_user(db, user_id)
    if db_user:
        db_user.name = name
        db.commit()
        db.refresh(db_user)
        return db_user
    return None

def search_users_by_name(db, name: str):
    return db.query(User).filter(User.name.ilike(f"%{name}%")).all()