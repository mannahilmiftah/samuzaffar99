from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from . import crud, model, database
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from functools import lru_cache
import cachetools
from fastapi import FastAPI, UploadFile, File
import mediapipe as mp
import cv2
import numpy as np

app = FastAPI()

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/", response_model=model.User)
def create_user(name: str, db: Session = Depends(get_db)):
    return crud.create_user(db=db, name=name)

@app.get("/users/{user_id}", response_model=model.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db=db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.put("/users/{user_id}", response_model=model.User)
def update_user(user_id: int, name: str, db: Session = Depends(get_db)):
    db_user = crud.update_user(db=db, user_id=user_id, name=name)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "admin"
    correct_password = "secret"
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


cache = cachetools.TTLCache(maxsize=100, ttl=300)

@lru_cache(maxsize=None)
def get_cache():
    return cache

@app.post("/users/", response_model=model.User)
def create_user(name: str, db: Session = Depends(get_db)):
    cache = get_cache()
    cache.clear()  # Clear cache after new user creation
    return crud.create_user(db=db, name=name)

@app.post("/image/")
async def process_image(file: UploadFile = File(...)):
    with mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        image = await file.read()
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        results = face_detection.process(image)

        # Crop detected faces
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cropped_img = image[y:y + h, x:x + w]

        # Get facial landmarks
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            landmarks = results.multi_face_landmarks[0].landmark  # Extract landmarks
            # Process the landmarks...

    return {"cropped_image": cropped_img, "landmarks": landmarks}