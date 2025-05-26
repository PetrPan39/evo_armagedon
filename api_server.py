import pkgutil
import importlib

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, status
from auth import verify_token
from quickstar_connect import init_astra_db
from evo_logic import EvolucniOptimalizace
import modules

# Fail-fast: před vytvořením instancí importujeme všechny modulové pluginy
for finder, name, ispkg in pkgutil.walk_packages(modules.__path__, modules.__name__ + "."):
    importlib.import_module(name)

app = FastAPI()
db = init_astra_db()
evo = EvolucniOptimalizace(db=db)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ask")
async def ask_text(request: Request):
    auth = request.headers.get("Authorization", "")
    verify_token(auth)
    payload = await request.json()
    text = payload.get("text", "")
    return {"result": evo.zpracuj(text)}

@app.post("/solve")
async def solve_task(request: Request):
    auth = request.headers.get("Authorization", "")
    verify_token(auth)
    payload = await request.json()
    task = payload.get("task", "")
    return {"result": evo.solve_task(task)}

@app.post("/process_image")
async def process_image(request: Request, file: UploadFile = File(...)):
    auth = request.headers.get("Authorization", "")
    verify_token(auth)
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    return evo.process_image(path)

@app.post("/process_audio")
async def process_audio(request: Request, file: UploadFile = File(...)):
    auth = request.headers.get("Authorization", "")
    verify_token(auth)
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    return evo.process_audio(path)
