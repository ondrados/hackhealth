import io
import numpy as np
import pandas as pd
import onnxruntime as ort

from fastapi import FastAPI, Request, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/post-file")
async def post_file(file: bytes = File(...)):
    df = pd.read_csv(io.BytesIO(file))
    df = df.drop(" ", 1)
    df = df.to_numpy()
    input = np.expand_dims(df, axis=0)
    input = input.transpose(0, 2, 1)
    ort_sess = ort.InferenceSession('resnet11.onnx')
    outputs = ort_sess.run(None, {'input': input.astype("float32")})
    return {"outputs": str(np.argmax(outputs))}
