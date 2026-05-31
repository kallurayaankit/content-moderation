from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from transformers import pipeline
from PIL import Image
import io
import asyncio
import time

app = FastAPI()

# ---------- Local models (download once at startup) ----------
print("Loading text model (toxic-bert)...")
text_pipe = pipeline("text-classification", model="unitary/toxic-bert")

print("Loading image model (nsfw detection)...")
img_pipe = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

def text_toxicity(text: str):
    return text_pipe(text)

def image_nsfw(image: Image.Image):
    results = img_pipe(image)
    for r in results:
        if r['label'] == 'nsfw':
            return [{"label": "nsfw", "score": r['score']}]
    return [{"label": "normal", "score": 1.0 - results[0]['score']}]

def combine_scores(text_result, img_result):
    text_score = 0.0
    for pred in text_result:
        if pred['label'] == 'toxic' and pred['score'] > text_score:
            text_score = pred['score']
    img_score = 0.0
    if img_result and len(img_result) > 0:
        img_score = img_result[0]['score']
    final_score = max(text_score, img_score)
    return {
        "final_decision": "flagged" if final_score > 0.5 else "safe",
        "text_toxicity_score": text_score,
        "image_nsfw_score": img_score,
        "text_explanation": [],
        "image_explanation": "Heatmap not available (using local model)"
    }

@app.post("/moderate")
async def moderate(
    text: str = Form(...),
    image: UploadFile = File(None)
):
    start = time.time()
    loop = asyncio.get_event_loop()

    # Run both models concurrently
    text_task = loop.run_in_executor(None, text_toxicity, text)
    img_task = None
    if image:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_task = loop.run_in_executor(None, image_nsfw, img)

    if img_task:
        text_result, img_result = await asyncio.gather(text_task, img_task)
    else:
        text_result = await text_task
        img_result = None

    combined = combine_scores(text_result, img_result)

    # Simple word‑level explanation
    if combined["text_toxicity_score"] > 0.5:
        words = text.split()
        if words:
            per_word = combined["text_toxicity_score"] / len(words)
            combined["text_explanation"] = [{"word": w, "impact": round(per_word, 4)} for w in words]

    latency = time.time() - start
    combined["latency_ms"] = round(latency * 1000, 2)
    return combined

@app.get("/health")
def health():
    return {"status": "ok"}