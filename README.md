# 🛡️ Real‑Time Multi‑Modal Content Moderation

[![Live Demo](https://img.shields.io/badge/Live-Demo-green?logo=huggingface)](https://kallurayaankit-content-moderation.hf.space)

A high‑speed API that accepts text and images, runs toxicity and NSFW detection **in parallel**, and returns a final moderation decision with explanations.

---

## 📌 Features

- **Async multi‑modal inference** – text + image models run concurrently
- **Pre‑trained models** – `unitary/toxic-bert` for text, `Falconsai/nsfw_image_detection` for images
- **Explainability** – word‑level impact scores for toxic text
- **Low latency** – parallel inference keeps total time low
- **Dockerized** – ready for deployment on Hugging Face Spaces or any cloud

---

## ⚡ Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

---

## 🔄 (Optional) CI/CD with GitHub Actions

Create `.github/workflows/ci.yml` with a simple syntax check:

```yaml
name: Validate

on: push

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python -m py_compile app/main.py
git add .github/workflows/ci.yml
git commit -m "Add CI pipeline"
git push
