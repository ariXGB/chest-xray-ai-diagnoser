# Chest X-Ray AI Diagnoser

A two-stage deep learning system that takes a chest X-ray image, first checks
whether it is actually a chest X-ray, and only then predicts whether the
patient is **Normal**, has **Pneumonia**, or has **Tuberculosis**. Built as an
end-to-end pipeline — training, testing, inference, a FastAPI backend, and a
Streamlit frontend — rather than just a notebook that ends at `model.fit()`.

I built this project to understand how a machine learning model actually goes
from "trained on Kaggle data" to "usable behind an API and a UI", and this
repo is basically the result of that. It is not meant to be a production
medical device (please don't use it to diagnose anyone), but it is a fairly
complete example of how such a system is structured.

---

## Why two models instead of one?

The first version of this project only had a single classifier trained on
chest X-rays. The problem: if someone uploads a random photo — say, a picture
of their knee, or a screenshot, or literally anything that isn't a chest
X-ray — the classifier will still confidently output a diagnosis, because
that is all it knows how to do. It has no concept of "this input doesn't
belong here."

So the pipeline was split into two stages:

1. **Gatekeeper** — a binary classifier (`Chest_x_ray` vs `Not_Chest_X_ray`)
   whose only job is to decide if the uploaded image is even a valid chest
   X-ray in the first place.
2. **Diagnoser** — the actual 3-class classifier (`Normal`, `Pneumonia`,
   `Tuberculosis`) that only runs if the Gatekeeper is confident enough that
   the input is legitimate.

This way, garbage input gets rejected early with an honest message instead of
a made-up diagnosis.

---

## How a request flows through the system

```
User uploads image (Streamlit UI)
            │
            ▼
   FastAPI  /predict/  endpoint
            │
            ▼
     predict_pipeline()
            │
            ▼
  ┌───────────────────┐
  │   Gatekeeper CNN    │   → "Is this even a chest X-ray?"
  └───────────────────┘
            │
   ┌────────┴────────┐
   │                 │
 Rejected          Accepted
 (not an X-ray,   (confidence ≥ 0.85 AND
  or low            predicted class is
  confidence)        chest_x_ray)
   │                 │
   ▼                 ▼
 Return           ┌───────────────────┐
 rejection        │   Diagnoser CNN     │   → "Normal / Pneumonia / TB?"
 message          └───────────────────┘
                          │
                          ▼
                  Return diagnosis +
                  confidence + full
                  class probabilities
```

The 0.85 confidence threshold for the Gatekeeper is a deliberate design
choice — it is better to occasionally reject a borderline-valid image than to
let something questionable slip through to the Diagnoser and get a confident
sounding but meaningless answer.

---

## Project structure

```
├── app.py                        # Streamlit frontend
├── main.py                       # FastAPI app entrypoint
│
├── api/
│   └── routers/
│       └── predict.py            # POST /predict/ endpoint
│
├── model_configs/
│   ├── diagnoser.py               # ChestClassifier (DenseNet-161 based)
│   └── gatekeeper.py              # Gatekeeper (ResNet-101 based)
│
├── datasets_dataloaders/
│   ├── dataset.py                 # ImageFolder datasets + transforms
│   └── dataloader.py              # DataLoader creation for both models
│
├── predict/
│   ├── predict_diagnoser.py       # single-image inference, Diagnoser
│   ├── predict_gatekeeper.py      # single-image inference, Gatekeeper
│   └── prediction_pipeline.py     # combines both into one pipeline
│
├── train_diagnoser.py             # training script, Diagnoser
├── train_gatekeeper.py            # training script, Gatekeeper
├── test_diagnoser.py              # evaluation script, Diagnoser
├── test_gatekeeper.py             # evaluation script, Gatekeeper
│
└── responses.py                   # Pydantic response schema
```

> Note: a few of the imports (like `api.routers.predict`, `utils.log_metrics`,
> `model_configs.*`, `datasets_dataloaders.*`) assume this file layout. If
> you're reorganising folders, keep the import paths in sync or things will
> not run.

---

## The models

Both models are built on top of pretrained ImageNet backbones using
**PyTorch Lightning**, with only the final classification layer trainable at
first (backbone frozen) — mainly because training the full backbone from
scratch on a relatively small medical imaging dataset would overfit badly and
also take forever on a single GPU.

| | Gatekeeper | Diagnoser |
|---|---|---|
| Backbone | ResNet-101 | DenseNet-161 |
| Task | Binary (X-ray vs not) | 3-class (Normal / Pneumonia / TB) |
| Classes | `Not_Chest_X_ray`, `Chest_x_ray` | `Normal`, `Pneumonia`, `Tuberculosis` |
| Optimizer | AdamW | AdamW |
| LR Scheduler | ReduceLROnPlateau (on `val_acc`) | ReduceLROnPlateau (on `val_acc`) |
| Metrics tracked | Accuracy, F1 (macro), Recall (macro) | Accuracy, F1 (macro), Recall (macro) |

Both classes expose an `unfreeze_backbone()` method, meant for a second
fine-tuning phase once the classifier head has settled — this is not wired
into the training scripts yet, so it currently has to be called manually if
you want to try it.

---

## Dataset expectations

The dataloader expects data arranged in the standard `torchvision.ImageFolder`
layout, separately for each model:

```
data/dataset/
├── gatekeeper/
│   ├── train/
│   │   ├── Chest_x_ray/
│   │   └── Not_Chest_X_ray/
│   ├── val/
│   └── test/
│
└── diagnoser/
    ├── train/
    │   ├── Normal/
    │   ├── Pneumonia/
    │   └── Tuberculosis/
    ├── val/
    └── test/
```

Images go through grayscale → resize (224×224) → (only for training:
horizontal flip + small rotation) → tensor → ImageNet normalization. This is
handled in `dataset.py` and is the same for both models.

---

## Running it

**1. Train the models** (requires a CUDA GPU — the trainers are hardcoded to
`accelerator="gpu"`):

```bash
python train_gatekeeper.py
python train_diagnoser.py
```

Checkpoints get saved to `models/gatekeeper/best.ckpt` and
`models/diagnoser/best.ckpt` respectively, picking the epoch with the best
`val_acc`.

**2. Evaluate on the test set:**

```bash
python test_gatekeeper.py
python test_diagnoser.py
```

**3. Serve the API:**

```bash
uvicorn main:app --reload
```

**4. Set up the environment variable for the frontend.** The Streamlit app
needs to know where the API is running, via a `.env` file:

```
FAST_API_URL=http://localhost:8000
```

**5. Run the frontend:**

```bash
streamlit run app.py
```

Upload an image, hit **Run Analysis**, and it'll call the API and show the
result — a rejection message if the Gatekeeper isn't convinced, or the
diagnosis with confidence and per-class probabilities if it is.

---

## A few honest limitations

- This was trained on a limited public dataset, so it will not generalise
  well to every real-world X-ray machine, patient population, or imaging
  artifact. It should not be treated as a diagnostic tool outside of a
  learning/demo context.
- The Gatekeeper's 0.85 threshold was picked by hand, not tuned
  systematically — it's a reasonable starting point rather than an optimised
  value.
- The GPU-only assumption in `Trainer(accelerator="gpu", devices=1)` across
  training, testing, and inference scripts means this currently won't run on
  a machine without a CUDA-capable GPU without editing those lines.
- There's a small naming inconsistency between `test_diagnoser.py` (device
  detection assigned but unused) and `test_gatekeeper.py` — harmless, just
  something I noticed while writing this up and haven't cleaned yet.

---

## Tech stack

`PyTorch` · `PyTorch Lightning` · `torchvision` (DenseNet-161, ResNet-101) ·
`torchmetrics` · `FastAPI` · `Streamlit` · `Pydantic`

---
