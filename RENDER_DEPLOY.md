# Render Deployment

This repo is ready to deploy to Render as a Python web service with `render.yaml`.

## Included

- `render.yaml` installs from `requirements.txt`
- Render starts the API with `uvicorn fastapi_server:app --host 0.0.0.0 --port $PORT`
- A persistent disk is mounted at `/var/data`
- `FRIDGE_MODEL_PATH` defaults to `/var/data/best.pt`
- `/fridge/info` reports the resolved model path and environment settings

## What you still need to do in Render

1. Create the web service from this repo.
2. Let Render apply `render.yaml`.
3. After the disk is mounted, place your fridge checkpoint at `/var/data/best.pt`.

Your current local checkpoint is an unpacked directory-style model. That is fine:

- if the mounted path is a real file checkpoint, set `FRIDGE_MODEL_PATH` to that file path
- if the mounted path is an unpacked checkpoint directory, set `FRIDGE_MODEL_PATH` to that directory path instead

## Verify after deploy

Open:

- `/`
- `/fridge/info`

`/fridge/info` should show:

- `"available": true`
- `"model_path"` with the resolved Render path

If it shows `"available": false`, check:

- the model exists on the mounted disk
- `FRIDGE_MODEL_PATH` matches the actual path
- the deploy logs for the startup message about the fridge model
