# Render Deployment

This repo is ready to deploy to Render as a Python web service with `render.yaml`.

## Included

- `render.yaml` installs from `requirements.txt`
- Render starts the API with `uvicorn fastapi_server:app --host 0.0.0.0 --port $PORT`
- A persistent disk is mounted at `/var/data`
- `FRIDGE_MODEL_PATH` defaults to `/var/data/best.pt`
- The API will auto-seed `/var/data/best.pt` from `models/best.pt` when the service starts
- `/fridge/info` reports the resolved model path and environment settings

## What you still need to do in Render

1. Create the web service from this repo.
2. Let Render apply `render.yaml`.
3. Redeploy after `models/best.pt` is present in the repo so startup can copy it onto the disk.

If you want to replace the checkpoint later, either:

- commit a new `models/best.pt`, or
- place a replacement file directly at `/var/data/best.pt`, or
- point `FRIDGE_MODEL_PATH` at another absolute checkpoint path

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
