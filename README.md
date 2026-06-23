---
title: ArtistMatch
emoji: 🎵
colorFrom: green
colorTo: purple
sdk: docker
pinned: false
---

# ArtistMatch

Upload a festival poster image and ArtistMatch will extract every artist name using YOLOv8 + EasyOCR, then rank them by how many of your Spotify liked songs feature each artist. The results are displayed as a festival poster layout — headliners at the top, supporting acts in the middle, and the rest below. You can create a Spotify playlist of your matched songs in one click.

## Features

- AI-powered artist name detection from poster images (PNG, JPG, PDF)
- Spotify OAuth integration — ranked by your actual liked songs
- Festival poster-style results view with tier grouping
- One-click Spotify playlist creation from your matched tracks

## Setup for Hugging Face Spaces

### Required secrets

Set these in your Space's **Settings → Repository secrets**:

| Secret | Description |
|--------|-------------|
| `SPOTIFY_CLIENT_ID` | From the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) |
| `SPOTIFY_CLIENT_SECRET` | From the Spotify Developer Dashboard |
| `FLASK_SECRET_KEY` | Any long random string (used to sign session cookies) |

### Redirect URI

In the Spotify Developer Dashboard, add your Space's public URL as a Redirect URI:

```
https://<your-username>-artistmatch.hf.space/callback
```

Then set `SPOTIFY_REDIRECT_URI` to that same URL as a Space secret (or update the default in `app.py`).

### Model weights

The YOLO detector expects a trained model at:

```
runs/detect/train/weights/best.pt
```

This path is excluded from Docker builds via `.dockerignore`. Before building the Docker image, place your `best.pt` file at that path, then remove or comment out the `runs/` line in `.dockerignore` so the weights are included in the image.

Alternatively, modify the `Dockerfile` to download the weights from an external source (e.g., HF Hub, S3) during the build step.

## Local development

```bash
cp .env.example .env   # fill in your Spotify credentials
python app.py          # runs on http://127.0.0.1:8000
```
