import os, io, re, sys, time
from pathlib import Path
from flask import Flask, request, render_template, url_for, session, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman


import numpy as np
from PIL import Image

# YOLO detector + OCR
from ultralytics import YOLO
import easyocr
from rapidfuzz import process, fuzz

# Spotify OAuth
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

from dotenv import load_dotenv
load_dotenv()

from festivals import FESTIVALS


# Optional: PDF support
try:
    from pdf2image import convert_from_bytes
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False

# ───────── Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB per upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'detections'), exist_ok=True)

# Secret key (needed for session-based Spotify tokens)
# Must be set via FLASK_SECRET_KEY env var — a weak or missing key breaks session isolation.
_secret = os.getenv("FLASK_SECRET_KEY")
if not _secret:
    raise RuntimeError("FLASK_SECRET_KEY environment variable is not set. "
                       "Set a long random string in your .env or Space secrets.")
app.secret_key = _secret

# ───────── Rate limiting (in-memory; swap storage_uri for Redis in production)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri="memory://",
)

# ───────── Security headers
Talisman(
    app,
    force_https=False,               # HF Spaces / reverse proxy handles TLS termination
    strict_transport_security=False, # ditto
    content_security_policy={
        'default-src': "'self'",
        'script-src':  ["'self'", 'cdnjs.cloudflare.com'],
        'style-src':   ["'self'", 'cdnjs.cloudflare.com', 'fonts.googleapis.com'],
        'font-src':    ["'self'", 'cdnjs.cloudflare.com', 'fonts.gstatic.com'],
        'img-src':     ["'self'", 'data:'],  # data: needed for upload thumbnail preview
    },
)

# ───────── Spotify config
SPOTIFY_CLIENT_ID     = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI  = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
# Minimal default scope; playlist scopes are requested lazily via /login?extended=1
SPOTIFY_SCOPE          = "user-library-read"
SPOTIFY_SCOPE_EXTENDED = "user-library-read playlist-modify-public playlist-modify-private"

# Print Spotify config for debugging (remove in production)
print(f"Spotify Config:")
print(f"  Client ID: {SPOTIFY_CLIENT_ID[:10]}...")
print(f"  Redirect URI: {SPOTIFY_REDIRECT_URI}")
print(f"  Scope: {SPOTIFY_SCOPE}")

def allowed_file(fname: str) -> bool:
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ───────── Load models once
_WEIGHTS = Path('runs/detect/train/weights/best.pt')
if not _WEIGHTS.exists():
    print(
        f"\n[ERROR] Model weights not found at: {_WEIGHTS.resolve()}\n"
        "        Run `python download_model.py` for instructions.\n"
        "        The app cannot start without best.pt.\n",
        file=sys.stderr,
    )
    sys.exit(1)
DETECTOR = YOLO(str(_WEIGHTS))
READER = easyocr.Reader(['en'], gpu=False)

# Apple Silicon acceleration (Ultralytics will auto-select if None)
DEVICE = 'mps' if (sys.platform == 'darwin') else None

# ───────── B2B / cleanup controls
REMOVE_TRAILING_LIVE = True  # drop ... "LIVE" suffixes from artist names

# Tokens to always remove when they appear as separate words
BLACKLIST_TOKENS = {
    "dj set",
    "presents",
    # "live",  # handled separately as a trailing suffix
}

# Pattern that finds "B2B" including OCR variants:
# - B2B / b2b / B 2 B
# - BZB (Z often OCRs for '2')
# - B28 / B23 (common OCR confusions)
# - any punctuation/spaces between the characters
B2B_SPLIT_RE = re.compile(
    r'\bB\W*[2Z]\W*B\b',  # B ... 2/Z ... B
    flags=re.IGNORECASE
)

# Also normalize any stray "B?B" gibberish variants into a single canonical token before splitting
B2B_CANONICALIZE_RE = re.compile(
    r'\bB[\W_]*([2Z]|2[38]|23|28)[\W_]*B\b',  # B (2/Z/23/28) B
    flags=re.IGNORECASE
)

def strip_blacklist_tokens(s: str) -> str:
    if not BLACKLIST_TOKENS:
        return s
    pat = re.compile(r'\b(?:' + "|".join(map(re.escape, BLACKLIST_TOKENS)) + r')\b', re.IGNORECASE)
    s = pat.sub(" ", s)
    return re.sub(r'\s+', ' ', s).strip()

def remove_trailing_live(s: str) -> str:
    if not REMOVE_TRAILING_LIVE:
        return s
    # remove trailing "LIVE" in a forgiving way (handles casing and punctuation)
    s = re.sub(r'\bLIVE\b\.?$', '', s, flags=re.IGNORECASE).strip()
    return re.sub(r'\s+', ' ', s).strip()

def explode_b2b(lines):
    """
    Turn lines like 'CHRIS LIEBING B2B SPEEDY J' into ['CHRIS LIEBING', 'SPEEDY J'].
    Handles spacing/casing and OCR variants (BZB, B 2 B, B28, etc.).
    If a line has multiple B2B separators, we explode all segments.
    """
    out = []
    for raw in lines:
        if not raw or not raw.strip():
            continue
        # First canonicalize any weird OCR B2B variants to a single ' B2B '
        s = B2B_CANONICALIZE_RE.sub(' B2B ', raw)
        # Now split on any B2B form
        parts = B2B_SPLIT_RE.split(s)
        # Clean and keep non-empty parts
        for p in parts:
            p = normalize(p)
            if p:
                out.append(p)
    return out

# ───────── Core helpers

def read_image_from_upload(file_storage):
    """Return a PIL.Image in RGB from upload (PNG/JPG or first page of PDF)."""
    suffix = file_storage.filename.rsplit('.', 1)[-1].lower()
    data = file_storage.read()
    if suffix == 'pdf':
        if not HAVE_PDF:
            raise RuntimeError("PDFs require 'pdf2image' + poppler. Install them or upload PNG/JPG.")
        pages = convert_from_bytes(data, fmt='jpeg', dpi=300)
        if not pages:
            raise RuntimeError("Could not read PDF.")
        img = pages[0].convert('RGB')
    else:
        # Verify the file is a real image before passing to the ML pipeline.
        # verify() closes the internal file handle, so we must re-open afterward.
        try:
            probe = Image.open(io.BytesIO(data))
            probe.verify()
        except Exception:
            raise ValueError("Invalid image file")
        img = Image.open(io.BytesIO(data)).convert('RGB')
    return img

def detect_text_boxes(pil_img, conf=0.25, iou=0.5, imgsz=1024):
    """Run YOLO on a PIL image; return (boxes, scores, raw_result)."""
    rgb = np.array(pil_img)
    res = DETECTOR.predict(source=rgb, conf=conf, iou=iou, imgsz=imgsz, device=DEVICE, verbose=False)[0]
    boxes, scores = [], []
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), sc in zip(xyxy, confs):
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            scores.append(float(sc))
    return boxes, scores, res

def pad_box(box, pad, W, H):
    x1, y1, x2, y2 = box
    return (max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad))

def ocr_boxes(pil_img, boxes, pad_px=4):
    """Crop each box with padding, OCR it, return (list_of_texts, [(box, text), ...])."""
    W, H = pil_img.size
    texts, box_texts = [], []
    for b in boxes:
        bx = pad_box(b, pad_px, W, H)
        crop = pil_img.crop(bx)
        raw = READER.readtext(np.array(crop), detail=0, paragraph=True)
        text = " ".join(raw).strip()
        texts.append(text)
        box_texts.append((bx, text))
    return texts, box_texts

def normalize(s: str) -> str:
    # remove bullets/typographic dots
    s = s.replace('•', ' ').replace('·', ' ')
    s = re.sub(r'[_•·\u2022]', ' ', s)

    # keep typical festival chars (letters, numbers, &, +, . ' : / , - ( ) and spaces)
    s = re.sub(r'[^A-Za-z0-9&+.\'’:/,\-() ]+', ' ', s)

    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()

    # remove blacklisted tokens appearing standalone
    s = strip_blacklist_tokens(s)

    # optionally drop trailing LIVE
    s = remove_trailing_live(s)

    return s

def fuzzy_dedupe(lines, threshold=92):
    """Remove near-duplicates with fuzzy match."""
    uniq = []
    for s in lines:
        if not s:
            continue
        match, score, _ = process.extractOne(s, uniq, scorer=fuzz.token_set_ratio) if uniq else (None, 0, None)
        if score < threshold:
            uniq.append(s)
    return uniq

def dedupe_and_sort(lines):
    cleaned = [normalize(s) for s in lines if s and normalize(s)]
    uniq = fuzzy_dedupe(cleaned, threshold=92)
    return sorted(uniq, key=lambda x: x.lower())

def cleanup_old_detections(max_age_seconds=3600):
    """Delete detection images older than max_age_seconds from static/detections/."""
    detections_dir = Path(app.root_path) / 'static' / 'detections'
    now = time.time()
    for f in detections_dir.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
            try:
                f.unlink()
            except OSError:
                pass

def save_debug_image(pil_img, boxes, out_path):
    """Draw YOLO boxes for visual debugging."""
    import cv2
    arr = np.array(pil_img).copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    Image.fromarray(arr).save(out_path)
    return out_path

# ───────── Spotify helpers

def get_sp_oauth(scope=None) -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=scope or SPOTIFY_SCOPE,
        cache_path=None,       # keep token in session, not on disk
        show_dialog=True,
    )

def _token_valid(token_info: dict) -> bool:
    if not token_info:
        return False
    # refresh 60s early to be safe
    return time.time() < float(token_info.get("expires_at", 0)) - 60

def spotify_client_from_session() -> Spotify | None:
    token_info = session.get("token_info")
    if not _token_valid(token_info):
        if token_info and token_info.get("refresh_token"):
            try:
                token_info = get_sp_oauth().refresh_access_token(token_info["refresh_token"])
                session["token_info"] = token_info
            except Exception:
                session.pop("token_info", None)
                return None
        else:
            return None
    return Spotify(auth=token_info["access_token"])

def current_spotify_user():
    sp = spotify_client_from_session()
    if not sp:
        return None
    try:
        return sp.current_user()
    except Exception:
        return None

def get_user_liked_tracks():
    """Fetch all liked tracks; return list of {artist_name, track_uri, track_name} dicts."""
    sp = spotify_client_from_session()
    if not sp:
        return []

    liked_tracks = []
    offset = 0
    limit = 50

    try:
        while True:
            results = sp.current_user_saved_tracks(limit=limit, offset=offset)
            items = results.get('items', [])
            if not items:
                break

            for item in items:
                track = item.get('track') or {}
                if not track:
                    continue
                track_uri  = track.get('uri')
                track_name = track.get('name', '')
                for artist in track.get('artists', []):
                    liked_tracks.append({
                        'artist_name': artist['name'],
                        'track_uri':   track_uri,
                        'track_name':  track_name,
                    })

            offset += limit
            if len(items) < limit:
                break

    except Exception as e:
        print(f"Error fetching liked tracks: {e}")
        return []

    return liked_tracks

def match_artists_with_liked(extracted_artists, liked_tracks):
    """Match extracted artists with liked tracks.

    Returns (artist_scores, matched_tracks) where:
      artist_scores  — list of (artist_name, score) tuples, sorted desc then alpha
      matched_tracks — dict {artist_name: [track_uri, ...]} for artists with score > 0,
                       in the same order as artist_scores
    """
    if not liked_tracks:
        return [(artist, 0) for artist in extracted_artists], {}

    artist_scores = []
    per_artist_uris = {}

    for artist in extracted_artists:
        exact_uris  = []
        fuzzy_uris  = []
        seen_uris   = set()

        for track in liked_tracks:
            liked_name = track['artist_name']
            uri        = track.get('track_uri')
            if not uri:
                continue

            if liked_name.lower() == artist.lower():
                if uri not in seen_uris:
                    seen_uris.add(uri)
                    exact_uris.append(uri)
            elif fuzz.ratio(artist.lower(), liked_name.lower()) > 85:
                if uri not in seen_uris:
                    seen_uris.add(uri)
                    fuzzy_uris.append(uri)

        score = len(exact_uris) + int(len(fuzzy_uris) * 0.8)
        artist_scores.append((artist, score))
        all_uris = exact_uris + fuzzy_uris
        if all_uris:
            per_artist_uris[artist] = all_uris

    artist_scores.sort(key=lambda x: (-x[1], x[0].lower()))

    # matched_tracks ordered by score descending (dict preserves insertion order)
    matched_tracks = {
        artist: per_artist_uris[artist]
        for artist, score in artist_scores
        if score > 0 and artist in per_artist_uris
    }
    return artist_scores, matched_tracks

# ───────── Spotify routes

@app.route('/login')
@limiter.limit("10 per minute")
def login():
    session.clear()  # wipe any previous user's data before starting a new OAuth flow
    scope = SPOTIFY_SCOPE_EXTENDED if request.args.get('extended') == '1' else SPOTIFY_SCOPE
    return redirect(get_sp_oauth(scope=scope).get_authorize_url())

@app.route('/callback')
@limiter.limit("10 per minute")
def callback():
    code = request.args.get('code')
    error = request.args.get('error')

    if error:
        print(f"Spotify OAuth error: {error}")
        session.clear()
        return render_template('upload.html',
                               error=f"Spotify authentication failed: {error}",
                               user=None)

    if not code:
        session.clear()
        return render_template('upload.html',
                               error="No authorization code received from Spotify",
                               user=None)
    try:
        session.clear()  # drop any previous user's data before writing new token
        token_info = get_sp_oauth().get_access_token(code, as_dict=True)
        session['token_info'] = token_info
        return render_template('upload.html',
                               success="Successfully connected to Spotify!",
                               user=current_spotify_user())
    except Exception as e:
        print("Spotify auth error:", e)
        session.clear()
        return render_template('upload.html',
                               error=f"Failed to connect to Spotify: {str(e)}",
                               user=None)

@app.route('/logout')
def logout():
    session.clear()  # clear all session keys, not just token_info
    return redirect(url_for('upload'))

# ───────── App routes

@app.route('/', methods=['GET', 'POST'])
@limiter.limit("5 per minute", methods=["POST"])
def upload():
    if request.method == 'POST':
        cleanup_old_detections()

        f = request.files.get('screenshot')
        if not f or not allowed_file(f.filename):
            return render_template('upload.html', error="Please upload a PNG, JPEG, or PDF.", user=current_spotify_user())

        try:
            img = read_image_from_upload(f)
        except ValueError:
            return render_template('upload.html', error="Invalid image file.", user=current_spotify_user())
        except Exception as e:
            return render_template('upload.html', error=f"Could not read image: {e}", user=current_spotify_user())

        # 1) detect
        boxes, scores, res = detect_text_boxes(img, conf=0.25, iou=0.5, imgsz=1024)

        # 2) OCR
        raw_texts, box_texts = ocr_boxes(img, boxes, pad_px=6)

        # 3) handle B2B (with OCR variants) -> explode into clean names
        split_lines = explode_b2b(raw_texts)

        # 4) finalize (clean, dedupe, sort)
        artists = dedupe_and_sort(split_lines)

        # 5) Get user's liked songs and match with extracted artists
        current_user = current_spotify_user()
        if current_user:
            liked_tracks = get_user_liked_tracks()
            artist_scores, matched_tracks = match_artists_with_liked(artists, liked_tracks)
            session['artist_scores'] = [[a, s] for a, s in artist_scores]
        else:
            artist_scores = [(artist, 0) for artist in artists]
            session.pop('artist_scores', None)

        # 6) debugger image
        dbg_name = f"det_{secure_filename(f.filename)}.jpg"
        dbg_path = os.path.join(app.root_path, 'static', 'detections', dbg_name)
        save_debug_image(img, boxes, dbg_path)

        return render_template(
            'result.html',
            artist_scores=artist_scores,
            debug_image=url_for('static', filename=f'detections/{dbg_name}'),
            count=len(artist_scores),
            user=current_spotify_user()
        )

    # GET
    return render_template('upload.html', user=current_spotify_user())

@app.route('/festival/<slug>')
@limiter.limit("5 per minute")
def festival(slug):
    fest = FESTIVALS.get(slug)
    if not fest:
        return redirect(url_for('upload'))

    artists = fest['artists']
    current_user = current_spotify_user()

    if current_user:
        liked_tracks = get_user_liked_tracks()
        artist_scores, matched_tracks = match_artists_with_liked(artists, liked_tracks)
        session['artist_scores'] = [[a, s] for a, s in artist_scores]
    else:
        artist_scores = [(artist, 0) for artist in artists]
        session.pop('artist_scores', None)

    return render_template(
        'result.html',
        artist_scores=artist_scores,
        count=len(artist_scores),
        user=current_user,
        debug_image=None,
        festival_name=fest['name'],
    )


@app.route('/create_playlist', methods=['POST'])
def create_playlist():
    sp = spotify_client_from_session()
    if not sp:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    # Check that the token actually has playlist-modify scopes (lazy scope upgrade)
    token_info = session.get('token_info', {})
    granted = set(token_info.get('scope', '').split())
    if not granted & {'playlist-modify-public', 'playlist-modify-private'}:
        return jsonify({'success': False, 'error': 'playlist_scope_required'}), 403

    stored_scores = session.get('artist_scores')
    if not stored_scores:
        return jsonify({'success': False, 'error': 'No matched songs found. Connect Spotify first.'})

    # Re-fetch liked tracks and rebuild matched_tracks live (avoids session size limits)
    artists_in_order = [name for name, score in stored_scores if score > 0]
    if not artists_in_order:
        return jsonify({'success': False, 'error': 'No matched songs found. Connect Spotify first.'})

    liked_tracks = get_user_liked_tracks()
    _, matched_tracks = match_artists_with_liked(artists_in_order, liked_tracks)

    if not matched_tracks:
        return jsonify({'success': False, 'error': 'No matched songs found. Connect Spotify first.'})

    # Flatten URIs: rank-1 artist first, rank-2 next, etc.
    all_uris = [uri for uris in matched_tracks.values() for uri in uris]

    try:
        user_id  = sp.current_user()['id']
        playlist = sp.user_playlist_create(
            user_id,
            'Your Electric Forest 2026 Playlist',
            public=False,
            description='Generated by ArtistMatch — your liked songs from this festival lineup',
        )
        playlist_id = playlist['id']

        for i in range(0, len(all_uris), 100):  # Spotify max 100 per call
            sp.playlist_add_items(playlist_id, all_uris[i:i + 100])

        return jsonify({'success': True, 'playlist_url': playlist['external_urls']['spotify']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    on_hf = bool(os.getenv('SPACE_ID') or os.getenv('HF_SPACE_ID'))
    host  = '0.0.0.0'   if on_hf else '127.0.0.1'
    port  = 7860        if on_hf else 8000
    # debug=True only in local dev — never on HF Spaces regardless of other settings
    app.run(host=host, port=port, debug=not on_hf)
