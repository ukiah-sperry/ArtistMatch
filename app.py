import os, io, re, sys, time
from pathlib import Path
from flask import Flask, request, render_template, url_for, session, redirect
from werkzeug.utils import secure_filename


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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace-with-secure-random-in-prod")

# Spotify config
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
SPOTIFY_SCOPE = "user-library-read user-follow-read"

# Print Spotify config for debugging (remove in production)
print(f"Spotify Config:")
print(f"  Client ID: {SPOTIFY_CLIENT_ID[:10]}...")
print(f"  Redirect URI: {SPOTIFY_REDIRECT_URI}")
print(f"  Scope: {SPOTIFY_SCOPE}")

def allowed_file(fname: str) -> bool:
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ───────── Load models once
DETECTOR = YOLO(str(Path('runs/detect/train/weights/best.pt')))
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

def save_debug_image(pil_img, boxes, out_path):
    """Draw YOLO boxes for visual debugging."""
    import cv2
    arr = np.array(pil_img).copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    Image.fromarray(arr).save(out_path)
    return out_path

# ───────── Spotify helpers

def get_sp_oauth() -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=SPOTIFY_SCOPE,
        cache_path=None,       # keep token in session, not on disk
        show_dialog=False
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
    """Fetch all liked tracks from user's Spotify library."""
    sp = spotify_client_from_session()
    if not sp:
        return []
    
    liked_tracks = []
    offset = 0
    limit = 50
    
    try:
        while True:
            results = sp.current_user_saved_tracks(limit=limit, offset=offset)
            tracks = results.get('items', [])
            if not tracks:
                break
                
            for item in tracks:
                track = item.get('track', {})
                if track:
                    artists = [artist['name'] for artist in track.get('artists', [])]
                    liked_tracks.extend(artists)
            
            offset += limit
            if len(tracks) < limit:
                break
                
    except Exception as e:
        print(f"Error fetching liked tracks: {e}")
        return []
    
    return liked_tracks

def match_artists_with_liked(extracted_artists, liked_artists):
    """Match extracted artists with user's liked artists and rank by match count."""
    if not liked_artists:
        return [(artist, 0) for artist in extracted_artists]
    
    artist_scores = []
    
    for artist in extracted_artists:
        # Count exact matches and fuzzy matches
        exact_matches = sum(1 for liked in liked_artists if liked.lower() == artist.lower())
        
        # Also check for fuzzy matches (in case of slight variations)
        fuzzy_matches = 0
        if exact_matches == 0:
            for liked in liked_artists:
                similarity = fuzz.ratio(artist.lower(), liked.lower())
                if similarity > 85:  # High threshold for fuzzy matching
                    fuzzy_matches += 1
        
        total_score = exact_matches + (fuzzy_matches * 0.8)  # Weight fuzzy matches slightly less
        artist_scores.append((artist, int(total_score)))
    
    # Sort by score (descending), then alphabetically
    artist_scores.sort(key=lambda x: (-x[1], x[0].lower()))
    return artist_scores

# ───────── Spotify routes

@app.route('/login')
def login():
    return redirect(get_sp_oauth().get_authorize_url())

@app.route('/callback')
def callback():
    code = request.args.get('code')
    error = request.args.get('error')
    
    if error:
        print(f"Spotify OAuth error: {error}")
        return render_template('upload.html', 
                             error=f"Spotify authentication failed: {error}", 
                             user=current_spotify_user())
    
    if not code:
        return render_template('upload.html', 
                             error="No authorization code received from Spotify", 
                             user=current_spotify_user())
    try:
        token_info = get_sp_oauth().get_access_token(code, as_dict=True)
        session['token_info'] = token_info
        return render_template('upload.html', 
                             success="Successfully connected to Spotify!", 
                             user=current_spotify_user())
    except Exception as e:
        print("Spotify auth error:", e)
        return render_template('upload.html', 
                             error=f"Failed to connect to Spotify: {str(e)}", 
                             user=current_spotify_user())

@app.route('/logout')
def logout():
    session.pop('token_info', None)
    return redirect(url_for('upload'))

# ───────── App routes

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('screenshot')
        if not f or not allowed_file(f.filename):
            return render_template('upload.html', error="Please upload a PNG, JPEG, or PDF.", user=current_spotify_user())

        try:
            img = read_image_from_upload(f)
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
        artist_scores = []
        if current_user:
            liked_artists = get_user_liked_tracks()
            artist_scores = match_artists_with_liked(artists, liked_artists)
        else:
            artist_scores = [(artist, 0) for artist in artists]

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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
