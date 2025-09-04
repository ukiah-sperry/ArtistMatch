import json, os, shutil, random, argparse, csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def load_via_v2(via_json_path: Path):
    d = json.loads(via_json_path.read_text())
    meta = d.get("_via_img_metadata", {})
    out = {}
    for _id, item in meta.items():
        fname = item.get("filename")
        regions = item.get("regions", [])
        if not fname or not regions:
            continue
        boxes = []
        for r in regions:
            sa = r.get("shape_attributes", {})
            ra = r.get("region_attributes", {})
            if sa.get("name") == "rect" and all(k in sa for k in ("x","y","width","height")):
                boxes.append({
                    "x": float(sa["x"]),
                    "y": float(sa["y"]),
                    "w": float(sa["width"]),
                    "h": float(sa["height"]),
                    "artist": (ra.get("artist") or "").strip()
                })
        if boxes:
            out.setdefault(fname, []).extend(boxes)
    return out

def yolo_line(x, y, w, h, W, H, cls=0):
    xc = (x + w/2) / W
    yc = (y + h/2) / H
    wn = w / W
    hn = h / H
    # clamp to [0,1]
    xc = min(max(xc, 0), 1); yc = min(max(yc, 0), 1)
    wn = min(max(wn, 0), 1); hn = min(max(hn, 0), 1)
    return f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--via_json", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", default="dataset")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    args = ap.parse_args()

    via = Path(args.via_json)
    img_root = Path(args.images_dir)
    out = Path(args.out_dir)

    ann = load_via_v2(via)
    if not ann:
        raise SystemExit("No boxes found — is this a VIA 2.x JSON with rectangle regions?")

    # gather (image_path, boxes)
    items = []
    missing = []
    for fname, boxes in ann.items():
        p = img_root / fname
        if not p.exists():
            # try to find by basename anywhere under images_dir
            hits = list(img_root.rglob(Path(fname).name))
            if hits:
                p = hits[0]
            else:
                missing.append(fname); continue
        items.append((p, boxes))

    if missing:
        print("\n[WARN] Missing image files (not found in images_dir):")
        for m in missing: print("  -", m)
        print("Continuing with found images...\n")

    random.shuffle(items)
    n_val = max(1, int(len(items) * args.val_ratio))
    val_items = items[:n_val]
    train_items = items[n_val:]

    for split in ("train","val"):
        (out / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (out / f"labels/{split}").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    csv_path = out / "meta" / "boxes_with_text.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["split","image","x","y","w","h","artist"])

        for split_name, split_items in (("train", train_items), ("val", val_items)):
            for img_path, boxes in tqdm(split_items, desc=f"Writing {split_name}"):
                # link/copy image
                dst_img = out / f"images/{split_name}" / img_path.name
                if args.copy:
                    shutil.copy2(img_path, dst_img)
                else:
                    try:
                        if dst_img.exists(): dst_img.unlink()
                        os.symlink(img_path.resolve(), dst_img)
                    except OSError:
                        shutil.copy2(img_path, dst_img)

                # image size
                with Image.open(img_path) as im:
                    W, H = im.size

                # labels
                lbl_path = out / f"labels/{split_name}" / (img_path.stem + ".txt")
                lines = []
                for b in boxes:
                    lines.append(yolo_line(b["x"], b["y"], b["w"], b["h"], W, H, cls=0))
                    writer.writerow([split_name, img_path.name, int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"]), b.get("artist","")])
                lbl_path.write_text("\n".join(lines))

    # YOLO data config
    (out / "poster.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames: [text]\n"
    )

    print("\nDone.")
    print(f"  → Train images: {len(train_items)}")
    print(f"  → Val images:   {len(val_items)}")
    print(f"  → YOLO config:  {(out/'poster.yaml').resolve()}")
    print(f"  → Text CSV:     {(out/'meta'/'boxes_with_text.csv').resolve()}")

if __name__ == "__main__":
    main()
