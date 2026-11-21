"""Regenerate neutral placeholder catalog photos (no third-party images).

Run from the repo root:  python make_placeholders.py
Creates pricematch/data/photos/<name>.jpg for every name in NAMES.
To use real photos, put your OWN photos (or open-license ones) there instead.
"""
import colorsys, hashlib, textwrap
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / "pricematch" / "data" / "photos"
NAMES = Path(__file__).parent / "photo_names.txt"


def make(name: str):
    h = int(hashlib.md5(name.split("-")[0:2].__str__().encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = [int(255 * x) for x in colorsys.hsv_to_rgb(h, 0.35, 0.92)]
    img = Image.new("RGB", (320, 320), (r, g, b))
    d = ImageDraw.Draw(img)
    label = "\n".join(textwrap.wrap(name.replace("-", " "), 18))
    d.multiline_text((24, 120), label, fill=(40, 40, 40), spacing=6)
    d.text((24, 285), "placeholder image", fill=(90, 90, 90))
    return img


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in NAMES.read_text().split() if n.strip()]
    for n in names:
        make(n).save(OUT / f"{n}.jpg", quality=85)
    print(f"wrote {len(names)} placeholders to {OUT}")
