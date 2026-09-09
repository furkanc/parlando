# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///
"""Generate the parlando brand assets.

- Menu bar template icons (black + alpha; macOS recolors them for
  light/dark): a three-bar speech waveform with state badges.
- README banner with the wordmark and a live insertion caret.

Drawn at 8x and downsampled for crisp edges.

Usage: uv run scripts/make_icons.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ICON_OUT = ROOT / "src" / "parlando" / "assets"
BANNER_OUT = ROOT / "assets"
SCALE = 8
SIZE = 36          # 18pt @2x, the standard menu bar template size
S = SIZE * SCALE   # 288
BLACK = (0, 0, 0, 255)

# Brand colors (banner only; menu bar icons stay template-monochrome)
BG = (13, 13, 18, 255)
ACCENT = (124, 108, 255, 255)   # violet
FG = (245, 245, 250, 255)
MUTED = (150, 152, 166, 255)


def canvas(w: int = S, h: int = S, bg=(0, 0, 0, 0)):
    img = Image.new("RGBA", (w, h), bg)
    return img, ImageDraw.Draw(img)


# The mark: a three-bar speech waveform (short-tall-medium), vertically
# centered. Musical *and* vocal — "parlando" is a musical direction meaning
# "in a speaking manner".
BAR_W = 44
BAR_GAP = 30
BAR_HEIGHTS = [104, 216, 148]   # left to right
MARK_W = 3 * BAR_W + 2 * BAR_GAP  # 192


def draw_wave(d: ImageDraw.ImageDraw, ox: int, cy: int, scale: float, color) -> None:
    x = ox
    for h in BAR_HEIGHTS:
        bw, bh = BAR_W * scale, h * scale
        d.rounded_rectangle(
            [x, cy - bh / 2, x + bw, cy + bh / 2],
            radius=bw / 2,
            fill=color,
        )
        x += (BAR_W + BAR_GAP) * scale


def icon_idle():
    img, d = canvas()
    draw_wave(d, (S - MARK_W) // 2, S // 2, 1.0, BLACK)
    return img


def icon_recording():
    img, d = canvas()
    # Bars nudged left; record dot at the top right.
    draw_wave(d, 24, S // 2, 1.0, BLACK)
    d.ellipse([232, 36, 284, 88], fill=BLACK)
    return img


def icon_paused():
    img, d = canvas()
    # Classic pause: two equal bars (distinct from the 3-bar uneven wave).
    for x in (88, 160):
        d.rounded_rectangle([x, 64, x + BAR_W, 224], radius=BAR_W // 2, fill=BLACK)
    return img


def save_icon(img: Image.Image, name: str) -> None:
    ICON_OUT.mkdir(parents=True, exist_ok=True)
    img.resize((SIZE, SIZE), Image.LANCZOS).save(ICON_OUT / name)
    print(f"wrote {ICON_OUT / name}")


def load_font(px: int) -> ImageFont.FreeTypeFont:
    candidates = [
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
        ("/System/Library/Fonts/Helvetica.ttc", 1),  # index 1: bold
        ("/System/Library/Fonts/Supplemental/Verdana Bold.ttf", 0),
    ]
    for path, index in candidates:
        try:
            return ImageFont.truetype(path, px, index=index)
        except OSError:
            continue
    raise SystemExit("no usable system font found for the banner")


def banner():
    """README hero image: wave mark + wordmark + tagline on a dark card."""
    W, H = 2560, 800
    img, d = canvas(W, H, bg=(0, 0, 0, 0))
    d.rounded_rectangle([0, 0, W, H], radius=48, fill=BG)

    mscale = 2.0
    name_font = load_font(210)
    tag_font = load_font(64)
    tagline = "Local voice dictation for macOS.  Tap, speak, it types."
    name_w = int(d.textlength("parlando", font=name_font))
    tag_w = int(d.textlength(tagline, font=tag_font))
    text_w = max(name_w, tag_w)
    mark_w = int(MARK_W * mscale)
    gap = 110
    total_w = mark_w + gap + text_w
    ox = (W - total_w) // 2

    draw_wave(d, ox, H // 2, mscale, ACCENT)

    # Wordmark with a live insertion caret right after it — "parlando▎",
    # a word being typed.
    tx = ox + mark_w + gap
    name_y = H // 2 - 210
    d.text((tx, name_y), "parlando", font=name_font, fill=FG)
    caret_x = tx + name_w + 34
    d.rounded_rectangle(
        [caret_x, name_y + 26, caret_x + 18, name_y + 240],
        radius=9,
        fill=ACCENT,
    )
    d.text((tx + 8, H // 2 + 55), tagline, font=tag_font, fill=MUTED)

    BANNER_OUT.mkdir(parents=True, exist_ok=True)
    out = BANNER_OUT / "banner.png"
    img.resize((W // 2, H // 2), Image.LANCZOS).save(out)
    print(f"wrote {out}")


def main() -> None:
    save_icon(icon_idle(), "mic.png")
    save_icon(icon_recording(), "mic-recording.png")
    save_icon(icon_paused(), "mic-paused.png")
    banner()


if __name__ == "__main__":
    main()
