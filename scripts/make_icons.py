# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow"]
# ///
"""Generate the parlando brand assets.

- Menu bar template icons (black + alpha; macOS recolors them for
  light/dark): a three-bar speech waveform with state badges.
- README banner with the wordmark and a live insertion caret.
- macOS app icon (Parlando.icns) for the generated Parlando.app bundle:
  Big Sur-style rounded square on the brand card, wave mark in the accent.

Drawn at 8x and downsampled for crisp edges.

Usage: uv run scripts/make_icons.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

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


def menubar_states_strip():
    """Small README-embeddable strip showing the three menu bar states.

    Drawn on the brand's dark card so it is visible on both GitHub themes
    (the raw template icons are black-on-transparent and would vanish in
    dark mode).
    """
    cell, gap, mx, my = 288, 96, 84, 56
    W, H = mx * 2 + 3 * cell + 2 * gap, 288 + my * 2
    img, d = canvas(W, H, (0, 0, 0, 0))
    d.rounded_rectangle([0, 0, W, H], radius=64, fill=BG)

    x = mx
    # Idle: centered wave
    draw_wave(d, x + (cell - MARK_W) // 2, H // 2, 1.0, FG)
    x += cell + gap
    # Recording: wave + accent dot
    draw_wave(d, x + 24, H // 2, 1.0, FG)
    d.ellipse([x + 232, my + 36, x + 284, my + 88], fill=ACCENT)
    x += cell + gap
    # Paused: two equal bars
    for bx in (88, 160):
        d.rounded_rectangle(
            [x + bx, my + 64, x + bx + BAR_W, my + 224],
            radius=BAR_W // 2,
            fill=FG,
        )

    BANNER_OUT.mkdir(parents=True, exist_ok=True)
    out = BANNER_OUT / "menubar-states.png"
    img.resize((W // 4, H // 4), Image.LANCZOS).save(out)
    print(f"wrote {out}")


def app_icon():
    """macOS app icon for Parlando.app (Big Sur grid: 824 px tile on 1024).

    Drawn at 2x and downsampled. Pillow writes the multi-size .icns
    directly, so the bundle needs no iconutil step at install time.
    """
    C = 1024 * 2                      # 2x canvas for anti-aliasing
    tile = 824 * 2
    margin = (C - tile) // 2
    radius = int(tile * 0.2237)       # Apple's continuous-corner ratio
    img = Image.new("RGBA", (C, C), (0, 0, 0, 0))

    # Soft drop shadow, as in Apple's icon template.
    shadow = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle(
        [margin, margin + 28, margin + tile, margin + tile + 28],
        radius=radius, fill=(0, 0, 0, 110),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(36))
    img.alpha_composite(shadow)

    # Tile: vertical gradient from a slightly lifted top to the brand card.
    top, bottom = (30, 29, 46), BG[:3]
    grad = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    gp = grad.load()
    for y in range(margin, margin + tile):
        t = (y - margin) / tile
        col = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)) + (255,)
        for x in range(margin, margin + tile):
            gp[x, y] = col
    mask = Image.new("L", (C, C), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [margin, margin, margin + tile, margin + tile], radius=radius, fill=255
    )
    img.paste(grad, (0, 0), mask)

    # Thin inner highlight along the top edge for depth.
    hl = Image.new("RGBA", (C, C), (0, 0, 0, 0))
    ImageDraw.Draw(hl).rounded_rectangle(
        [margin + 3, margin + 3, margin + tile - 3, margin + tile - 3],
        radius=radius - 3, outline=(255, 255, 255, 28), width=6,
    )
    img.alpha_composite(hl)

    # The wave mark, accent colored, ~52% of the tile width.
    d = ImageDraw.Draw(img)
    mscale = tile * 0.52 / MARK_W
    ox = int((C - MARK_W * mscale) / 2)
    draw_wave(d, ox, C // 2, mscale, ACCENT)

    return img.resize((1024, 1024), Image.LANCZOS)


def save_app_icon(img: Image.Image) -> None:
    ICON_OUT.mkdir(parents=True, exist_ok=True)
    icns = ICON_OUT / "Parlando.icns"
    img.save(icns, format="ICNS")
    print(f"wrote {icns}")
    BANNER_OUT.mkdir(parents=True, exist_ok=True)
    preview = BANNER_OUT / "app-icon.png"
    img.resize((256, 256), Image.LANCZOS).save(preview)
    print(f"wrote {preview}")


def main() -> None:
    save_icon(icon_idle(), "mic.png")
    save_icon(icon_recording(), "mic-recording.png")
    save_icon(icon_paused(), "mic-paused.png")
    banner()
    menubar_states_strip()
    save_app_icon(app_icon())


if __name__ == "__main__":
    main()
