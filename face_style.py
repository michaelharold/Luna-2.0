# face_style.py
"""
face_style.py — palette, geometry and shared drawing helpers for Luna's face.

Everything that answers "what colour is this and how round is it" lives here,
so the renderer never hard-codes a colour and a new look is one dict entry.

    from face_style import theme, apply_style
    pygame.draw.rect(surf, theme.EYE_OUTER, rect)

`theme` is a single mutable object rather than a pile of module globals.
That matters: styles switch at runtime (keys 1 / 2), and `from x import COLOUR`
would bind the old value forever, so a live switch only repainted half the face.
Attribute access on one object always sees the current style.

Styles
  1 — Luna classic: purple, rounded, organic mouth
  2 — Robo:         cyan, sharp corners, square pupils, equalizer-bar mouth
  3 — Aurora:       Face 3 — EMO/Vector-style solid gradient eyes with
                    catchlights, eyes-only at rest. The product face.
"""

import pygame

# ── Canvas ────────────────────────────────────────────────────────────────────
# Fixed logical canvas. pygame's SCALED flag maps it onto the real display, so
# the same numbers work on a desktop window and the Pi's 800x480 panel.
WIDTH  = 1400
HEIGHT = 800

# Style-independent constants — black is black in every palette.
BG        = (0, 0, 0)
PUPIL_COL = (0, 0, 0)


# ── Style table ───────────────────────────────────────────────────────────────
STYLES = {
    1: dict(
        name="luna",
        EYE_OUTER=(55, 0, 200),    EYE_MID=(90, 30, 255),
        EYE_INNER=(160, 90, 255),  IRIS_SHINE=(220, 190, 255),
        GLOW_COL=(80, 20, 200),    MOUTH_COL=(70, 10, 220),
        ZZZ_COL=(120, 60, 220),    LISTEN_COL=(0, 180, 255),
        BLUSH_COL=(255, 100, 150), TEAR_COL=(100, 180, 255),
        STEAM_COL=(255, 120, 60),  STAR_COL=(255, 220, 80),
        TEETH_COL=(240, 240, 255), WAVE_COL=(120, 40, 255),
        ANGRY_COL=(220, 30, 30),   HEART_COL=(255, 80, 130),
        AWAKE_COL=(0, 200, 255),
        EYE_RADIUS=20, IRIS_SQUARE=False, MOUTH_STYLE="organic",
    ),
    2: dict(
        name="robo",
        EYE_OUTER=(0, 120, 150),   EYE_MID=(0, 190, 220),
        EYE_INNER=(140, 240, 255), IRIS_SHINE=(230, 255, 255),
        GLOW_COL=(0, 110, 150),    MOUTH_COL=(0, 200, 230),
        ZZZ_COL=(0, 160, 200),     LISTEN_COL=(255, 170, 40),
        BLUSH_COL=(0, 220, 200),   TEAR_COL=(120, 220, 255),
        STEAM_COL=(255, 120, 60),  STAR_COL=(180, 255, 255),
        TEETH_COL=(210, 250, 255), WAVE_COL=(0, 220, 255),
        ANGRY_COL=(255, 60, 40),   HEART_COL=(90, 240, 210),
        AWAKE_COL=(120, 255, 220),
        EYE_RADIUS=8, IRIS_SQUARE=True, MOUTH_STYLE="bars",
        EYE_STYLE="iris", CATCHLIGHT=False,
        EYE_GRAD_TOP=(140, 240, 255), EYE_GRAD_BOTTOM=(0, 120, 150),
    ),
    3: dict(
        name="aurora",
        EYE_OUTER=(60, 225, 210),  EYE_MID=(0, 170, 190),
        EYE_INNER=(150, 250, 240), IRIS_SHINE=(245, 255, 255),
        GLOW_COL=(0, 130, 140),    MOUTH_COL=(70, 230, 215),
        ZZZ_COL=(90, 200, 220),    LISTEN_COL=(120, 190, 255),
        BLUSH_COL=(255, 120, 160), TEAR_COL=(120, 200, 255),
        STEAM_COL=(255, 130, 80),  STAR_COL=(255, 230, 120),
        TEETH_COL=(235, 255, 252), WAVE_COL=(0, 210, 200),
        ANGRY_COL=(255, 70, 60),   HEART_COL=(255, 105, 140),
        AWAKE_COL=(140, 255, 230),
        EYE_RADIUS=44, IRIS_SQUARE=False, MOUTH_STYLE="organic",
        # Solid eyes: no iris/pupil — the whole eye is one vertical-gradient
        # block whose SHAPE carries the emotion (squash, slant, arc), the way
        # EMO / Vector / Cozmo faces work. Catchlights make it look alive.
        EYE_STYLE="solid", CATCHLIGHT=True,
        EYE_GRAD_TOP=(175, 255, 240), EYE_GRAD_BOTTOM=(0, 185, 195),
    ),
}

# Styles 1 keeps its original look — the new geometry keys just describe it.
STYLES[1].update(EYE_STYLE="iris", CATCHLIGHT=False,
                 EYE_GRAD_TOP=(160, 90, 255), EYE_GRAD_BOTTOM=(55, 0, 200))

_KEYS = tuple(k for k in STYLES[1] if k != "name")


class Theme:
    """Live palette. Attributes are whatever the active style defines."""

    __slots__ = ("style", "name") + _KEYS

    def __repr__(self):
        return f"<Theme {self.style} ({self.name})>"


theme = Theme()


def apply_style(n):
    """Swap the whole palette + geometry at runtime.
    1 = classic, 2 = robo, 3 = aurora (Face 3)."""
    style = STYLES.get(n)
    if style is None:
        return False
    theme.style = n
    theme.name  = style["name"]
    for key in _KEYS:
        setattr(theme, key, style[key])
    clear_caches()           # cached glow surfaces are baked per-palette
    print(f"[face] Style {n} ({style['name']}) active")
    return True


# ── Interpolation helpers ─────────────────────────────────────────────────────
def lerp(a, b, t):
    """Move `a` a fraction `t` of the way toward `b`. Every animation here is
    built from this — it is what keeps transitions smooth instead of snapping."""
    return a + (b - a) * t


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def blend(col_a, col_b, t):
    """Mix two RGB colours. `t` = 0 gives col_a, 1 gives col_b."""
    return (int(lerp(col_a[0], col_b[0], t)),
            int(lerp(col_a[1], col_b[1], t)),
            int(lerp(col_a[2], col_b[2], t)))


# ── Caches ────────────────────────────────────────────────────────────────────
# Glow surfaces and fonts were being rebuilt from scratch every frame, which was
# the single biggest cost on the Pi. Both are now cached; sizes are bucketed so
# nearby sizes share one entry instead of filling the cache with near-duplicates.
_GLOW_CACHE = {}
_GLOW_CACHE_MAX = 220
_FONT_CACHE = {}

# Small sprites (motes, particles, catchlights, overlay pills, mouth shapes).
# These were each a fresh SRCALPHA Surface plus a few draw calls EVERY frame —
# a few hundred allocations a second, which on the Pi's CPU-only pygame is real
# time. They are rendered once per (shape, size bucket, colour, alpha bucket)
# and reused. Kept apart from the glow cache because there are many more of
# them, and a full glow cache is cleared wholesale.
_SPRITE_CACHE = {}
_SPRITE_CACHE_MAX = 1800


def clear_caches():
    """Drop every baked surface. Called on a style switch, since all of them
    have the old palette's colours burned in."""
    _GLOW_CACHE.clear()
    _SPRITE_CACHE.clear()


def alpha_bucket(alpha, steps=16):
    """Quantise a 0-255 alpha to `steps` evenly spaced levels (0 and 255 kept
    exact). Fading things then share a handful of cached sprites instead of
    one per alpha value; 16 levels is below what the eye picks out on a fade
    that lasts a second or more."""
    a = int(alpha)
    if a <= 0:
        return 0
    if a >= 255:
        return 255
    step = 255.0 / (steps - 1)
    return int(round(round(a / step) * step))


def sprite(key, build):
    """Return the cached sprite for `key`, calling `build()` to make it once.

    `key` must capture everything that changes the pixels (shape, size,
    colour, alpha bucket). A full cache is cleared rather than LRU-evicted:
    it only fills when something keys on a continuously varying value, and
    a clear is cheaper than bookkeeping on every hit."""
    surf = _SPRITE_CACHE.get(key)
    if surf is None:
        if len(_SPRITE_CACHE) >= _SPRITE_CACHE_MAX:
            _SPRITE_CACHE.clear()
        surf = build()
        _SPRITE_CACHE[key] = surf
    return surf


def with_alpha(surf, alpha):
    """Copy of a per-pixel-alpha surface with every alpha scaled by alpha/255.

    Baking the fade into the pixels (instead of set_alpha on a shared surface)
    keeps cached sprites immutable and blits on pygame's plain per-pixel path."""
    out = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    out.blit(surf, (0, 0))          # onto fully transparent pixels = exact copy
    if alpha < 255:
        out.fill((255, 255, 255, max(0, int(alpha))),
                 special_flags=pygame.BLEND_RGBA_MULT)
    return out


def _cache_put(key, surf):
    if len(_GLOW_CACHE) >= _GLOW_CACHE_MAX:
        _GLOW_CACHE.clear()
    _GLOW_CACHE[key] = surf


def get_font(size):
    font = _FONT_CACHE.get(size)
    if font is None:
        font = pygame.font.SysFont("monospace", size, bold=True)
        _FONT_CACHE[size] = font
    return font


# ── Glow drawing ──────────────────────────────────────────────────────────────
def draw_glow_rect(surf, color, rect, radius=20, layers=5, max_alpha=70):
    """Soft halo behind a rounded rect, built from concentric fading layers."""
    bw = max(8, (rect.width  // 8) * 8)      # bucket to 8 px
    bh = max(8, (rect.height // 8) * 8)
    key = ("r", color, bw, bh, radius, layers, max_alpha)

    glow = _GLOW_CACHE.get(key)
    if glow is None:
        pad  = (layers + 1) * 5
        glow = pygame.Surface((bw + pad * 2, bh + pad * 2), pygame.SRCALPHA)
        for i in range(layers, 0, -1):
            alpha  = int(max_alpha * (i / layers) ** 2)
            expand = (layers - i + 1) * 5
            layer  = pygame.Surface((bw + expand * 2, bh + expand * 2),
                                    pygame.SRCALPHA)
            pygame.draw.rect(layer, (*color, alpha), layer.get_rect(),
                             border_radius=radius + expand)
            glow.blit(layer, (pad - expand, pad - expand))
        _cache_put(key, glow)

    surf.blit(glow, (rect.centerx - glow.get_width()  // 2,
                     rect.centery - glow.get_height() // 2))


def draw_grad_rect(surf, top_col, bottom_col, rect, radius=20):
    """Rounded rect filled with a vertical gradient — the body of a Face 3 eye.

    Built row-by-row once and cached (bucketed to 8 px like the glows), because
    regenerating a ~200-row gradient every frame for both eyes is exactly the
    kind of per-frame cost the Pi cannot afford."""
    bw = max(8, (rect.width  // 8) * 8)
    bh = max(8, (rect.height // 8) * 8)
    key = ("g", top_col, bottom_col, bw, bh, radius)

    grad = _GLOW_CACHE.get(key)
    if grad is None:
        grad = pygame.Surface((bw, bh), pygame.SRCALPHA)
        for y in range(bh):
            t = y / max(1, bh - 1)
            pygame.draw.line(grad, (*blend(top_col, bottom_col, t), 255),
                             (0, y), (bw, y))
        # Punch the rounded corners: BLEND_RGBA_MIN against a white rounded
        # mask zeroes everything outside the shape and leaves the rest intact.
        mask = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(),
                         border_radius=min(radius, bh // 2, bw // 2))
        grad.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        _cache_put(key, grad)

    surf.blit(grad, (rect.centerx - bw // 2, rect.centery - bh // 2))


def draw_glow_circle(surf, color, center, radius, layers=4, max_alpha=55):
    """Soft halo behind a circle."""
    br  = max(4, (radius // 4) * 4)          # bucket to 4 px
    key = ("c", color, br, layers, max_alpha)

    glow = _GLOW_CACHE.get(key)
    if glow is None:
        pad  = (layers + 1) * 5 + 2
        size = br * 2 + pad * 2
        glow = pygame.Surface((size, size), pygame.SRCALPHA)
        for i in range(layers, 0, -1):
            alpha = int(max_alpha * (i / layers) ** 2)
            layer = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(layer, (*color, alpha), (size // 2, size // 2),
                               br + (layers - i + 1) * 5)
            glow.blit(layer, (0, 0))
        _cache_put(key, glow)

    surf.blit(glow, (center[0] - glow.get_width()  // 2,
                     center[1] - glow.get_height() // 2))
