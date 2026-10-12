# face_cards.py
"""
face_cards.py — the full-screen cards Luna shows in place of her face.

When a question is better ANSWERED ON SCREEN than with a face — the date, the
time, who a teacher is — the face fades out and one of these fades in:

    date      a calendar tile like the macOS widget: weekday in red, month
              in grey, a huge day number
    time      an analog clock tile: white dial, black hands, orange seconds
    person    a name in big rounded bold, the role under it in grey, one
              highlight line under that (a recent achievement, a PhD)
    greeting  the name of someone Luna has just recognised, with a short
              line ("Good afternoon · welcome back")

Every function here is pure drawing: it takes the content and returns a
WIDTH x HEIGHT Surface on black, and never reads shared state. robot_face
decides WHEN a card shows and cross-fades it with the face; face.py is how
the rest of Luna asks for one.

Fonts: Apple's SF Pro Rounded when the machine has it (a Mac) — a Bold and a
Semibold instance are cut from the system's variable font once and cached
per machine, never copied into the repo, since Apple's licence does not allow
it. Everywhere else (the Pi) the bundled Nunito (assets/fonts, SIL Open Font
Licence) — the closest open rounded face. Then pygame's own font, so a card
can always be drawn.
"""

import math
import os
import tempfile

import pygame

from face_style import WIDTH, HEIGHT

# ── Palette: the macOS widget colours ─────────────────────────────────────────
BLACK  = (0, 0, 0)
TILE   = (28, 28, 30)        # widget background
WHITE  = (242, 242, 247)
GREY   = (142, 142, 147)
SILVER = (199, 199, 204)     # the clock's minute ticks
RED    = (255, 69, 58)
ORANGE = (255, 159, 10)
BLUE   = (100, 210, 255)     # a person card's highlight line
INK    = (17, 17, 17)        # clock numerals and hands

TILE_SIZE   = 560            # date / clock tile, px on the 1400x800 canvas
TILE_RADIUS = 124
SUPERSAMPLE = 2              # shapes drawn at 2x, then smoothed down

# ── Fonts ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SF_ROUNDED = "/System/Library/Fonts/SFNSRounded.ttf"
_SF_WEIGHT = {"bold": 700, "semibold": 590}
_BUNDLED = {"bold": os.path.join(_HERE, "assets", "fonts", "Nunito-ExtraBold.ttf"),
            "semibold": os.path.join(_HERE, "assets", "fonts", "Nunito-SemiBold.ttf")}
_font_paths = {}
_fonts = {}


def _font_cache_dir():
    base = os.path.expanduser("~/Library/Caches")
    if not os.path.isdir(base):
        base = os.environ.get("XDG_CACHE_HOME") or tempfile.gettempdir()
    path = os.path.join(base, "luna-fonts")
    os.makedirs(path, exist_ok=True)
    return path


def _sf_rounded(weight):
    """A static SF Pro Rounded instance at this weight, cut from the system's
    variable font into a per-machine cache. None when unavailable."""
    if not os.path.exists(_SF_ROUNDED):
        return None
    out = os.path.join(_font_cache_dir(), f"SFRounded-{_SF_WEIGHT[weight]}.ttf")
    if os.path.exists(out):
        return out
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
        font = instancer.instantiateVariableFont(
            TTFont(_SF_ROUNDED), {"wght": _SF_WEIGHT[weight]},
            updateFontNames=False)
        tmp = out + ".tmp"
        font.save(tmp)
        os.replace(tmp, out)
        return out
    except Exception as e:                                  # noqa: BLE001
        print(f"[face] SF Pro Rounded unavailable ({e}) — using Nunito")
        return None


def font_path(weight="bold"):
    """Path of the card font at this weight ("bold" | "semibold"), or None
    for pygame's built-in font."""
    if weight not in _font_paths:
        path = _sf_rounded(weight)
        if path is None and os.path.exists(_BUNDLED[weight]):
            path = _BUNDLED[weight]
        _font_paths[weight] = path
    return _font_paths[weight]


def font(size, weight="bold"):
    key = (int(size), weight)
    f = _fonts.get(key)
    if f is None:
        if not pygame.font.get_init():
            pygame.font.init()
        path = font_path(weight)
        f = pygame.font.Font(path, key[0]) if path else \
            pygame.font.SysFont(None, key[0], bold=True)
        _fonts[key] = f
    return f


def fit_font(text, max_width, start, smallest, weight="bold"):
    """The largest size from `start` down to `smallest` at which `text` fits."""
    for size in range(int(start), int(smallest) - 1, -4):
        f = font(size, weight)
        if f.size(text)[0] <= max_width:
            return f
    return font(smallest, weight)


def wrap(text, f, max_width, max_lines=2):
    """Greedy word wrap; the last line is ellipsised if text is left over."""
    words, lines, line = (text or "").split(), [], ""
    for i, w in enumerate(words):
        trial = f"{line} {w}".strip()
        if f.size(trial)[0] <= max_width:
            line = trial
            continue
        if line:
            lines.append(line)
        line = w
        if len(lines) == max_lines:
            line = ""
            rest = " ".join(words[i:])
            lines[-1] = _ellipsis(f"{lines[-1]} {rest}", f, max_width)
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    return lines


def _ellipsis(text, f, max_width):
    if f.size(text)[0] <= max_width:
        return text
    while text and f.size(text + "…")[0] > max_width:
        text = text[:-1]
    return text.rstrip(" ,·-") + "…"


# ── Drawing helpers ───────────────────────────────────────────────────────────
def _canvas():
    s = pygame.Surface((WIDTH, HEIGHT))
    s.fill(BLACK)
    return s


def _tile():
    """(canvas, tile rect): a smooth rounded dark tile centred on black."""
    k = SUPERSAMPLE
    big = pygame.Surface((TILE_SIZE * k, TILE_SIZE * k), pygame.SRCALPHA)
    pygame.draw.rect(big, TILE, big.get_rect(), border_radius=TILE_RADIUS * k)
    tile = pygame.transform.smoothscale(big, (TILE_SIZE, TILE_SIZE))
    canvas = _canvas()
    rect = tile.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    canvas.blit(tile, rect)
    return canvas, rect


def _blit_centred(surf, text_surf, cx, cy):
    surf.blit(text_surf, text_surf.get_rect(center=(int(cx), int(cy))))


# ── Cards ─────────────────────────────────────────────────────────────────────
def date_card(now):
    """`now` = a datetime. The macOS calendar widget: 'Mon' in red, 'Oct' in
    grey on one line, the day number huge and white beneath."""
    canvas, rect = _tile()
    head = font(92, "bold")
    day_name = now.strftime("%a")
    month = now.strftime("%b")
    left = head.render(day_name, True, RED)
    gap = head.size(" ")[0]
    right = head.render(month, True, GREY)
    total = left.get_width() + gap + right.get_width()
    x = rect.centerx - total // 2
    y = rect.top + 58
    canvas.blit(left, (x, y))
    canvas.blit(right, (x + left.get_width() + gap, y))
    num = font(330, "bold").render(str(now.day), True, WHITE)
    _blit_centred(canvas, num, rect.centerx, rect.top + 345)
    return canvas


_clock_base_cache = {}


def _clock_base(r):
    """Tile + dial + ticks + numerals: everything but the hands. Static, so
    it is drawn once and reused — only the hands change each second."""
    base = _clock_base_cache.get(r)
    if base is not None:
        return base
    canvas, rect = _tile()
    k = SUPERSAMPLE
    size = (r * 2 + 8) * k
    dial = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2
    pygame.draw.circle(dial, (255, 255, 255), (c, c), r * k)
    for i in range(60):                       # minute ticks
        a = math.radians(i * 6)
        hour = i % 5 == 0
        outer, inner = (r - 10) * k, (r - (30 if hour else 22)) * k
        pygame.draw.line(dial, INK if hour else SILVER,
                         (c + inner * math.sin(a), c - inner * math.cos(a)),
                         (c + outer * math.sin(a), c - outer * math.cos(a)),
                         (5 if hour else 3) * k)
    dial = pygame.transform.smoothscale(dial, (size // k, size // k))
    canvas.blit(dial, dial.get_rect(center=rect.center))
    numerals = font(46, "bold")
    for n in range(1, 13):                    # numerals at 1x: crisp
        a = math.radians(n * 30)
        rr = r - 64
        _blit_centred(canvas, numerals.render(str(n), True, INK),
                      rect.centerx + rr * math.sin(a),
                      rect.centery - rr * math.cos(a) + 2)
    _clock_base_cache[r] = base = (canvas, rect)
    return base


def clock_card(now):
    """`now` = a datetime. The macOS clock widget: white dial on the dark
    tile, numerals, minute ticks, black hands and an orange second hand."""
    r = 236                                   # dial radius
    base, rect = _clock_base(r)
    canvas = base.copy()
    hands = _hands_layer(now, r)
    canvas.blit(hands, hands.get_rect(center=rect.center))
    return canvas


def _hands_layer(now, r):
    """The three hands alone, smoothed — drawn over the numerals."""
    k = SUPERSAMPLE
    size = (r * 2 + 8) * k
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    c = size / 2

    def hand(angle_deg, length, width, colour, tail=0.0):
        a = math.radians(angle_deg)
        dx, dy = math.sin(a), -math.cos(a)
        x0, y0 = c - dx * tail * k, c - dy * tail * k
        x1, y1 = c + dx * length * k, c + dy * length * k
        pygame.draw.line(s, colour, (x0, y0), (x1, y1), width * k)
        for (x, y) in ((x0, y0), (x1, y1)):
            pygame.draw.circle(s, colour, (x, y), width * k / 2)

    sec = now.second
    minute = now.minute + sec / 60.0
    hour = (now.hour % 12) + minute / 60.0
    hand(hour * 30, r * 0.52, 14, INK)
    hand(minute * 6, r * 0.80, 10, INK)
    hand(sec * 6, r * 0.88, 3, ORANGE, tail=r * 0.16)
    pygame.draw.circle(s, ORANGE, (c, c), 9 * k)
    pygame.draw.circle(s, (255, 255, 255), (c, c), 3 * k)
    return pygame.transform.smoothscale(s, (size // k, size // k))


def person_card(name, role="", highlight=""):
    """A name in big rounded bold, the role beneath in grey, then one
    highlight line (up to two lines) — vertically centred as one block."""
    canvas = _canvas()
    max_w = WIDTH - 160
    name_font = fit_font(name, max_w, 124, 64, "bold")
    role_font = font(50, "semibold")
    high_font = font(42, "semibold")
    blocks = [(name_font.render(name, True, WHITE), 26)]
    if role:
        blocks.append((role_font.render(_ellipsis(role, role_font, max_w),
                                        True, GREY), 34))
    for line in wrap(highlight, high_font, max_w - 80, 2):
        blocks.append((high_font.render(line, True, BLUE), 10))
    height = sum(s.get_height() + gap for s, gap in blocks) - blocks[-1][1]
    y = (HEIGHT - height) // 2
    for s, gap in blocks:
        canvas.blit(s, s.get_rect(midtop=(WIDTH // 2, y)))
        y += s.get_height() + gap
    return canvas


def greeting_card(name, line=""):
    """Someone Luna has just recognised: the name, huge, and a short line."""
    canvas = _canvas()
    max_w = WIDTH - 200
    name_s = fit_font(name, max_w, 176, 80, "bold").render(name, True, WHITE)
    blocks = [(name_s, 22)]
    if line:
        f = font(56, "semibold")
        blocks.append((f.render(_ellipsis(line, f, max_w), True, GREY), 0))
    height = sum(s.get_height() + gap for s, gap in blocks)
    y = (HEIGHT - height) // 2
    for s, gap in blocks:
        canvas.blit(s, s.get_rect(midtop=(WIDTH // 2, y)))
        y += s.get_height() + gap
    return canvas


def render(card, now):
    """The Surface for a card spec — {"kind": ..., ...} as face.py stores it —
    at datetime `now`. None for an unknown kind."""
    kind = (card or {}).get("kind")
    if kind == "date":
        return date_card(now)
    if kind == "time":
        return clock_card(now)
    if kind == "person":
        return person_card(card.get("name", ""), card.get("role", ""),
                           card.get("highlight", ""))
    if kind == "greeting":
        return greeting_card(card.get("name", ""), card.get("line", ""))
    return None


def cache_key(card, now):
    """What makes a card's pixels change: the clock every second, the date
    once a day, the others never."""
    kind = (card or {}).get("kind")
    if kind == "time":
        return ("time", now.strftime("%H%M%S"))
    if kind == "date":
        return ("date", now.strftime("%Y%m%d"))
    return (kind,) + tuple(sorted((k, str(v)) for k, v in (card or {}).items()))
