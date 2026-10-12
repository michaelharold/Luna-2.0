# face_parts.py
"""
face_parts.py — the physical pieces of Luna's face: Eye, Mouth, MicroExpressions.

Each part owns its own smoothing state and knows how to draw itself. None of
them contain a single `if emotion == "happy"` branch: they are handed an
`Expression` (see expressions.py) and render whatever it asks for. Adding a
mood therefore never means editing this file — only adding a shape does.

Shapes are dispatched through small tables (`_BROWS`, `_MOUTHS`) rather than
if/elif ladders, so a new brow or mouth is one method plus one dict entry.
"""

import math
import time
import random
import pygame

from face_style import (theme, BG, PUPIL_COL,
                        lerp, clamp, blend, draw_glow_rect, draw_glow_circle,
                        draw_grad_rect, sprite, alpha_bucket)
from config import (MOUTH_FILLED_TALK, MOUTH_MIN_DEPTH, MOUTH_MAX_DEPTH,
                    MOUTH_CLOSED_H)


# ══════════════════════════════════════════════════════════════════════════════
# Eye
# ══════════════════════════════════════════════════════════════════════════════
class Eye:
    """One eye, including its brow.

    Every visible property is stored twice over: the current value, and the
    target it is easing toward. `update()` only ever moves current -> target,
    which is what makes expression changes glide instead of snap.
    """

    def __init__(self, rel_x, rel_y, w, h):
        self.rel_x = rel_x
        self.rel_y = rel_y
        self.w = float(w)
        self.h = float(h)
        self.blink_t     = 1.0
        self.pupil_ox    = 0.0
        self.pupil_oy    = 0.0
        self.pupil_scale = 1.0
        self.squint      = 0.0
        self.widen       = 0.0
        self.droop       = 0.0
        self.slant       = 0.0   # 0-1, the angry cut across the top (Face 3)

    # ── state ─────────────────────────────────────────────────────────────
    def update(self, w, h, blink_t, pupil_ox, pupil_oy,
               pupil_scale=1.0, squint=0.0, widen=0.0, droop=0.0, slant=0.0):
        # Rates are tuned per property: the blink is fast, the pupil follows
        # quickly, and the shape changes are slow so moods settle rather than
        # flick. Do not flatten these into one constant.
        self.w           = lerp(self.w,           w,           0.10)
        self.h           = lerp(self.h,           h,           0.10)
        self.blink_t     = lerp(self.blink_t,     blink_t,     0.30)
        self.pupil_ox    = lerp(self.pupil_ox,    pupil_ox,    0.18)
        self.pupil_oy    = lerp(self.pupil_oy,    pupil_oy,    0.18)
        self.pupil_scale = lerp(self.pupil_scale, pupil_scale, 0.10)
        self.squint      = lerp(self.squint,      squint,      0.08)
        self.widen       = lerp(self.widen,       widen,       0.08)
        self.droop       = lerp(self.droop,       droop,       0.06)
        self.slant       = lerp(self.slant,       slant,       0.08)

    # ── drawing ───────────────────────────────────────────────────────────
    def draw(self, surf, cx, cy, brow="flat", color=None,
             red=0.0, arc=0.0, nudge=None):
        """`red` (0-1) is how far the eye is recoloured red — angry.

        `nudge` = (pupil_dx, scale) is a one-frame additive offset (the
        recognition double-take). It is applied around the draw and undone
        afterwards, so it never leaks into the eased state that update()
        smooths — otherwise the lerps would swallow a 0.15 s dart."""
        if nudge is not None and (nudge[0] or nudge[1] != 1.0):
            saved = (self.pupil_ox, self.w, self.h)
            self.pupil_ox += nudge[0]
            self.w *= nudge[1]
            self.h *= nudge[1]
            try:
                self._draw(surf, cx, cy, brow, color, red, arc)
            finally:
                self.pupil_ox, self.w, self.h = saved
            return
        self._draw(surf, cx, cy, brow, color, red, arc)

    def _draw(self, surf, cx, cy, brow, color, red, arc):
        eye_col = color or theme.EYE_OUTER
        pal = _red_palette(red)
        if red > 0.01:
            eye_col = blend(eye_col, theme.ANGRY_COL, red)

        # '∩' arc eyes take over completely once the blend is committed, so the
        # swap into and out of the shape happens mid-transition and isn't seen.
        if arc > 0.55 and self.blink_t > 0.75:
            self._draw_arc(surf, cx, cy, eye_col, arc)
            return

        if theme.EYE_STYLE == "solid":
            # Face 3: one gradient block whose shape carries the emotion.
            # No separate brow — the slanted top edge IS the brow.
            self._draw_solid(surf, cx, cy, eye_col, pal, red > 0.01)
            return

        rect = self._body_rect(cx, cy)
        self._draw_body(surf, rect, eye_col, pal)
        self._draw_iris(surf, cx, cy, rect, pal)
        if self.blink_t > 0.4:
            self._draw_brow(surf, cx, rect, brow)

    def _body_rect(self, cx, cy):
        h = self.h * (1.0 - self.squint * 0.4) * (1.0 + self.widen * 0.2)
        vis_h = max(4, int(h * self.blink_t))
        w = int(self.w)
        return pygame.Rect(cx - w // 2, cy - vis_h // 2, w, vis_h)

    # ── Face 3: solid gradient eye ────────────────────────────────────────
    def _draw_solid(self, surf, cx, cy, eye_col, pal, red):
        """EMO/Vector-style eye: a rounded gradient block, animated with
        squash-and-stretch. The gaze moves the WHOLE eye (a screen-robot's
        eyes slide, they don't roll), a blink squashes it flat while bulging
        it wider, and the top edge slants for angry/sad instead of a brow."""
        # Solid eyes run larger and squarer than the iris styles — expression
        # sizes are shared across styles, and what reads as an eyeball housing
        # there reads as the whole eye here.
        h = (self.h * 1.30) * (1.0 - self.squint * 0.45) * (1.0 + self.widen * 0.22)
        vis_h = max(6, int(h * self.blink_t))
        # Squash-and-stretch: as the lid drops the eye widens, so a blink has
        # the weight of a soft body rather than a shutter.
        w = int(self.w * 1.06 * (1.0 + 0.12 * (1.0 - self.blink_t)))

        # The gaze slides the whole block, damped so it stays composed.
        ex = cx + int(self.pupil_ox * 0.55)
        ey = cy + int(self.pupil_oy * 0.45)
        rect = pygame.Rect(ex - w // 2, ey - vis_h // 2, w, vis_h)

        top, bottom = pal["top"], pal["bottom"]
        if not red and eye_col != theme.EYE_OUTER:
            # A status tint (listening pulse / awake glow) recolours the
            # gradient toward the cue colour rather than replacing it.
            top    = blend(top,    eye_col, 0.45)
            bottom = blend(bottom, eye_col, 0.45)

        draw_glow_rect(surf, pal["glow"], rect,
                       radius=theme.EYE_RADIUS, layers=5)
        draw_grad_rect(surf, top, bottom, rect, radius=theme.EYE_RADIUS)

        # Emotion lives in the top edge: angry slants it down toward the nose,
        # sad droops it down toward the OUTSIDE — mirrored per eye. Both are
        # eased amounts, so the cut grows in rather than appearing whole.
        is_left = self.rel_x < 0
        if self.slant > 0.05:
            self._slant_top(surf, rect, inner_down=True,  is_left=is_left,
                            amount=0.42 * self.slant)
        elif self.droop > 0.05:
            self._slant_top(surf, rect, inner_down=False, is_left=is_left,
                            amount=0.38 * self.droop)

        if theme.CATCHLIGHT and self.blink_t > 0.35:
            self._draw_catchlights(surf, rect)

    def _slant_top(self, surf, rect, inner_down, is_left, amount):
        """Paint a background wedge over the eye's top edge. inner_down=True
        is the angry cut (drops toward the nose); False is the sad droop
        (drops toward the temple)."""
        drop = int(rect.h * clamp(amount, 0.0, 0.9))
        if drop < 3:
            return
        inner_on_right = is_left     # the nose side of the LEFT eye is its right
        high_on_left = inner_on_right if inner_down else not inner_on_right
        x0, x1 = rect.left - 2, rect.right + 2
        if high_on_left:
            pts = [(x0, rect.top - 8), (x1, rect.top - 8),
                   (x1, rect.top + drop), (x0, rect.top - 2)]
        else:
            pts = [(x0, rect.top - 8), (x1, rect.top - 8),
                   (x1, rect.top - 2), (x0, rect.top + drop)]
        pygame.draw.polygon(surf, BG, pts)

    def _draw_catchlights(self, surf, rect):
        """Two specular dots — one big high-left, one small low-right. They sit
        at fixed positions IN the eye (not tracking the gaze), like a room
        light reflecting off a curved screen; this is most of the 'alive'."""
        big = max(6, int(rect.w * 0.14))
        sml = max(3, int(rect.w * 0.06))
        bx = rect.left + int(rect.w * 0.28)
        by = rect.top  + int(rect.h * 0.26)
        sx = rect.left + int(rect.w * 0.68)
        sy = rect.top  + int(rect.h * 0.66)
        for r, x, y, a in ((big, bx, by, 235), (sml, sx, sy, 170)):
            if y + r > rect.bottom or y - r < rect.top:
                continue                      # squashed shut — no room left
            surf.blit(_dot_sprite(r, theme.IRIS_SHINE, a), (x - r, y - r))

    def _draw_body(self, surf, rect, eye_col, pal):
        draw_glow_rect(surf, pal["glow"], rect,
                       radius=theme.EYE_RADIUS + 2, layers=5)
        pygame.draw.rect(surf, eye_col, rect, border_radius=theme.EYE_RADIUS)

        # Squinting is drawn by painting the background back over the top of
        # the eye — cheaper and cleaner than reshaping the rounded rect.
        if self.squint > 0.05:
            clip_h = int(rect.h * self.squint * 0.5)
            if clip_h > 0:
                pygame.draw.rect(surf, BG,
                                 pygame.Rect(rect.x, rect.y, rect.w, clip_h),
                                 border_radius=max(4, theme.EYE_RADIUS // 2))

        inset = rect.inflate(-20, -20)
        if inset.w > 8 and inset.h > 8:
            pygame.draw.rect(surf, pal["mid"], inset,
                             border_radius=max(4, theme.EYE_RADIUS - 6))

    def _draw_iris(self, surf, cx, cy, rect, pal):
        r  = int(min(rect.w, rect.h) * 0.27 * self.pupil_scale)
        ix = cx + int(self.pupil_ox)
        iy = cy + int(self.pupil_oy)
        if r <= 4 or self.blink_t <= 0.25:
            return

        inner = pal["inner"]
        if theme.IRIS_SQUARE:
            # Robo style — a square sensor rather than an organic iris.
            box = pygame.Rect(ix - r, iy - r, r * 2, r * 2)
            draw_glow_rect(surf, inner, box, radius=6, layers=3,
                           max_alpha=45)
            pygame.draw.rect(surf, inner, box, border_radius=6)
            pr = int(r * 0.55)
            pygame.draw.rect(surf, PUPIL_COL,
                             pygame.Rect(ix - pr, iy - pr, pr * 2, pr * 2),
                             border_radius=4)
            self._draw_shine(surf, ix, iy, r, square=True)
        else:
            draw_glow_circle(surf, inner, (ix, iy), r)
            pygame.draw.circle(surf, inner, (ix, iy), r)
            pygame.draw.circle(surf, PUPIL_COL, (ix, iy), int(r * 0.50))
            self._draw_shine(surf, ix, iy, r, square=False)

    def _draw_shine(self, surf, ix, iy, r, square):
        """Small specular dot up and to the left — what makes the eye look wet
        rather than printed."""
        sr  = max(3, int(r * (0.18 if square else 0.20)))
        off = r * (0.45 if square else 0.30)
        if square:
            col = (*theme.IRIS_SHINE, 190)

            def build():
                s = pygame.Surface((sr * 2, sr * 2), pygame.SRCALPHA)
                pygame.draw.rect(s, col, s.get_rect(), border_radius=2)
                return s
            s = sprite(("shine_sq", sr, col), build)
        else:
            s = _dot_sprite(sr, theme.IRIS_SHINE, 180)
        surf.blit(s, (ix - int(off) - sr, iy - int(off) - sr))

    def _draw_arc(self, surf, cx, cy, color, amount):
        """Happy '∩' eye — the classic upside-down-U smiling eye.

        Built as a filled polygon between two half-ellipses, NOT with
        pygame.draw.arc. A thick arc renders with ragged, speckled edges —
        visibly broken at this size — whereas a polygon fills cleanly.
        """
        rx = self.w * 0.40 * (0.94 + 0.06 * amount)
        ry = max(20.0, self.h * 0.46)

        # The arch is ~70 trig calls plus three polygon fills; once the eye has
        # settled its size barely moves, so it is cached at 1 px resolution.
        # Sub-pixel radii are rounded — a polygon rasterises to the same
        # pixels to within an edge pixel either way.
        rx, ry = round(rx), round(ry)
        th = max(11.0, ry * 0.44)                 # band thickness
        key = ("arc", rx, ry, color, theme.GLOW_COL, theme.IRIS_SHINE)
        s = sprite(key, lambda: self._arc_sprite(rx, ry, th, color))
        pad = int(th) + 14
        ox, oy = s.get_width() // 2, s.get_height() - pad
        surf.blit(s, (cx - ox, cy - oy + int(ry * 0.35)))

    @staticmethod
    def _arc_sprite(rx, ry, th, color):
        def arch(rx_o, ry_o, rx_i, ry_i, n=34):
            pts = [(rx_o * math.cos(math.pi * (1 - i / n)),
                    -ry_o * math.sin(math.pi * (1 - i / n)))
                   for i in range(n + 1)]         # outer edge, left -> right
            pts += [(rx_i * math.cos(math.pi * (i / n)),
                     -ry_i * math.sin(math.pi * (i / n)))
                    for i in range(n + 1)]        # inner edge, right -> left
            return pts

        pad = int(th) + 14
        w   = int(rx * 2) + pad * 2
        h   = int(ry) + pad * 2
        s   = pygame.Surface((w, h), pygame.SRCALPHA)
        ox, oy = w // 2, h - pad

        def band(rx_o, ry_o, rx_i, ry_i, col, alpha):
            pygame.draw.polygon(s, (*col, alpha),
                                [(ox + px, oy + py)
                                 for px, py in arch(rx_o, ry_o, rx_i, ry_i)])

        # A soft halo behind, then ONE solid band. Deliberately not three
        # stacked colours — concentric dark/light/dark bands read as a rainbow
        # rather than an eye.
        band(rx + 7, ry + 7, rx - th - 5, ry - th - 5, theme.GLOW_COL, 70)
        band(rx, ry, rx - th, ry - th, color, 255)

        # One small highlight at the crown of the arch, for a bit of gloss.
        hl_w, hl_h = max(4, int(rx * 0.26)), max(3, int(th * 0.30))
        hl = pygame.Surface((hl_w, hl_h), pygame.SRCALPHA)
        pygame.draw.ellipse(hl, (*theme.IRIS_SHINE, 150), hl.get_rect())
        s.blit(hl, (ox - hl_w // 2 - int(rx * 0.22),
                    oy - int(ry) + int(th * 0.28)))
        return s

    # ── brows ─────────────────────────────────────────────────────────────
    def _draw_brow(self, surf, cx, rect, style):
        y  = rect.top - 16
        w  = int(self.w * 0.70)
        draw = _BROWS.get(style, _BROWS["flat"])
        draw(self, surf, cx, y, w)

    def _brow_flat(self, surf, cx, y, w):
        pygame.draw.rect(surf, theme.EYE_MID,
                         pygame.Rect(cx - w // 2, y, w, 9), border_radius=4)

    def _brow_raised(self, surf, cx, y, w):
        lift = int(10 * self.widen)
        pygame.draw.rect(surf, theme.EYE_MID,
                         pygame.Rect(cx - w // 2, y - lift, w, 9),
                         border_radius=4)

    def _brow_angry(self, surf, cx, y, w):
        # Inner ends drop toward the nose — the universal anger cue. The side
        # comes from the eye itself, not the screen: while the head is tilted
        # the face is drawn on a layer where both eyes sit left of WIDTH/2.
        sign = 1 if self.rel_x < 0 else -1
        pygame.draw.line(surf, theme.EYE_MID,
                         (cx - w // 2, y + sign * 10),
                         (cx + w // 2, y - sign * 10), 9)

    def _brow_sad(self, surf, cx, y, w):
        # Inner corners RAISED — the exact inverse of angry.
        drop = int(self.droop * 14)
        if self.rel_x < 0:
            pygame.draw.line(surf, theme.EYE_MID,
                             (cx - w // 2, y), (cx + w // 2, y - drop), 9)
        else:
            pygame.draw.line(surf, theme.EYE_MID,
                             (cx - w // 2, y - drop), (cx + w // 2, y), 9)

    def _brow_happy(self, surf, cx, y, w):
        col = (*theme.EYE_MID, 200)

        def build():
            arc = pygame.Surface((w + 20, 30), pygame.SRCALPHA)
            pygame.draw.arc(arc, col, pygame.Rect(10, 0, w, 28),
                            math.radians(10), math.radians(170), 7)
            return arc
        surf.blit(sprite(("brow_happy", w, col), build),
                  (cx - w // 2 - 10, y - 10))


def _red_palette(red):
    """The eye's colours with `red` (0-1) of anger mixed in.

    Every part of the eye moves toward a red of its own — light red where the
    style is light, deep red where it is dark — so a red eye keeps its
    gradient and depth instead of becoming a flat red block. Tinting only the
    outline (the old behaviour in the iris styles) barely showed at all."""
    if red <= 0.01:
        return {"top": theme.EYE_GRAD_TOP, "bottom": theme.EYE_GRAD_BOTTOM,
                "glow": theme.GLOW_COL, "mid": theme.EYE_MID,
                "inner": theme.EYE_INNER}
    c = theme.ANGRY_COL
    light = blend(c, (255, 255, 255), 0.35)
    deep  = blend(c, (0, 0, 0), 0.25)
    glow  = blend(c, (0, 0, 0), 0.45)
    return {"top":    blend(theme.EYE_GRAD_TOP,    light, red),
            "bottom": blend(theme.EYE_GRAD_BOTTOM, deep,  red),
            "glow":   blend(theme.GLOW_COL,        glow,  red),
            "mid":    blend(theme.EYE_MID,         deep,  red),
            "inner":  blend(theme.EYE_INNER,       light, red)}


def _dot_sprite(r, col, a):
    """Cached translucent disc — catchlights and iris shines."""
    def build():
        dot = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(dot, (*col, a), (r, r), r)
        return dot
    return sprite(("dot", r, col, a), build)


#: Brow shapes. Add a `_brow_x` method above and one entry here.
_BROWS = {
    "flat":   Eye._brow_flat,
    "raised": Eye._brow_raised,
    "angry":  Eye._brow_angry,
    "sad":    Eye._brow_sad,
    "happy":  Eye._brow_happy,
}


# ══════════════════════════════════════════════════════════════════════════════
# Mouth
# ══════════════════════════════════════════════════════════════════════════════
class Mouth:
    """The mouth has two completely separate jobs, and keeping them apart is
    what stops this class becoming a mess:

      • SPEAKING — shape is driven by live audio energy, emotion is ignored.
      • RESTING  — shape is whatever the current Expression names.
    """

    def __init__(self, rel_x, rel_y):
        self.rel_x       = rel_x
        self.rel_y       = rel_y
        self.talk_open   = 0.0
        self.mouth_w     = 140.0
        self.visible     = 0.0
        self.shape       = "none"   # resting shape currently on screen
        self.corner_lift = 0.0
        self.jaw_bounce  = 0.0
        self.prev_energy = 0.0

    # ── state ─────────────────────────────────────────────────────────────
    def update(self, energy, expr, speaking):
        if speaking:
            self.shape = expr.mouth
            target_visible = 1.0
        elif expr.mouth != self.shape:
            # A new resting shape waits for the old one to fade out, then
            # fades in: the mouth relaxes before it re-forms. Swapping the
            # shape in place made a smile vanish (or turn into a frown) in a
            # single frame.
            target_visible = 0.0
            if self.visible < 0.05:
                self.shape = expr.mouth
        else:
            target_visible = expr.mouth_visible
        self.visible     = lerp(self.visible, target_visible, 0.20)
        self.corner_lift = lerp(self.corner_lift, expr.corner_lift, 0.08)

        if speaking:
            # The energy is the real voice now (lip_sync), so silence means a
            # closed mouth — between sentences, and while the next one is
            # still being synthesised. Sounding speech keeps a small floor so
            # quiet syllables still visibly open it.
            want = max(0.15, energy) if energy > 0.03 else 0.0
            self.talk_open = lerp(self.talk_open, want, 0.30)
            self.mouth_w   = lerp(self.mouth_w,
                                  190.0 if energy > 0.5 else 150.0, 0.12)
            delta = energy - self.prev_energy
            if delta > 0.15:
                self.jaw_bounce = min(1.0, self.jaw_bounce + delta * 2)
            self.jaw_bounce = lerp(self.jaw_bounce, 0.0, 0.20)
        else:
            self.talk_open  = lerp(self.talk_open, 0.0, 0.15)
            self.jaw_bounce = 0.0

        self.prev_energy = energy

    # ── drawing ───────────────────────────────────────────────────────────
    def draw(self, surf, cx, cy, expr, speaking):
        if self.visible < 0.02:
            return
        alpha = int(clamp(self.visible * 255, 0, 255))

        if speaking:
            self._draw_speaking(surf, cx, cy, alpha)
            return

        if self.visible < 0.05:
            return
        shape = _MOUTHS.get(self.shape)
        if shape:
            self._blit_resting(surf, shape, self.shape, cx, cy, alpha)

    def _blit_resting(self, surf, shape, name, cx, cy, alpha):
        """Resting mouths are fixed shapes that only fade, so each is drawn
        once per (shape, palette, alpha bucket) into a sprite and reused.

        The shape methods below still just 'draw at (cx, cy)'; rendering one
        onto a transparent canvas centred on a known point and blitting that
        canvas back gives the same pixels, because every shape is itself
        built on its own alpha surface and blitted — onto a transparent pixel
        pygame copies the source exactly."""
        a = alpha_bucket(alpha, steps=64)
        key = ("mouth", name, theme.MOUTH_COL, theme.EYE_INNER, a)

        def build():
            canvas = pygame.Surface((_MOUTH_CANVAS[0], _MOUTH_CANVAS[1]),
                                    pygame.SRCALPHA)
            shape(self, canvas, _MOUTH_ORIGIN[0], _MOUTH_ORIGIN[1], a)
            return canvas
        surf.blit(sprite(key, build),
                  (cx - _MOUTH_ORIGIN[0], cy - _MOUTH_ORIGIN[1]))

    def _draw_speaking(self, surf, cx, cy, alpha):
        if theme.MOUTH_STYLE == "bars":
            self._draw_bars(surf, cx, cy, alpha)
        elif MOUTH_FILLED_TALK:
            # corner_lift bows the top edge, so she still looks pleased while
            # talking instead of stamping out a neutral rectangle every frame.
            self._draw_d_mouth(surf, cx, cy, alpha,
                               clamp(self.talk_open, 0.0, 1.0),
                               lift=self.corner_lift * 14)
        else:
            self._draw_legacy_jaw(surf, cx, cy, alpha)

    # ── speaking shapes ───────────────────────────────────────────────────
    def _draw_bars(self, surf, cx, cy, alpha):
        """Robotic talking mouth — 7 equalizer bars driven by audio energy."""
        n, bw, gap = 7, 16, 10
        total = n * bw + (n - 1) * gap
        t = time.time() * 9.0
        e = max(0.12, self.talk_open)
        for i in range(n):
            # Centre bars taller; a per-bar phase offset makes it dance.
            centre = 1.0 - abs(i - (n - 1) / 2) / ((n - 1) / 2) * 0.45
            h = int(10 + e * 78 * centre
                    * (0.55 + 0.45 * abs(math.sin(t + i * 0.9))))
            x = cx - total // 2 + i * (bw + gap)
            pygame.draw.rect(surf, theme.MOUTH_COL,
                             pygame.Rect(x, cy - h // 2, bw, h),
                             border_radius=4)
            tip_h = max(3, h // 5)
            tip_col = (*theme.TEETH_COL, alpha_bucket(alpha * 0.8))

            def build(tip_h=tip_h, tip_col=tip_col):
                tip = pygame.Surface((bw, tip_h), pygame.SRCALPHA)
                pygame.draw.rect(tip, tip_col, tip.get_rect(), border_radius=3)
                return tip
            surf.blit(sprite(("bar_tip", bw, tip_h, tip_col), build),
                      (x, cy - h // 2))

    def _d_points(self, cx, cy, half_w, depth, lift=0.0, steps=26):
        """Outline of a 'D' lying on its back: a flat-ish top edge and a curved
        bottom that bulges by `depth`.

        Driving ONE continuous shape from the audio energy gives a C -> D -> C
        cycle: shallow at low energy, a full round D at high energy. The mouth
        this replaced switched between four discrete shapes, which is what made
        it jump around.
        """
        pts = []
        for i in range(steps + 1):                 # top edge, left -> right
            u = i / steps
            pts.append((cx - half_w + 2 * half_w * u,
                        cy - lift * math.sin(math.pi * u)))
        for i in range(steps + 1):                 # bottom arc, right -> left
            u = 1.0 - i / steps
            k = 2.0 * u - 1.0
            # A half ELLIPSE, not a sine bulge. sin(pi*u) meets the corners at
            # a shallow angle and gives a pointed V; sqrt(1-k^2) meets them
            # vertically, which is what makes it read as a rounded D.
            pts.append((cx - half_w + 2 * half_w * u,
                        cy + depth * math.sqrt(max(0.0, 1.0 - k * k))))
        return pts

    def _draw_d_mouth(self, surf, cx, cy, alpha, energy, lift=0.0):
        """Filled 'D on its back', opening and closing with `energy`."""
        half_w = int(self.mouth_w * 0.5)
        depth  = MOUTH_MIN_DEPTH + energy * (MOUTH_MAX_DEPTH - MOUTH_MIN_DEPTH)

        # Nearly shut — a soft rounded bar rather than a degenerate polygon, so
        # the closed mouth still has weight and doesn't flicker.
        if depth <= MOUTH_MIN_DEPTH + 1.5:
            col = (*theme.MOUTH_COL, alpha_bucket(alpha, steps=64))

            def build():
                bar = pygame.Surface((half_w * 2, MOUTH_CLOSED_H * 2),
                                     pygame.SRCALPHA)
                pygame.draw.rect(bar, col, bar.get_rect(),
                                 border_radius=MOUTH_CLOSED_H)
                return bar
            surf.blit(sprite(("closed_bar", half_w, col), build),
                      (cx - half_w, cy - MOUTH_CLOSED_H // 2))
            return

        pad = 40
        s   = pygame.Surface((half_w * 2 + pad * 2,
                              int(depth + lift) + pad * 2), pygame.SRCALPHA)
        ox, oy = pad + half_w, pad + int(lift)
        pts = self._d_points(ox, oy, half_w, depth, lift)

        # Solid fill. Deliberately NO dark interior shape: an inset dark polygon
        # is exactly what made the old mouth look hollow.
        pygame.draw.polygon(s, (*theme.MOUTH_COL, alpha), pts)
        pygame.draw.polygon(s, (*theme.EYE_INNER, alpha), pts, 4)

        # Warm tongue low in the mouth, only when properly open — kept well
        # clear of the rim so it reads as part of the mouth.
        if depth > MOUTH_MAX_DEPTH * 0.55:
            t_depth = depth * 0.30
            pygame.draw.polygon(
                s, (*theme.BLUSH_COL, int(alpha * 0.75)),
                self._d_points(ox, oy + depth - t_depth * 1.15,
                               int(half_w * 0.55), t_depth))

        # Small glint on the upper lip — cheap, and it stops the flat fill
        # looking like a sticker.
        gl_w, gl_h = max(6, int(half_w * 0.28)), max(4, int(depth * 0.13))
        if gl_w > 6 and gl_h > 3:
            gl = pygame.Surface((gl_w, gl_h), pygame.SRCALPHA)
            pygame.draw.ellipse(gl, (*theme.TEETH_COL, int(alpha * 0.30)),
                                gl.get_rect())
            s.blit(gl, (ox - half_w // 2, oy + max(4, int(depth * 0.16))))

        surf.blit(s, (cx - ox, cy - oy))

    def _draw_legacy_jaw(self, surf, cx, cy, alpha):
        """The original four-shape talking jaw, kept for MOUTH_FILLED_TALK=False."""
        mw = int(self.mouth_w)
        e  = self.talk_open

        if e > 0.55:                                   # open jaw with teeth
            mh   = max(20, int((e + self.jaw_bounce * 0.3) * 90))
            rect = pygame.Rect(cx - mw // 2, cy - mh // 2, mw, mh)
            draw_glow_rect(surf, theme.GLOW_COL, rect, radius=26, layers=4,
                           max_alpha=50)
            body = pygame.Surface((mw, mh), pygame.SRCALPHA)
            pygame.draw.rect(body, (*theme.MOUTH_COL, alpha), body.get_rect(),
                             border_radius=22)
            surf.blit(body, rect.topleft)

            th = max(6, mh // 4)
            teeth = pygame.Surface((mw - 8, th), pygame.SRCALPHA)
            pygame.draw.rect(teeth, (*theme.TEETH_COL, int(alpha * 0.7)),
                             teeth.get_rect(), border_radius=8)
            surf.blit(teeth, (rect.x + 4, rect.y + 2))

            inner = pygame.Rect(rect.x + 8, rect.y + th + 8,
                                rect.w - 16, rect.h - th - 16)
            if inner.w > 6 and inner.h > 4:
                dark = pygame.Surface((inner.w, inner.h), pygame.SRCALPHA)
                pygame.draw.rect(dark, (*PUPIL_COL, alpha), dark.get_rect(),
                                 border_radius=14)
                surf.blit(dark, inner.topleft)

            cl = int(self.corner_lift * 12)
            pygame.draw.line(surf, theme.EYE_MID, (rect.left, rect.top + cl),
                             (rect.left + 20, rect.top - cl), 4)
            pygame.draw.line(surf, theme.EYE_MID, (rect.right, rect.top + cl),
                             (rect.right - 20, rect.top - cl), 4)

        elif e > 0.28:                                 # mid-open oval
            mh = max(14, int(e * 70))
            self._blit_ellipse(surf, cx, cy, mw, mh, theme.MOUTH_COL, alpha)
            self._blit_ellipse(surf, cx, cy, max(8, mw - 24), max(4, mh - 10),
                               PUPIL_COL, alpha)

        elif e > 0.08:                                 # pursed
            self._blit_ellipse(surf, cx, cy, max(40, int(mw * 0.45)),
                               max(8, int(e * 40)), theme.MOUTH_COL, alpha)

        else:                                          # closed line
            cl = int(self.corner_lift * 8)
            pygame.draw.lines(surf, theme.MOUTH_COL, False, [
                (cx - 70, cy + cl), (cx - 20, cy - cl // 2), (cx, cy - cl),
                (cx + 20, cy - cl // 2), (cx + 70, cy + cl)], 10)

    @staticmethod
    def _blit_ellipse(surf, cx, cy, w, h, color, alpha):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (*color, alpha), s.get_rect())
        surf.blit(s, (cx - w // 2, cy - h // 2))

    # ── resting shapes — one method per Expression.mouth value ────────────
    def _mouth_smile(self, surf, cx, cy, alpha):
        """Filled U band. A thick pygame.draw.arc renders with ragged, speckled
        edges at this size, which is half of why the old mouth looked scruffy —
        so this is a polygon between two half-ellipses instead."""
        rx, ry, th = 128.0, 58.0, 21.0
        n   = 34
        pts = [(rx * -math.cos(math.pi * (i / n)),
                ry * math.sin(math.pi * (i / n))) for i in range(n + 1)]
        pts += [((rx - th) * -math.cos(math.pi * (1 - i / n)),
                 (ry - th) * math.sin(math.pi * (1 - i / n)))
                for i in range(n + 1)]
        s   = pygame.Surface((int(rx * 2) + 40, int(ry) + 60), pygame.SRCALPHA)
        ox  = int(rx) + 20
        pygame.draw.polygon(s, (*theme.MOUTH_COL, alpha),
                            [(px + ox, py + 20) for px, py in pts])
        surf.blit(s, (cx - ox, cy - 20))

    def _mouth_frown(self, surf, cx, cy, alpha):
        w, h = 200, 100
        s = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
        pygame.draw.arc(s, (*theme.MOUTH_COL, alpha), pygame.Rect(2, 2, w, h),
                        math.radians(20), math.radians(160), 18)
        surf.blit(s, (cx - w // 2 - 2, cy + 4))

    def _mouth_tight(self, surf, cx, cy, alpha):
        """Angry — a flat line, angled. The lighter overlay gives it an edge."""
        s = pygame.Surface((240, 40), pygame.SRCALPHA)
        pygame.draw.line(s, (*theme.MOUTH_COL, alpha), (10, 28), (230, 12), 18)
        pygame.draw.line(s, (*theme.EYE_INNER, int(alpha * 0.5)),
                         (10, 28), (230, 12), 5)
        surf.blit(s, (cx - 120, cy - 20))

    def _mouth_o(self, surf, cx, cy, alpha):
        """Surprised — a round open O with a dark centre."""
        r = 52
        s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*theme.MOUTH_COL, alpha), (r + 2, r + 2), r)
        pygame.draw.circle(s, (*PUPIL_COL, alpha), (r + 2, r + 2),
                           int(r * 0.55))
        surf.blit(s, (cx - r - 2, cy - r - 2))

    def _mouth_sleep(self, surf, cx, cy, alpha):
        """A small peaceful closed curve, drawn faint."""
        w = 160
        s = pygame.Surface((w + 4, 30), pygame.SRCALPHA)
        pygame.draw.lines(s, (*theme.MOUTH_COL, int(alpha * 0.6)), False,
                          [(2, 20), (w // 4, 10), (w // 2, 6),
                           (w * 3 // 4, 10), (w + 2, 20)], 8)
        surf.blit(s, (cx - w // 2 - 2, cy - 10))

    def _mouth_shy(self, surf, cx, cy, alpha):
        """Small hesitant squiggle — narrower than a smile and slightly wavy,
        so it reads as an embarrassed half-grin rather than a confident one."""
        w = 104
        s = pygame.Surface((w + 8, 34), pygame.SRCALPHA)
        pygame.draw.lines(s, (*theme.MOUTH_COL, alpha), False,
                          [(4, 12), (w // 4, 22), (w // 2, 14),
                           (w * 3 // 4, 24), (w + 4, 14)], 11)
        surf.blit(s, (cx - w // 2 - 4, cy - 12))


#: Canvas every resting mouth is baked onto, and where (cx, cy) sits in it.
#: Big enough for the widest (tight, 240 px) and deepest (frown, cy+108).
_MOUTH_CANVAS = (300, 180)
_MOUTH_ORIGIN = (150, 60)

#: Resting mouth shapes. Add a `_mouth_x` method above and one entry here,
#: then any expression can ask for it by name.
_MOUTHS = {
    "smile": Mouth._mouth_smile,
    "frown": Mouth._mouth_frown,
    "tight": Mouth._mouth_tight,
    "o":     Mouth._mouth_o,
    "sleep": Mouth._mouth_sleep,
    "shy":   Mouth._mouth_shy,
}


# ══════════════════════════════════════════════════════════════════════════════
# Micro-expressions
# ══════════════════════════════════════════════════════════════════════════════
class MicroExpressions:
    """Small involuntary eye movements layered on top of the current expression.

    This is the difference between a face and a picture of a face. The current
    Expression sets the floor (an angry face is always somewhat squinted); the
    idle twitches below can only ever add to it, never fight it.
    """

    #: name -> (squint, widen, pupil_scale)
    IDLE_TICS = {
        "squint":  (0.5, 0.0, 0.9),
        "widen":   (0.0, 0.7, 1.2),
        "dilate":  (0.0, 0.1, 1.3),
        "neutral": (0.0, 0.0, 1.0),
    }

    def __init__(self):
        self.squint = 0.0
        self.widen  = 0.0
        self.droop  = 0.0
        self.pupil_scale = 1.0
        self._next_tic = time.time() + random.uniform(3, 8)
        self._tic_ends = 0.0
        self._tic      = None

    def update(self, expr, listening, speaking):
        # Talking keeps the mood's own eye shaping, minus the idle tics, so an
        # angry or sad face stays angry or sad mid-sentence instead of
        # resetting to blank the moment the voice starts. Listening is a
        # state of attention, not a mood, and overrides the shaping.
        if speaking:
            self._set(expr.squint, expr.widen, expr.droop, expr.pupil_scale)
            return
        if listening:
            self._set(0.0, 0.6, 0.0, 1.15)
            return

        squint, widen, pupil = self._idle_tic()

        # The expression is the floor; a tic can only add to it.
        self.squint = max(squint, expr.squint)
        self.widen  = max(widen,  expr.widen)
        self.pupil_scale = max(pupil, expr.pupil_scale) \
            if expr.pupil_scale >= 1.0 else min(pupil, expr.pupil_scale)
        self.droop  = expr.droop

    def _idle_tic(self):
        now = time.time()
        if now > self._next_tic:
            self._next_tic = now + random.uniform(4, 12)
            self._tic_ends = now + random.uniform(0.8, 2.0)
            self._tic      = random.choice(tuple(self.IDLE_TICS))
        if self._tic and now < self._tic_ends:
            return self.IDLE_TICS[self._tic]
        self._tic = None
        return self.IDLE_TICS["neutral"]

    def _set(self, squint, widen, droop, pupil):
        self.squint, self.widen = squint, widen
        self.droop, self.pupil_scale = droop, pupil
