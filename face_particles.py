# face_particles.py
"""
face_particles.py — every short-lived visual effect on Luna's face.

All particles share one tiny contract, so the renderer can hold them in a list
and never care which kind it is looking at:

    p.update()              advance one frame
    p.alive()               False once it should be culled
    p.draw(surf, dx, dy)    paint it, shifted by (dx, dy)
    p.reach()               px from (x, y) to the far edge of what it draws

The shift and the reach exist for the renderer's head tilt: particles live in
screen coordinates, but while the head is tilted most of them are painted into
the (smaller) face layer that gets rotated, and the few that have drifted off
it are painted straight onto the screen at their rotated position.

Every sprite comes from face_style's sprite cache (keyed by size, colour and a
16-step alpha bucket) instead of a fresh Surface per particle per frame.

`SPAWNERS` maps the particle names used in expressions.py ("tear", "zzz", ...)
to a factory that places one near the face, which is the only thing the
renderer needs to know to honour an expression's particle list.

Colours come from `theme` at draw time, never at import time, so particles
already in flight recolour correctly when the style is switched live.
"""

import math
import random
import pygame

from face_style import (theme, get_font, lerp, WIDTH, HEIGHT,
                        sprite, alpha_bucket, with_alpha)


def _alpha_col(surf, col, alpha):
    """Colour for drawing a translucent shape DIRECTLY onto `surf`.

    On a per-pixel-alpha surface (the face layer) the alpha is written as-is
    and composited later; on the opaque screen the colour is pre-multiplied,
    which is exactly what blending it over the black background gives."""
    if surf.get_flags() & pygame.SRCALPHA:
        return (*col, alpha)
    k = alpha / 255.0
    return (int(col[0] * k), int(col[1] * k), int(col[2] * k))


# ── Background ────────────────────────────────────────────────────────────────
class AmbientMote:
    """A slow drifting speck of light behind the face.

    Unlike every other particle here these never die — they wrap around the
    screen edges and live for the whole session. A fixed-size pool means no
    allocation per frame, which matters on the Pi.
    """

    def __init__(self):
        self._reset(anywhere=True)

    def _reset(self, anywhere=False):
        self.x     = random.uniform(0, WIDTH)
        self.y     = random.uniform(0, HEIGHT) if anywhere else HEIGHT + 10
        self.vx    = random.uniform(-0.18, 0.18)
        self.vy    = random.uniform(-0.35, -0.10)
        self.size  = random.uniform(1.5, 3.6)
        self.phase = random.uniform(0, math.tau)
        self.speed = random.uniform(0.010, 0.028)

    def update(self):
        self.x     += self.vx
        self.y     += self.vy
        self.phase += self.speed
        # Wrap rather than respawn, so the field never visibly thins out.
        if self.y < -10:
            self.y = HEIGHT + 10
            self.x = random.uniform(0, WIDTH)
        if self.x < -10:
            self.x = WIDTH + 10
        elif self.x > WIDTH + 10:
            self.x = -10

    def alive(self):
        return True

    def reach(self):
        return max(1, int(self.size)) * 3

    def draw(self, surf, tint, dx=0, dy=0):
        # Twinkle: alpha rides a slow sine so the field shimmers gently. The
        # alpha range is narrow (40-85), so it gets fine 4-unit buckets rather
        # than the usual 16 steps, keeping the shimmer smooth.
        a = int(40 + 45 * (0.5 + 0.5 * math.sin(self.phase)))
        a = (a // 4) * 4
        r = max(1, int(self.size))
        surf.blit(sprite(("mote", r, tint, a), lambda: _mote_sprite(r, tint, a)),
                  (int(self.x) - r * 3 + dx, int(self.y) - r * 3 + dy))


def _mote_sprite(r, tint, a):
    s = pygame.Surface((r * 6, r * 6), pygame.SRCALPHA)
    pygame.draw.circle(s, (*tint, a // 3), (r * 3, r * 3), r * 3)
    pygame.draw.circle(s, (*tint, a),      (r * 3, r * 3), r)
    return s


# ── Shared base ───────────────────────────────────────────────────────────────
class Particle:
    """Common motion + fade. Subclasses supply `draw` and their own constants."""

    fade  = 3.0     # alpha lost per frame
    gravity = 0.0   # added to vy per frame

    def __init__(self, x, y, alpha=255):
        self.x     = float(x)
        self.y     = float(y)
        self.vx    = 0.0
        self.vy    = 0.0
        self.alpha = float(alpha)

    def update(self):
        self.vy   += self.gravity
        self.x    += self.vx
        self.y    += self.vy
        self.alpha = max(0.0, self.alpha - self.fade)

    def alive(self):
        return self.alpha > 0

    def reach(self):
        return 40

    def draw(self, surf, dx=0, dy=0):
        raise NotImplementedError


# ── Sleeping ──────────────────────────────────────────────────────────────────
class ZzzParticle(Particle):
    """Drifting 'z' — the sleep tell."""

    fade = 3.0

    def __init__(self, x, y):
        super().__init__(x, y)
        self.size = random.randint(18, 36)
        self.vx   = random.uniform(-0.4, 0.4)
        self.vy   = random.uniform(-1.2, -0.6)

    def reach(self):
        return self.size * 2

    def draw(self, surf, dx=0, dy=0):
        size, col = self.size, theme.ZZZ_COL
        a = alpha_bucket(self.alpha)
        if a <= 0:
            return
        base = sprite(("z", size, col, 255),
                      lambda: get_font(size).render("z", True, col))
        txt = base if a >= 255 else sprite(("z", size, col, a),
                                           lambda: with_alpha(base, a))
        surf.blit(txt, (int(self.x) + dx, int(self.y) + dy))


# ── Sad ───────────────────────────────────────────────────────────────────────
class TearParticle(Particle):
    """Teardrop that accelerates as it falls."""

    fade    = 2.0
    gravity = 0.1

    def __init__(self, x, y):
        super().__init__(x, y, alpha=200)
        self.r  = random.randint(6, 10)
        self.vy = random.uniform(1.0, 2.0)
        self.vx = random.uniform(-0.2, 0.2)

    def reach(self):
        return self.r * 2 + 4

    def draw(self, surf, dx=0, dy=0):
        r, a = self.r, alpha_bucket(self.alpha)
        col = (*theme.TEAR_COL, a)

        def build():
            s = pygame.Surface((r * 2 + 4, r * 3 + 4), pygame.SRCALPHA)
            # A circle with a small triangle on top reads as a teardrop.
            pygame.draw.circle(s, col, (r + 2, r * 2 + 2), r)
            pygame.draw.polygon(s, col, [(r + 2, 2), (r - 4, r * 2 - 4),
                                         (r + 8, r * 2 - 4)])
            return s

        surf.blit(sprite(("tear", r, col), build),
                  (int(self.x) - r - 2 + dx, int(self.y) - r - 2 + dy))


# ── Angry ─────────────────────────────────────────────────────────────────────
class SteamParticle(Particle):
    """Puff of steam venting off the top of the head."""

    fade = 4.0

    def __init__(self, x, y):
        super().__init__(x, y, alpha=180)
        self.r  = random.randint(5, 12)
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-1.5, -0.8)

    def update(self):
        super().update()
        self.r = max(1.0, self.r - 0.1)      # thins as it rises

    def reach(self):
        return int(self.r) + 4

    def draw(self, surf, dx=0, dy=0):
        r = int(self.r)
        col = (*theme.STEAM_COL, alpha_bucket(self.alpha))

        def build():
            s = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(s, col, (r + 2, r + 2), r)
            return s

        surf.blit(sprite(("steam", r, col), build),
                  (int(self.x) - r - 2 + dx, int(self.y) - r - 2 + dy))


# ── Surprised / excited ───────────────────────────────────────────────────────
class StarParticle(Particle):
    """Five-pointed spark flung outward, decelerating as it fades."""

    fade = 6.0

    def __init__(self, x, y):
        super().__init__(x, y)
        self.size = random.randint(8, 18)
        angle     = random.uniform(0, math.tau)
        speed     = random.uniform(2.0, 5.0)
        self.vx   = math.cos(angle) * speed
        self.vy   = math.sin(angle) * speed

    def update(self):
        super().update()
        self.vx *= 0.92       # drag, so the burst settles instead of flying off
        self.vy *= 0.92

    def reach(self):
        return self.size * 2 + 2

    def draw(self, surf, dx=0, dy=0):
        size = self.size
        if size < 2:
            return
        col = (*theme.STAR_COL, alpha_bucket(self.alpha))

        def build():
            s = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
            cx = cy = size * 2
            for i in range(5):
                out = math.pi / 2 + i * (math.tau / 5)
                inn = out + math.pi / 5
                pygame.draw.line(s, col, (cx, cy),
                                 (int(cx + math.cos(out) * size),
                                  int(cy - math.sin(out) * size)), 2)
                pygame.draw.line(s, col, (cx, cy),
                                 (int(cx + math.cos(inn) * size * 0.4),
                                  int(cy - math.sin(inn) * size * 0.4)), 1)
            return s

        surf.blit(sprite(("star", size, col), build),
                  (int(self.x) - size * 2 + dx, int(self.y) - size * 2 + dy))


# ── Love ──────────────────────────────────────────────────────────────────────
class HeartParticle(Particle):
    """Heart that sways as it floats up. Spawned by compliments."""

    fade = 3.0

    def __init__(self, x, y):
        super().__init__(x, y, alpha=235)
        self.size = random.randint(10, 22)
        self.vx   = random.uniform(-0.6, 0.6)
        self.vy   = random.uniform(-1.8, -0.9)
        self.sway = random.uniform(0, math.tau)

    def update(self):
        self.sway += 0.1
        self.x    += self.vx + math.sin(self.sway) * 0.6
        self.y    += self.vy
        self.alpha = max(0.0, self.alpha - self.fade)

    def reach(self):
        return self.size + 6

    def draw(self, surf, dx=0, dy=0):
        r   = self.size
        col = (*theme.HEART_COL, alpha_bucket(self.alpha))

        def build():
            cr = max(2, r // 2)
            s  = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            # Two circles and a triangle make a heart.
            pygame.draw.circle(s, col, (2 + cr, 2 + cr), cr)
            pygame.draw.circle(s, col, (2 + r + cr - 1, 2 + cr), cr)
            pygame.draw.polygon(s, col, [(2, 2 + cr), (2 + r * 2, 2 + cr),
                                         (2 + r, 2 + r * 2)])
            return s

        surf.blit(sprite(("heart", r, col), build),
                  (int(self.x) - r + dx, int(self.y) - r + dy))


# ── Speaking ──────────────────────────────────────────────────────────────────
class SoundWave:
    """Ring rippling out from the mouth while Luna talks.

    Rings must START outside the mouth. They used to begin at r=20 and die by
    r=140, which sat entirely within the (now much larger) filled mouth — so
    instead of sound radiating away they looked like a stray circle drawn on
    top of her face. The lifetime is kept short for the same reason: long-lived
    rings grew wide enough to cross the eyes, which reads as clutter.
    """

    def __init__(self, x, y, start_r=20.0):
        self.x     = x
        self.y     = y
        self.r     = float(start_r)
        self.start = float(start_r)
        self.alpha = 95
        self.max_r = start_r + random.randint(48, 78)

    def update(self):
        self.r    += 2.6
        span       = max(1.0, self.max_r - self.start)
        self.alpha = max(0, int(95 * (1 - (self.r - self.start) / span)))

    def alive(self):
        return self.r < self.max_r

    def reach(self):
        return int(self.r) + 3

    def draw(self, surf, dx=0, dy=0):
        """A 2 px ring drawn straight onto the target. It used to be a fresh
        ~400x400 alpha Surface per ring per frame — mostly empty pixels — just
        to get translucency; `_alpha_col` gets the same pixels for free (the
        ring sits on the black background, behind the mouth)."""
        if self.r < 1 or self.alpha < 5:
            return
        pygame.draw.circle(surf, _alpha_col(surf, theme.WAVE_COL, self.alpha),
                           (int(self.x) + dx, int(self.y) + dy), int(self.r), 2)


# ── Blush (persistent, not a particle) ────────────────────────────────────────
class Blush:
    """A cheek patch that eases toward a target alpha instead of popping on."""

    def __init__(self, rel_x, rel_y):
        self.rel_x = rel_x
        self.rel_y = rel_y
        self.alpha = 0.0

    def update(self, target_alpha):
        self.alpha = lerp(self.alpha, target_alpha, 0.04)

    def draw(self, surf, face_cx, face_cy):
        if self.alpha < 3:
            return
        # 32 alpha steps: the blush eases in over about a second, and 16
        # visible steps on a large soft patch would read as flicker.
        a = alpha_bucket(self.alpha, steps=32)
        col = theme.BLUSH_COL
        patch = sprite(("blush", col, a), lambda: _blush_sprite(col, a))
        surf.blit(patch, (face_cx + self.rel_x - 38, face_cy + self.rel_y - 38))


def _blush_sprite(col, alpha):
    """Three stacked ellipses give a soft edge without a blur pass. Composited
    once into one sprite — same colour at every layer, so stacking them here
    and blitting once is pixel-for-pixel the same as blitting all three."""
    out = pygame.Surface((76, 76), pygame.SRCALPHA)
    for r, mult in ((38, 0.4), (28, 0.7), (18, 1.0)):
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(s, (*col, int(alpha * mult)), s.get_rect())
        out.blit(s, (38 - r, 38 - r))
    return out


# ── Spawn table ───────────────────────────────────────────────────────────────
# Maps the particle names used in expressions.py to "make one near the face".
# Each factory takes the face centre and returns a particle already positioned,
# so the renderer stays free of per-emotion placement rules.

def _spawn_zzz(fcx, fcy):
    return ZzzParticle(fcx + random.randint(-80, 80), fcy - 120)


def _spawn_tear(fcx, fcy, eye_dx=-190, eye_dy=30):
    return TearParticle(fcx + eye_dx, fcy + eye_dy)


def _spawn_steam(fcx, fcy):
    side = random.choice((-190, 190))
    return SteamParticle(fcx + side + random.randint(-20, 20), fcy - 80)


def _spawn_star(fcx, fcy):
    return StarParticle(fcx + random.randint(-260, 260),
                        fcy + random.randint(-180, 120))


def _spawn_heart(fcx, fcy):
    return HeartParticle(fcx + random.randint(-240, 240),
                         fcy + random.randint(-60, 100))


SPAWNERS = {
    "zzz":   _spawn_zzz,
    "tear":  _spawn_tear,
    "steam": _spawn_steam,
    "star":  _spawn_star,
    "heart": _spawn_heart,
}

#: How many to make when an expression asks for a one-shot burst (period 0).
BURST_COUNT = 12
