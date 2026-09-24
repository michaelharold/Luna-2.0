# expressions.py
"""
expressions.py — one function per expression. This is the file you edit.

Every emotion Luna can wear is described in exactly ONE place: the function
named after it. Want a sadder sad? Open `def sad()` and change the numbers.
Want a brand new mood? Copy a function, rename it, add it to EXPRESSIONS.

This used to be spread across six separate if/elif chains inside the renderer
(eye size, arc eyes, blush, micro-expressions, mouth visibility, particles),
so changing one emotion meant finding and editing six places and hoping you
found them all. Now the renderer asks this module what to draw and never
branches on an emotion name at all.

    from expressions import get
    expr = get("happy")
    expr.eye_w, expr.blush, expr.particles   # everything about happy

Nothing here imports pygame or touches the screen — these are pure
descriptions, which makes them trivial to read, diff and unit-test.
"""

from dataclasses import dataclass, field


# ── The shape of a face ───────────────────────────────────────────────────────
@dataclass(frozen=True)
class Expression:
    """A complete description of Luna's face for one emotion.

    Grouped the way you actually think about a face: eyes, brows, mouth, then
    the extras. Every field has a sane default, so an expression only states
    what makes it different from neutral.
    """

    name: str

    # ── Eyes ──────────────────────────────────────────────────────────────
    eye_w: int = 175          # target eye width  in px
    eye_h: int = 125          # target eye height in px
    arc_eyes: bool = False    # draw '∩' smiling eyes instead of rounded rects
    squint: float = 0.0       # 0-1, lower lid rises
    widen: float = 0.0        # 0-1, eye opens beyond normal
    droop: float = 0.0        # 0-1, inner brow falls (the sad tell)
    pupil_scale: float = 1.0  # iris size multiplier; >1 reads as interested
    gaze: tuple = (0.0, 0.0)  # forced pupil offset in px (x, y); shy looks away
    red_pulse: bool = False   # angry red throb over the eye colour

    # ── Brow ──────────────────────────────────────────────────────────────
    brow: str = "flat"        # flat | angry | happy | sad | raised

    # ── Mouth ─────────────────────────────────────────────────────────────
    mouth: str = "none"       # none | smile | frown | tight | o | sleep | shy
    mouth_visible: float = 0.0  # 0-1 opacity target when not speaking
    corner_lift: float = 0.0    # 0-1, bows the talking mouth into a smile

    # ── Whole face ────────────────────────────────────────────────────────
    blush: float = 0.0        # 0-255 cheek alpha
    tilt: float = 0.0         # extra head tilt in degrees (shy tips away)
    bounce: float = 0.0       # one-shot hop amplitude on entering this state
    asym: tuple = (1.0, 1.0)  # per-eye size multiplier (left, right) — one
                              # raised eye is the whole of "curious"

    # ── Particles ─────────────────────────────────────────────────────────
    # (kind, every_n_frames). A period of 0 means "one burst on entry".
    # Kinds: zzz | tear | steam | star | heart
    particles: tuple = ()


# ══════════════════════════════════════════════════════════════════════════════
# The expressions themselves — one function each, in mood order.
# ══════════════════════════════════════════════════════════════════════════════

def neutral():
    """Resting face. Eyes only, no mouth — deliberately calm and unremarkable
    so that every other expression reads as a change."""
    return Expression(name="neutral")


def happy():
    """Warm and open. The '∩' arc eyes do most of the work here: a cheerful
    face squints its eyes shut into an arch, which a squashed rounded
    rectangle simply cannot express no matter how far it is flattened."""
    return Expression(
        name="happy",
        eye_w=205, eye_h=150,
        arc_eyes=True, squint=0.35,
        brow="happy",
        mouth="smile", mouth_visible=0.85, corner_lift=0.8,
        blush=140.0,
        bounce=22.0,
    )


def sad():
    """Small, low and slow. Narrow eyes plus a full inner-brow droop, with an
    occasional tear so the state still reads from across a room."""
    return Expression(
        name="sad",
        eye_w=158, eye_h=88,
        droop=1.0, pupil_scale=0.9,
        brow="sad",
        mouth="frown", mouth_visible=0.7,
        particles=(("tear", 90),),
    )


def angry():
    """Hard squint, angled brows, a red throb in the eyes and steam venting
    off the top of the head. The steam is what stops angry from reading as
    merely 'sleepy but cross'."""
    return Expression(
        name="angry",
        eye_w=185, eye_h=78,
        squint=0.45, red_pulse=True,
        brow="angry",
        mouth="tight", mouth_visible=0.7,
        particles=(("steam", 12),),
    )


def surprised():
    """Everything springs open — wide eyes, blown pupils, a round O mouth and
    a one-shot starburst. The burst fires once on entry rather than looping,
    because surprise is an event, not a mood you sit in."""
    return Expression(
        name="surprised",
        eye_w=240, eye_h=190,
        widen=0.85, pupil_scale=1.25,
        brow="raised",
        mouth="o", mouth_visible=0.7,
        particles=(("star", 0),),
    )


def excited():
    """Surprise plus delight: wide AND arched, bouncing, and trailing stars
    for as long as it lasts."""
    return Expression(
        name="excited",
        eye_w=230, eye_h=175,
        arc_eyes=True, widen=0.8, pupil_scale=1.3,
        brow="happy",
        mouth="smile", mouth_visible=0.85, corner_lift=0.8,
        blush=140.0,
        bounce=22.0,
        particles=(("star", 0), ("star", 25)),
    )


def love():
    """Happy with the blush pushed most of the way up and hearts drifting off
    the face. Triggered by compliments."""
    return Expression(
        name="love",
        eye_w=200, eye_h=150,
        arc_eyes=True, squint=0.35,
        brow="happy",
        mouth="smile", mouth_visible=0.85, corner_lift=0.8,
        blush=210.0,
        bounce=22.0,
        particles=(("heart", 18),),
    )


def shy():
    """Bashful — the one expression built from where she ISN'T looking.

    Heavy blush, a soft squint and a head tipped away, but the tell is `gaze`:
    the pupils sit down and off to one side, so she is avoiding your eyes
    rather than staring through you. Without that offset this is just 'happy
    with red cheeks'."""
    return Expression(
        name="shy",
        eye_w=165, eye_h=105,
        squint=0.30, droop=0.25, pupil_scale=0.95,
        gaze=(26.0, 16.0),
        brow="sad",
        mouth="shy", mouth_visible=0.55,
        blush=235.0,
        tilt=7.0,
    )


def sleeping():
    """Eyes closed to slits, a peaceful curve for a mouth, a faint blush and
    a slow stream of z's. Entering this state also arms the waking animation
    in the renderer."""
    return Expression(
        name="sleeping",
        eye_w=155, eye_h=10,
        mouth="sleep", mouth_visible=0.5,
        blush=80.0,
        particles=(("zzz", 40),),
    )


def curious():
    """Head cocked, one eye bigger than the other, pupils dilated — the
    'ooh, what's that?' face. The asymmetry does all the work: two matched
    eyes at any size just read as surprised."""
    return Expression(
        name="curious",
        eye_w=185, eye_h=140,
        asym=(1.18, 0.86), pupil_scale=1.15,
        brow="raised",
        tilt=9.0,
    )


def annoyed():
    """Flat, half-lidded stare with an angled mouth — deadpan displeasure.
    Deliberately NO red pulse or steam: annoyed is the quiet sibling of
    angry, and sharing those effects would collapse the two."""
    return Expression(
        name="annoyed",
        eye_w=182, eye_h=72,
        squint=0.55, pupil_scale=0.9,
        brow="angry",
        mouth="tight", mouth_visible=0.55,
        tilt=-3.0,
    )


def amazed():
    """Beyond surprised — everything maxed, a starburst AND a trailing star
    stream, and a bounce. What Luna wears when something is genuinely cool."""
    return Expression(
        name="amazed",
        eye_w=235, eye_h=185,
        widen=1.0, pupil_scale=1.45,
        brow="raised",
        mouth="o", mouth_visible=0.8,
        bounce=26.0,
        particles=(("star", 0), ("star", 20)),
    )


def _boop(name, left_big):
    """Tapping an eye on the touchscreen pops THAT eye — a quick cartoonish
    cross-eyed startle with a blush and a couple of stars. Two mirrored
    registrations so the reaction happens on the side that was touched."""
    big, small = 1.45, 0.72
    return Expression(
        name=name,
        eye_w=190, eye_h=150,
        asym=(big, small) if left_big else (small, big),
        widen=0.9, pupil_scale=1.3,
        brow="raised",
        blush=120.0,
        bounce=18.0,
        particles=(("star", 0),),
    )


def giggle():
    """The mouth-tap reaction: scrunched-happy with a wobble of a smile —
    being tickled, basically."""
    return Expression(
        name="giggle",
        eye_w=200, eye_h=145,
        arc_eyes=True, squint=0.4,
        brow="happy",
        mouth="smile", mouth_visible=0.9, corner_lift=0.8,
        blush=185.0,
        bounce=16.0,
        tilt=4.0,
    )


def recognised():
    """The "I know you!" moment — worn for a few seconds when Luna recognises
    a face she has met before. Happy, but brighter and bigger than happy:
    wider eyes and blown pupils (the delighted double-take), a big smile, a
    strong blush, a proper hop, and a starburst followed by drifting hearts.
    The renderer layers a left-right eye dart and a sliding name banner on
    top (see robot_face: double_take_offset / slide_params)."""
    return Expression(
        name="recognised",
        eye_w=215, eye_h=160,
        arc_eyes=True, widen=0.5, pupil_scale=1.3,
        brow="happy",
        mouth="smile", mouth_visible=0.9, corner_lift=0.9,
        blush=170.0,
        bounce=28.0,
        particles=(("star", 0), ("heart", 30)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

#: Every expression, built once at import. Add new moods here.
EXPRESSIONS = {
    expr.name: expr
    for expr in (
        neutral(), happy(), sad(), angry(), surprised(),
        excited(), love(), shy(), sleeping(),
        curious(), annoyed(), amazed(),
        _boop("boop_left", True), _boop("boop_right", False), giggle(),
        recognised(),
    )
}

#: Names in a stable, presentable order — used by the demo and by `face.list()`.
NAMES = tuple(EXPRESSIONS)

#: Moods that count as cheerful. The renderer uses this instead of repeating
#: `emotion in ("happy", "excited", "love")` in five different places.
POSITIVE = frozenset(n for n, e in EXPRESSIONS.items() if e.arc_eyes)


def get(name):
    """Look up an expression by name, falling back to neutral.

    Never raises: an unknown emotion arriving from vision or the LLM should
    leave Luna looking blank, not crash the render thread."""
    return EXPRESSIONS.get(name, EXPRESSIONS["neutral"])


def register(expr):
    """Add or replace an expression at runtime. Handy for experimenting
    without editing this file."""
    EXPRESSIONS[expr.name] = expr
    return expr
