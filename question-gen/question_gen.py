#!/usr/bin/env python3
"""
Generate PAT orthographic projection questions with arcs, angled faces,
smooth shaded rendering, and hidden-line drawings.

Output: questions/qNN/input/top_view.png, end_view.png
        questions/qNN/answers/A-D.png
        questions/solutions.json
        ../analysis/analysis_NN.png  (composite PAT images, isometric rendered in-memory)
"""

import io
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import trimesh
import trimesh.creation
import trimesh.util

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
HERE         = Path(__file__).parent
OUT_DIR      = HERE / "generated_questions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Analysis composite constants ──────────────────────────────────────────────
CELL   = 240
PAD    = 12
GAP    = 6
SEP    = 18
LABEL  = 26
BORDER = 5
BG_COL      = (255, 255, 255)
FG_COL      = (30,  30,  30)
CORRECT_COL = (34, 139,  34)
SECTION_COL = (180, 180, 180)

N_QUESTIONS = 10
LIGHT_DIR = np.array([0.4, -1.0, 0.8])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
AMBIENT = 0.3

# Edge angle threshold (degrees) for crease detection
CREASE_ANGLE_DEG = 60.0


# ---------------------------------------------------------------------------
# Mesh primitives
# ---------------------------------------------------------------------------

def box(w, d, h, ox=0.0, oy=0.0, oz=0.0):
    """Box with corner at (ox, oy, oz), extents (w, d, h)."""
    m = trimesh.creation.box(extents=(w, d, h))
    m.apply_translation([ox + w/2, oy + d/2, oz + h/2])
    return m


def cylinder(r, h, ox=0.0, oy=0.0, oz=0.0, segs=32):
    """Cylinder centered at (ox, oy, oz+h/2)."""
    m = trimesh.creation.cylinder(radius=r, height=h, sections=segs)
    m.apply_translation([ox, oy, oz + h/2])
    return m


def wedge(w, d, h_front, h_back, ox=0.0, oy=0.0, oz=0.0):
    """
    Wedge: 4 bottom vertices at z=oz, 4 top vertices with sloped top.
    h_front = height at y=oy (front), h_back = height at y=oy+d (back).
    """
    # 8 vertices: corners of the wedge
    # bottom: (ox,oy,oz), (ox+w,oy,oz), (ox+w,oy+d,oz), (ox,oy+d,oz)
    # top:    (ox,oy,oz+h_front), (ox+w,oy,oz+h_front),
    #         (ox+w,oy+d,oz+h_back), (ox,oy+d,oz+h_back)
    verts = np.array([
        [ox,     oy,     oz],         # 0
        [ox+w,   oy,     oz],         # 1
        [ox+w,   oy+d,   oz],         # 2
        [ox,     oy+d,   oz],         # 3
        [ox,     oy,     oz+h_front], # 4
        [ox+w,   oy,     oz+h_front], # 5
        [ox+w,   oy+d,   oz+h_back],  # 6
        [ox,     oy+d,   oz+h_back],  # 7
    ], dtype=float)

    faces = np.array([
        # bottom
        [0, 2, 1], [0, 3, 2],
        # front (y=oy)
        [0, 1, 5], [0, 5, 4],
        # back (y=oy+d)
        [3, 6, 2], [3, 7, 6],
        # left (x=ox)
        [0, 4, 7], [0, 7, 3],
        # right (x=ox+w)
        [1, 2, 6], [1, 6, 5],
        # top (sloped)
        [4, 5, 6], [4, 6, 7],
    ], dtype=np.int64)

    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return m


def angled_prism(w, d, h, cut_angle_deg, ox=0.0, oy=0.0, oz=0.0):
    """
    Block with angled right face. The top-right edge is shifted inward by h*tan(angle).
    cut_angle_deg: angle of the right face from vertical (0=vertical, 45=45deg slope).
    """
    ang = math.radians(cut_angle_deg)
    shift = h * math.tan(ang)
    # Clamp shift so we don't get degenerate geometry
    shift = min(shift, w * 0.9)

    # 6-vertex prism: extruded trapezoid in x-z plane, depth d in y
    # Bottom: full width. Top-right is shifted left by shift.
    # vertices at y=oy and y=oy+d
    x_br = ox + w          # bottom-right x
    x_tr = ox + w - shift  # top-right x
    verts = np.array([
        [ox,    oy,     oz],      # 0 bottom-left front
        [x_br,  oy,     oz],      # 1 bottom-right front
        [x_tr,  oy,     oz+h],    # 2 top-right front
        [ox,    oy,     oz+h],    # 3 top-left front
        [ox,    oy+d,   oz],      # 4 bottom-left back
        [x_br,  oy+d,   oz],      # 5 bottom-right back
        [x_tr,  oy+d,   oz+h],    # 6 top-right back
        [ox,    oy+d,   oz+h],    # 7 top-left back
    ], dtype=float)

    faces = np.array([
        # bottom
        [0, 1, 5], [0, 5, 4],
        # top
        [3, 6, 2], [3, 7, 6],
        # front (y=oy)
        [0, 2, 1], [0, 3, 2],
        # back (y=oy+d)
        [4, 5, 6], [4, 6, 7],
        # left (x=ox)
        [0, 4, 7], [0, 7, 3],
        # right (angled face)
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)

    m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return m


def combine(*meshes):
    """Concatenate multiple meshes into one."""
    valid = [m for m in meshes if m is not None and len(m.faces) > 0]
    if not valid:
        return trimesh.creation.box(extents=(1, 1, 1))
    if len(valid) == 1:
        return valid[0]
    return trimesh.util.concatenate(valid)


# ---------------------------------------------------------------------------
# Shape generators
# ---------------------------------------------------------------------------

def gen_block_cylinder(params=None, rng=None):
    """Base box + cylinder on top."""
    if rng is None:
        rng = np.random.default_rng(0)
    if params is None:
        bw = float(rng.uniform(2.0, 4.0))
        bd = float(rng.uniform(2.0, 4.0))
        bh = float(rng.uniform(1.0, 2.0))
        r = float(rng.uniform(0.3, min(bw, bd) * 0.4))
        cx = float(rng.uniform(r, bw - r))
        cy = float(rng.uniform(r, bd - r))
        ch = float(rng.uniform(0.5, 2.0))
        params = dict(type='block_cylinder', bw=bw, bd=bd, bh=bh, r=r, cx=cx, cy=cy, ch=ch)

    bw, bd, bh = params['bw'], params['bd'], params['bh']
    r, cx, cy, ch = params['r'], params['cx'], params['cy'], params['ch']

    base = box(bw, bd, bh, 0, 0, 0)
    cyl = cylinder(r, ch, cx, cy, bh)
    return combine(base, cyl), params


def gen_wedge_step(params=None, rng=None):
    """Box + wedge side by side (step profile)."""
    if rng is None:
        rng = np.random.default_rng(0)
    if params is None:
        bw = float(rng.uniform(1.5, 3.0))
        bd = float(rng.uniform(2.0, 4.0))
        h1 = float(rng.uniform(1.0, 2.0))
        sw = float(rng.uniform(1.0, 2.5))
        h2 = float(rng.uniform(0.5, h1))
        h_back = float(rng.uniform(h2 * 0.3, h2))
        params = dict(type='wedge_step', bw=bw, bd=bd, h1=h1, sw=sw, h2=h2, h_back=h_back)

    bw, bd, h1 = params['bw'], params['bd'], params['h1']
    sw, h2, h_back = params['sw'], params['h2'], params['h_back']

    b = box(bw, bd, h1, 0, 0, 0)
    w = wedge(sw, bd, h2, h_back, bw, 0, 0)
    return combine(b, w), params


def gen_angled_step(params=None, rng=None):
    """Box + angled_prism side by side."""
    if rng is None:
        rng = np.random.default_rng(0)
    if params is None:
        bw = float(rng.uniform(1.5, 3.0))
        bd = float(rng.uniform(2.0, 4.0))
        h1 = float(rng.uniform(1.0, 2.5))
        sw = float(rng.uniform(1.0, 2.5))
        h2 = float(rng.uniform(0.5, h1))
        ang = float(rng.uniform(10, 40))
        params = dict(type='angled_step', bw=bw, bd=bd, h1=h1, sw=sw, h2=h2, ang=ang)

    bw, bd, h1 = params['bw'], params['bd'], params['h1']
    sw, h2, ang = params['sw'], params['h2'], params['ang']

    b = box(bw, bd, h1, 0, 0, 0)
    ap = angled_prism(sw, bd, h2, ang, bw, 0, 0)
    return combine(b, ap), params


def gen_l_cylinder(params=None, rng=None):
    """Wide base box + narrow upper box + cylinder on top (L-shape)."""
    if rng is None:
        rng = np.random.default_rng(0)
    if params is None:
        bw = float(rng.uniform(3.0, 5.0))
        bd = float(rng.uniform(2.0, 4.0))
        bh = float(rng.uniform(0.8, 1.5))
        lw = float(rng.uniform(1.0, bw * 0.6))
        lh = float(rng.uniform(0.8, 2.0))
        r = float(rng.uniform(0.3, min(lw, bd) * 0.35))
        cx = float(rng.uniform(r, lw - r))
        cy = float(rng.uniform(r, bd - r))
        ch = float(rng.uniform(0.5, 1.5))
        params = dict(type='l_cylinder', bw=bw, bd=bd, bh=bh, lw=lw, lh=lh,
                      r=r, cx=cx, cy=cy, ch=ch)

    bw, bd, bh = params['bw'], params['bd'], params['bh']
    lw, lh = params['lw'], params['lh']
    r, cx, cy, ch = params['r'], params['cx'], params['cy'], params['ch']

    base = box(bw, bd, bh, 0, 0, 0)
    upper = box(lw, bd, lh, 0, 0, bh)
    cyl = cylinder(r, ch, cx, cy, bh + lh)
    return combine(base, upper, cyl), params


def gen_multi_wedge(params=None, rng=None):
    """Box + wedge + angled_prism in sequence."""
    if rng is None:
        rng = np.random.default_rng(0)
    if params is None:
        bd = float(rng.uniform(2.0, 4.0))
        w1 = float(rng.uniform(1.5, 3.0))
        w2 = float(rng.uniform(1.0, 2.5))
        w3 = float(rng.uniform(1.0, 2.0))
        h1 = float(rng.uniform(1.5, 3.0))
        h2 = float(rng.uniform(0.8, h1))
        h3 = float(rng.uniform(0.4, h2))
        h_back = float(rng.uniform(h2 * 0.3, h2))
        ang = float(rng.uniform(15, 40))
        params = dict(type='multi_wedge', bd=bd, w1=w1, w2=w2, w3=w3,
                      h1=h1, h2=h2, h3=h3, h_back=h_back, ang=ang)

    bd = params['bd']
    w1, w2, w3 = params['w1'], params['w2'], params['w3']
    h1, h2, h3 = params['h1'], params['h2'], params['h3']
    h_back, ang = params['h_back'], params['ang']

    b = box(w1, bd, h1, 0, 0, 0)
    wg = wedge(w2, bd, h2, h_back, w1, 0, 0)
    ap = angled_prism(w3, bd, h3, ang, w1 + w2, 0, 0)
    return combine(b, wg, ap), params


SHAPE_GENERATORS = [
    gen_block_cylinder,
    gen_wedge_step,
    gen_angled_step,
    gen_l_cylinder,
    gen_multi_wedge,
]


def rebuild_mesh(params):
    """Rebuild mesh from params dict."""
    t = params['type']
    if t == 'block_cylinder':
        return gen_block_cylinder(params)[0]
    elif t == 'wedge_step':
        return gen_wedge_step(params)[0]
    elif t == 'angled_step':
        return gen_angled_step(params)[0]
    elif t == 'l_cylinder':
        return gen_l_cylinder(params)[0]
    elif t == 'multi_wedge':
        return gen_multi_wedge(params)[0]
    else:
        raise ValueError(f"Unknown type: {t}")


# ---------------------------------------------------------------------------
# Distractor generation
# ---------------------------------------------------------------------------

def _drastic_scale(rng):
    """Return a scale factor that is clearly noticeable: either 0.3-0.5 or 1.8-2.5."""
    if int(rng.integers(0, 2)) == 0:
        return float(rng.uniform(0.3, 0.5))
    else:
        return float(rng.uniform(1.8, 2.5))


def _mirror_mesh(mesh):
    """Mirror a mesh about its own X centre."""
    verts = mesh.vertices.copy()
    cx = (verts[:, 0].max() + verts[:, 0].min()) / 2
    verts[:, 0] = 2 * cx - verts[:, 0]
    m = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=True)
    m.invert()
    return m


def make_distractor(params, seed, attempt):
    """
    Return (mesh, tag_string) or (None, None).

    Operations are grouped into three families so every question sees a mix:
      A — structural: remove / add / swap a whole component
      B — drastic param: change a dimension by ≥50 % so the difference is obvious
      C — arrangement: mirror, reverse slope/angle direction, swap two heights
    """
    rng = np.random.default_rng(seed * 1000 + attempt)
    t = params['type']
    p = params.copy()

    # ── per-type operation lists ──────────────────────────────────────────────
    ops_map = {
        'block_cylinder': [
            # structural
            'remove_cylinder',
            'replace_cyl_with_box',
            'double_cylinder',
            'add_top_shelf',
            # drastic param
            'drastic_bh',
            'drastic_ch',
            'drastic_r',
            # arrangement
            'shift_cx_far',
            'mirror_x',
        ],
        'wedge_step': [
            # structural
            'remove_step',         # drop wedge entirely → plain box
            'wedge_to_box',
            'add_top_notch',
            'add_top_shelf',
            'flatten_wedge',
            # drastic param
            'drastic_h1',
            'drastic_h2',
            'drastic_sw',
            # arrangement
            'reverse_slope',
            'mirror_x',
        ],
        'angled_step': [
            # structural
            'remove_step',         # drop angled_prism → plain box
            'add_top_notch',
            'add_top_shelf',
            'make_vertical',
            # drastic param
            'drastic_ang',
            'drastic_h1',
            'drastic_h2',
            'drastic_sw',
            # arrangement
            'mirror_x',
        ],
        'l_cylinder': [
            # structural
            'remove_cylinder',
            'replace_cyl_with_box',
            'add_second_level',
            'remove_upper_tier',   # drop upper box + cylinder → just base
            # drastic param
            'drastic_lw',
            'drastic_lh',
            'drastic_ch',
            # arrangement
            'shift_cx_far',
            'mirror_x',
        ],
        'multi_wedge': [
            # structural
            'remove_right',        # drop angled_prism → box + wedge only
            'remove_middle',       # drop wedge → box + angled_prism (gap filled)
            'add_top_notch',
            'flatten_middle',
            # drastic param
            'drastic_h1',
            'drastic_h2',
            'drastic_h3',
            'drastic_ang',
            # arrangement
            'swap_h1_h3',
            'mirror_x',
        ],
    }

    ops = ops_map.get(t, ['mirror_x'])
    op = ops[int(rng.integers(0, len(ops)))]

    custom_mesh = None   # built directly (structural ops)
    do_mirror   = False  # applied as post-step

    try:
        # ── A: structural ────────────────────────────────────────────────────

        if op == 'remove_step':
            # Drop the step (wedge or angled_prism) entirely → just the base box
            if t in ('wedge_step', 'angled_step'):
                custom_mesh = box(p['bw'], p['bd'], p['h1'])

        elif op == 'remove_upper_tier' and t == 'l_cylinder':
            # Drop upper box + cylinder → just the wide base box
            custom_mesh = box(p['bw'], p['bd'], p['bh'])

        elif op == 'remove_right' and t == 'multi_wedge':
            # Drop the angled_prism (rightmost section) → box + wedge only
            custom_mesh = combine(
                box(p['w1'], p['bd'], p['h1']),
                wedge(p['w2'], p['bd'], p['h2'], p['h_back'], p['w1'], 0, 0),
            )

        elif op == 'remove_middle' and t == 'multi_wedge':
            # Drop the wedge (middle section), slide angled_prism up to meet box
            custom_mesh = combine(
                box(p['w1'], p['bd'], p['h1']),
                angled_prism(p['w3'], p['bd'], p['h3'], p['ang'], p['w1'], 0, 0),
            )

        elif op == 'remove_cylinder':
            # Drop the cylinder entirely — all its crease/hidden lines vanish
            if t == 'block_cylinder':
                custom_mesh = box(p['bw'], p['bd'], p['bh'])
            elif t == 'l_cylinder':
                custom_mesh = combine(
                    box(p['bw'], p['bd'], p['bh']),
                    box(p['lw'], p['bd'], p['lh'], oz=p['bh']),
                )

        elif op == 'replace_cyl_with_box':
            # Cylinder → same-footprint square prism: same silhouette but no circular arcs
            r = p['r']
            bx = box(r * 2, r * 2, p['ch'], p['cx'] - r, p['cy'] - r,
                     p['bh'] if t == 'block_cylinder' else p['bh'] + p['lh'])
            if t == 'block_cylinder':
                custom_mesh = combine(box(p['bw'], p['bd'], p['bh']), bx)
            elif t == 'l_cylinder':
                custom_mesh = combine(
                    box(p['bw'], p['bd'], p['bh']),
                    box(p['lw'], p['bd'], p['lh'], oz=p['bh']),
                    bx,
                )

        elif op == 'double_cylinder' and t == 'block_cylinder':
            # Add a second cylinder on the opposite side — extra hidden circle appears
            cx2 = p['bw'] - p['cx']
            cx2 = float(np.clip(cx2, p['r'] + 0.05, p['bw'] - p['r'] - 0.05))
            cyl2 = cylinder(p['r'] * float(rng.uniform(0.7, 1.1)),
                            p['ch'] * float(rng.uniform(0.6, 1.0)),
                            cx2, p['cy'], p['bh'])
            base_mesh, _ = gen_block_cylinder(p, rng=rng)
            custom_mesh = combine(base_mesh, cyl2)

        elif op == 'add_top_shelf':
            # Thin extra slab on top of the tallest part — adds a new visible crease line
            base_mesh = rebuild_mesh(p)
            # Find correct top_z for each shape type
            if t == 'block_cylinder':
                top_z = p['bh'] + p['ch']
            elif t == 'l_cylinder':
                top_z = p['bh'] + p['lh'] + p['ch']
            elif t in ('wedge_step', 'angled_step'):
                top_z = p['h1']
            elif t == 'multi_wedge':
                top_z = p['h1']
            else:
                top_z = p.get('h1', 1.5)
            shelf_h = float(rng.uniform(0.25, 0.5))
            ref_w = p.get('bw', p.get('w1', 3.0))
            shelf_w = ref_w * float(rng.uniform(0.45, 0.8))
            shelf_ox = (ref_w - shelf_w) * float(rng.uniform(0.0, 1.0))
            shelf = box(shelf_w, p.get('bd', 2.5), shelf_h, shelf_ox, 0, top_z)
            custom_mesh = combine(base_mesh, shelf)

        elif op == 'flatten_wedge' and t == 'wedge_step':
            # Make the ramp horizontal (h_back = h2): the slope line disappears
            p['h_back'] = p['h2']

        elif op == 'wedge_to_box' and t == 'wedge_step':
            # Replace wedge with a simple box of the same max height: slope becomes vertical step
            custom_mesh = combine(
                box(p['bw'], p['bd'], p['h1']),
                box(p['sw'], p['bd'], p['h2'], p['bw'], 0, 0),
            )

        elif op == 'add_top_notch':
            # Add a box notch on top of the tallest part — extra horizontal crease line appears
            base_mesh = rebuild_mesh(p)
            top_z = {'wedge_step': p['h1'], 'angled_step': p['h1'],
                     'l_cylinder': p['bh'] + p['lh'], 'multi_wedge': p['h1']}.get(t, 1.5)
            notch_d = p.get('bd', 2.5) * float(rng.uniform(0.3, 0.55))
            notch_w = p.get('bw', p.get('w1', 2.0)) * float(rng.uniform(0.25, 0.45))
            notch_h = float(rng.uniform(0.3, 0.65))
            notch_ox = float(rng.uniform(0.1, max(0.11, p.get('bw', p.get('w1', 2.0)) - notch_w - 0.1)))
            custom_mesh = combine(base_mesh,
                                  box(notch_w, notch_d, notch_h, notch_ox, 0, top_z))

        elif op == 'make_vertical' and t == 'angled_step':
            # Collapse the angle to nearly 0: the sloped line becomes a vertical step
            p['ang'] = float(rng.uniform(1.0, 4.0))

        elif op == 'add_second_level' and t == 'l_cylinder':
            # Extra narrow tier on top of the upper block: new horizontal line appears
            base_mesh = rebuild_mesh(p)
            lv_w = p['lw'] * float(rng.uniform(0.35, 0.6))
            lv_h = float(rng.uniform(0.4, 0.9))
            custom_mesh = combine(base_mesh, box(lv_w, p['bd'], lv_h, oz=p['bh'] + p['lh']))

        elif op == 'flatten_middle' and t == 'multi_wedge':
            # Collapse the middle wedge to the height of the left box: ramp line disappears
            p['h2']     = p['h1']
            p['h_back'] = p['h1']

        # ── B: drastic param ─────────────────────────────────────────────────

        elif op == 'drastic_bh':
            key = 'bh' if 'bh' in p else 'h1'
            p[key] = max(0.3, p[key] * _drastic_scale(rng))

        elif op == 'drastic_ch' and 'ch' in p:
            p['ch'] = max(0.2, p['ch'] * _drastic_scale(rng))

        elif op == 'drastic_r' and 'r' in p:
            p['r'] = max(0.1, p['r'] * _drastic_scale(rng))

        elif op == 'drastic_h1' and 'h1' in p:
            p['h1'] = max(0.4, p['h1'] * _drastic_scale(rng))

        elif op == 'drastic_h2' and 'h2' in p:
            p['h2'] = max(0.2, p['h2'] * _drastic_scale(rng))

        elif op == 'drastic_h3' and 'h3' in p:
            p['h3'] = max(0.15, p['h3'] * _drastic_scale(rng))

        elif op == 'drastic_sw' and 'sw' in p:
            p['sw'] = max(0.4, p['sw'] * _drastic_scale(rng))

        elif op == 'drastic_lw' and 'lw' in p:
            p['lw'] = max(0.4, p['lw'] * _drastic_scale(rng))

        elif op == 'drastic_lh' and 'lh' in p:
            p['lh'] = max(0.3, p['lh'] * _drastic_scale(rng))

        elif op == 'drastic_ang' and 'ang' in p:
            delta = float(rng.uniform(25, 40)) * (1 if int(rng.integers(0, 2)) == 0 else -1)
            p['ang'] = float(np.clip(p['ang'] + delta, 3, 62))

        # ── C: arrangement ───────────────────────────────────────────────────

        elif op == 'shift_cx_far':
            # Move cylinder from one side of the base to the other
            r   = p['r']
            avail = p['bw'] - 2 * r - 0.1
            if p['cx'] > avail / 2 + r:
                p['cx'] = r + avail * float(rng.uniform(0.05, 0.2))
            else:
                p['cx'] = r + avail * float(rng.uniform(0.8, 0.95))

        elif op == 'reverse_slope' and t == 'wedge_step':
            # Swap front and back heights of the wedge
            p['h2'], p['h_back'] = p['h_back'], p['h2']

        elif op == 'swap_h1_h3' and t == 'multi_wedge':
            # Tall left becomes short, short right becomes tall
            p['h1'], p['h3'] = p['h3'], p['h1']

        elif op == 'mirror_x':
            do_mirror = True

        else:
            # Fallback if operation key didn't match (e.g. wrong type)
            do_mirror = True

        # ── build final mesh ─────────────────────────────────────────────────
        if custom_mesh is None:
            mesh = rebuild_mesh(p)
        else:
            mesh = custom_mesh

        if do_mirror:
            mesh = _mirror_mesh(mesh)

        if mesh is None or len(mesh.faces) == 0:
            return None, None

        return mesh, op   # return op string as tag (not used for generation)

    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Voxelization helpers
# ---------------------------------------------------------------------------

def safe_voxelize(mesh, pitch=0.25):
    """Voxelize mesh, return matrix or None on failure."""
    try:
        vox = mesh.voxelized(pitch)
        return vox.matrix
    except Exception:
        return None


def silhouette_from_vox(g, axis):
    """Project voxel grid along axis, return 2D bool array."""
    return np.any(g, axis=axis)


def silhouettes_differ(m1, m2, axis, pitch=0.25):
    """Return True if the two meshes have different silhouettes along `axis`."""
    g1 = safe_voxelize(m1, pitch)
    g2 = safe_voxelize(m2, pitch)
    if g1 is None or g2 is None:
        return True  # assume different if can't voxelize

    s1 = silhouette_from_vox(g1, axis)
    s2 = silhouette_from_vox(g2, axis)

    # Pad to same shape
    sh = tuple(max(a, b) for a, b in zip(s1.shape, s2.shape))
    p1 = np.zeros(sh, dtype=bool)
    p2 = np.zeros(sh, dtype=bool)
    p1[:s1.shape[0], :s1.shape[1]] = s1
    p2[:s2.shape[0], :s2.shape[1]] = s2
    return not np.array_equal(p1, p2)


def front_view_edge_counts(mesh):
    """Return (n_crease, n_hidden) edge counts in the front view."""
    try:
        vu = build_visible_union(mesh, 'front')
        _, crease, hidden = classify_edges(mesh, 'front', vu)
        return (len(crease), len(hidden))
    except Exception:
        return (0, 0)


def front_views_differ(m1, m2, pitch=0.25):
    """
    True if the two meshes look visually different from the front.
    Checks outer silhouette first, then falls back to internal edge structure
    so that 'dropped line' distractors (same outline, fewer crease/hidden lines)
    are accepted rather than silently rejected.
    """
    if silhouettes_differ(m1, m2, axis=1, pitch=pitch):
        return True
    # Same outer silhouette — check internal line structure
    ec1 = front_view_edge_counts(m1)
    ec2 = front_view_edge_counts(m2)
    return ec1 != ec2


def fv_is_consistent(fv_mesh, ref_mesh, pitch=0.25):
    """
    Check if fv_mesh's front view is consistent with ref_mesh's top and end views.
    Returns True if consistent (bad distractor), False if inconsistent (good distractor).
    """
    try:
        ref_g = safe_voxelize(ref_mesh, pitch)
        fv_g = safe_voxelize(fv_mesh, pitch)
        if ref_g is None or fv_g is None:
            return False

        # ref projections
        # Top view: project along Z (axis=2), result shape (nx, ny)
        tv_ref = np.any(ref_g, axis=2)      # (nx, ny)
        # End view: project along X (axis=0), result shape (ny, nz)
        ev_ref = np.any(ref_g, axis=0)      # (ny, nz) → flip to get (nz, ny)
        ev_ref = ev_ref.T[::-1, :]          # (nz, ny)
        # Front view: project along Y (axis=1), result shape (nx, nz)
        fv_ref = np.any(ref_g, axis=1)      # (nx, nz) → flip
        fv_ref = fv_ref.T[::-1, :]          # (nz, nx)

        # fv_mesh front view
        fv_fv = np.any(fv_g, axis=1)        # (nx, nz)
        fv_fv = fv_fv.T[::-1, :]            # (nz, nx)

        # Pad all to same shapes
        def pad2d(a, shape):
            out = np.zeros(shape, dtype=bool)
            out[:min(a.shape[0], shape[0]), :min(a.shape[1], shape[1])] = \
                a[:min(a.shape[0], shape[0]), :min(a.shape[1], shape[1])]
            return out

        # fv shape: (nz, nx) — need to match fv_ref and fv_fv
        nz = max(fv_ref.shape[0], fv_fv.shape[0])
        nx = max(fv_ref.shape[1], fv_fv.shape[1])
        ny = max(tv_ref.shape[1], ev_ref.shape[1])

        # Use fv from distractor mesh
        fv = pad2d(fv_fv, (nz, nx))
        tv = pad2d(tv_ref, (nx, ny))   # (nx, ny)
        ev = pad2d(ev_ref, (nz, ny))   # (nz, ny)

        # Build maximal solid: maximal[x,y,z] = fv[nz-1-z,x] and tv[x,y] and ev[nz-1-z,y]
        # Note: array indexing is [z_idx, x_idx] for fv, etc.
        maximal = np.zeros((nx, ny, nz), dtype=bool)
        for xi in range(nx):
            for yi in range(ny):
                for zi in range(nz):
                    fv_val = fv[nz - 1 - zi, xi] if (nz - 1 - zi) < fv.shape[0] and xi < fv.shape[1] else False
                    tv_val = tv[xi, yi] if xi < tv.shape[0] and yi < tv.shape[1] else False
                    ev_val = ev[nz - 1 - zi, yi] if (nz - 1 - zi) < ev.shape[0] and yi < ev.shape[1] else False
                    maximal[xi, yi, zi] = fv_val and tv_val and ev_val

        # Check if maximal solid's projections match
        max_fv = np.any(maximal, axis=1).T[::-1, :]   # (nz, nx)
        max_tv = np.any(maximal, axis=2)               # (nx, ny)
        max_ev = np.any(maximal, axis=0).T[::-1, :]   # (nz, ny)

        fv_match = np.array_equal(max_fv, fv)
        tv_match = np.array_equal(pad2d(max_tv, tv.shape), tv)
        ev_match = np.array_equal(pad2d(max_ev, ev.shape), ev)

        return fv_match and tv_match and ev_match

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def get_face_normals(mesh):
    """Return per-face normals (Nx3)."""
    return mesh.face_normals


def project_face_2d(face_verts, view):
    """
    Project 3D face vertices to 2D based on view direction.
    view: 'front' -> (x,z), 'top' -> (x,y), 'end' -> (y,z)
    Returns (N,2) array.
    """
    if view == 'front':
        return face_verts[:, [0, 2]]
    elif view == 'top':
        return face_verts[:, [0, 1]]
    elif view == 'end':
        return face_verts[:, [1, 2]]


def face_depth(face_verts, view):
    """Depth value for painter's algorithm (higher = farther, draw first)."""
    if view == 'front':
        # viewer at -Y, farther = higher Y
        return np.mean(face_verts[:, 1])
    elif view == 'top':
        # viewer at +Z (above), farther = lower Z
        return -np.mean(face_verts[:, 2])
    elif view == 'end':
        # viewer at +X (right), farther = smaller X
        return -np.mean(face_verts[:, 0])


def is_visible_face(normal, view):
    """True if face is facing the viewer."""
    if view == 'front':
        return normal[1] < -0.01
    elif view == 'top':
        return normal[2] > 0.01
    elif view == 'end':
        return normal[0] > 0.01


def lambertian_color(normal, view):
    """Return RGBA face color with Lambertian shading."""
    # Light direction toward light source
    brightness = AMBIENT + (1 - AMBIENT) * max(0, np.dot(normal, LIGHT_DIR))
    brightness = min(1.0, brightness)
    c = brightness
    return (c, c, c, 1.0)


def poly_to_shapely(pts_2d):
    """Convert 2D point array to shapely Polygon, or None if degenerate."""
    try:
        if len(pts_2d) < 3:
            return None
        poly = Polygon(pts_2d)
        if poly.is_valid and poly.area > 1e-10:
            return poly
        # Try buffer(0) to fix
        poly = poly.buffer(0)
        if poly.is_valid and poly.area > 1e-10:
            return poly
        return None
    except Exception:
        return None


def build_visible_union(mesh, view):
    """Build shapely union of all visible face triangles projected to 2D."""
    polys = []
    for i, face in enumerate(mesh.faces):
        normal = mesh.face_normals[i]
        if not is_visible_face(normal, view):
            continue
        verts = mesh.vertices[face]
        pts = project_face_2d(verts, view)
        p = poly_to_shapely(pts)
        if p is not None:
            polys.append(p)

    if not polys:
        return None

    try:
        union = unary_union(polys)
        return union
    except Exception:
        return None


def get_edge_info(mesh):
    """
    Return dict: edge (i,j) (i<j) -> list of face indices adjacent to it.
    """
    edge_faces = {}
    for fi, face in enumerate(mesh.faces):
        for k in range(3):
            a = int(face[k])
            b = int(face[(k+1) % 3])
            key = (min(a, b), max(a, b))
            edge_faces.setdefault(key, []).append(fi)
    return edge_faces


def classify_edges(mesh, view, visible_union):
    """
    Classify edges and return lists: silhouette_edges, crease_edges, hidden_edges.
    Each edge is ((v0_3d, v1_3d), (p0_2d, p1_2d)).
    """
    edge_faces = get_edge_info(mesh)
    face_normals = mesh.face_normals

    silhouette_edges = []
    crease_edges = []
    hidden_edges = []

    for (va_idx, vb_idx), face_list in edge_faces.items():
        va_3d = mesh.vertices[va_idx]
        vb_3d = mesh.vertices[vb_idx]

        # Classify each adjacent face
        face_visibility = [is_visible_face(face_normals[fi], view) for fi in face_list]
        n_visible = sum(face_visibility)
        n_faces = len(face_list)

        pa = project_face_2d(va_3d.reshape(1, 3), view)[0]
        pb = project_face_2d(vb_3d.reshape(1, 3), view)[0]

        # Skip degenerate (zero length in 2D)
        if np.allclose(pa, pb, atol=1e-6):
            continue

        # Skip coplanar edges: triangulation diagonals inside a flat rectangle.
        # These are never real structural edges and must not be drawn.
        if len(face_list) >= 2:
            n0 = face_normals[face_list[0]]
            if all(abs(float(np.dot(face_normals[fi], n0))) > 0.9999
                   for fi in face_list[1:]):
                continue

        if n_faces == 1:
            # Boundary edge -> silhouette
            if face_visibility[0]:
                silhouette_edges.append((pa, pb))
            else:
                # Hidden boundary
                pass
        elif n_visible == n_faces:
            # All visible: check angle
            n0 = face_normals[face_list[0]]
            n1 = face_normals[face_list[1]] if len(face_list) > 1 else n0
            cos_angle = np.clip(np.dot(n0, n1), -1, 1)
            angle_deg = math.degrees(math.acos(cos_angle))
            if angle_deg > CREASE_ANGLE_DEG:
                crease_edges.append((pa, pb))
            # else smooth, don't draw
        elif n_visible == 0:
            # All hidden: check if midpoint inside visible union
            mid = (pa + pb) / 2
            if visible_union is not None:
                try:
                    pt = Point(mid)
                    if visible_union.contains(pt) or visible_union.boundary.distance(pt) < 1e-4:
                        hidden_edges.append((pa, pb))
                except Exception:
                    pass
        else:
            # Mixed: silhouette boundary between visible and hidden
            silhouette_edges.append((pa, pb))

    return silhouette_edges, crease_edges, hidden_edges


def get_silhouette_polygon(mesh, view):
    """Get outer silhouette boundary polygon from visible union."""
    union = build_visible_union(mesh, view)
    if union is None:
        return None, union
    return union, union


def render_view(mesh, view, ax, title=None):
    """
    Render one orthographic view onto matplotlib axes.
    """
    # Collect face data
    faces_data = []
    for i, face in enumerate(mesh.faces):
        normal = mesh.face_normals[i]
        if not is_visible_face(normal, view):
            continue
        verts = mesh.vertices[face]
        pts2d = project_face_2d(verts, view)
        depth = face_depth(verts, view)
        color = lambertian_color(normal, view)
        faces_data.append((depth, pts2d, color))

    # Build visible union for hidden edge check
    visible_union = build_visible_union(mesh, view)

    # Fill background silhouette
    if visible_union is not None:
        try:
            def draw_shapely_poly(geom, ax):
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    pts = list(zip(x, y))
                    patch = mpatches.Polygon(pts, closed=True,
                                             facecolor='#d8d8d8', edgecolor='none', zorder=0)
                    ax.add_patch(patch)
                elif geom.geom_type in ('MultiPolygon', 'GeometryCollection'):
                    for g in geom.geoms:
                        if hasattr(g, 'exterior'):
                            draw_shapely_poly(g, ax)

            draw_shapely_poly(visible_union, ax)
        except Exception:
            pass

    # Painter's algorithm: sort by depth (farthest first)
    faces_data.sort(key=lambda x: x[0], reverse=True)

    for depth, pts2d, color in faces_data:
        if len(pts2d) < 3:
            continue
        try:
            patch = mpatches.Polygon(pts2d, closed=True,
                                     facecolor=color, edgecolor='none', zorder=1)
            ax.add_patch(patch)
        except Exception:
            pass

    # Classify and draw edges
    silhouette_edges, crease_edges, hidden_edges = classify_edges(mesh, view, visible_union)

    for pa, pb in hidden_edges:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color='black', linewidth=0.7, linestyle=(0, (4, 3)), zorder=3)

    for pa, pb in crease_edges:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color='black', linewidth=1.0, linestyle='-', zorder=4)

    for pa, pb in silhouette_edges:
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]],
                color='black', linewidth=2.0, linestyle='-', zorder=5)

    # Auto-scale axes
    all_verts = mesh.vertices
    if view == 'front':
        xs = all_verts[:, 0]
        ys = all_verts[:, 2]
    elif view == 'top':
        xs = all_verts[:, 0]
        ys = all_verts[:, 1]
    else:  # end
        xs = all_verts[:, 1]
        ys = all_verts[:, 2]

    margin = 0.3
    ax.set_xlim(xs.min() - margin, xs.max() + margin)
    ax.set_ylim(ys.min() - margin, ys.max() + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9)


def render_mesh_to_file(mesh, view, filepath, fig_size=4):
    """Render one view of a mesh and save to PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
    fig.patch.set_facecolor('white')
    render_view(mesh, view, ax)
    fig.savefig(filepath, dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


def render_isometric_to_file(mesh, filepath, fig_size=5):
    """
    Render a smooth-shaded isometric view of the mesh and save to PNG.
    Uses matplotlib's 3D axes with elev=35.264°, azim=45° (true isometric angles).
    Faces are drawn without edges; only structural (non-coplanar) edges are overlaid,
    so triangulation diagonals on flat faces are never shown.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    fig = plt.figure(figsize=(fig_size, fig_size))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # ── Shaded faces (no per-triangle edges) ─────────────────────────────────
    verts_list = []
    face_colors = []
    for i, face in enumerate(mesh.faces):
        verts = mesh.vertices[face]
        normal = mesh.face_normals[i]
        brightness = AMBIENT + (1 - AMBIENT) * max(0.0, float(np.dot(normal, LIGHT_DIR)))
        brightness = min(1.0, brightness)
        verts_list.append(verts)
        face_colors.append((brightness, brightness, brightness, 1.0))

    poly = Poly3DCollection(verts_list, facecolors=face_colors,
                            edgecolors='none', linewidths=0)
    ax.add_collection3d(poly)

    # ── Structural edges only (skip coplanar / triangulation diagonals) ───────
    import math as _math
    iso_view_dir = np.array([1.0, 1.0, 1.0]) / _math.sqrt(3)
    edge_faces = get_edge_info(mesh)
    face_normals = mesh.face_normals
    structural_edges = []

    for (va_idx, vb_idx), face_list in edge_faces.items():
        # Skip coplanar edges (triangulation diagonals inside a flat face)
        if len(face_list) >= 2:
            n0 = face_normals[face_list[0]]
            if all(abs(float(np.dot(face_normals[fi], n0))) > 0.9999
                   for fi in face_list[1:]):
                continue
        # Skip edges where all adjacent faces point away from the isometric viewer
        any_visible = any(float(np.dot(face_normals[fi], iso_view_dir)) > 0.01
                          for fi in face_list)
        if not any_visible:
            continue
        va = mesh.vertices[va_idx]
        vb = mesh.vertices[vb_idx]
        structural_edges.append([va, vb])

    if structural_edges:
        lc = Line3DCollection(structural_edges, colors='black', linewidths=0.8)
        ax.add_collection3d(lc)

    # ── Axis limits and view ──────────────────────────────────────────────────
    v = mesh.vertices
    cx, cy, cz = v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean()
    half = max(v[:, 0].ptp(), v[:, 1].ptp(), v[:, 2].ptp()) * 0.55
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

    ax.view_init(elev=35.264, azim=45)
    ax.set_proj_type('ortho')
    ax.set_axis_off()
    fig.savefig(filepath, dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

def generate_question(q_id, seed):
    """
    Generate one question: answer mesh + 3 distractors.
    Returns dict or None if failed.
    """
    rng_np = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    # Pick shape type
    gen_idx = int(rng_np.integers(0, len(SHAPE_GENERATORS)))
    gen_fn = SHAPE_GENERATORS[gen_idx]

    try:
        answer_mesh, params = gen_fn(rng=rng_np)
        if answer_mesh is None or len(answer_mesh.faces) == 0:
            return None
    except Exception as e:
        print(f"  [WARN] gen failed for {q_id}: {e}")
        return None

    # Collect 3 distractors
    distractors = []  # list of meshes

    accepted_ops = []
    drastic_count = 0  # at most 2 drastic ops among the 3 distractors (ensures ≥1 structural)
    for attempt in range(200):
        d_mesh, d_tag = make_distractor(params, seed, attempt)
        if d_mesh is None or len(d_mesh.faces) == 0:
            continue

        # (a) Distractor front view must look different from answer
        #     (checks outer silhouette AND internal crease/hidden lines)
        if not front_views_differ(d_mesh, answer_mesh):
            continue

        # (b) Distractor must look different from existing distractors
        differs_from_all = True
        for prev in distractors:
            if not front_views_differ(d_mesh, prev):
                differs_from_all = False
                break
        if not differs_from_all:
            continue

        # (c) Distractor must NOT be consistent with answer's TV+EV —
        #     if it is, it's a valid alternative answer and the question is ambiguous.
        #     This applies to both silhouette-changing and same-silhouette distractors.
        if fv_is_consistent(d_mesh, answer_mesh, pitch=0.25):
            continue

        # (d) Diversity: at most 2 drastic-param ops among 3 distractors
        is_drastic = (d_tag is not None and d_tag.startswith('drastic_'))
        if is_drastic and drastic_count >= 2:
            continue

        if is_drastic:
            drastic_count += 1
        accepted_ops.append(d_tag)
        distractors.append(d_mesh)

        if len(distractors) == 3:
            break

    if len(distractors) < 3:
        print(f"  [WARN] Only found {len(distractors)} distractors for {q_id} (ops: {accepted_ops})")
        return None

    # Assign letters
    letters = ['A', 'B', 'C', 'D']
    correct_pos = rng_py.randint(0, 3)
    choices = {}
    dist_iter = iter(distractors)
    for i, letter in enumerate(letters):
        if i == correct_pos:
            choices[letter] = answer_mesh
        else:
            choices[letter] = next(dist_iter)

    answer = letters[correct_pos]

    return {
        'id': q_id,
        'answer_mesh': answer_mesh,
        'choices': choices,
        'answer': answer,
    }


# ---------------------------------------------------------------------------
# Analysis composite
# ---------------------------------------------------------------------------

def _load_square(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize((CELL, CELL), Image.LANCZOS)


def _add_border(img: Image.Image, colour, width: int) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for i in range(width):
        draw.rectangle([i, i, img.width - 1 - i, img.height - 1 - i], outline=colour)
    return img


def make_analysis_image(q_id: str, answer: str, q_num: int,
                        answer_mesh, q_dir: Path) -> Image.Image:
    """Composite one PAT analysis image; isometric rendered in-memory."""
    tv = _load_square(q_dir / "input" / "top_view.png")
    ev = _load_square(q_dir / "input" / "end_view.png")

    buf = io.BytesIO()
    render_isometric_to_file(answer_mesh, buf)
    buf.seek(0)
    iso = Image.open(buf).convert("RGB").resize((CELL, CELL), Image.LANCZOS)

    choices = {}
    for letter in "ABCD":
        img = _load_square(q_dir / "answers" / f"{letter}.png")
        if letter == answer:
            img = _add_border(img.copy(), CORRECT_COL, BORDER)
        choices[letter] = img

    section1_w = 2 * CELL + GAP
    section2_w = 4 * CELL + 3 * GAP
    section3_w = CELL
    total_w = PAD + section1_w + SEP + section2_w + SEP + section3_w + PAD
    total_h = PAD + CELL + LABEL + PAD

    canvas = Image.new("RGB", (total_w, total_h), BG_COL)
    draw   = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except Exception:
        font = ImageFont.load_default()

    def paste_panel(img, x, y, label, label_colour=FG_COL):
        canvas.paste(img, (x, y))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (CELL - tw) // 2, y + CELL + 4), label,
                  fill=label_colour, font=font)

    y0 = PAD
    x  = PAD
    paste_panel(tv, x, y0, "Top View");  x += CELL + GAP
    paste_panel(ev, x, y0, "End View")

    sep_x = PAD + section1_w + SEP // 2
    draw.line([(sep_x, PAD), (sep_x, PAD + CELL + LABEL)], fill=SECTION_COL, width=2)

    x = PAD + section1_w + SEP
    for letter in "ABCD":
        colour = CORRECT_COL if letter == answer else FG_COL
        paste_panel(choices[letter], x, y0, letter, label_colour=colour)
        x += CELL + GAP

    sep_x2 = PAD + section1_w + SEP + section2_w + SEP // 2
    draw.line([(sep_x2, PAD), (sep_x2, PAD + CELL + LABEL)], fill=SECTION_COL, width=2)

    paste_panel(iso, PAD + section1_w + SEP + section2_w + SEP, y0, "Isometric")

    draw.text((PAD, 2), f"Q{q_num:02d}  —  answer: {answer}", fill=FG_COL, font=font)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    solutions = {}
    generated = 0
    seed = 0

    while generated < N_QUESTIONS:
        seed += 1
        q_id = f"q{generated + 1:02d}"
        print(f"Generating {q_id} (seed={seed})...")

        q = generate_question(q_id, seed)
        if q is None:
            print(f"  [SKIP] {q_id} failed, trying next seed")
            continue

        answer = q['answer']
        solutions[q_id] = answer

        q_dir = OUT_DIR / q_id
        input_dir = q_dir / "input"
        answers_dir = q_dir / "answers"
        input_dir.mkdir(parents=True, exist_ok=True)
        answers_dir.mkdir(parents=True, exist_ok=True)

        mesh = q['answer_mesh']

        # Render input views (top and end of the ANSWER mesh)
        print(f"  Rendering top_view...")
        render_mesh_to_file(mesh, 'top', input_dir / "top_view.png")
        print(f"  Rendering end_view...")
        render_mesh_to_file(mesh, 'end', input_dir / "end_view.png")

        # Render answer choices (front views)
        for letter, choice_mesh in q['choices'].items():
            print(f"  Rendering answer {letter}...")
            render_mesh_to_file(choice_mesh, 'front', answers_dir / f"{letter}.png")

        # Composite analysis image (isometric rendered in-memory, not saved separately)
        print(f"  Rendering analysis image...")
        analysis_img = make_analysis_image(q_id, answer, generated + 1, mesh, q_dir)
        analysis_img.save(q_dir / "full_question.png")

        # PAT-format composite image (what the vision agent is given)
        print(f"  Rendering composite...")
        try:
            from make_composite import make_composite as _make_composite
            _make_composite(q_dir, q_dir / "composite.png")
        except Exception as e:
            print(f"  [WARN] composite generation failed: {e}")

        print(f"  Done: {q_id}, answer={answer}")
        generated += 1

    with open(OUT_DIR / "solutions.json", "w") as f:
        json.dump(solutions, f, indent=2)

    print(f"\nDone! {generated} questions in {OUT_DIR}")
    print("Solutions:", solutions)


if __name__ == "__main__":
    main()
