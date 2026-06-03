import math, random, time, sys
import numpy as np
import torch, torch.nn as nn
from torch.utils.dlpack import to_dlpack, from_dlpack
import pygame
import moderngl
from species_config import (KIND_NAMES, SPECIES_COLORS,
                             SPECIES_NAMES, SPECIES_EVENT_COLORS)

# ── GPU device ────────────────────────────────────────────────────────────────
def select_device():  # picks best available compute device (CUDA > XP> DirectML > CPU)
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), f"XPU ({torch.xpu.get_device_name(0)})"
    try:
        import torch_directml as dml
        if dml.is_available() and dml.device_count() > 0:
            return dml.device(0), f"DirectML ({dml.device_name(0)})"
    except Exception:
        pass
    return torch.device("cpu"), "CPU"

device, _device_label = select_device()
print(f"PyTorch device : {_device_label}")

CUPY_AVAILABLE = False
if device.type == 'cuda':
    try:
        import cupy as cp
        def t2c(t): return cp.from_dlpack(to_dlpack(t.contiguous()))
        def c2t(c): return from_dlpack(c.toDlpack())
        CUPY_AVAILABLE = True
    except (ImportError, RuntimeError):
        pass

if not CUPY_AVAILABLE:
    import numpy as cp
    def t2c(t): return t.detach().cpu().numpy()
    def c2t(c): return torch.from_numpy(c).to(device)

try:
    import sim_core
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

# ── Simulation constants ──────────────────────────────────────────────────────
WORLD_W, WORLD_H       = 2000, 2000
INIT_OMNIVORE, INIT_PRED = 80, 30
INIT_APEX, INIT_MICRO  = 6, 25
FOOD_COUNT, FOOD_MAX   = 500, 700
PLANT_GROWTH_RATE      = 0.04
EGG_TICKS              = 60
THIRST_DEATH           = 120.0
OMNIVORE_MAX_AGE       = 1800
PRED_MAX_AGE           = 2000
APEX_MAX_AGE           = 4000
MICRO_MAX_AGE          = 1500
OMNIVORE_REPRO_H       = 200.0
PRED_REPRO_H           = 270.0
APEX_REPRO_H           = 300.0
MICRO_REPRO_H          = 160.0
OMNIVORE_START_H       = 100.0
PRED_START_H           = 150.0
APEX_START_H           = 250.0
MICRO_START_H          = 95.0
THIRST_INC             = 0.15
TARGET_OMNIVORE        = 150
TARGET_PRED            = 60
TARGET_MICRO           = 80
TARGET_APEX            = 20
SPECIES_EVENT_MIN      = 250
SPECIES_EVENT_MAX      = 500
SPECIES_EVENT_SIZE     = (8, 10)
MAX_EVENT_LOG          = 50
W_MARGIN               = 150.0
MAX_POP                = 2000

PREFERRED_HEIGHT = {"omnivore": 5.0, "pred": 5.0, "micro": 5.0, "apex": 10.0}
HEIGHT_SPRING    = 0.025
HEIGHT_DAMPING   = 0.12
MAX_DISPLAY_H    = 120.0

WIN_W, WIN_H  = 1280, 720
HUD_W         = 280          
FOV_Y         = 55.0
NEAR, FAR     = 0.5, 6000.0

rng = random.Random()

def _avg(lst, fn):  # population-average of a stat across a creature list
    return round(sum(fn(c) for c in lst) / len(lst), 2) if lst else 0

def carrying_clutch(count, target, boost, normal):  # returns larger clutch when population is critically low
    return boost if count < target * 0.3 else normal

def hex_to_rgb(h):  # hex color string to normalized float RGB tuple
    return (int(h[1:3], 16)/255.0, int(h[3:5], 16)/255.0, int(h[5:7], 16)/255.0)

SPECIES_COLORS_RGB = {k: hex_to_rgb(v) for k, v in SPECIES_COLORS.items()}
FOOD_RIPE_RGB = hex_to_rgb("#66BB6A")
FOOD_UNRIPE_RGB = hex_to_rgb("#827717")
FOOD_POLL_RGB = hex_to_rgb("#FFF176")

# ── GLSL shaders ──────────────────────────────────────────────────────────────
_SPHERE_VERT = """
#version 330 core
in vec3 in_pos;
in vec3 in_norm;
in vec3 inst_pos;
in vec3 inst_color;
in vec3 inst_scale;

uniform mat4 u_mvp;
uniform vec3 u_light;

out vec3 v_color;
out float v_light;

void main() {
    vec3 world = in_pos * inst_scale + inst_pos;
    gl_Position = u_mvp * vec4(world, 1.0);
    float d = max(dot(normalize(in_norm), normalize(u_light)), 0.0);
    v_light = 0.45 + 0.55 * d;
    v_color = inst_color;
}
"""
_SPHERE_FRAG = """
#version 330 core
in vec3 v_color;
in float v_light;
out vec4 out_color;
void main() {
    out_color = vec4(v_color * v_light, 1.0);
}
"""

_TERRAIN_VERT = """
#version 330 core
in vec3 in_pos;
uniform mat4 u_mvp;
out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(in_pos.x, in_pos.y - 0.1, in_pos.z, 1.0);
    v_uv = in_pos.xz / vec2(2000.0, 2000.0);
}
"""
_TERRAIN_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 out_color;
void main() {
    vec2 g = fract(v_uv * vec2(20.0, 20.0));
    float line = step(0.96, max(g.x, g.y));
    vec3 base = vec3(0.20, 0.25, 0.22);
    vec3 grid = vec3(0.30, 0.40, 0.35);
    out_color = vec4(mix(base, grid, line * 0.6), 1.0);
}
"""

_WATER_VERT = """
#version 330 core
in vec2 in_pos;
in vec3 inst_pos;
in float inst_radius;

uniform mat4 u_mvp;

void main() {
    vec3 world = vec3(inst_pos.x + in_pos.x * inst_radius,
                      0.2,
                      inst_pos.z + in_pos.y * inst_radius);
    gl_Position = u_mvp * vec4(world, 1.0);
}
"""
_WATER_FRAG = """
#version 330 core
out vec4 out_color;
void main() {
    out_color = vec4(0.10, 0.38, 0.82, 0.55);
}
"""

_HUD_VERT = """
#version 330 core
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""
_HUD_FRAG = """
#version 330 core
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 out_color;
void main() {
    out_color = texture(u_tex, v_uv);
}
"""

_PATH_VERT = """
#version 330 core
in vec3 in_pos;
in vec3 in_color;
uniform mat4 u_mvp;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_color = in_color;
}
"""
_PATH_FRAG = """
#version 330 core
in vec3 v_color;
out vec4 out_color;
void main() {
    out_color = vec4(v_color, 0.6);
}
"""

# ── Geometry helpers ──────────────────────────────────────────────────────────
def _make_uv_sphere(stacks=12, slices=18):  # sphere mesh used for omnivores and food nodes
    verts, norms, idx = [], [], []
    ring = slices + 1
    for i in range(stacks + 1):
        phi = math.pi / 2 - i * math.pi / stacks
        y   = math.sin(phi)
        r   = math.cos(phi)
        for j in range(ring):
            t = j * 2 * math.pi / slices
            x, z = r * math.cos(t), r * math.sin(t)
            verts += [x, y, z]; norms += [x, y, z]
    for i in range(stacks):
        for j in range(slices):
            a = i * ring + j; b = a + ring
            idx += [a, b, a+1, b, b+1, a+1]
    vv = np.array(verts, np.float32).reshape(-1, 3)
    nn_ = np.array(norms, np.float32).reshape(-1, 3)
    combined = np.hstack([vv, nn_]).astype(np.float32)
    return combined, np.array(idx, np.uint32)

def _make_circle(segments=48):  # disc fan mesh for water pool rendering
    verts = [(0.0, 0.0)]
    for i in range(segments + 1):
        a = i * 2 * math.pi / segments
        verts.append((math.cos(a), math.sin(a)))
    return np.array(verts, np.float32)

def _make_ground():  # flat ground-plane quad covering the world
    v = np.array([
        0,       0, 0,
        WORLD_W, 0, 0,
        WORLD_W, 0, WORLD_H,
        0,       0, WORLD_H,
    ], dtype=np.float32)
    i = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return v, i

def _make_cube(size=1.0):  # cube mesh used for micro creatures
    half_size = size / 2.0
    # Vertices and normals for each face (duplicated for hard edges)
    interleaved_verts = np.array([
        # Front face (+Z)
        -half_size, -half_size,  half_size,  0.0, 0.0, 1.0,
         half_size, -half_size,  half_size,  0.0, 0.0, 1.0,
         half_size,  half_size,  half_size,  0.0, 0.0, 1.0,
        -half_size,  half_size,  half_size,  0.0, 0.0, 1.0,
        -half_size, -half_size, -half_size,  0.0, 0.0, -1.0,
         half_size, -half_size, -half_size,  0.0, 0.0, -1.0,
         half_size,  half_size, -half_size,  0.0, 0.0, -1.0,
        -half_size,  half_size, -half_size,  0.0, 0.0, -1.0,
        -half_size,  half_size,  half_size,  0.0, 1.0, 0.0,
         half_size,  half_size,  half_size,  0.0, 1.0, 0.0,
         half_size,  half_size, -half_size,  0.0, 1.0, 0.0,
        -half_size,  half_size, -half_size,  0.0, 1.0, 0.0,
        -half_size, -half_size,  half_size,  0.0, -1.0, 0.0,
         half_size, -half_size,  half_size,  0.0, -1.0, 0.0,
         half_size, -half_size, -half_size,  0.0, -1.0, 0.0,
        -half_size, -half_size, -half_size,  0.0, -1.0, 0.0,
         half_size, -half_size,  half_size,  1.0, 0.0, 0.0,
         half_size, -half_size, -half_size,  1.0, 0.0, 0.0,
         half_size,  half_size, -half_size,  1.0, 0.0, 0.0,
         half_size,  half_size,  half_size,  1.0, 0.0, 0.0,
        -half_size, -half_size,  half_size, -1.0, 0.0, 0.0,
        -half_size, -half_size, -half_size, -1.0, 0.0, 0.0,
        -half_size,  half_size, -half_size, -1.0, 0.0, 0.0,
        -half_size,  half_size,  half_size, -1.0, 0.0, 0.0,
    ], dtype=np.float32)

    indices = []
    for i in range(6):
        offset = i * 4
        indices.extend([
            offset + 0, offset + 1, offset + 2,
            offset + 2, offset + 3, offset + 0
        ])
    return interleaved_verts, np.array(indices, np.uint32)

def _make_cylinder(segments=16, radius=0.5, height=1.0):  # cylinder mesh used for pred creatures
    verts = []
    norms = []
    indices = []

    half_height = height / 2.0

    top_center_idx = 0
    verts.extend([0.0, half_height, 0.0])
    norms.extend([0.0, 1.0, 0.0])
    for i in range(segments):
        angle = i * 2 * math.pi / segments
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        verts.extend([x, half_height, z])
        norms.extend([0.0, 1.0, 0.0])
        indices.extend([top_center_idx, top_center_idx + i + 1, top_center_idx + (i + 1) % segments + 1])

    bottom_center_idx = len(verts) // 3
    verts.extend([0.0, -half_height, 0.0])
    norms.extend([0.0, -1.0, 0.0])
    for i in range(segments):
        angle = i * 2 * math.pi / segments
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        verts.extend([x, -half_height, z])
        norms.extend([0.0, -1.0, 0.0])
        indices.extend([bottom_center_idx, bottom_center_idx + (i + 1) % segments + 1, bottom_center_idx + i + 1])

    for i in range(segments):
        angle0 = i * 2 * math.pi / segments
        angle1 = (i + 1) * 2 * math.pi / segments

        nx0, nz0 = math.cos(angle0), math.sin(angle0)
        nx1, nz1 = math.cos(angle1), math.sin(angle1)

        v_tl = [radius * nx0, half_height, radius * nz0]
        v_bl = [radius * nx0, -half_height, radius * nz0]
        v_tr = [radius * nx1, half_height, radius * nz1]
        v_br = [radius * nx1, -half_height, radius * nz1]

        current_idx = len(verts) // 3
        verts.extend(v_tl); norms.extend([nx0, 0.0, nz0])
        verts.extend(v_bl); norms.extend([nx0, 0.0, nz0])
        verts.extend(v_tr); norms.extend([nx1, 0.0, nz1])
        verts.extend(v_br); norms.extend([nx1, 0.0, nz1])

        indices.extend([current_idx, current_idx + 1, current_idx + 2])
        indices.extend([current_idx + 2, current_idx + 1, current_idx + 3])

    interleaved = []
    for i in range(len(verts) // 3):
        interleaved.extend(verts[i*3 : i*3+3])
        interleaved.extend(norms[i*3 : i*3+3])

    return np.array(interleaved, np.float32), np.array(indices, np.uint32)

def _make_cone(segments=16, radius=0.5, height=1.0):  # cone mesh used for apex creatures
    verts = []
    norms = []
    indices = []

    verts.extend([0.0, height / 2.0, 0.0])
    norms.extend([0.0, 1.0, 0.0])

    base_center_idx = len(verts) // 3
    verts.extend([0.0, -height / 2.0, 0.0])
    norms.extend([0.0, -1.0, 0.0])
    for i in range(segments):
        angle = i * 2 * math.pi / segments
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        verts.extend([x, -height / 2.0, z])
        norms.extend([0.0, -1.0, 0.0])
        indices.extend([base_center_idx, base_center_idx + (i + 1) % segments + 1, base_center_idx + i + 1])

    for i in range(segments):
        angle0 = i * 2 * math.pi / segments
        angle1 = (i + 1) * 2 * math.pi / segments

        v0_base_pos = np.array([radius * math.cos(angle0), -height / 2.0, radius * math.sin(angle0)])
        v1_base_pos = np.array([radius * math.cos(angle1), -height / 2.0, radius * math.sin(angle1)])
        apex_pos = np.array([0.0, height / 2.0, 0.0])

        vec1 = v0_base_pos - apex_pos
        vec2 = v1_base_pos - apex_pos
        
        side_normal = np.cross(vec1, vec2)
        side_normal = side_normal / np.linalg.norm(side_normal)

        current_idx = len(verts) // 3
        verts.extend(apex_pos.tolist()); norms.extend(side_normal.tolist())
        verts.extend(v0_base_pos.tolist()); norms.extend(side_normal.tolist())
        verts.extend(v1_base_pos.tolist()); norms.extend(side_normal.tolist())

        indices.extend([current_idx, current_idx + 1, current_idx + 2])

    interleaved = []
    for i in range(len(verts) // 3):
        interleaved.extend(verts[i*3 : i*3+3])
        interleaved.extend(norms[i*3 : i*3+3])

    return np.array(interleaved, np.float32), np.array(indices, np.uint32)

# ── Matrix math (no external dep) ────────────────────────────────────────────
def _perspective(fov_deg, aspect, near, far):  # perspective projection matrix
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    return np.array([
        [f/aspect,0,0,0],
        [0,f,0,0],
        [0,0,(far+near)/(near-far),-1],
        [0,0,2*far*near/(near-far),0],
    ], dtype=np.float32)

def _look_at(eye, target, up):  # view matrix from camera position, look-at target, and up vector
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.array([
        [ r[0],r[1],r[2],-np.dot(r, eye)],
        [ u[0],u[1],u[2],-np.dot(u, eye)],
        [-f[0],-f[1],-f[2],np.dot(f, eye)],
        [0,0,0,1],
    ], dtype=np.float32)

# ── Neural network ────────────────────────────────────────────────────────────
class BatchedBrains(nn.Module):  # shared neural network for all creatures; 19 sensory inputs → 11 hidden → 3 motor outputs
    def __init__(self, pop=MAX_POP):
        super().__init__()  # allocates weight tensors for the full creature population
        self.w1 = nn.Parameter(torch.randn(pop, 19, 11, device=device) * 0.3)
        self.w2 = nn.Parameter(torch.randn(pop, 11,  3, device=device) * 0.3)
    def forward(self, x, idx):  # two-layer tanh pass; outputs turn angle, acceleration, and spare channel
        h = torch.tanh(torch.bmm(x.view(-1, 1, 19), self.w1[idx]).view(-1, 11))
        return torch.tanh(torch.bmm(h.view(-1, 1, 11), self.w2[idx]).view(-1, 3))

brains = BatchedBrains().to(device)

def _mutate(parent, child, rate=0.12):  # inherits parent weights then injects Gaussian noise at mutation rate
    with torch.no_grad():
        brains.w1[child].copy_(brains.w1[parent])
        brains.w2[child].copy_(brains.w2[parent])
        m1 = (torch.rand_like(brains.w1[child]) < rate).float()
        m2 = (torch.rand_like(brains.w2[child]) < rate).float()
        brains.w1[child] += torch.randn_like(brains.w1[child]) * 0.45 * m1
        brains.w2[child] += torch.randn_like(brains.w2[child]) * 0.45 * m2

def _randomize(indices):  # random weight initialization for freshly spawned lineages with no ancestor
    if not indices: return
    idx = torch.tensor(indices, dtype=torch.long, device=device)
    with torch.no_grad():
        brains.w1[idx] = torch.randn(len(indices), 19, 11, device=device) * 0.3
        brains.w2[idx] = torch.randn(len(indices), 11,  3, device=device) * 0.3

# ── Simulation classes ────────────────────────────────────────────────────────
class Creature:  # base entity for omnivore, pred, micro, and apex; holds position, hunger, thirst, age, and brain
    _cid = 0
    def __init__(self, kind, x=None, y=None, brain_idx=-1, gen=0, name=None, color=None):  # assigns species-specific speed, body radius, starting hunger, and max lifespan
        Creature._cid += 1
        self.id   = Creature._cid
        self.kind = kind
        self.x    = x if x is not None else rng.uniform(20, WORLD_W-20)
        self.y    = y if y is not None else rng.uniform(20, WORLD_H-20)
        self.angle = rng.uniform(0, math.tau)
        self.vx = self.vy = 0.0
        if kind == "omnivore":
            self.speed, self.radius, self.hunger, self.max_age = rng.uniform(2.0,3.0), 5, OMNIVORE_START_H, OMNIVORE_MAX_AGE
        elif kind == "pred":
            self.speed, self.radius, self.hunger, self.max_age = rng.uniform(2.0,2.7), 7, PRED_START_H, PRED_MAX_AGE
        elif kind == "apex":
            self.speed, self.radius, self.hunger, self.max_age = rng.uniform(2.8,3.75),9, APEX_START_H, APEX_MAX_AGE
        else:
            self.speed, self.radius, self.hunger, self.max_age = rng.uniform(2.5,3.5), 3, MICRO_START_H, MICRO_MAX_AGE
        self.thirst = 0.0; self.age = 0; self.alive = True
        self.fitness = 0; self.gen = gen; self.brain_idx = brain_idx
        self.name = name if name else KIND_NAMES[kind]
        self.color = color if color else SPECIES_COLORS[kind]
        self.color_rgb = hex_to_rgb(self.color); self.target_x = None; self.target_y = None
        self.repro_cooldown = 0
        self.display_y  = PREFERRED_HEIGHT[kind]
        self.display_vy = 0.0

    def apply_nn_output(self, _nn_turn, _accel, _inputs, ws, skip_physics=False):  # per-tick driver: syncs GPU movement, then routes to species food/combat/reproduction block
        self.age += 1; self.thirst += THIRST_INC
        self.repro_cooldown = max(0, self.repro_cooldown - 1)
        i = ws['creature_idx']; r = self.radius

        if not skip_physics and 'gpu_physics' in ws:
            gp = ws['gpu_physics']
            self.x, self.y = gp['x'][i], gp['y'][i]
            self.vx, self.vy = gp['vx'][i], gp['vy'][i]
            self.angle = gp['angle'][i]; self.hunger = gp['hunger'][i]

        pref = PREFERRED_HEIGHT[self.kind]
        self.display_vy += (pref - self.display_y) * HEIGHT_SPRING
        self.display_vy *= (1.0 - HEIGHT_DAMPING)
        self.display_y   = max(-20.0, min(MAX_DISPLAY_H, self.display_y + self.display_vy))

        if (ws['dist_cw_np'][i] < r + 45).any():
            self.thirst = max(0.0, self.thirst - (0.8 if self.kind=="pred" else 2.5))

        children = []; eat_r = r + 5
        # grazes ripe plants and opportunistically eats nearby micro; moves toward nearest food via NN steering; no combat, relies on speed and distance to avoid threats
        if self.kind == "omnivore":
            for h in cp.where(ws['dist_cf_cp'][i] < eat_r)[0].tolist():
                f = ws['food_list'][h]
                if f.alive and f.growth >= 1.0:
                    f.alive=False; self.hunger=min(self.hunger+f.energy,250); self.fitness+=1
            for h in cp.where(ws['dist_micro_cp'][i] < eat_r)[0].tolist():
                mo = ws['micro_list'][h]
                if mo.alive: mo.alive=False; self.hunger=min(self.hunger+15,250); self.fitness+=1
            self.fitness += 0.01
            if self.hunger >= OMNIVORE_REPRO_H and self.repro_cooldown == 0:
                children = [self._egg(OMNIVORE_REPRO_H*0.35) for _ in range(
                    carrying_clutch(ws['counts']['omnivore'], TARGET_OMNIVORE, 2, 1))]

        # hunts omnivores as primary food source; steers toward nearest prey when hungry; skirmishes apex with chip damage (20% hit chance) but rarely kills one outright
        elif self.kind == "pred":
            if self.hunger < PRED_REPRO_H:
                for h in cp.where(ws['dist_omnivore_only_cp'][i] < r+8)[0].tolist():
                    p = ws['omnivore_list'][h]
                    if p.alive:
                        p.alive=False; self.hunger=min(self.hunger+(110 if p.kind=="apex" else 75),300)
                        self.fitness+=2; break
            for h in cp.where(ws['dist_apex_only_cp'][i] < r+12)[0].tolist():
                ap = ws['apex_list'][h]
                if ap.alive:
                    if rng.random()<0.2: ap.hunger-=20; self.hunger-=3; self.fitness+=0.8
                    if ap.hunger<=0: ap.alive=False; self.hunger=min(self.hunger+110,300)
                    break
            if self.hunger >= PRED_REPRO_H and self.repro_cooldown == 0:
                children = [self._egg(PRED_REPRO_H*0.35) for _ in range(
                    carrying_clutch(ws['counts']['pred'], TARGET_PRED, 3, 2))]

        # feeds on unripe plants and pollinates flowers to spawn new food nodes; no offensive combat; swarm crossover blends weights with nearby micro for collective learning
        elif self.kind == "micro":
            for h in cp.where(ws['dist_cf_all_cp'][i] < r+10)[0].tolist():
                f = ws['all_food_list'][h]
                if f.alive:
                    if f.needs_pollination:
                        f.needs_pollination=False; self.fitness+=1.5
                        for _ in range(rng.randint(2,4)):
                            sx=min(max(f.x+rng.uniform(-40,40),10),WORLD_W-10)
                            sy=min(max(f.y+rng.uniform(-40,40),10),WORLD_H-10)
                        ws['new_foods'].append(Food(sx,sy))
                    if f.growth<1.0:
                        f.growth=min(1.0,f.growth+0.03); self.hunger=min(self.hunger+20,150); self.fitness+=0.2
            if self.hunger >= MICRO_REPRO_H and self.repro_cooldown == 0:
                c = math.ceil(carrying_clutch(ws['counts']['micro'], TARGET_MICRO, 3, 2) * 0.8)
                if c: self.repro_cooldown=30; self.hunger*=0.6
                children = [self._egg(self.hunger) for _ in range(c)]

        # hunts whichever species is most numerous (dynamically ranked each tick); steers toward highest-density prey type; brawls pred with 80% win rate, can lose and be killed on the remaining 20%
        elif self.kind == "apex":
            if self.hunger < APEX_REPRO_H:
                for h in cp.where(ws['dist_apex_targets_cp'][i] < r+12)[0].tolist():
                    t = ws['apex_target_list'][h]
                    if t.alive:
                        gain={"omnivore":100,"pred":110,"micro":20,"apex":90}.get(t.kind,70)
                        t.alive=False; self.hunger=min(self.hunger+gain,700); self.fitness+=3; break
            for h in cp.where(ws['dist_pred_only_cp'][i] < r+12)[0].tolist():
                pr = ws['pred_list'][h]
                if pr.alive:
                    if rng.random()<0.8:
                        pr.hunger-=34; self.hunger-=6; self.fitness+=1
                        if pr.hunger<=0: pr.alive=False; self.hunger=min(self.hunger+70,700)
                    else:
                        self.hunger-=45; pr.hunger-=2; pr.fitness+=1.5
                        if self.hunger<=0: self.alive=False; pr.hunger=min(pr.hunger+110,300)
                    break
            if self.hunger >= APEX_REPRO_H and self.repro_cooldown == 0:
                children = [self._egg(APEX_REPRO_H*0.35) for _ in range(
                    carrying_clutch(ws['counts']['apex'], TARGET_APEX, 2, 1))]

        if self.hunger<=0 or self.thirst>=THIRST_DEATH or self.age>self.max_age:
            self.alive=False
        return children

    def _egg(self, c_hunger):  # spawns a mutated offspring near the parent; deducts hunger as reproductive cost
        self.repro_cooldown = 160 if self.kind=="omnivore" else 110
        if self.kind!="micro": self.hunger -= c_hunger
        return Egg(self.kind, self.x+rng.uniform(-20,20), self.y+rng.uniform(-20,20),
                   self.brain_idx, self.gen+1, self.speed+rng.uniform(-0.1,0.1), 
                   c_hunger, name=self.name, color=self.color, radius=self.radius)

class Food:  # plant food node; starts unripe or needs pollination; omnivores harvest ripe nodes, micro pollinates and accelerates growth
    _fid = 0
    def __init__(self, x=None, y=None, water_pools=None):  # places food away from water; randomly assigns kind (0-2) and initial growth/pollination state
        Food._fid+=1; self.id=Food._fid; self.alive=True
        self.kind=rng.randint(0,2); self.needs_pollination=rng.random()<0.2
        self.growth=0.0 if self.needs_pollination else rng.uniform(0.0,1.0)
        self.energy=[40,50,60][self.kind]
        if x is not None: self.x,self.y=x,y
        else:
            while True:
                self.x,self.y=rng.uniform(10,WORLD_W-10),rng.uniform(10,WORLD_H-10)
                if not water_pools or all(math.hypot(self.x-w.x,self.y-w.y)>w.radius+3 for w in water_pools): break

class WaterPool:  # static thirst source; creatures reduce thirst when within radius each tick
    _wid=0
    def __init__(self,x,y,radius):  # records pool center and radius; used for distance checks and rendering
        WaterPool._wid+=1; self.id=WaterPool._wid
        self.x=x; self.y=y; self.radius=radius

class Egg:  # pending hatchling; incubates for EGG_TICKS then mutates the parent brain into a new creature
    _eid=0
    def __init__(self,kind,x,y,parent_idx,gen,speed,hunger,name=None,color=None,radius=None):  # stores inherited traits (speed, hunger, color) until the egg hatches
        Egg._eid+=1; self.id=Egg._eid
        self.kind=kind; self.x=x; self.y=y
        self.parent_idx=parent_idx; self.gen=gen
        self.speed=speed; self.hunger=hunger; self.age=0; self.alive=True; self.name=name
        self.color=color; self.radius=radius 

# ── World ─────────────────────────────────────────────────────────────────────
class World:  # simulation container; manages creature populations, food ecology, water, eggs, evolution, and species events
    def __init__(self):  # seeds best-brain records per species then delegates to _init
        self.best_records={
        "omnivore": {"w1":None,"w2":None,"speed":2.5,"max_age":OMNIVORE_MAX_AGE,"fitness":0.0},
        "pred": {"w1":None,"w2":None,"speed":2.4,"max_age":PRED_MAX_AGE,"fitness":0.0},
        "micro":{"w1":None,"w2":None,"speed":3.0,"max_age":MICRO_MAX_AGE,"fitness":0.0},
        "apex": {"w1":None,"w2":None,"speed":3.2,"max_age":APEX_MAX_AGE,"fitness":0.0},
    }; self._init()

    def _init(self):  # cold-starts all populations, food nodes, water pools, brain index pool, and event timers
        Creature._cid=Food._fid=Egg._eid=WaterPool._wid=0
        self.running=False; self.tick=0
        self.mut_rate=0.12
        self.available_brain_indices=list(range(MAX_POP)); random.shuffle(self.available_brain_indices)
        self.creatures=[]
        for k,n in [("omnivore",INIT_OMNIVORE),("pred",INIT_PRED),("micro",INIT_MICRO),("apex",INIT_APEX)]:
            for _ in range(n):
                if self.available_brain_indices:
                    self.creatures.append(Creature(k,brain_idx=self.available_brain_indices.pop(), name="Primal"))
        self.foods=[Food() for _ in range(FOOD_COUNT)]
        cols,rows=3,2
        cells=[(cx,cy) for cx in range(cols) for cy in range(rows)]; rng.shuffle(cells)
        cw,ch=WORLD_W/cols,WORLD_H/rows
        self.water_pools=[WaterPool((cx+rng.uniform(.3,.7))*cw,(cy+rng.uniform(.3,.7))*ch,rng.uniform(40,64))
                          for cx,cy in cells[:rng.randint(4,len(cells))]]
        self.eggs=[]; self.pop_history=[]; self.fitness_history=[]
        self.extinction_log=[]; self.extinct_kinds=set()
        self.species_events=[]; self.species_event_count=0
        self._last_counts={"omnivore":0,"pred":0,"micro":0,"apex":0}
        self.lineage_counts = {}
        for c in self.creatures:
            l_key = (c.name, c.color, c.kind)
            self.lineage_counts[l_key] = self.lineage_counts.get(l_key, 0) + 1
        self.selected_custom_kind = 0
        self.custom_name = rng.choice(SPECIES_NAMES)
        self.custom_color_rgb = [rng.random(), rng.random(), rng.random()]
        self.custom_speed_mul = 1.0
        self.custom_age_mul = 1.0
        self.custom_spawn_count = 15
        self.editing_name = False
        self.next_species_event=rng.randint(SPECIES_EVENT_MIN,SPECIES_EVENT_MAX)
        self.next_pollination_tick=rng.randint(600,900)

    def reset(self): self._init()  # full world restart; resets all IDs and state

    def step(self):  # one simulation tick: GPU distance sensing, NN inference, physics integration, species feeding/combat, egg hatching, food growth, and record keeping
        self.tick+=1
        alive=self.creatures
        ws = {'new_foods': []}

        if alive:
            _N=len(alive)
            _cxy=np.empty((_N,2),np.float32)
            _kinds=np.empty(_N,dtype='<U8')
            for _i,_c in enumerate(alive):
                _cxy[_i,0]=_c.x; _cxy[_i,1]=_c.y; _kinds[_i]=_c.kind
            c_pos=torch.from_numpy(_cxy).to(device,non_blocking=True)
            f_all=self.foods
            f_ripe=[f for f in f_all if f.growth>=1.0 and not f.needs_pollination]
            f_poll=[f for f in f_all if f.needs_pollination]
            f_pos =torch.from_numpy(np.array([[f.x,f.y] for f in f_ripe],dtype=np.float32)).to(device,non_blocking=True) if f_ripe else torch.empty((0,2),device=device)
            f_all_pos=torch.from_numpy(np.array([[f.x,f.y] for f in f_all],dtype=np.float32)).to(device,non_blocking=True) if f_all else torch.empty((0,2),device=device)
            f_poll_pos=torch.from_numpy(np.array([[f.x,f.y] for f in f_poll],dtype=np.float32)).to(device,non_blocking=True) if f_poll else torch.empty((0,2),device=device)
            w_pos=torch.from_numpy(np.array([[w.x,w.y] for w in self.water_pools],dtype=np.float32)).to(device,non_blocking=True)

            dist_cc=torch.cdist(c_pos,c_pos).to(device)
            dist_ns=dist_cc.clone(); dist_ns.fill_diagonal_(1e6)
            dist_cf =torch.cdist(c_pos,f_pos).to(device) if f_pos.shape[0]>0 else torch.empty((len(alive),0),device=device)
            dist_cfa=torch.cdist(c_pos,f_all_pos).to(device) if f_all_pos.shape[0]>0 else torch.empty((len(alive),0),device=device)
            dist_cw =torch.cdist(c_pos,w_pos).to(device)
            dist_cp =torch.cdist(c_pos,f_poll_pos).to(device) if f_poll_pos.shape[0]>0 else torch.empty((len(alive),0),device=device)

            om=torch.from_numpy(_kinds=="omnivore").to(device,non_blocking=True)
            dm=torch.from_numpy(_kinds=="pred").to(device,non_blocking=True)
            am=torch.from_numpy(_kinds=="apex").to(device,non_blocking=True)
            mm=torch.from_numpy(_kinds=="micro").to(device,non_blocking=True)
            threat=dm|am; pred_tgt=om|am

            _huntable=[("omnivore",om),("pred",dm),("micro",mm),("apex",am)]
            _ranked=sorted(_huntable,key=lambda kv:self._last_counts.get(kv[0],0),reverse=True)
            valid=[m for k,m in _ranked if self._last_counts.get(k,0)>0]
            apex_hunt=valid[0] if valid else om
            apex_eat =apex_hunt|(valid[1] if len(valid)>1 else apex_hunt)
            avoid_micro=torch.cat([f_all_pos[torch.tensor([f.growth<1.0 for f in f_all],dtype=torch.bool,device=device)],c_pos[~mm]]) if f_all_pos.shape[0]>0 else c_pos[~mm]

            ws.update({
                'tick':self.tick,
                'food_list':f_ripe,'all_food_list':f_all,'micro_list':[c for i,c in enumerate(alive) if mm[i]],
                'omnivore_list':[c for i,c in enumerate(alive) if pred_tgt[i]],
                'pred_list':[c for i,c in enumerate(alive) if dm[i]],
                'apex_list':[c for i,c in enumerate(alive) if am[i]],
                'apex_target_list':[c for i,c in enumerate(alive) if apex_eat[i]],
                'water_pos':w_pos,'counts':self._last_counts,
                'dist_apex_only':dist_cc[:,am],
                'dist_threats':dist_cc[:,threat],'dist_omnivore':dist_cc[:,om],
                'dist_pred_only':dist_ns[:,dm],'dist_micro':dist_ns[:,mm],
                'dist_apex_targets':dist_ns[:,apex_eat],'dist_apex_social':dist_ns[:,am],
                'avoid_micro_pos':avoid_micro,
                'threat_pos':c_pos[threat],'omnivore_pos':c_pos[om],'pred_pos':c_pos[dm],
                'apex_pos':c_pos[am],'micro_pos':c_pos[mm],
                'apex_target_pos':c_pos[apex_hunt],
                'dist_avoid_micro':torch.cdist(c_pos,avoid_micro).to(device)
            })

            # ── Batched sensing ──
            N=_N
            cpos_cp = t2c(c_pos); cx_ = cpos_cp[:,0]; cy_ = cpos_cp[:,1]
            _sense=cp.array([[c.angle,c.hunger,c.thirst] for c in alive],cp.float32)
            ang_=_sense[:,0]
            om_ = t2c(om); dm_ = t2c(dm); mm_ = t2c(mm); am_ = t2c(am)

            def near(D,P):  # find the closest thing and return how far away it is + where it is
                if D.shape[1]==0: return cp.full(N,cp.inf,cp.float32),cp.full(N,cp.nan,cp.float32),cp.full(N,cp.nan,cp.float32)
                md,mi=torch.min(D,dim=1); tgt=P[mi]
                return t2c(md), t2c(tgt[:, 0]), t2c(tgt[:, 1])

            def comb(pv,dv,mv,av):  # fill in each creature's nearest target info based on what type it is
                d_=cp.empty(N,cp.float32); tx=cp.empty(N,cp.float32); ty=cp.empty(N,cp.float32)
                for msk, v in ((om_, pv), (dm_, dv), (mm_, mv), (am_, av)):
                    d_[msk] = v[0][msk]; tx[msk] = v[1][msk]; ty[msk] = v[2][msk]
                return d_,tx,ty

            def asc(tx,ty):  # figure out which direction the target is relative to where the creature is facing
                a=cp.arctan2(ty-cy_,tx-cx_)-ang_; a=(a+cp.pi)%(2*cp.pi)-cp.pi
                s,c_=cp.sin(a),cp.cos(a); miss=cp.isnan(tx); s[miss]=0.0; c_[miss]=1.0
                return s,c_

            nd=lambda d: cp.clip(d/1200.0,-1.0,1.0)

            td,tx,ty=comb(near(dist_cf,f_pos),near(dist_cc[:,pred_tgt],c_pos[pred_tgt]),
                        near(dist_cp,f_poll_pos),near(dist_ns[:,apex_hunt],c_pos[apex_hunt]))
            ad,ax,ay=comb(near(ws['dist_threats'],ws['threat_pos']),near(ws['dist_apex_only'],ws['apex_pos']),
                        near(ws['dist_avoid_micro'],avoid_micro),near(ws['dist_apex_social'],ws['apex_pos']))
            sd,sx,sy=comb(near(ws['dist_omnivore'],ws['omnivore_pos']),near(ws['dist_pred_only'],ws['pred_pos']),
                        near(ws['dist_micro'],ws['micro_pos']),near(ws['dist_apex_social'],ws['apex_pos']))
            wd,wx_,wy_=near(dist_cw,w_pos)

            ts,tc=asc(tx,ty); as_,ac=asc(ax,ay); ss,sc=asc(sx,sy); ws_,wc=asc(wx_,wy_)
            h_in=cp.clip(_sense[:,1]/180.,-1,1)
            t_in=cp.clip(_sense[:,2]/100.,-1,1)

            wL = cp.clip(1.0 - (cx_ / W_MARGIN), 0.0, 1.0)
            wR = cp.clip(1.0 - ((WORLD_W - cx_) / W_MARGIN), 0.0, 1.0)
            wT = cp.clip(1.0 - (cy_ / W_MARGIN), 0.0, 1.0)
            wB = cp.clip(1.0 - ((WORLD_H - cy_) / W_MARGIN), 0.0, 1.0)

            inp=cp.stack([nd(td),ts,tc,nd(ad),as_,ac,nd(sd),ss,sc,nd(wd),ws_,wc,h_in,t_in,wL,wR,wT,wB,cp.ones(N,cp.float32)],axis=1).astype(cp.float32)
            tx_np, ty_np = tx.get() if CUPY_AVAILABLE else tx, ty.get() if CUPY_AVAILABLE else ty
            for i,c in enumerate(alive):
                c.target_x=(None if np.isnan(tx_np[i]) else float(tx_np[i])); c.target_y=(None if np.isnan(ty_np[i]) else float(ty_np[i]))

            ws.update({
                'dist_cw_np': t2c(dist_cw), 'dist_cf_cp': t2c(dist_cf),
                'dist_omnivore_only_cp': t2c(dist_cc[:,pred_tgt]),
                'dist_cf_all_cp': t2c(dist_cfa),
                'dist_apex_targets_cp': t2c(ws['dist_apex_targets']),
                'dist_micro_cp': t2c(ws['dist_micro']),
                'dist_apex_only_cp': t2c(ws['dist_apex_only']),
                'dist_pred_only_cp': t2c(ws['dist_pred_only']),
            })

            ai=torch.from_numpy(np.array([c.brain_idx for c in alive],dtype=np.int64)).to(device,non_blocking=True)
            with torch.no_grad(): out=brains(c2t(inp),ai)
            out_cpu=out.cpu().numpy()

            # ── GPU physics ──
            with torch.no_grad():
                turns=out[:,0]; accels=(out[:,1]+1.0)*0.5

                has_target = c2t((~cp.isnan(tx)).astype(cp.float32)).bool()
                steering = torch.clamp(torch.atan2(c2t(ts), c2t(tc)) / 0.28, -1.0, 1.0)
                turns = torch.where(has_target, steering, turns)
                accels = torch.where(has_target, torch.ones_like(accels), accels)

                _phys=np.array([[c.vx,c.vy,c.angle,c.speed,c.hunger,c.radius] for c in alive],dtype=np.float32)
                _pt=torch.from_numpy(_phys).to(device,non_blocking=True)
                vx_,vy_,ang_t,spd,hun,rad=_pt[:,0],_pt[:,1],_pt[:,2],_pt[:,3],_pt[:,4],_pt[:,5]
                new_ang=ang_t+turns*0.28
                spd_v=spd*(0.55+accels*0.45)
                mv=torch.where(mm,0.92,0.78)
                nvx=vx_*mv+torch.cos(new_ang)*spd_v*(1.-mv); nvy=vy_*mv+torch.sin(new_ang)*spd_v*(1.-mv)

                s_m = 150.0
                px, py = c_pos[:,0], c_pos[:,1]
                push_l, push_r = torch.clamp((s_m - px)/s_m, 0, 1), torch.clamp((px - (WORLD_W - s_m))/s_m, 0, 1)
                push_t, push_b = torch.clamp((s_m - py)/s_m, 0, 1), torch.clamp((py - (WORLD_H - s_m))/s_m, 0, 1)
                nvx += (push_l - push_r) * 0.5
                nvy += (push_t - push_b) * 0.5

                nx=c_pos[:,0]+nvx; ny=c_pos[:,1]+nvy
                lh=nx<rad; rh=nx>WORLD_W-rad; nx=torch.where(lh,rad,torch.where(rh,WORLD_W-rad,nx)); nvx=torch.where(lh|rh,nvx*-0.8,nvx)
                th=ny<rad; bh=ny>WORLD_H-rad; ny=torch.where(th,rad,torch.where(bh,WORLD_H-rad,ny)); nvy=torch.where(th|bh,nvy*-0.8,nvy)
                bm=torch.where(dm,0.01,torch.where(am,0.004,0.007))
                cost=torch.where(am,0.015,0.025)
                nh=hun-(spd_v**1.5)*cost-bm
                _gp=torch.stack([nx,ny,nvx,nvy,new_ang,nh],dim=1).cpu().numpy()
                ws['gpu_physics']={'x':_gp[:,0],'y':_gp[:,1],'vx':_gp[:,2],'vy':_gp[:,3],'angle':_gp[:,4],'hunger':_gp[:,5]}

            new_eggs=[]
            for i,c in enumerate(alive):
                ws['creature_idx']=i
                clutch=c.apply_nn_output(out_cpu[i][0],out_cpu[i][1],inp[i],ws)
                if clutch: new_eggs.extend(clutch)
            for c in alive:
                if not c.alive: self.available_brain_indices.append(c.brain_idx)
            if ws['new_foods']: self.foods.extend(ws['new_foods'])

            mi_idx=torch.where(mm)[0]
            if mi_idx.shape[0]>1:
                for _ in range(min(mi_idx.shape[0],5)):
                    i1=mi_idx[random.randint(0,mi_idx.shape[0]-1)].item()
                    nb=(dist_cc[i1]<15)&mm; nb[i1]=False; pts=torch.where(nb)[0]
                    if pts.shape[0]>0:
                        i2=pts[random.randint(0,pts.shape[0]-1)].item()
                        with torch.no_grad():
                            a=random.random()
                            brains.w1[alive[i1].brain_idx]=a*brains.w1[alive[i1].brain_idx]+(1-a)*brains.w1[alive[i2].brain_idx]
                            brains.w2[alive[i1].brain_idx]=a*brains.w2[alive[i1].brain_idx]+(1-a)*brains.w2[alive[i2].brain_idx]
                        alive[i1].fitness+=0.02

            self.eggs+=new_eggs

        hatchlings=[]
        for e in self.eggs:
            e.age+=1
            if e.age>=EGG_TICKS and self.available_brain_indices:
                ci=self.available_brain_indices.pop(); _mutate(e.parent_idx,ci,self.mut_rate)
                c=Creature(e.kind,e.x,e.y,ci,e.gen, name=e.name, color=e.color)
                c.speed=e.speed; c.hunger=e.hunger
                if c.color: c.color_rgb = hex_to_rgb(c.color)
                if e.radius: c.radius=e.radius
                hatchlings.append(c); e.alive=False
        self.eggs=[e for e in self.eggs if e.alive]

        self.creatures=[c for c in self.creatures if c.alive]+hatchlings
        lc={"omnivore":0,"pred":0,"micro":0,"apex":0}
        for c in self.creatures: lc[c.kind]+=1
        self._last_counts=lc
        
        # ── Lineage-based Extinction Tracking ──
        curr_lineages = {}
        for c in self.creatures:
            l_key = (c.name, c.color, c.kind)
            curr_lineages[l_key] = curr_lineages.get(l_key, 0) + 1
        for l_key in self.lineage_counts:
            if l_key not in curr_lineages:
                self.extinction_log.append({"tick": self.tick, "name": l_key[0], "color": l_key[1], "kind": l_key[2]})
                if len(self.extinction_log) > MAX_EVENT_LOG: self.extinction_log = self.extinction_log[-MAX_EVENT_LOG:]
        self.lineage_counts = curr_lineages

        if len(self.eggs)>150: self.eggs=self.eggs[-150:]
        for f in self.foods:
            if f.alive and f.growth<1.0: f.growth+=PLANT_GROWTH_RATE
        self.foods=[f for f in self.foods if f.alive]
        if len(self.foods)<FOOD_COUNT:
            self.foods+=[Food(water_pools=self.water_pools) for _ in range(FOOD_COUNT-len(self.foods))]
        elif len(self.foods)>FOOD_MAX: self.foods=self.foods[-FOOD_MAX:]

        if self.tick>=self.next_pollination_tick:
            av=[f for f in self.foods if f.alive and not f.needs_pollination]
            if av: rng.choice(av).needs_pollination=True
            self.next_pollination_tick=self.tick+rng.randint(600,900)

        if self.tick%40==0:
            for k in ("omnivore","pred","micro","apex"):
                mb=[c for c in self.creatures if c.kind==k and c.alive]
                if mb:
                    bn=max(mb,key=lambda c:c.fitness)
                    if bn.fitness>=self.best_records[k]["fitness"]:
                        self.best_records[k].update({"w1":brains.w1[bn.brain_idx].clone(),
                            "w2":brains.w2[bn.brain_idx].clone(),"speed":bn.speed,
                            "max_age":bn.max_age,"fitness":bn.fitness})

        if self.tick%20==0:
            lc=self._last_counts
            self.pop_history.append({"t":self.tick,"p":lc["omnivore"],"d":lc["pred"],"m":lc["micro"],"a":lc["apex"]})
            if len(self.pop_history)>400: self.pop_history=self.pop_history[-400:]

        if self.tick>=self.next_species_event:
            self._species_event()
            self.next_species_event=self.tick+rng.randint(SPECIES_EVENT_MIN,SPECIES_EVENT_MAX)

    def _species_event(self):  # injects a new named lineage for each kind using the best recorded brain as evolutionary seed
        if not self.available_brain_indices: return
        for kind in ["omnivore","pred","micro","apex"]:
            if not self.available_brain_indices: break
            self.extinct_kinds.discard(kind)
            color=SPECIES_EVENT_COLORS[self.species_event_count%len(SPECIES_EVENT_COLORS)]
            self.species_event_count+=1
            name=f"{rng.choice(SPECIES_NAMES)}-{self.species_event_count:03d}"
            n=min(rng.randint(*SPECIES_EVENT_SIZE),len(self.available_brain_indices))
            if n<=0: continue
            cx,cy=rng.uniform(60,WORLD_W-60),rng.uniform(60,WORLD_H-60)
            idxs=[self.available_brain_indices.pop() for _ in range(n)]
            rec=self.best_records[kind]; has=rec["w1"] is not None
            if has:
                ti=torch.tensor(idxs,dtype=torch.long,device=device)
                with torch.no_grad(): brains.w1[ti]=rec["w1"].clone(); brains.w2[ti]=rec["w2"].clone()
            else: _randomize(idxs)
            for bi in idxs:
                c=Creature(kind,cx+rng.uniform(-40,40),cy+rng.uniform(-40,40),brain_idx=bi,gen=0, name=name, color=color)
                c.color=color; c.color_rgb=hex_to_rgb(color)
                if has: c.speed=min(rec["speed"]*1.3,1500.0); c.max_age=int(rec["max_age"]*1.5); c.fitness=rec["fitness"]*3.
                self.creatures.append(c)
            self.species_events=getattr(self,"species_events",[]); self.species_events.append({"tick":self.tick,"kind":kind,"name":name,"n":n,"color":color})
            if len(self.species_events)>MAX_EVENT_LOG: self.species_events=self.species_events[-MAX_EVENT_LOG:]

    def spawn_custom(self, kind_idx):  # user-triggered species drop with designer name, color, speed multiplier, and age multiplier
        if not self.available_brain_indices: return
        kinds = ["omnivore", "pred", "micro", "apex"]
        k, name = kinds[kind_idx], self.custom_name
        c_rgb = tuple(self.custom_color_rgb)
        color_hex = "#%02x%02x%02x" % (int(c_rgb[0]*255), int(c_rgb[1]*255), int(c_rgb[2]*255))
        
        self.species_event_count += 1
        name = f"{name}-{self.species_event_count:02d}"

        n = min(self.custom_spawn_count, len(self.available_brain_indices))
        if n <= 0: return
        
        cx, cy = rng.uniform(100, WORLD_W-100), rng.uniform(100, WORLD_H-100)
        idxs = [self.available_brain_indices.pop() for _ in range(n)]
        
        rec = self.best_records[k]
        has = rec["w1"] is not None
        if has:
            ti = torch.tensor(idxs, dtype=torch.long, device=device)
            with torch.no_grad():
                brains.w1[ti] = rec["w1"].clone()
                brains.w2[ti] = rec["w2"].clone()
        else:
            _randomize(idxs)
            
        for bi in idxs:
            c = Creature(k, cx+rng.uniform(-40,40), cy+rng.uniform(-40,40), brain_idx=bi, gen=0, name=name, color=color_hex)
            c.color = color_hex; c.color_rgb = c_rgb
            c.speed = min(c.speed * self.custom_speed_mul, 1500.0)
            c.max_age = int(c.max_age * self.custom_age_mul)
            if has:
                c.fitness = rec["fitness"] * 2.0
            self.creatures.append(c)
        
        self.species_events.append({"tick":self.tick,"kind":k,"name":name,"n":n,"color":color_hex})

    def stats(self):  # returns per-kind averages of speed, generation, and fitness for HUD display
        def grp(k): return [c for c in self.creatures if c.kind==k]  # filters creature list to a single species kind
        return {f"{k}_speed":_avg(grp(k),lambda c:c.speed) for k in("omnivore","pred","micro","apex")} | \
               {f"{k}_gen":_avg(grp(k),lambda c:c.gen)     for k in("omnivore","pred","micro","apex")} | \
               {f"{k}_fit":_avg(grp(k),lambda c:c.fitness) for k in("omnivore","pred","micro","apex")}

# ── Orbit camera ──────────────────────────────────────────────────────────────
class OrbitCamera:  # 3D orbit camera; left-drag to rotate, scroll to zoom
    def __init__(self):  # sets default orbit position centered on the world
        self.cx=WORLD_W/2; self.cz=WORLD_H/2
        self.dist=900.0; self.azim=45.0; self.elev=35.0
        self._drag=False; self._last=None

    def handle_event(self, ev):  # processes mouse drag for azimuth/elevation orbit and scroll wheel for zoom distance
        if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
            self._drag=True; self._last=ev.pos
        elif ev.type==pygame.MOUSEBUTTONUP and ev.button==1:
            self._drag=False
        elif ev.type==pygame.MOUSEMOTION and self._drag:
            dx,dy=ev.pos[0]-self._last[0],ev.pos[1]-self._last[1]
            self.azim-=dx*0.35; self.elev=max(5.,min(89.,self.elev-dy*0.25))
            self._last=ev.pos
        elif ev.type==pygame.MOUSEWHEEL:
            self.dist=max(100.,min(3000.,self.dist-ev.y*40))

    def get_mvp(self, w, h):  # computes combined projection × view matrix from current orbit angles and distance
        ar=math.radians(self.azim); er=math.radians(self.elev)
        ex=self.cx+self.dist*math.cos(er)*math.cos(ar)
        ey=        self.dist*math.sin(er)
        ez=self.cz+self.dist*math.cos(er)*math.sin(ar)
        eye=np.array([ex,ey,ez],np.float32)
        tgt=np.array([self.cx,0.,self.cz],np.float32)
        up =np.array([0.,1.,0.],np.float32)
        V=_look_at(eye,tgt,up)
        P=_perspective(FOV_Y,w/h,NEAR,FAR)
        return (P@V).T.copy()

    @property
    def eye(self):  # world-space camera position derived from orbit azimuth, elevation, and distance
        ar=math.radians(self.azim); er=math.radians(self.elev)
        return np.array([self.cx+self.dist*math.cos(er)*math.cos(ar),
                         self.dist*math.sin(er),
                         self.cz+self.dist*math.cos(er)*math.sin(ar)],np.float32)

# ── 3D Renderer ───────────────────────────────────────────────────────────────
class Renderer3D:  # OpenGL instanced renderer; draws terrain, water, per-species meshes, path lines, and dual HUD panels
    def __init__(self, ctx: moderngl.Context):  # compiles shaders, allocates per-species VAOs and instance buffers, loads fonts for HUD
        self.ctx=ctx
        self.ctx.enable(moderngl.DEPTH_TEST|moderngl.BLEND)
        self.ctx.blend_func=moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA

        # ── Sphere geometry ──
        self.sphere_verts, self.sphere_indices = _make_uv_sphere(8, 16)
        self.sphere_vbo = ctx.buffer(self.sphere_verts.tobytes())
        self.sphere_ebo = ctx.buffer(self.sphere_indices.tobytes())
        self.sp_vbo, self.sp_ebo = self.sphere_vbo, self.sphere_ebo

        self.sp_prog=ctx.program(vertex_shader=_SPHERE_VERT,fragment_shader=_SPHERE_FRAG)
        self.inst_buf=ctx.buffer(reserve=MAX_POP*(3+3+3)*4)
        self.food_inst_buf=ctx.buffer(reserve=FOOD_MAX*(3+3+3)*4)

        # ── Terrain ──
        gv,gi=_make_ground()
        self.terr_prog=ctx.program(vertex_shader=_TERRAIN_VERT,fragment_shader=_TERRAIN_FRAG)
        _terr_vbo=ctx.buffer(gv.tobytes()); _terr_ibo=ctx.buffer(gi.tobytes())
        self.terr_vao=ctx.vertex_array(self.terr_prog,[(_terr_vbo,'3f','in_pos')],_terr_ibo)

        # ── Water pools ──
        self.water_prog=ctx.program(vertex_shader=_WATER_VERT,fragment_shader=_WATER_FRAG)
        circ=_make_circle(56)
        self.water_circ_vbo=ctx.buffer(circ.tobytes())
        self.water_inst_buf=ctx.buffer(reserve=12*4*4)
        self.water_vao=ctx.vertex_array(self.water_prog,[
            (self.water_circ_vbo,'2f','in_pos'),
            (self.water_inst_buf,'3f 1f/i','inst_pos','inst_radius'),
        ])

        # ── HUD quad ──
        self.hud_prog=ctx.program(vertex_shader=_HUD_VERT,fragment_shader=_HUD_FRAG)
        xl = 1.0 - 2.0 * HUD_W / WIN_W
        hud_quad = np.array([xl, -1, 0, 1,  1, -1, 1, 1,  1, 1, 1, 0,  xl, 1, 0, 0], np.float32)
        hud_idx =np.array([0,1,2,0,2,3],np.uint32)
        _hud_vbo=ctx.buffer(hud_quad.tobytes()); self._hud_ibo=ctx.buffer(hud_idx.tobytes())
        self.hud_vao=ctx.vertex_array(self.hud_prog,[(_hud_vbo,'2f 2f','in_pos','in_uv')],self._hud_ibo)
        self.hud_tex=ctx.texture((HUD_W,WIN_H),4)
        self.hud_tex.filter=moderngl.LINEAR,moderngl.LINEAR
        self.pause_btn_rect = None
        self.kind_btn_rects = []
        self.rgb_slider_rects = []
        self.stat_btn_rects = []
        self.name_btn_rect = None
        self.custom_spawn_btn_rect = None

        xr = -1.0 + 2.0 * HUD_W / WIN_W
        left_hud_quad = np.array([-1, -1, 0, 1, xr, -1, 1, 1, xr, 1, 1, 0, -1, 1, 0, 0], np.float32)
        _l_hud_vbo = ctx.buffer(left_hud_quad.tobytes())
        self.left_hud_vao = ctx.vertex_array(self.hud_prog, [(_l_hud_vbo, '2f 2f', 'in_pos', 'in_uv')], self._hud_ibo)
        self.left_hud_tex = ctx.texture((HUD_W, WIN_H), 4)
        self.left_hud_tex.filter = moderngl.LINEAR, moderngl.LINEAR

        pygame.font.init()
        self.font_sm=pygame.font.SysFont("Courier New",13,bold=False)
        self.font_md=pygame.font.SysFont("Courier New",15,bold=True)
        self.hud_surf=pygame.Surface((HUD_W,WIN_H),pygame.SRCALPHA)
        self.left_hud_surf=pygame.Surface((HUD_W,WIN_H),pygame.SRCALPHA)

        self._hud_tick = 0
        # ── Target path lines (GL_LINES, one line per creature) ──
        self.path_prog = ctx.program(vertex_shader=_PATH_VERT, fragment_shader=_PATH_FRAG)
        self.path_buf  = ctx.buffer(reserve=MAX_POP * 2 * 6 * 4)
        self.path_vao  = ctx.vertex_array(self.path_prog,
                             [(self.path_buf, '3f 3f', 'in_pos', 'in_color')])
        self.paths_btn_rect = None

        self.light_dir=np.array([-0.4,-1.0,-0.3],np.float32)
        self.light_dir/=np.linalg.norm(self.light_dir)
        self._light_bytes=self.light_dir.tobytes()
        self.species_log_scroll = 0
        self.extinction_log_scroll = 0

        # ── Creature and Food VAOs ──
        self.cube_verts, self.cube_indices = _make_cube(1.0)
        self.cube_vbo = ctx.buffer(self.cube_verts.tobytes())
        self.cube_ebo = ctx.buffer(self.cube_indices.tobytes())

        self.cylinder_verts, self.cylinder_indices = _make_cylinder(segments=16, radius=0.5, height=1.0)
        self.cylinder_vbo = ctx.buffer(self.cylinder_verts.tobytes())
        self.cylinder_ebo = ctx.buffer(self.cylinder_indices.tobytes())

        self.cone_verts, self.cone_indices = _make_cone(segments=16, radius=0.5, height=1.0)
        self.cone_vbo = ctx.buffer(self.cone_verts.tobytes())
        self.cone_ebo = ctx.buffer(self.cone_indices.tobytes())

        self.omnivore_vao = ctx.vertex_array(self.sp_prog, [
            (self.sphere_vbo, '3f 3f', 'in_pos', 'in_norm'),
            (self.inst_buf, '3f 3f 3f/i', 'inst_pos', 'inst_color', 'inst_scale'),
        ], self.sphere_ebo)

        self.pred_vao = ctx.vertex_array(self.sp_prog, [
            (self.cylinder_vbo, '3f 3f', 'in_pos', 'in_norm'),
            (self.inst_buf, '3f 3f 3f/i', 'inst_pos', 'inst_color', 'inst_scale'),
        ], self.cylinder_ebo)

        self.apex_vao = ctx.vertex_array(self.sp_prog, [
            (self.cone_vbo, '3f 3f', 'in_pos', 'in_norm'),
            (self.inst_buf, '3f 3f 3f/i', 'inst_pos', 'inst_color', 'inst_scale'),
        ], self.cone_ebo)

        self.micro_vao = ctx.vertex_array(self.sp_prog, [
            (self.cube_vbo, '3f 3f', 'in_pos', 'in_norm'),
            (self.inst_buf, '3f 3f 3f/i', 'inst_pos', 'inst_color', 'inst_scale'),
        ], self.cube_ebo)

        self.food_vao = ctx.vertex_array(self.sp_prog, [
            (self.sphere_vbo, '3f 3f', 'in_pos', 'in_norm'),
            (self.food_inst_buf, '3f 3f 3f/i', 'inst_pos', 'inst_color', 'inst_scale'),
        ], self.sphere_ebo)

    # ── pack instance buffer ──
    @staticmethod
    def _pack(creatures_or_food, scale_fn, color_fn, height_fn):  # packs position, color, and scale into a flat float32 array for GPU instanced draw calls
        if not creatures_or_food: return None
        n = len(creatures_or_food)
        arr = np.empty((n, 9), dtype=np.float32)
        arr[:, 0] = [c.x for c in creatures_or_food]
        arr[:, 1] = [height_fn(c) for c in creatures_or_food]
        arr[:, 2] = [c.y for c in creatures_or_food]
        arr[:, 3:6] = [color_fn(c) for c in creatures_or_food]
        arr[:, 6:9] = [scale_fn(c) for c in creatures_or_food]
        return arr.ravel()

    def render(self, world: World, camera: OrbitCamera, fps: float, show_paths: bool = False):  # full frame: terrain, water pools, food spheres, per-species creature meshes, optional path lines, then HUD
        self.ctx.clear(0.60, 0.75, 0.85, 1.0)
        mvp=camera.get_mvp(WIN_W,WIN_H)
        mvp_bytes=mvp.tobytes()

        self.terr_prog['u_mvp'].write(mvp_bytes)
        self.terr_vao.render(moderngl.TRIANGLES)

        pools=world.water_pools
        if pools:
            pd=np.array([[w.x,0.,w.y,w.radius] for w in pools],np.float32)
            self.water_inst_buf.write(pd.tobytes())
            self.water_prog['u_mvp'].write(mvp_bytes)
            self.water_vao.render(moderngl.TRIANGLE_FAN,instances=len(pools))

        self.sp_prog['u_mvp'].write(mvp_bytes)
        self.sp_prog['u_light'].write(self._light_bytes)

        alive_food=world.foods
        if alive_food:
            def fcol(f):
                if f.needs_pollination: return FOOD_POLL_RGB
                return FOOD_RIPE_RGB if f.growth>=1.0 else FOOD_UNRIPE_RGB
            fsc=lambda f: (2.5, 2.5, 2.5) if f.growth>=1.0 else (1.5+f.growth, 1.5+f.growth, 1.5+f.growth)
            fht=lambda f: 0.5+f.growth*1.5
            fd=self._pack(alive_food,fsc,fcol,fht)
            self.food_inst_buf.orphan(len(alive_food)*(3+3+3)*4)
            self.food_inst_buf.write(fd.tobytes())
            self.food_vao.render(moderngl.TRIANGLES,instances=len(alive_food))

        alive_creatures = world.creatures
        if alive_creatures:
            def ccol(c): return c.color_rgb
            cht=lambda c: c.display_y

            creatures_by_kind = {
                "omnivore": [],
                "pred": [],
                "apex": [],
                "micro": []
            }
            for c in alive_creatures:
                creatures_by_kind[c.kind].append(c)

            omnivore_creatures = creatures_by_kind["omnivore"]
            if omnivore_creatures:
                csc_omnivore = lambda c: (c.radius * 1.5, c.radius * 1.5, c.radius * 1.5)
                omnivore_data = self._pack(omnivore_creatures, csc_omnivore, ccol, cht)
                self.inst_buf.orphan(len(omnivore_creatures)*(3+3+3)*4)
                self.inst_buf.write(omnivore_data.tobytes())
                self.omnivore_vao.render(moderngl.TRIANGLES, instances=len(omnivore_creatures))

            pred_creatures = creatures_by_kind["pred"]
            if pred_creatures:
                csc_pred = lambda c: (c.radius * 1.3, c.radius * 2.2, c.radius * 1.3)
                pred_data = self._pack(pred_creatures, csc_pred, ccol, cht)
                self.inst_buf.orphan(len(pred_creatures)*(3+3+3)*4)
                self.inst_buf.write(pred_data.tobytes())
                self.pred_vao.render(moderngl.TRIANGLES, instances=len(pred_creatures))

            apex_creatures = creatures_by_kind["apex"]
            if apex_creatures:
                csc_apex = lambda c: (c.radius * 1.8, c.radius * 2.5, c.radius * 1.8)
                apex_data = self._pack(apex_creatures, csc_apex, ccol, cht)
                self.inst_buf.orphan(len(apex_creatures)*(3+3+3)*4)
                self.inst_buf.write(apex_data.tobytes())
                self.apex_vao.render(moderngl.TRIANGLES, instances=len(apex_creatures))

            micro_creatures = creatures_by_kind["micro"]
            if micro_creatures:
                csc_micro = lambda c: (c.radius * 1.0, c.radius * 1.0, c.radius * 1.0)
                micro_data = self._pack(micro_creatures, csc_micro, ccol, cht)
                self.inst_buf.orphan(len(micro_creatures)*(3+3+3)*4)
                self.inst_buf.write(micro_data.tobytes())
                self.micro_vao.render(moderngl.TRIANGLES, instances=len(micro_creatures))

        if show_paths and alive_creatures:
            self._render_paths(alive_creatures, mvp_bytes)

        self._draw_hud(world, fps, show_paths)

    def _render_paths(self, creatures, mvp_bytes):  # draws GL_LINES from each creature to its current navigation target
        valid_c = [c for c in creatures if c.target_x is not None]
        if not valid_c: return
        n = len(valid_c)
        xs  = np.array([c.x         for c in valid_c], np.float32)
        ys  = np.array([c.display_y for c in valid_c], np.float32)
        zs  = np.array([c.y         for c in valid_c], np.float32)
        txs = np.array([c.target_x  for c in valid_c], np.float32)
        tys = np.array([c.target_y  for c in valid_c], np.float32)
        clr = np.array([c.color_rgb for c in valid_c], np.float32)
        ones = np.ones(n, np.float32)
        starts = np.column_stack([xs, ys, zs, clr])
        ends   = np.column_stack([txs, ones, tys, clr])
        data = np.empty((n * 2, 6), np.float32)
        data[0::2] = starts
        data[1::2] = ends
        n_verts = n * 2
        self.path_buf.orphan(n_verts * 6 * 4)
        self.path_buf.write(data.tobytes())
        self.path_prog['u_mvp'].write(mvp_bytes)
        self.ctx.line_width = 1.5
        self.ctx.enable(moderngl.BLEND)
        self.path_vao.render(moderngl.LINES, vertices=n_verts)

    def _draw_hud(self, world: World, fps: float, show_paths: bool = False):  # renders right HUD (stats, designer, species log) and left HUD (extinction log, controls) onto pygame surfaces then uploads as textures
        s=self.hud_surf; s.fill((15,18,25,195))
        lc=world._last_counts
        y=10; pad=8
        def txt(text,row,col=(200,220,200),bold=False):  # blits a text line onto the right HUD surface and returns the next y offset
            f=self.font_md if bold else self.font_sm
            surf=f.render(text,True,col); s.blit(surf,(pad,row))
            return row+surf.get_height()+2

        y=txt("NEURAL ECOSYSTEM 3D",y,(130,200,255),bold=True)
        y=txt(f"Tick:{world.tick:>6}  FPS:{fps:>5.1f}",y)

        # ── Play/Pause toggle button ──
        y += 8
        is_running = world.running
        p_bg   = (40, 60, 100) if is_running else (100, 40, 40)
        p_bord = (100, 150, 255) if is_running else (255, 100, 100)
        p_txt  = (200, 220, 255) if is_running else (255, 200, 200)
        p_rect = pygame.Rect(pad, y, HUD_W - pad*2, 30)
        pygame.draw.rect(s, p_bg,   p_rect, border_radius=5)
        pygame.draw.rect(s, p_bord, p_rect, 1, border_radius=5)
        icon = "⏸" if is_running else "▶"
        p_label = self.font_md.render(f" {icon}  {'RUNNING' if is_running else 'PAUSED'}", True, p_txt)
        s.blit(p_label, (p_rect.x + 6, p_rect.y + (p_rect.height - p_label.get_height()) // 2))
        self.pause_btn_rect = p_rect
        y += p_rect.height + 8

        # ── Paths toggle button ──
        btn_on   = show_paths
        btn_bg   = (30, 80, 50)  if btn_on else (40, 45, 55)
        btn_bord = (80, 200,110) if btn_on else (80, 90,110)
        btn_txt  = (120,240,150) if btn_on else (160,170,185)
        btn_rect = pygame.Rect(pad, y, HUD_W - pad*2, 24)
        pygame.draw.rect(s, btn_bg,   btn_rect, border_radius=5)
        pygame.draw.rect(s, btn_bord, btn_rect, 1, border_radius=5)
        dot = "●" if btn_on else "○"
        label_surf = self.font_sm.render(f"  {dot}  TARGET PATHS  [P]", True, btn_txt)
        s.blit(label_surf, (btn_rect.x + 6, btn_rect.y + (btn_rect.height - label_surf.get_height()) // 2))
        self.paths_btn_rect = btn_rect
        y += btn_rect.height + 8

        # ── CUSTOM SPECIES CREATOR ──
        y += 4
        y = txt("── DESIGNER ──", y, (200, 200, 100), bold=True)
        
        # 1. Kind Selection
        kinds = ["omnivore", "pred", "micro", "apex"]
        self.kind_btn_rects = []
        bw = (HUD_W - pad*2) // 4
        for i, k in enumerate(kinds):
            r = pygame.Rect(pad + i*bw, y, bw-2, 22)
            self.kind_btn_rects.append(r)
            active = (i == world.selected_custom_kind)
            bg = (60, 60, 80) if active else (30, 30, 40)
            pygame.draw.rect(s, bg, r, border_radius=3)
            if active: pygame.draw.rect(s, (200, 200, 100), r, 1, border_radius=3)
            label = self.font_sm.render(k[:1].upper(), True, (255,255,255))
            s.blit(label, (r.centerx - label.get_width()//2, r.centery - label.get_height()//2))
        y += 28

        # 2. RGB Color Sliders
        self.rgb_slider_rects = []
        preview_rect = pygame.Rect(pad, y, 40, 58)
        cur_rgb_255 = tuple(int(c*255) for c in world.custom_color_rgb)
        pygame.draw.rect(s, cur_rgb_255, preview_rect, border_radius=4)
        pygame.draw.rect(s, (150, 150, 150), preview_rect, 1, border_radius=4)

        slider_x = pad + 72
        slider_w = HUD_W - slider_x - pad
        for i, (label, val, col) in enumerate([("R", world.custom_color_rgb[0], (255, 100, 100)), 
                                              ("G", world.custom_color_rgb[1], (100, 255, 100)), 
                                              ("B", world.custom_color_rgb[2], (100, 100, 255))]):
            sy = y + i * 20
            r = pygame.Rect(slider_x, sy + 8, slider_w, 6)
            self.rgb_slider_rects.append(r)
            pygame.draw.rect(s, (40, 40, 50), r, border_radius=3)
            fill_rect = pygame.Rect(r.x, r.y, int(r.width * val), r.height)
            pygame.draw.rect(s, col, fill_rect, border_radius=3)
            l_surf = self.font_sm.render(label, True, (200, 200, 200))
            s.blit(l_surf, (slider_x - 18, sy + 2))
        y += 64

        # 3. Name & Randomize
        self.name_btn_rect = pygame.Rect(pad, y, HUD_W - pad*2, 22)
        bg_col = (60, 80, 110) if world.editing_name else (40, 45, 55)
        pygame.draw.rect(s, bg_col, self.name_btn_rect, border_radius=4)
        if world.editing_name: pygame.draw.rect(s, (130, 200, 255), self.name_btn_rect, 1, border_radius=4)
        
        display_name = world.custom_name + ("_" if world.editing_name and (time.time()*2)%2 > 1 else "")
        name_txt = self.font_sm.render(f"Name: {display_name}", True, (255, 255, 255) if world.editing_name else (200, 200, 200))
        s.blit(name_txt, (self.name_btn_rect.x + 6, self.name_btn_rect.y + 4))
        rand_label = self.font_sm.render("RAND", True, (130, 200, 255))
        s.blit(rand_label, (self.name_btn_rect.right - 45, self.name_btn_rect.y + 4))
        y += 28

        # 4. Stats (Speed & Age & Count)
        self.stat_btn_rects = []
        stat_cfg = [("SPD", world.custom_speed_mul, True), ("AGE", world.custom_age_mul, True), ("NUM", world.custom_spawn_count, False)]
        for i, (label, val, is_float) in enumerate(stat_cfg):
            val_str = f"{val:.1f}x" if is_float else str(val)
            txt(f"{label}: {val_str}", y)
            r_minus = pygame.Rect(pad + 100, y, 22, 18)
            r_plus = pygame.Rect(pad + 130, y, 22, 18)
            self.stat_btn_rects.extend([r_minus, r_plus])
            for r, char in [(r_minus, "-"), (r_plus, "+")]:
                pygame.draw.rect(s, (50, 55, 70), r, border_radius=3)
                l = self.font_sm.render(char, True, (255,255,255))
                s.blit(l, (r.centerx-l.get_width()//2, r.centery-l.get_height()//2))
            y += 22

        # 5. Spawn Button
        self.custom_spawn_btn_rect = pygame.Rect(pad, y, HUD_W - pad*2, 26)
        pygame.draw.rect(s, (50, 50, 100), self.custom_spawn_btn_rect, border_radius=5)
        pygame.draw.rect(s, (100, 100, 255), self.custom_spawn_btn_rect, 1, border_radius=5)
        lab = self.font_sm.render("SPAWN CUSTOM SPECIES", True, (220, 220, 255))
        s.blit(lab, (self.custom_spawn_btn_rect.centerx - lab.get_width()//2, y + (26-lab.get_height())//2))
        y += 32

        y+=6; y=txt("── POPULATION ──",y,(180,180,180),bold=True)
        for k,label,col in [("omnivore","Omnivore","#4CAF50"),("pred","Predator","#F44336"),
                              ("micro","Micro","#FF9800"),("apex","Apex","#9C27B0")]:
            c3=tuple(int(col.lstrip("#")[i:i+2],16) for i in(0,2,4))
            bar=int(lc[k]/max(1,TARGET_OMNIVORE)*80)
            pygame.draw.rect(s,(40,50,40),(pad,y+2,80,12),border_radius=2)
            pygame.draw.rect(s,c3,(pad,y+2,min(bar,HUD_W-pad*2-50),12),border_radius=2)
            y=txt(f"{label:<10}{lc[k]:>4}",y+14,c3)

        y+=6; y=txt("── SPECIES LOG ──",y,(180,180,180),bold=True)
        evs=getattr(world,"species_events",[])
        for ev in reversed(evs[-6:]):
            c3=tuple(int(ev["color"].lstrip("#")[i:i+2],16) for i in(0,2,4))
            y=txt(f"T{ev['tick']:>5} {ev['name'][:14]} ×{ev['n']}",y,c3)
        
        species_log_area_rect = pygame.Rect(pad, y, HUD_W - pad*2, 180)
        self.species_log_rect = species_log_area_rect
        
        species_log_surf = pygame.Surface((species_log_area_rect.width, species_log_area_rect.height), pygame.SRCALPHA)
        
        evs = getattr(world, "species_events", [])
        item_height = self.font_sm.get_height() + 2
        total_log_height = len(evs) * item_height
        max_scroll_offset_species = max(0, total_log_height - species_log_area_rect.height)
        self.species_log_scroll = max(0, min(self.species_log_scroll, max_scroll_offset_species))

        current_item_y = 0 - self.species_log_scroll
        for ev in reversed(evs):
            if current_item_y + item_height > 0 and current_item_y < species_log_area_rect.height:
                c3 = tuple(int(ev["color"].lstrip("#")[i:i+2],16) for i in(0,2,4))
                label_surf = self.font_sm.render(f"T{ev['tick']:>5} {ev['name'][:14]} ×{ev['n']}", True, c3)
                species_log_surf.blit(label_surf, (0, current_item_y))
            current_item_y += item_height
        s.blit(species_log_surf, (species_log_area_rect.x, species_log_area_rect.y))
        y = species_log_area_rect.bottom + 6

        # ── Left HUD ──
        sl = self.left_hud_surf; sl.fill((15,18,25,195))
        def txt_l(text, row, col=(200,220,200), bold=False):  # blits a text line onto the left HUD surface and returns the next y offset
            f = self.font_md if bold else self.font_sm
            surf = f.render(text, True, col); sl.blit(surf, (pad, row))
            return row + surf.get_height() + 2

        ly = 10
        ly = txt_l("── EXTINCTIONS ──", ly, (220, 100, 100), bold=True)
        extinction_log_area_rect = pygame.Rect(pad, ly, HUD_W - pad * 2, 250)
        self.extinction_log_rect = extinction_log_area_rect

        extinction_log_surf = pygame.Surface((extinction_log_area_rect.width, extinction_log_area_rect.height), pygame.SRCALPHA)
        ext_evs = getattr(world, "extinction_log", [])
        total_ext_log_height = len(ext_evs) * item_height
        max_scroll_offset_extinction = max(0, total_ext_log_height - extinction_log_area_rect.height)
        self.extinction_log_scroll = max(0, min(self.extinction_log_scroll, max_scroll_offset_extinction))

        current_item_y = 0 - self.extinction_log_scroll
        for ev in reversed(ext_evs):
            if current_item_y + item_height > 0 and current_item_y < extinction_log_area_rect.height:
                hex_c = ev["color"]
                c3 = tuple(int(hex_c.lstrip("#")[i:i+2],16) for i in(0,2,4))
                k_name = KIND_NAMES.get(ev.get("kind", ""), "Unknown")
                label_surf = self.font_sm.render(f"T{ev['tick']:>5} {ev['name'][:10]} {k_name}", True, c3)
                extinction_log_surf.blit(label_surf, (0, current_item_y))
            current_item_y += item_height
        sl.blit(extinction_log_surf, (extinction_log_area_rect.x, extinction_log_area_rect.y))

        controls_start_y = WIN_H - 110
        txt_l("── CONTROLS ──", controls_start_y, (150,150,150), bold=True); controls_start_y += 18
        for line in ["SPACE pause  R reset","P target paths  E event","Mouse drag: orbit","Scroll: zoom"]:
            controls_start_y = txt_l(line, controls_start_y, (120,130,120))

        self._hud_tick += 1
        if self._hud_tick % 4 == 0:
            raw=pygame.image.tobytes(s,"RGBA",False)
            self.hud_tex.write(raw)
            l_raw=pygame.image.tobytes(self.left_hud_surf,"RGBA",False)
            self.left_hud_tex.write(l_raw)
            
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.hud_prog['u_tex']=0
        self.hud_tex.use(0)
        self.hud_vao.render(moderngl.TRIANGLES)
        self.left_hud_tex.use(0)
        self.left_hud_vao.render(moderngl.TRIANGLES)
        self.ctx.enable(moderngl.DEPTH_TEST)

# ── Application ───────────────────────────────────────────────────────────────
class App:  # main application; owns the pygame window, world, camera, and renderer
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Neural Ecosystem Simulator 3D")
        pygame.display.set_mode((WIN_W,WIN_H), pygame.OPENGL|pygame.DOUBLEBUF|pygame.HWSURFACE)
        self.ctx=moderngl.create_context()
        self.world=World()
        self.camera=OrbitCamera()
        self.renderer=Renderer3D(self.ctx)
        self.clock=pygame.time.Clock()
        self.show_paths=False
        self._fps=0.0; self._fc=0; self._ft=time.time()

    def run(self):
        while True:
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                self.camera.handle_event(ev)
                if ev.type==pygame.KEYDOWN:
                    if self.world.editing_name:
                        if ev.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                            self.world.editing_name = False
                        elif ev.key == pygame.K_BACKSPACE:
                            self.world.custom_name = self.world.custom_name[:-1]
                        elif len(self.world.custom_name) < 16:
                            if ev.unicode.isprintable():
                                self.world.custom_name += ev.unicode
                        continue

                    if ev.key==pygame.K_SPACE:
                        self.world.running=not self.world.running
                    elif ev.key==pygame.K_r:
                        self.world.reset()
                    elif ev.key==pygame.K_e:
                        self.world._species_event()
                    elif ev.key==pygame.K_p:
                        self.show_paths=not self.show_paths
                if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
                    mx, my = ev.pos
                    if mx >= WIN_W - HUD_W:
                        lx, ly = mx - (WIN_W - HUD_W), my
                        btn_p = self.renderer.paths_btn_rect
                        if btn_p and btn_p.collidepoint(lx, ly):
                            self.show_paths = not self.show_paths
                        btn_s = self.renderer.pause_btn_rect
                        if btn_s and btn_s.collidepoint(lx, ly):
                            self.world.running = not self.world.running
                        for i, r in enumerate(self.renderer.kind_btn_rects):
                            if r.collidepoint(lx, ly):
                                self.world.selected_custom_kind = i
                        btn_c = self.renderer.custom_spawn_btn_rect
                        if btn_c and btn_c.collidepoint(lx, ly):
                            self.world.spawn_custom(self.world.selected_custom_kind)
                        for i, r in enumerate(self.renderer.rgb_slider_rects):
                            if r.inflate(0, 10).collidepoint(lx, ly):
                                val = max(0.0, min(1.0, (lx - r.x) / r.width))
                                self.world.custom_color_rgb[i] = val

                        nbr = self.renderer.name_btn_rect
                        if nbr and nbr.collidepoint(lx, ly):
                            if lx > nbr.right - 45:
                                self.world.custom_name = rng.choice(SPECIES_NAMES)
                                self.world.editing_name = False
                            else:
                                self.world.editing_name = not self.world.editing_name
                                if self.world.editing_name and self.world.custom_name in SPECIES_NAMES:
                                    self.world.custom_name = ""
                        else:
                            self.world.editing_name = False

                        for i, r in enumerate(self.renderer.stat_btn_rects):
                            if r.collidepoint(lx, ly):
                                if i == 0: self.world.custom_speed_mul = max(0.2, self.world.custom_speed_mul - 0.1)
                                if i == 1: self.world.custom_speed_mul = min(3.0, self.world.custom_speed_mul + 0.1)
                                if i == 2: self.world.custom_age_mul = max(0.2, self.world.custom_age_mul - 0.1)
                                if i == 3: self.world.custom_age_mul = min(4.0, self.world.custom_age_mul + 0.1)
                                if i == 4: self.world.custom_spawn_count = max(1, self.world.custom_spawn_count - 1)
                                if i == 5: self.world.custom_spawn_count = min(100, self.world.custom_spawn_count + 5)
                                self.world.custom_speed_mul = round(self.world.custom_speed_mul, 1)
                                self.world.custom_age_mul = round(self.world.custom_age_mul, 1)
                                
                elif ev.type==pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if mx >= WIN_W - HUD_W:
                        lx, ly = mx - (WIN_W - HUD_W), my
                        if self.renderer.species_log_rect and self.renderer.species_log_rect.collidepoint(lx, ly):
                            self.renderer.species_log_scroll -= ev.y * 10
                    elif mx <= HUD_W:
                        lx, ly = mx, my
                        if self.renderer.extinction_log_rect and self.renderer.extinction_log_rect.collidepoint(lx, ly):
                            self.renderer.extinction_log_scroll -= ev.y * 10

            if self.world.running:
                self.world.step()

            self._fc+=1; now=time.time()
            if now-self._ft>=1.0:
                self._fps=self._fc/(now-self._ft); self._fc=0; self._ft=now

            self.renderer.render(self.world,self.camera,self._fps,self.show_paths)
            pygame.display.flip()
            self.clock.tick(60)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__=="__main__":
    App().run()