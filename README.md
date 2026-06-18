# 🌍 Neural Ecosystem Simulator 3D

A real-time **3D evolutionary ecosystem simulation** powered by neural networks and rendered with OpenGL. Hundreds of creatures — omnivores, predators, microbials, and apex predators — compete, evolve, and speciate across generations on a living terrain.

---

## ✨ Features

- **3D OpenGL renderer** — instanced geometry per species (spheres, cylinders, cones, cubes), GLSL shaders, orbit camera
- **Batched neural brains** — every creature runs a tiny 2-layer neural network, all evaluated in one GPU batch per tick
- **Evolutionary pressure** — creatures reproduce, mutate, and die; fittest lineages persist across species events
- **Emergent ecology** — predator/prey cycles, pollination mechanics, horizontal gene transfer for microbials
- **Live HUD** — population bars, species log, extinction log, custom species designer — all rendered as OpenGL texture overlays
- **Optional GPU acceleration** — CuPy zero-copy tensors on CUDA, DirectML support on Windows AMD/Intel

---

## 🐍 Python Version

**Python 3.8 – 3.12** supported.  
**Minimum:** Python 3.8  
**Recommended:** Python 3.10 – 3.12 (PyTorch best tested on these versions)

Windows users: **`run.bat`** will auto-download Python 3.12 if not found.

---

## 📦 Dependencies

### Required

| Package | Purpose | Install |
|---|---|---|
| `torch` | Neural networks, GPU batching, physics | `pip install torch` |
| `numpy` | Array math, distance matrices | `pip install numpy` |
| `pygame` | Window, input handling, HUD surface rendering | `pip install pygame` |
| `moderngl` | OpenGL 3.3 context, shaders, instanced rendering | `pip install moderngl` |

### Optional but Recommended

| Package | Purpose | Install |
|---|---|---|
| `cupy` | Zero-copy GPU arrays (CUDA only, faster sensing) | `pip install cupy-cuda12x` |
| `torch-directml` | GPU on Windows AMD/Intel | `pip install torch-directml` |
| `cython` + `sim_core` | Compiled simulation core | See [Cython Setup](#cython-setup-optional) |

---

## ⚙️ Installation

### Windows (Easiest)

Simply double-click **`run.bat`** — it will:
- ✅ Auto-detect or download Python 3.12 (if not found)
- ✅ Verify Python 3.8+ is installed
- ✅ Install all required dependencies (pip, torch, numpy, pygame, moderngl, cython)
- ✅ Launch the simulator

### Manual Installation (Windows, macOS, Linux)

#### 1. Clone

```bash
git clone https://github.com/your-username/neural-ecosystem-3d.git
cd neural-ecosystem-3d
```

#### 2. Verify Python (3.8 or higher)

```bash
python --version  # Should show 3.8+
```

If not installed, download from [python.org](https://www.python.org/downloads/)

#### 3. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

#### 4. Install core dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install torch numpy pygame moderngl cython
```

#### 5. PyTorch with CUDA (NVIDIA GPU - Optional)

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the right wheel for your CUDA version. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then optionally install CuPy for zero-copy GPU sensing:

```bash
pip install cupy-cuda12x   # match your CUDA version
```

#### 6. Run

```bash
python neural_system_3d.py
```

### Note: Visual C++ Build Tools (Windows)

If Cython compilation fails, install **Visual C++ Build Tools** from [Microsoft](https://visualstudio.microsoft.com/downloads/) (free), then reinstall Cython:

```bash
pip install --force-reinstall cython
```

---

## 🖥️ GPU Device Selection

The simulator auto-detects the best device at startup:

| Priority | Device | Requirement |
|---|---|---|
| 1st | CUDA (NVIDIA) | PyTorch + CUDA |
| 2nd | XPU (Intel Arc) | PyTorch XPU build |
| 3rd | DirectML (AMD/Intel on Windows) | `pip install torch-directml` |
| Fallback | CPU | Always available |

Printed on launch:
```
PyTorch device : CUDA (NVIDIA GeForce RTX 3060)
```

CuPy is used automatically when CUDA is available, enabling zero-copy tensor↔array transfers for faster distance sensing.

---

## ⚡ Cython Setup (Optional Acceleration)

Cython compiles the inner simulation loop to C for a speed boost.

**Windows users:** `run.bat` installs Cython automatically. If compilation fails, install Visual C++ Build Tools and restart.

**Manual compilation:**

```bash
pip install cython
python setup.py build_ext --inplace
```

If successful:
```
Cython acceleration: ON
```

Falls back silently to pure Python if not built.

---

## 🎮 Controls

| Input | Action |
|---|---|
| `Space` | Pause / Resume |
| `R` | Reset simulation |
| `E` | Trigger a species event immediately |
| `P` | Toggle target path lines |
| `+` / `-` | Increase / decrease simulation speed |
| **Mouse drag** | Orbit camera |
| **Scroll wheel** | Zoom in / out |
| **Left click HUD** | Interact with buttons, sliders, species designer |

---

## 🧬 Species

| Type | 3D Shape | Role |
|---|---|---|
| **Omnivore** | Sphere | Eats plants and microbials |
| **Predator** | Cylinder | Hunts omnivores |
| **Micro** | Cube | Pollinates plants; shares genes via HGT |
| **Apex** | Cone | Hunts all other species |

---

## 🖼️ Rendering Pipeline

```
Terrain (ground quad + GLSL grid)
  └── Water pools (instanced circle fans, alpha blend)
      └── Food (instanced UV spheres, growth-colored)
          └── Creatures (instanced per-species geometry)
              └── Target paths (GL_LINES, per creature)
                  └── HUD panels (Pygame → texture → fullscreen quad)
```

All creature rendering uses **GPU instancing** — one draw call per species type per frame.

---

## 📁 Project Structure

```
neural-ecosystem-3d/
├── run.bat                # Windows auto-installer & launcher
├── neural_system_3d.py    # Main app (World, Renderer3D, OrbitCamera)
├── species_config.py      # Names, hex colors, environment palette
├── setup.py               # Cython build script
├── sim_core.pyx           # Cython simulation core (optional acceleration)
└── README.md
```

---

## 📋 Requirements Summary

**Minimum Python:** 3.8+  
**Recommended:** Python 3.10 – 3.12

### Required Packages

```
python >= 3.8
torch           # Neural networks & GPU batching
numpy           # Array math
pygame          # Window & HUD rendering
moderngl        # OpenGL 3.3 rendering
cython          # Simulation core compilation
```

### Optional but Recommended

```
cupy-cuda12x       # Zero-copy GPU arrays (CUDA only, faster sensing)
torch-directml     # GPU on Windows AMD/Intel
visual-c++-build   # Windows: for Cython compilation
```

Or as `requirements.txt`:

```txt
torch
numpy
pygame
moderngl
cython
```

**Quick install on Windows:** Just run `run.bat`! It handles everything automatically.

---

## STI Pasay-edsa

Created by Hakuryu Acosta Kato
