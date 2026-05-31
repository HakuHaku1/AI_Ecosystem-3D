# 🌿 Neural Ecosystem Simulator

A real-time evolutionary ecosystem simulation powered by neural networks, where creatures (prey, predators, microbials, and apex predators) evolve and compete across generations — all running in a Tkinter GUI.

---

## 🐍 Python Version

**Python 3.9 or higher** is recommended. Python 3.10–3.12 works best for PyTorch compatibility.

---

## 📦 Required Packages

### Core (required)

| Package | Purpose | Install |
|---|---|---|
| `torch` | Neural networks & batched brain simulation | `pip install torch` |
| `numpy` | Fast distance calculations & input arrays | `pip install numpy` |
| `tkinter` | GUI (canvas, controls, panels) | Built into Python (see note below) |

### Optional but Recommended

| Package | Purpose | Install |
|---|---|---|
| `Cython` + `sim_core` | Accelerated simulation core | See [Cython Setup](#cython-setup-optional) below |
| `torch-directml` | GPU acceleration on Windows (AMD/Intel) | `pip install torch-directml` |

> **tkinter note:** On Linux/Ubuntu, tkinter may not be included. Install it with:
> ```bash
> sudo apt-get install python3-tk
> ```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/neural-ecosystem-simulator.git
cd neural-ecosystem-simulator
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch numpy
```

For PyTorch with CUDA (NVIDIA GPU support), visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your OS + CUDA version. Example for CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Running the Simulator

```bash
python main.py
```

> Replace `main.py` with whatever your entry-point file is named.

---

## ⚡ Cython Setup (Optional)

Cython compiles the simulation core to C for faster tick processing. This is optional — the simulator runs fine without it.

### Install Cython

```bash
pip install cython
```

### Build the extension

```bash
python setup.py build_ext --inplace
```

If successful, you'll see:

```
Cython acceleration: ON
```

Otherwise it falls back gracefully:

```
Cython acceleration: OFF — run: python setup.py build_ext --inplace
```

---

## 🖥️ GPU Support

The simulator auto-detects the best available device in this order:

| Priority | Device | Requirement |
|---|---|---|
| 1st | CUDA (NVIDIA) | `torch` with CUDA |
| 2nd | XPU (Intel Arc) | `torch` with XPU support |
| 3rd | DirectML (AMD/Intel on Windows) | `pip install torch-directml` |
| Fallback | CPU | Always available |

You'll see the selected device printed on startup:
```
Using device: CUDA (NVIDIA GeForce RTX 3060)
```

---

## 📁 Project Structure

```
neural-ecosystem-simulator/
├── main.py               # Entry point — World + EcosystemGUI
├── species_config.py     # Species names, colors, environment colors
├── setup.py              # Cython build script (optional)
├── sim_core.pyx          # Cython source (optional)
└── README.md
```

---

## 🧬 Species Overview

| Type | Symbol | Role |
|---|---|---|
| **Omnivore (Prey)** | Triangle | Eats plants and microbials |
| **Predator** | Hollow triangle | Hunts prey |
| **Micro** | Diamond | Pollinates plants, enables HGT gene sharing |
| **Apex** | Hexagon | Hunts all other species |

---

## 📋 Quick Requirements Summary

```
python >= 3.9
torch
numpy
tkinter (built-in or via system package)

# Optional
cython
torch-directml  # Windows AMD/Intel GPU only
```

Or create a `requirements.txt`:

```txt
torch
numpy
```

---

## 📜 License

MIT — feel free to use, modify, and share.
