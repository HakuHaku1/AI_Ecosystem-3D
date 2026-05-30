# ── Species Configuration & Visual Assets ──

# Internal kind ID mapped to Display Names
KIND_NAMES = {
    "omnivore": "Omnivore",
    "pred": "Pred",
    "micro": "Micro",
    "apex": "Apex"
}

# Default colors for the primary species
SPECIES_COLORS = {
    "omnivore": "#4ade80",   # Light Green
    "pred": "#ef4444",   # Red
    "apex": "#f97316",   # Orange
    "micro": "#a78bfa"   # Purple
}

# Environmental and state-based colors
ENVIRONMENT_COLORS = {
    "theme_bg": "#0d1117",
    "theme_fg": "#e6edf3",
    "canvas_bg": "#0b0f17",
    "food_immature": "#d9f99d",
    "food_mature": "#16a34a",
    "food_pollinate": "#ca8a04",
    "egg_standard": "#fef3c7",
    "egg_predator": "#fca5a5",
    "water": "#2563eb"
}

# Names used for randomly arriving immigrant lineages
SPECIES_NAMES = [ 
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
    "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho", "Sigma", "Tau", "Upsilon",
    "Phi", "Chi", "Psi", "Omega", "Sirius", "Vega", "Altair", "Rigel", "Antares", "Spica",
    "Pollux", "Arcturus", "Betelgeuse", "Aldebaran", "Deneb", "Regulus", "Castor", "Orion", "Lyra", "Draco",
    "Cygnus", "Aquila", "Phoenix", "Hydra", "Lupus", "Volans", "Pavo", "Cetus", "Vex", "Nox",
    "Kira", "Lumen", "Strix", "Vorta", "Xylos", "Zora", "Nyx", "Eon", "Void", "Flux",
    "Apex", "Core", "Spore", "Mycel", "Helix", "Prime", "Nova", "Pulsar", "Quasar", "Nebula",
    "Zenith", "Nadir", "Rift", "Glitch", "Echo", "Prism", "Shard", "Spark", "Blaze", "Frost",
    "Storm", "Terra", "Aero", "Hydro", "Pyro", "Geo", "Chronos", "Aeon", "Titan", "Gaia",
    "Solis", "Luna", "Astra", "Cosmo", "Xenon", "Argon", "Neon", "Cobalt", "Quartz", "Onyx"
] 

# Distinct color palette for immigrant lineages
SPECIES_EVENT_COLORS = [ 
    "#c3445a", "#81c344", "#445ac3", "#c38144", "#44c381", "#8144c3", "#c34481", "#4481c3", "#81c344", "#c3445a",
    "#44c35a", "#c344c3", "#44c3c3", "#c38181", "#81c381", "#8181c3", "#c34444", "#44c344", "#4444c3", "#c3c344",
    "#44c3c3", "#c344c3", "#44c344", "#c3c344", "#4444c3", "#c34444", "#8181c3", "#81c381", "#c38181", "#44c3c3",
    "#c3445a", "#81c344", "#445ac3", "#c38144", "#44c381", "#8144c3", "#c34481", "#4481c3", "#81c344", "#c3445a",
    "#44c35a", "#c344c3", "#44c3c3", "#c38181", "#81c381", "#8181c3", "#c34444", "#44c344", "#4444c3", "#c3c344",
    "#44c3c3", "#c344c3", "#44c344", "#c3c344", "#4444c3", "#c34444", "#8181c3", "#81c381", "#c38181", "#44c3c3",
    "#c3445a", "#81c344", "#445ac3", "#c38144", "#44c381", "#8144c3", "#c34481", "#4481c3", "#81c344", "#c3445a",
    "#44c35a", "#c344c3", "#44c3c3", "#c38181", "#81c381", "#8181c3", "#c34444", "#44c344", "#4444c3", "#c3c344",
    "#44c3c3", "#c344c3", "#44c344", "#c3c344", "#4444c3", "#c34444", "#8181c3", "#81c381", "#c38181", "#44c3c3",
    "#c3445a", "#81c344", "#445ac3", "#c38144", "#44c381", "#8144c3", "#c34481", "#4481c3", "#81c344", "#c3445a"
] 

# Helper function to get color by kind
def get_species_color(kind, custom_color=None):
    if custom_color:
        return custom_color
    return SPECIES_COLORS.get(kind, "#ffffff")