# sim_core.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
# These directives make Cython as fast as C

import math
import cython
from libc.math cimport cos, sin, atan2, sqrt, fabs
from libc.stdlib cimport rand

# Typed struct for creature state — Cython needs static types to be fast
@cython.cclass
class CyCreature:
    cdef public double x, y, vx, vy, angle, speed
    cdef public double hunger, thirst, fitness
    cdef public int age, radius, alive, repro_cooldown
    cdef public str kind

# Fast boundary bounce — replaces the if/elif chain in apply_nn_output
cpdef void apply_movement(
    double[:] xs, double[:] ys,
    double[:] vxs, double[:] vys,
    double[:] angles,
    double[:] speeds,
    double[:] nn_turns,
    double[:] nn_accels,
    double world_w, double world_h,
    int[:] radii, int n
):
    cdef int i
    cdef double spd, m, new_x, new_y, final_turn
    for i in range(n):
        final_turn = nn_turns[i] * 0.28
        angles[i] += final_turn
        spd = speeds[i] * (0.55 + nn_accels[i] * 0.45)
        m = 0.78
        vxs[i] = vxs[i] * m + cos(angles[i]) * spd * (1.0 - m)
        vys[i] = vys[i] * m + sin(angles[i]) * spd * (1.0 - m)
        new_x = xs[i] + vxs[i]
        new_y = ys[i] + vys[i]
        
        if new_x < radii[i]:
            new_x = radii[i]; vxs[i] *= -0.8
        elif new_x > world_w - radii[i]:
            new_x = world_w - radii[i]; vxs[i] *= -0.8
            
        if new_y < radii[i]:
            new_y = radii[i]; vys[i] *= -0.8
        elif new_y > world_h - radii[i]:
            new_y = world_h - radii[i]; vys[i] *= -0.8
            
        xs[i] = new_x
        ys[i] = new_y

# Fast metabolism update
cpdef void apply_metabolism(
    double[:] hungers,
    double[:] speeds,
    double[:] accels,
    int[:] kinds,   # 0=prey, 1=pred, 2=micro, 3=apex
    int n
):
    cdef int i
    cdef double spd, bm
    for i in range(n):
        spd = speeds[i] * (0.55 + accels[i] * 0.45)
        if kinds[i] == 1: bm = 0.015
        elif kinds[i] == 3: bm = 0.025
        else: bm = 0.006
        hungers[i] -= (spd ** 1.5) * 0.025 + bm

# Fast angle normalization
cpdef double normalize_angle(double a):
    return (a + 3.14159265) % (2.0 * 3.14159265) - 3.14159265