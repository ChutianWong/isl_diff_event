"""Solver constants and defaults."""
import numpy as np
SENSOR = (260, 346)
SIGMA = 0.849            # splat width of the joint stage
SIGMA_SR = 1.0           # splat width of the SR stage
THRE = float(np.float32(1 + 1e-6))   # lin_log knee, rounded to float32
CONTRAST_0 = 0.2         # initial c
MIN_FLOW_MAG = 0.005     # segments below this displacement (sensor px) leave the event term
# event_sr None keeps the joint event weight at SR; tv_object (complex field) and laplacian
# (hologram) are the object priors, active with optics.
# Optional image priors are off by default.
KAPPA = dict(event=2e-1, frame=1.0, sog=1e-1, bg=4e-1, tv=0.0, tv_sr=2e-2, event_sr=None,
             tv_object=0.0, laplacian=0.0, iso_tv=0.0, iso_tv_sr=0.0, l0=0.0, l0_sr=0.0,
             flow_l0=0.0, flow_l0_sr=0.0, pnp=0.0, pnp_sr=0.0)
# DRUNet schedule of the PnP prior (defaults of Zhang et al., 2022)
PNP = dict(weight1=2.5, weight2=1.3, iter_num=16)
# distance and distance_sr are the learning rates of z at M = 1 and at SR
LR = dict(cmax=1e-2, image=4e-3, background=4e-3, motion=1e-4, contrast=1e-3, sr=1e-3,
          distance=1e-5, distance_sr=2e-6)
SCHEDULE = dict(cmax=(200, 0.9), joint=(200, 0.7), sr=(100, 0.7))   # StepLR (step_size, gamma)
