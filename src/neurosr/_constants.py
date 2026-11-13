"""Sensor facts and solver defaults shared by the two routes."""
import numpy as np

# DAVIS346: rows, columns and pixel pitch; HeNe wavelength of the inline holography setup.
SENSOR = (260, 346)
PIXEL_PITCH_M = 18.5e-6
WAVELENGTH_M = 632.8e-9

SIGMA = 0.849            # Gaussian splat width of the event image, in sensor pixels across the transport
SIGMA_SR = 1.0           # the width at M > 1 for an intensity latent; a complex field keeps SIGMA
THRE = float(np.float32(1 + 1e-6))   # knee of the linear-to-log intensity response, rounded to float32
CONTRAST_0 = 0.2         # initial contrast threshold c
MIN_FLOW_MAG = 0.005     # segments that move less than this (sensor px) leave the event term

# Weights of the objective. event_sr None keeps the native event weight at the super-resolved scale;
# tv_object (complex field) and laplacian (hologram) are the object priors used with optics.
KAPPA = dict(event=2e-1, frame=1.0, sog=1e-1, bg=4e-1, tv=0.0, tv_sr=2e-2, event_sr=None,
             tv_object=0.0, laplacian=0.0)
# Learning rates; distance and distance_sr are those of the propagation distance z.
LR = dict(cmax=1e-2, image=4e-3, background=4e-3, motion=1e-4, contrast=1e-3, sr=1e-3,
          distance=1e-5, distance_sr=2e-6)
SCHEDULE = dict(cmax=(200, 0.9), joint=(200, 0.7), sr=(100, 0.7))   # StepLR (step_size, gamma)
