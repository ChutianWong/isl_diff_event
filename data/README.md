# Dataset layout

## Joint presets

`tools/make_joint_window.py` extracts two consecutive APS exposures and a
100 ms event window. The NPZ stores grayscale frames, exposure intervals and
event timestamps relative to the selected exposure start. The joint solver
uses that clock to align both event terms to the exposure centre.

## Bundled holography example

`holography/usaf_veryclose` contains a 260 × 346 APS frame and 239,118 events
from a 100 ms resolution-target window. The event timestamps are in
microseconds from the first selected event, with x as the column and y as the
row. Polarity is stored as 0 or 1.

`config.yaml` describes the input. `config_neurosr.yaml` contains optional
solver settings. The bundled NPZ runs without the offline AEDAT4 reader.
Write results to a separate directory.
