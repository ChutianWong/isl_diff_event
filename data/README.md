# Data

The bundled windows are extracts of the public dataset
[10.6084/m9.figshare.33545614](https://doi.org/10.6084/m9.figshare.33545614), which is
distributed under CC BY 4.0. Each is 100 ms of DAVIS346 events (260 × 346 pixels) starting with an
APS exposure. Nothing is rescaled: events keep their sensor coordinates and frames their 8-bit
values. The earthworm recording is a colour recording; its frames are converted to grey.

| Window | Archive | Recording | SHA-256 of the recording | APS frame | Window start |
|---|---|---|---|---|---|
| `joint/earthworm.npz` | `01_microscopy.zip` | `microscopy_01/recording_012.aedat4` | `24b9f090f8611fcac9d38e0755f4be013113f15e6a66cbbf49777ab97253f12f` | 7 | 371,132 µs |
| `joint/testis.npz` | `03_holography.zip` | `holography/recording_003.aedat4` | `3c75122ec4a43242bb44362dba928eefe452767f8a25cf463ffb38b57ca12362` | 69 | 2,783,248 µs |
| `holography/usaf_veryclose` | `03_holography.zip` | `holography/recording_001.aedat4` | `1e0732d5c535f24d7a29d81056de521ec0fb2ee4ac933b235e821963647adbed` | 82 | 3,276,296 µs |

Frame indices are zero-based. The window start is the exposure start of that frame, in
microseconds after the first event of the recording. The checksums are those of `SHA256SUMS` in
each archive.

## Two-exposure windows (`python -m neurosr joint`)

One NPZ file, written by `tools/make_joint_window.py`.

| Key | Type and shape | Content |
|---|---|---|
| `xs`, `ys` | int16, `[N]` | event column and row |
| `ts` | int64, `[N]` | event time in microseconds after the exposure start of the first frame, sorted, 0 to 100,000 |
| `ps` | int8, `[N]` | polarity, 1 for an increase of intensity and 0 for a decrease |
| `frames` | uint8, `[2, 260, 346]` | the selected APS frame and the next one, grey values 0 to 255 |
| `exposures_us` | int64, `[2, 2]` | start and end of the two exposures on the clock of `ts` |
| `metadata` | JSON string | scene, archive, recording and its SHA-256, frame index, exposure start, window length |

The first exposure is the frame record. The second is used only to fit the contrast threshold
between the two exposure centres.

## Acquisition directories (`python -m neurosr`)

```
data/<group>/<name>/
    config.yaml            measurement facts; required
    config_neurosr.yaml    solver settings; optional, every key optional
    raw/window.npz         the frame and the events
    raw/window.json        name and SHA-256 of the recording, the selected frame
```

`raw/window.npz`:

| Key | Type and shape | Content |
|---|---|---|
| `xs`, `ys` | int16, `[N]` | event column and row |
| `ts` | int64, `[N]` | event time in microseconds after the first event of the window, sorted |
| `ps` | int8, `[N]` | polarity, 1 for an increase of intensity and 0 for a decrease |
| `frame` | uint8, `[260, 346]` | the APS frame, grey values 0 to 255 |
| `frame_ind` | integer | zero-based index of the frame in the recording |
| `start`, `end` | integer | exposure start and end, microseconds after the first event of the recording |
| `win_frame` | integer | exposure duration in microseconds, `end - start` |
| `win` | integer | window length in microseconds |

The solver takes the first event of the window as the exposure start. `raw/window.json` records
the time between the two as `first_event_offset_us`; it is 0 for the bundled window.
`config.yaml` repeats the facts of the window with the sensor shape, the wavelength and the pixel
pitch, and the loader refuses a window that disagrees with it. `config_neurosr.yaml` overrides
solver defaults; `_template/config_neurosr.yaml` lists every key with its default and meaning.

## Regenerating the bundled windows

Download and extract `01_microscopy.zip` and `03_holography.zip` from the dataset, then pass the
path of the recording:

```bash
python -m pip install dv opencv-python
python tools/make_joint_window.py earthworm path/to/microscopy_01/recording_012.aedat4 earthworm.npz
python tools/make_joint_window.py testis path/to/holography/recording_003.aedat4 testis.npz
python tools/make_window.py path/to/holography/recording_001.aedat4 data/holography/usaf_again
```

The two NPZ files and `data/holography/usaf_again/raw/window.npz` are identical, byte for byte,
to the bundled ones.

## A recording of your own

The second route takes any DAVIS recording of an inline hologram in which the sample translates
and the exposure is short enough to be sharp.

```bash
python tools/make_window.py my_recording.aedat4 data/mine/sample --start-us 2.5e6
python -m neurosr data/mine/sample --output results/sample
```

`make_window.py` selects the last frame whose exposure starts before `--start-us`, cuts
`--window-us` of events from that exposure start (100 ms by default) and writes the acquisition
directory, `config.yaml` included. Pass `--wavelength-m` and `--pixel-pitch-m` if they differ from
632.8 nm and 18.5 µm. Copy `_template/config_neurosr.yaml` into the directory to change M, the
autofocus range, the number of transport segments or the iteration counts. The transport model is
one trajectory for the whole field of view, so the motion in the window must be a translation.

The first route is tied to its two presets. To try it on another recording, add an entry to
`RECORDINGS` in `tools/make_joint_window.py` and to `PRESETS` in `src/neurosr/joint_demo.py`. The
window length, the number of segments and, for a complex field, the wavelength, the pixel pitch and
the autofocus range of 15 to 21 mm are constants of that route.

`dv` (AGPLv3) is needed only to read AEDAT4 in the two tools. The package itself reads NPZ.
