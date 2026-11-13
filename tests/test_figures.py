"""Where the zoom of a figure falls, and that panels at different M show the same pixels."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from neurosr.ops import plot_ax_roi_norm
from neurosr.visualize import auto_roi


def test_zoom_prefers_dense_fine_structure_to_a_strong_coarse_edge():
    M, height, width = 2, 120, 160
    rows, columns = np.mgrid[:height * M, :width * M]
    iwe = np.zeros((height * M, width * M))
    # Strong bars eight native pixels wide on the left, weak bars half a native pixel wide on the
    # right.  Without the square root the strong bars carry more energy above the sensor Nyquist.
    coarse = (rows < 60 * M) & (columns < 60 * M)
    fine = (rows >= 50 * M) & (rows < 100 * M) & (columns >= 90 * M) & (columns < 150 * M)
    iwe[coarse] = np.where((columns[coarse] // (8 * M)) % 2 == 0, 8.0, -8.0)
    iwe[fine] = np.where(columns[fine] % 2 == 0, 1.0, -1.0)
    (x0, x1), (y1, y0) = auto_roi(iwe, M)
    assert (x1 - x0, y1 - y0) == (60, 45)
    assert 90 <= x0 and x1 <= 150 and 50 <= y0 and y1 <= 100


def test_panels_at_different_sampling_show_the_same_native_pixels():
    roi = np.asarray([[10, 70], [55, 10]])
    figure, axes = plt.subplots(1, 2)
    for axis, M in zip(axes, (1, 3)):
        plot_ax_roi_norm(axis, np.zeros((260 * M, 346 * M)), roi=roi, M=M)
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        # Pixel k of an image is drawn from k - 0.5 to k + 0.5.
        assert (left + 0.5, right + 0.5) == (10 * M, 70 * M)
        assert (top + 0.5, bottom + 0.5) == (10 * M, 55 * M)
    plt.close(figure)


@pytest.mark.parametrize("sensor", [(24, 32), (2, 2), (200, 20), (20, 200)])
def test_zoom_fits_images_smaller_than_the_default_window(sensor):
    M = 2
    (x0, x1), (y1, y0) = auto_roi(np.ones((sensor[0] * M, sensor[1] * M)), M)
    assert 0 <= x0 < x1 <= sensor[1]
    assert 0 <= y0 < y1 <= sensor[0]
