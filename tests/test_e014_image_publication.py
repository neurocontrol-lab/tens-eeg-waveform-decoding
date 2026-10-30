"""Image export preserves temporal support and rendered pixels."""
import numpy as np
from PIL import Image
from experiments.response_characterization.e014_all_electrode_repeatability.image_report import (
    raster, ROW, BG, image, save_image, PALETTE)


def test_display_raster_retains_gaps_and_spectral_centre_support():
    # One pixel per millisecond lets us test actual recorded-trigger coordinates.
    values=np.array([[1.,-1.]])
    rendered=raster(values,np.array([100.,300.]),np.array([200.,400.]),1.,False,width=1138)
    assert rendered.shape==(ROW,1138)
    assert np.all(rendered[:,50:150]==120) and np.all(rendered[:,250:350]==0)
    assert np.all(rendered[:,:50]==BG) and np.all(rendered[:,150:250]==BG)
    assert np.all(rendered[:,350:]==BG)
    # A power marker at 300 ms represents a centre cell [275,325), not an epoch
    # beginning at 50 ms; its underlying spectral window is still 500 ms long.
    power=raster(np.array([[.5]]),np.array([275.]),np.array([325.]),1.,True,width=1138)
    assert np.all(power[:,:225]==BG) and np.all(power[:,225:275]==181)
    assert np.all(power[:,275:]==BG)


def test_webp_is_lossless_for_the_rendered_color_pixels(tmp_path):
    codes=np.arange(250,dtype=np.uint8).reshape(10,25)
    original=image((25,10));original.putdata(codes.ravel())
    path=tmp_path/'all_display_colors.webp';save_image(original,path)
    with Image.open(path) as decoded:
        np.testing.assert_array_equal(np.array(decoded.convert('RGB')),PALETTE[codes])
