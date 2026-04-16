"""
Scene loading utilities.
Loads RGB + required bands for index computation.
"""


def load_scene(item, aoi):
    """
    Load and clip bands needed for RGB + indices.
    Returns a dict of numpy arrays.
    """
    raise NotImplementedError