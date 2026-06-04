try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'Unknown'


from ._dataset import FLIMASCFile, FLIMSDTFile, FLIMSeries, FLIMSet
from ._reader import napari_get_reader

__all__ = (
    'napari_get_reader',
    'FLIMASCFile',
    'FLIMSDTFile',
    'FLIMSet',
    'FLIMSeries',
)
