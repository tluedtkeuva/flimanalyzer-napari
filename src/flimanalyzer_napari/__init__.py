try:
    from ._version import version as __version__
except ImportError:
    __version__ = '0.0.1'


from ._dataset import FLIMASCFile, FLIMSDTFile, FLIMSeries, FLIMSet
from ._reader import napari_get_reader
from ._widget import (
    ExampleQWidget,
    ImageThreshold,
    threshold_autogenerate_widget,
    threshold_magic_widget,
)
from ._writer import write_multiple, write_single_image

__all__ = (
    'napari_get_reader',
    'write_single_image',
    'write_multiple',
    'ExampleQWidget',
    'ImageThreshold',
    'threshold_autogenerate_widget',
    'threshold_magic_widget',
    'FLIMASCFile',
    'FLIMSDTFile',
    'FLIMSet',
    'FLIMSeries',
)
