"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/building_a_plugin/guides.html#readers
"""

import logging
from pathlib import Path

import numpy as np

from flimanalyzer_napari._dataset import FLIMSeries

log = logging.getLogger(__name__)
if __name__ == '__main__':
    import sys

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format='%(levelname)s %(message)s',
    )
    log.debug('Logger initialized for %(name)s', {'name': __name__})


def napari_get_reader(path: str | list[str]):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    #
    # For now hadle only Directories
    #
    if isinstance(path, str):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        opath = Path(path)
        log.debug(
            'Checking if path %s: is a directory with .asc files for FLIM reader.',
            opath.absolute(),
        )
        log.debug('opath.is_dir(): %s', opath.is_dir())
        log.debug("opath.glob('*.asc'): %s", len(list(opath.glob('*.asc'))))
        if opath.is_dir() and opath.glob('*.asc'):
            return reader_function
    else:
        return None

    # # the get_reader function should make as many checks as possible
    # # (without loading the full file) to determine if it can read
    # # the path. Here, we check the dtype of the array by loading
    # # it with memmap, so that we don't actually load the full array into memory.
    # # We pretend that this reader can only read integer arrays.
    # try:
    #     arr = np.load(path, mmap_mode='r')
    #     if arr.dtype != np.int_:
    #         return None
    # # napari_get_reader should never raise an exception, because napari
    # # raises its own specific errors depending on what plugins are
    # # available for the given path, so we catch
    # # the OSError that np.load might raise if the file is malformed
    # except OSError:
    #     return None

    # # otherwise we return the *function* that can read ``path``.
    # return reader_function


def reader_function(path: str | list[str]):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # # handle both a string and a list of strings
    # paths = [path] if isinstance(path, str) else path
    # # load all files into array
    # arrays = [np.load(_path) for _path in paths]
    # # stack arrays into single array
    # data = np.squeeze(np.stack(arrays))

    # For now, only support directories
    if not isinstance(path, str):
        raise ValueError(f'Path {path} is not a single directory.')
    opath = Path(path)
    if not opath.is_dir():
        raise ValueError(f'Path {path} is not a directory.')

    allSeries = FLIMSeries.series_from_directory(opath)
    assert allSeries, f'Could not find any FLIM series in directory {opath}.'
    assert len(allSeries) == 1, (
        f'Found multiple FLIM series in directory {opath}, but only one is supported for now.'
    )

    # load all files from first (only) series into array
    series = next(iter(allSeries.values()))
    seriesData = series.load()

    sets = list(seriesData.values())
    time_dim = len(sets)  # one set per time point
    # Assume channels and measures are consistent across time points,
    # so we can determine their dimensions from the first time point's data
    set1 = sets[0]
    log.debug('Examining set 0: %s', set1.keys())
    channel_dim = len(set1['data'])
    log.debug(
        'Examining set 0 channels: %s',
        set1['data'].keys(),
    )
    channel1 = next(iter(set1['data'].values()))
    measure_dim = len(channel1)
    log.debug('Examining set 0 measures: %s', channel1.keys())
    data = np.zeros((time_dim, channel_dim, measure_dim, 1, 256, 256))
    for t, (time, timeData) in enumerate(seriesData.items()):
        log.debug('Processing time point %s with data %s', t, timeData)
        for c, (channel, channelSet) in enumerate(timeData['data'].items()):
            for m, (measure, measureData) in enumerate(channelSet.items()):
                log.debug(
                    'Assigning data[%s, %s, %s][%s, %s, %s] => %s',
                    t,
                    c,
                    m,
                    time,
                    channel,
                    measure,
                    measureData,
                )
                # TODO?: Convert t, c, m to the correct indices for the data array based on the structure of the sets and channels
                # Maybe add "channel_index" and "measure_index" methods to the FileSet/FileSeries class to help with this?
                data[t, c, m, 0, :, :] = measureData

    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = 'image'  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]
