import logging

from flimanalyzer_napari import FLIMASCFile, FLIMSDTFile

log = logging.getLogger(__name__)


def test_sorting(tmp_path):
    # file names are in expected time sorted order
    files = [
        'pytest-ctrl_800_.sdt',
        'pytest-ctrl_800_-Ch1-_a1[%].asc',
        'pytest-t10_800_-Ch1-_a1[%].asc',
        'pytest-t20_800_-Ch1-_a1[%].asc',
        'pytest-t100_800_-Ch1-_a1[%].asc',
        'pytest-t180_800_-Ch1-_a1[%].asc',
        'pytest-t200_800_-Ch1-_a1[%].asc',
        'pytest2-ctrl_800_.sdt',
        'pytest2-ctrl_800_-Ch1-_a1[%].asc',
        'pytest2-t10_800_-Ch1-_a1[%].asc',
    ]

    flim_files = []
    for f in files:
        asc_file = (
            FLIMASCFile.parsePath(tmp_path / f)
            if f.endswith('.asc')
            else FLIMSDTFile.parsePath(tmp_path / f)
        )
        assert asc_file is not None
        flim_files.append(asc_file)
    unsorted_files = flim_files[::-1]
    log.debug('Unsorted files: %s', [f.title for f in unsorted_files])
    log.debug('Sorted files: %s', [f.title for f in sorted(unsorted_files)])
    # TODO:
    #  More extensive tests of the sorting, including edge cases like missing time points, missing measures, etc.
    #  Start wtin individual FLIMFile objects, then move up to sets and series.
    #  Numerical sorting of frequency values?
    #  Sorting of sets and series.

    assert unsorted_files != flim_files
    assert sorted(unsorted_files) == flim_files
