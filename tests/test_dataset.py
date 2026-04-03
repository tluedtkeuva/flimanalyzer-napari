import logging

from flimanalyzer_napari import FLIMASCFile

log = logging.getLogger(__name__)


def test_sorting(tmp_path):
    files = [
        'pytest-t10_800_-Ch1-_a1[%].asc',
        'pytest-t20_800_-Ch1-_a1[%].asc',
        'pytest-t100_800_-Ch1-_a1[%].asc',
        'pytest-t180_800_-Ch1-_a1[%].asc',
        'pytest-t200_800_-Ch1-_a1[%].asc',
    ]

    asc_files = []
    for f in files:
        asc_file = FLIMASCFile.parsePath(tmp_path / f)
        asc_files.append(asc_file)
    unsorted_files = asc_files[::-1]
    log.debug('Unsorted files: %s', [f.title for f in unsorted_files])
    log.debug('Sorted files: %s', [f.title for f in sorted(unsorted_files)])
    # TODO:
    #  More extensive tests of the sorting, including edge cases like missing time points, missing measures, etc.
    #  Numerical sorting of frequency values?
    #  Sorting of SDT files.
    #  Sorting of sets and series.

    assert unsorted_files != asc_files
    assert sorted(unsorted_files) != asc_files
