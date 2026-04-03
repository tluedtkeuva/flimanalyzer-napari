import logging
from pathlib import Path

import numpy as np

from flimanalyzer_napari import napari_get_reader

log = logging.getLogger(__name__)


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""
    log.debug(
        'Checking if path %s: is a directory with .asc files for FLIM reader.',
        tmp_path,
    )
    # # write some fake data using your supported file format
    # # we make the array an integer type to be compatible with the reader
    # my_test_file = str(tmp_path / 'myfile.npy')
    # original_data = np.random.rand(20, 20).astype(np.int_)
    # np.save(my_test_file, original_data)

    my_test_dir = Path(tmp_path) / 'ascdir'
    create_files(my_test_dir)
    reader = napari_get_reader(str(my_test_dir.absolute()))
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(str(my_test_dir.absolute()))
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # make sure it's the same as it started
    # original_data = []
    # np.testing.assert_allclose(original_data, layer_data_tuple[0])


def test_get_reader_pass(tmp_path):

    reader = napari_get_reader('fake.file')
    print(f'reader: {reader}')
    assert reader is None

    # the original_data is a float type, so the reader should return None
    # my_test_file = str("tmp_path / 'myfile.npy'")
    # original_data = np.random.rand(20, 20)
    # np.save(my_test_file, original_data)

    my_test_dir = Path(tmp_path) / 'ascdir'
    create_files(my_test_dir)
    reader = napari_get_reader(str(my_test_dir))
    print(f'reader: {reader}')
    assert reader is not None


def create_files(datadir: Path):
    # Creare directory of .asc files for testing
    # 3 time points, including the ctrl, 2 channels and 4 measures
    # 3 * 2 * 4 = 24 files
    # and 3 .sdt files
    array_size = 256

    # These values should be sorted as we expect the reader to load them
    # so that the images have the right marks.
    times = ('ctrl', 't10', 't20', 't100', 't200')
    channels = ('Ch1', 'Ch2')
    measures = ('a1', 'a1[%]', 'a2', 'chi')

    # Ensure the directory exists
    datadir.mkdir(exist_ok=True)

    # create base images (i.e. the 2 channels) as gradients in different directions
    l2r = np.full(
        (array_size, array_size),
        [x / (array_size - 1) for x in range(array_size)],
    )
    # r2l = np.fliplr(l2r.copy())
    t2b = np.rot90(l2r.copy(), 3)
    # b2t = np.flipud(t2b.copy())
    channel_bases = {
        channels[0]: l2r,
        channels[1]: t2b,
    }

    def mark_measure(a: np.ndarray, measure: int):
        if measure == 0:
            return a.copy()
        k = int(np.ceil(array_size / 2) - array_size // 8)
        if measure == 1:
            return np.tril(a, k)
        if measure == 2:
            return np.triu(a, -k)
        if measure == 3:
            return np.triu(np.tril(a, k), -k)
        raise ValueError(f'Measure {measure} not recognized.')

    def mark_time(a: np.ndarray, time: int):
        if time == 0:
            return a.copy()
        width = 8
        gap = width // 2
        cp = np.copy(a)
        for t in range(time):
            offset = gap + int(t * (width + gap))
            cp[:, offset : offset + width] = 1
        return cp

    # create the files
    for i, time in enumerate(times):
        title = f'pytest-{time}_800_'
        Path(
            datadir / f'{title}.sdt'
        ).touch()  # create empty .sdt file for each time point
        for _j, channel in enumerate(channels):
            for k, measure in enumerate(measures):
                array = mark_time(
                    mark_measure(
                        channel_bases.get(
                            channel, np.zeros((array_size, array_size))
                        ),
                        k,
                    ),
                    i,
                )
                np.savetxt(
                    datadir / f'{title}-{channel}-_{measure}.asc',
                    array,
                    delimiter=' ',
                    fmt='%0.4f',
                )


if __name__ == '__main__':
    create_files(Path('ascdir'))
    # test_get_reader_pass('/tmp')
    # test_reader('/tmp')
