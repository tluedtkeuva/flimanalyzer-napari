import logging
from pathlib import Path

from fixtures import create_files

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


if __name__ == '__main__':
    create_files(Path('ascdir'))
    # test_get_reader_pass('/tmp')
    # test_reader('/tmp')
