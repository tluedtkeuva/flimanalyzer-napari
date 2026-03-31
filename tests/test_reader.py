import logging

from flimanalyzer_napari import napari_get_reader

log = logging.getLogger(__name__)
if __name__ == '__main__':
    import sys

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
    )
    log.debug('Logger initialized for %(name)s', {'name': __name__})


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    """An example of how you might test your plugin."""

    # # write some fake data using your supported file format
    # # we make the array an integer type to be compatible with the reader
    # my_test_file = str(tmp_path / 'myfile.npy')
    # original_data = np.random.rand(20, 20).astype(np.int_)
    # np.save(my_test_file, original_data)

    my_test_file = '../../../data/fitted_raw_image_data/800-FoV-a-V5analysis'
    reader = napari_get_reader(my_test_file)
    assert callable(reader)

    # make sure we're delivering the right format
    layer_data_list = reader(my_test_file)
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

    my_test_file = '../../../data/fitted_raw_image_data/800-FoV-a-V5analysis'
    reader = napari_get_reader(my_test_file)
    print(f'reader: {reader}')
    assert reader is not None


if __name__ == '__main__':
    test_get_reader_pass('/tmp')
    test_reader('/tmp')
