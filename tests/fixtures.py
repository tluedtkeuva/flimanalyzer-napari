from pathlib import Path

import numpy as np


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
