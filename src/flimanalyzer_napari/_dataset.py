# %%
# imports
"""###
Structure of fitted FLIM datasets

The fitting process (with proprietary software) produces a series for data files. We care about:

    *_photons.asc
    *_chi.asc
    *_a2[%].asc
    *_a1[%].asc
    *_a2.asc
    *_a1.asc
    *_t2.asc
    *_t1.asc

From an image viewer’s perspective (like Napari) we can consider them as different channels that
we’d want to treat as different layers that can be overlayed.

There are examples in /standard/siller/Periasamy/fitted_raw_image_data. Note that in a given experiment
multiple fluorescent molecules may be imaged together. They will be labeled as channels,
like *_-Ch1-*, *_-Ch2-*, *_-Ch3-*. It is important to keep the channels separate.

Examples:

A Set: (title="a-t80_800_)
a-t80_800_.sdt
a-t80_800_-Ch1-_photons.asc     a-t80_800_-Ch2-_photons.asc
a-t80_800_-Ch3-_photons.asc     a-t80_800_-Ch1-_chi.asc
a-t80_800_-Ch2-_chi.asc	        a-t80_800_-Ch3-_chi.asc
a-t80_800_-Ch1-_a2[%].asc	    a-t80_800_-Ch2-_a2[%].asc
a-t80_800_-Ch3-_a2[%].asc       a-t80_800_-Ch1-_a1[%].asc
a-t80_800_-Ch2-_a1[%].asc   	a-t80_800_-Ch3-_a1[%].asc
a-t80_800_-Ch1-_a2.asc	        a-t80_800_-Ch2-_a2.asc
a-t80_800_-Ch3-_a2.asc          a-t80_800_-Ch1-_a1.asc
a-t80_800_-Ch2-_a1.asc	        a-t80_800_-Ch3-_a1.asc
a-t80_800_-Ch1-_t2.asc	        a-t80_800_-Ch2-_t2.asc
a-t80_800_-Ch3-_t2.asc          a-t80_800_-Ch1-_t1.asc
a-t80_800_-Ch2-_t1.asc	        a-t80_800_-Ch3-_t1.asc

A Series (just SDT files shown here): (seriesTitle="a-_800_.sdt")
a-ctrl_800_.sdt     a-t20_800_.sdt      a-t40_800_.sdt
a-t60_800_.sdt      a-t100_800_.sdt     a-t120_800_.sdt
a-t140_800_.sdt     a-t160_800_.sdt     a-t180_800_.sdt

###"""

import logging
import re
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import cast

import numpy as np

log = logging.getLogger(__name__)
log.debug('Importing FLIM dataset classes from _dataset.py')
log.warning(
    'This module is a work in progress. The API is not stable and may change without warning.'
)


# %%
class FLIMFile(ABC):
    # A single ASC or SDT file
    # The file is associated with the SDT file by its title,
    # which is a combination of the timecode and frequency information in the filename.
    # For example, for the SDT file "a-t80_800_.sdt", the associated ASC files would be those
    # that start with "a-t80_800_" and have the same timecode and frequency information in
    # their filenames.

    #
    # Class level defaults common to both SDT and ASC
    #
    frequency_re = re.compile(r'(?P<freq>\d+)')
    time_re = re.compile(r'(?P<time>ctrl|t\d+|\d+min)')

    # Title may be combination of name, timestamp and frequency, or just a name.
    # We want to be flexible in what we accept as title, but we want to make sure it doesn't capture the channel or measure information. For simplicity, we can assume that the title is everything before the channel and measure information.
    # # title_re = re.compile(f"(?P<title>(?:{time_re}|{frequency_re}|[_]|[\\w-]+)+|.+?)")
    simple_title_re = re.compile('(?P<title>.+?)')
    title_re = re.compile(
        f'(?P<title>(?:{time_re.pattern}|(?<=_){frequency_re.pattern}|.+?)+)'
    )

    def __init__(
        self, path, title, seriesTitle, timecode=None, frequency=None
    ):
        self._path = path
        self._title = title
        self._seriesTitle = seriesTitle
        self._timecode = timecode
        self._frequency = frequency

    @classmethod
    @abstractmethod
    def parsePath(
        cls, path, pathRegex=None, titleRegex=None
    ) -> 'FLIMFile|None':
        pass

    @abstractmethod
    def load(self) -> object:
        pass

    @property
    def path(self) -> Path:
        return self._path

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def seriesTitle(self) -> str | None:
        return self._seriesTitle

    @property
    def timecode(self) -> str | None:
        return self._timecode

    @property
    def frequency(self) -> str | None:
        return self._frequency

    # TODO: implement lt and eq for sorting (time/channel/measure order)
    #       Does this break equality? Consider same filename in different directory!
    def __lt__(self, other) -> bool:
        # Compare directory, seriesTitle, timecode, frequency, and if exists, channel, measure
        return (
            self._path.parent < other._path.parent
            or self._seriesTitle < other._seriesTitle
            or (
                self._timecode is not None
                and other._timecode is not None
                and self._timecode < other._timecode
            )
            or (
                self._frequency is not None
                and other._frequency is not None
                and self._frequency < other._frequency
            )
            or self.path.name < other.path.name
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, FLIMFile):
            return NotImplemented
        return self._path == other.path

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}: {repr(self._path)}'
            if self._path
            else self._path
        )


class FLIMASCFile(FLIMFile):
    #
    # Class level defaults specific to ASC
    #
    measures = sorted(
        ['photons', 'chi', 'a2[%]', 'a1[%]', 'a2', 'a1', 't2', 't1']
    )
    measures_re = re.compile(
        '(?P<measure>'
        + '|'.join(map(re.escape, sorted(measures, key=len, reverse=True)))
        + ')'
    )
    channel_re = re.compile(r'(?:-(?P<channel>Ch\d+)-)')

    asc_regex = re.compile(
        f"""
        {FLIMFile.simple_title_re.pattern} # Title
        (?:{channel_re.pattern})?   # Channel
        _{measures_re.pattern}      # Scene
        \\.asc
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        path,
        setTitle,
        seriesTitle,
        timecode=None,
        frequency=None,
        channel=None,
        measure=None,
    ):
        super().__init__(path, setTitle, seriesTitle, timecode, frequency)
        self._channel = channel
        self._measure = measure

    @classmethod
    def parsePath(
        cls, path, pathRegex=None, titleRegex=None
    ) -> 'FLIMASCFile|None':
        if pathRegex is None:
            pathRegex = cls.asc_regex
        if titleRegex is None:
            titleRegex = cls.title_re

        match = pathRegex.match(path.name)
        if match:
            title = match.group('title')
            channel = match.group('channel')
            measure = match.group('measure')
            match = titleRegex.match(title)
            if match:
                timecode = match.group('time')
                frequency = match.group('freq')
                seriesTitle = title.replace(timecode, '')
            else:
                timecode = None
                frequency = None
                seriesTitle = title
            return FLIMASCFile(
                path, title, seriesTitle, timecode, frequency, channel, measure
            )
        else:
            log.debug(
                'Filename %s does not match expected pattern for FLIM ASC files.',
                path.name,
            )
            return None

    def load(self) -> np.ndarray:
        return np.loadtxt(self._path, delimiter=' ')

    @property
    def channel(self) -> str | None:
        return self._channel

    @property
    def measure(self) -> str | None:
        return self._measure

    # TODO: implement lt and eq for sorting (time/channel/measure order)
    #       Does this break equality? Consider same filename in different directory!
    def __lt__(self, other) -> bool:
        if not isinstance(other, FLIMASCFile):
            return NotImplemented

        # Compare directory, seriesTitle, timecode, frequency, and if exists, channel, measure
        if (
            self._path.parent < other._path.parent
            or self._seriesTitle < other._seriesTitle
            or (
                self._timecode is not None
                and other._timecode is not None
                and self._timecode < other._timecode
            )
            or (
                self._frequency is not None
                and other._frequency is not None
                and self._frequency < other._frequency
            )
        ):
            return True

        if isinstance(other, FLIMSDTFile):
            return False  # ASC files should come after matching SDT file

        if (  # noqa: SIM103
            (
                self._channel is not None
                and other._channel is not None
                and self._channel < other._channel
            )
            or (
                self._measure is not None
                and other._measure is not None
                and self._measure < other._measure
            )
            or self.path.name < other.path.name
        ):
            return True

        return False


class FLIMSDTFile(FLIMFile):
    sdt_regex = re.compile(
        f"""
        {FLIMFile.simple_title_re.pattern}          # Title
        \\.sdt
        """,
        re.VERBOSE,
    )

    def __init__(
        self, path, title, seriesTitle, timecode=None, frequency=None
    ):
        super().__init__(path, title, seriesTitle, timecode, frequency)

    @classmethod
    def parsePath(
        cls, path, pathRegex=None, titleRegex=None
    ) -> 'FLIMSDTFile|None':
        log.debug('%s.parsePath %s', cls.__name__, path)
        if pathRegex is None:
            pathRegex = FLIMSDTFile.sdt_regex
        if titleRegex is None:
            titleRegex = FLIMFile.title_re

        match = pathRegex.match(path.name)
        if match:
            title = match.group('title')
            match = titleRegex.match(title)
            if match:
                timecode = match.group('time')
                frequency = match.group('freq')
                seriesTitle = title.replace(timecode, '')
            else:
                timecode = None
                frequency = None
                seriesTitle = title
            return FLIMSDTFile(path, title, seriesTitle, timecode, frequency)
        else:
            log.debug(
                'Filename %s does not match expected pattern for FLIM SDT files.',
                path.name,
            )
            return None

    @classmethod
    def findSDTforASC(
        cls, asc_file: FLIMASCFile, pathRegex=None, titleRegex=None
    ):  # -> 'FLIMSDTFile|None':
        path = asc_file.path.parent.joinpath(f'{asc_file.title}.sdt')
        if path.exists():
            return cls.parsePath(path, pathRegex, titleRegex)
        else:
            return None

    def load(self) -> dict:
        sdt_data = {}
        try:
            in_identification_section = False
            in_setup_section = False
            with open(self._path, errors='replace') as file:
                for line in file:
                    if 'BIN_PARA_BEGIN:' in line:
                        # Stop reading the file when the binary data starts
                        break

                    if '*IDENTIFICATION' in line:
                        in_identification_section = True
                        continue
                    if in_identification_section:
                        if '*END' in line:
                            in_identification_section = False
                        elif ':' in line:
                            key, value = line.split(':', 1)
                            sdt_data[key.strip()] = value.strip()
                        else:
                            log.warning(
                                'Invalid line in identification section: %s',
                                line.strip(),
                            )
                        continue

                    if '*SETUP' in line:
                        in_setup_section = True
                        sdt_data['Setup'] = []
                        continue
                    if in_setup_section:
                        if '*END' in line:
                            in_setup_section = False
                        else:
                            sdt_data['Setup'].append(line.strip())
                        continue
        except FileNotFoundError:
            log.error(
                "Error: The file '%s' was not found.",
                file_path,
            )
            # for Napari, should we rethrow the error?

        return sdt_data

    def __lt__(self, other) -> bool:
        if not isinstance(other, FLIMFile):
            return NotImplemented

        # Compare directory, seriesTitle, timecode, frequency, and if exists, channel, measure
        if (
            self._path.parent < other._path.parent
            or self._seriesTitle < other._seriesTitle
            or (
                self._timecode is not None
                and other._timecode is not None
                and self._timecode < other._timecode
            )
            or (
                self._frequency is not None
                and other._frequency is not None
                and self._frequency < other._frequency
            )
        ):
            return True

        if isinstance(other, FLIMASCFile):  # noqa: SIM103
            return True  # ASC files should come after matching SDT file

        return False


class FLIMSet:
    # A set of files associated with a single SDT file.
    # The files are associated with the SDT file by their title,
    # which is a combination of the timecode and frequency information in the filename.
    # For example, for the SDT file "a-t80_800_.sdt", the associated files would be those
    # that start with "a-t80_800_" and have the same timecode and frequency information in
    # their filenames.
    #
    # Following OME's default ordering, XYZCT.

    # Structure of data for ASC files:
    #   {Path(file): {channel: str, measure: str}}
    #
    def __init__(
        self,
        files: list[FLIMASCFile],
        sdt: FLIMSDTFile | None = None,
        ignore_missing_files: bool = False,
        ignore_missing_sdt: bool = False,
    ):
        # log.debug(
        #     'Creating FLIMSet with files: %s and sdt: %s',
        #     files,
        #     sdt,
        # )
        self._ignore_missing_files = ignore_missing_files
        self._ignore_missing_sdt = ignore_missing_sdt
        self._sdt = sdt
        if len(files) == 0 and sdt is None:
            log.info('FLIMSet created with no files or SDT file.')
            self._title = None
            self._seriesTitle = None
            self._timecode = None
            self._frequency = None
            return
        keyfile = sdt if sdt is not None else files[0]
        self._title = keyfile.title
        self._seriesTitle = keyfile.seriesTitle
        self._timecode = keyfile.timecode
        self._frequency = keyfile.frequency
        self._allFiles = cast(list[FLIMFile], files.copy())
        if sdt is not None:
            self._allFiles.append(sdt)

        self._channels = {}

        for file in files:
            log.debug('Adding file: %s', file)
            if file.channel not in self._channels:
                self._channels[file.channel] = {}
            channel = self._channels[file.channel]
            if file.measure in channel:
                # TODO: Should this be an exception?
                log.warning(
                    'File %s has the same channel and measure as %s',
                    file.title,
                    channel[file.measure].title,
                )
            channel[file.measure] = file

        # sort files, channels and measures
        self._allFiles.sort()
        self._channels = dict(sorted(self._channels.items()))
        for key, value in self._channels.items():
            self._channels[key] = dict(sorted(value.items()))

        # TODO: add checks for missing files and missing sdt file.
        if not self._ignore_missing_files:
            assert len(self._channels) > 0, (
                f'Dataset {self.title} has no files.'
            )
        if not self._ignore_missing_sdt:
            assert self._sdt is not None, (
                f'Dataset {self.title} has no sdt file.'
            )

    # TODO: Does it make sense to load data here, or
    # should the file loader walk through the series/sets to create the stacked image?
    # Ditto for FLIMSeries
    def load(self) -> dict:
        """
        Load FLIMSet data

        Returns
        -------
        dict
            A dictionary with two keys: 'data' and 'metadata'. The 'data' key
            will contain the loaded asc data in a dictionary of 2d arrays,
            and the 'metadata' key will contain the sdt metadata as a
            dictionary.
        """
        metadata = self._sdt.load() if self._sdt else {}
        data = {}
        for ch_name, channel in self._channels.items():
            if ch_name not in data:
                data[ch_name] = {}
            data_ch = data[ch_name]
            for m_name, measure in channel.items():
                data_ch[m_name] = measure.load()

        return {'data': data, 'metadata': metadata}

    @property
    def title(self) -> str | None:
        return self._title

    @property
    def seriesTitle(self) -> str | None:
        return self._seriesTitle

    @property
    def channels(self):
        return self._channels

    # @property
    # def measures(self):
    #     return self._measures

    @property
    def timecode(self):
        return self._timecode

    @property
    def frequency(self):
        return self._frequency

    @property
    def sdt(self):
        return self._sdt

    def __lt__(self, other):
        if not isinstance(other, FLIMSet):
            return NotImplemented

        return self._allFiles < other._allFiles

    def __eq__(self, other):
        if not isinstance(other, FLIMSet):
            return NotImplemented

        return self._allFiles == other._allFiles

    def __repr__(self) -> str:
        s = f'FLIMSet: title={self.title}, timecode={self.timecode}, frequency={self.frequency}\n'
        s += f'  SDT: {self._sdt}\n'
        s += '  Channels:\n'
        for ch_name, channel in self._channels.items():
            s += f'    {ch_name}:\n'
            for m_name, measure in channel.items():
                s += f'        {m_name}: {measure}\n'
        return s


class FLIMSeries:
    # A group of FileSet objects associated by SDT titles across timecodes.
    # The files are associated with multiple SDT files by their title,
    # which is a combination of the timecode and frequency information in the filename.
    # For example, for the SDT file "a-t80_800_.sdt", the associated files would be those
    # that start with "a-t80_800_" and have the same  frequency information in
    # their filenames. The timecode information can vary across the files in the series,
    # e.g. "a-t80_800_" and "a-t100_800_.asc" would both be associated with the same series.

    def __init__(self, sets: list[FLIMSet]):
        self._sets = sorted(sets)
        if len(sets) == 0:
            self._seriesTitle = None
        else:
            self._seriesTitle = sets[0].seriesTitle

    # TODO: Resolve naming confusionn of "series" between a single series (of sets) and an list of series
    @classmethod
    def series_from_directory(
        cls,
        dirpath,
        glob=None,
        recursive=True,
        ignore_missing_files=False,
        ignore_missing_sdt=False,
    ) -> 'dict[str,FLIMSeries]':

        # Scan directory for files, group by title, extract timecodes,
        # create FileSet objects and store them in a FileSeries object.
        # Return a list of FileSeries objects containing the grouped FileSet objects
        allFiles = {}
        for path in dirpath.glob(
            glob if glob is not None else '**/*' if recursive else '*'
        ):
            if not path.is_file():
                continue
            flimfile = FLIMASCFile.parsePath(path)
            if flimfile is not None:
                title = flimfile.title
                seriesTitle = flimfile.seriesTitle
                if seriesTitle not in allFiles:
                    allFiles[seriesTitle] = {}
                series = allFiles[seriesTitle]
                if title not in series:
                    series[title] = {'files': []}
                title = series[title]
                title['files'].append(flimfile)
            else:
                flimfile = FLIMSDTFile.parsePath(path)
                if flimfile is not None:
                    title = flimfile.title
                    seriesTitle = flimfile.seriesTitle
                    if seriesTitle not in allFiles:
                        allFiles[seriesTitle] = {}
                    series = allFiles[seriesTitle]
                    if title not in series:
                        series[title] = {'files': []}
                    title = series[title]
                    # TODO: check for multiple sdt files?
                    title['sdt'] = flimfile
                else:
                    log.info(
                        'Skipping file %s because it is not an ASC or SDT file.',
                        path,
                    )

        # Convert to FileSet objects
        allSeries = {}
        for seriesTitle, setData in allFiles.items():
            sets = []
            for title, data in setData.items():
                log.debug(
                    'Creating FileSet for %s\n%s',
                    title,
                    data,
                )
                files = data.get('files', [])
                sdt = data.get('sdt', None)
                assert sdt or files, (
                    'FileSet %s has neither files nor sdt file.',
                    title,
                )
                fileset = FLIMSet(
                    files=files,
                    sdt=sdt,
                    ignore_missing_files=ignore_missing_files,
                    ignore_missing_sdt=ignore_missing_sdt,
                )
                sets.append(fileset)
            if seriesTitle in allSeries:
                log.warning(
                    'Series %s already exists. Overwriting.',
                    seriesTitle,
                )
            allSeries[seriesTitle] = FLIMSeries(sets)

        return allSeries

    def load(self) -> dict:
        data = {}
        for fset in self._sets:
            data[fset.title] = fset.load()
        return data

    @property
    def sets(self):

        return self._sets

    @property
    def seriesTitle(self) -> str | None:
        return self._seriesTitle

    def __repr__(self) -> str:
        s = f'FLIMSeries: seriesTitle={self.seriesTitle} with {len(self.sets)} set(s)\n'
        for fset in self.sets:
            s += f'{textwrap.indent(repr(fset), "  ")}'
        return s


# %%
if __name__ == '__main__':
    # testing
    filenames = [
        'a_1_2_3.tif',
        'a-ctrl_800_-Ch1-_a1[%].asc',
        'a-t100_800_-Ch3-_t2.asc',
        'a-t120_800_-Ch1-_color coded value.asc',
        'a-t20_800_-Ch3-_a1.asc',
        'a-t20_800_.sdt',
        'APOE3_Male_30min_t2.asc',
    ]
    matches = []

    testdir = Path('flimanalyzer-napari/scratch/testdir')
    if not testdir.exists():
        print(f'testdir does not exist: {testdir.absolute()}')
    print(r'glob=**/*')
    f = FLIMSeries.series_from_directory(
        testdir,
        glob='**/*',
        ignore_missing_files=True,
        ignore_missing_sdt=True,
    )
    print(f'Found {len(f)} series:\n')
    for _title, series in f.items():
        print(f'{textwrap.indent(repr(series), "  ")}\n')
    # print("\n\nglob=*")
    # print(FLIMSeries.series_from_directory(testdir, glob="*", ignore_missing_files=True, ignore_missing_sdt=True))

    # %%
    # defining classes
    """ ###
    SDT files have a header section that contains metadata about the image acquisition and fitting
    process. Read the file until binary data is reached (indicated by the string "BIN_PARA_BEGIN:").
    The file contents look like this:


    è?*???¯?"???≥t’{????????’u??????UU?????ë*IDENTIFICATION
    ID        : ?SPC FCS Data File?
    Title     : a-t60_800_
    Version   : 3  980 M
    Revision  : 8 bits ADC
    Date      : 2019-07-20
    Time      : 17:30:12
    Author    : Unknown
    Company   : Unknown
    Contents  :
    *END

    *SETUP
    SYS_PARA_BEGIN:
    #PR [PR_PDEV,I,0]
    #PR [PR_PPORT,I,2]
    #PR [PR_PWHAT,I,1]
    #PR [PR_PF,B,0]
    ...
    #WI #4 *NO *6 [192,223]
    #WI #4 *NO *7 [224,255]
    WIND_PARA_END:
    *END

    BIN_PARA_BEGIN:?0Y??....
    ### """

    file_path = (
        '../../data/fitted_raw_image_data/800-FoV-a-V5analysis/a-ctrl_800_.sdt'
    )

    sdt_data = {}
    try:
        in_identification_section = False
        in_setup_section = False
        with open(file_path, errors='replace') as file:
            for line in file:
                if 'BIN_PARA_BEGIN:' in line:
                    # Stop reading the file when the binary data starts
                    break

                if '*IDENTIFICATION' in line:
                    in_identification_section = True
                    continue
                if in_identification_section:
                    if '*END' in line:
                        in_identification_section = False
                    elif ':' in line:
                        key, value = line.split(':', 1)
                        sdt_data[key.strip()] = value.strip()
                    else:
                        log.warning(
                            'Invalid line in identification section: %s',
                            line.strip(),
                        )
                    continue

                if '*SETUP' in line:
                    in_setup_section = True
                    sdt_data['Setup'] = []
                    continue
                if in_setup_section:
                    if '*END' in line:
                        in_setup_section = False
                    else:
                        sdt_data['Setup'].append(line.strip())
                    continue

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

    print(sdt_data)
