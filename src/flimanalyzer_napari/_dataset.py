# %%
# imports
"""
Module for handling FLIM (Fluorescence Lifetime Imaging Microscopy) dataset structures.

This module defines classes for representing and manipulating FLIM data files,
including ASC data files and SDT metadata files.

The FLIM data structure follows this hierarchy:
- FLIMFile: Abstract base class for FLIM data files
  - FLIMASCFile: Represents ASC data files containing measurements
  - FLIMSDTFile: Represents SDT metadata files containing acquisition parameters
- FLIMSet: A group of FLIM files associated with a single SDT file (same timepoint/frequency)
- FLIMSeries: A collection of FLIMSet objects sharing the same series title

Typical usage:
    >>> series = FLIMSeries.series_from_directory('/path/to/flim/data')
    >>> for series_title, flim_series in series.items():
    >>>     data = flim_series.loadAsTCMZYX()['data']
    >>>     # Process the 6D data array (time, channel, measure, z, y, x)
"""

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
    """Abstract base class for FLIM (Fluorescence Lifetime Imaging Microscopy) data files.

    This class represents a single FLIM data file (either ASC or SDT) and provides
    common functionality for parsing file paths, extracting metadata, and comparing files.

    The file is associated with the SDT file by its title,
    which is a combination of the timecode and frequency information in the filename.
    For example, for the SDT file "a-t80_800_.sdt", the associated ASC files would be those
    that start with "a-t80_800_" having the same timecode and frequency information in
    their filenames.

    """

    #
    # Class level defaults common to both SDT and ASC
    #
    frequency_re = re.compile(r'(?P<freq>\d+)')
    time_re = re.compile(r'(?P<time>ctrl|t\d+|\d+min)')

    # TODO: should this be more sophisticated? This matches "80", "t20", and "20min"
    timedigits_re = re.compile(r'^\D*(\d+)\D*$')

    # Title may be combination of name, timestamp and frequency, or just a name.
    # We want to be flexible in what we accept as title, but we want to make sure it doesn't
    # capture the channel or measure information. For simplicity, we can assume that the title
    # is everything before the channel and measure information.
    simple_title_re = re.compile('(?P<title>.+?)')
    # title_re = re.compile(f"(?P<title>(?:{time_re}|{frequency_re}|[_]|[\\w-]+)+|.+?)")
    title_re = re.compile(
        f'(?P<title>(?:{time_re.pattern}|(?<=_){frequency_re.pattern}|.+?)+)'
    )

    def __init__(
        self, path, title, seriesTitle, timecode=None, frequency=None
    ):
        """Initialize a FLIMFile instance.

        Args:
            path (Path): Full path to the file
            title (str): Base title of the file (timecode + frequency info)
            seriesTitle (str): Series identifier (title without timecode)
            timecode (str, optional): Timepoint identifier (e.g., 'ctrl', 't20')
            frequency (str, optional): Laser frequency identifier (e.g., '800')
        """
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
        """Parse a file path to identify FLIM files and extract filename metadata.

        Args:
            path (Path): File path to parse
            pathRegex (Pattern, optional): Regex pattern to match against filename
            titleRegex (Pattern, optional): Regex pattern to extract timecode/frequency

        Returns:
            FLIMFile|None: Parsed FLIM file object or None if path doesn't match expected pattern
        """

    @abstractmethod
    def load(self) -> object:
        """Load the data from the associated file.

        Returns:
            object: The loaded data, determined by the file type, e.g. numeric array for ASC files,
            metadata dictionary for SDT files.
        """

    @classmethod
    def timecode_value(cls, timecode: str) -> int | str | None:
        """Convert a timecode string to a numerical value for sorting.

        Special handling for 'ctrl' (returns -1) and numeric timecodes.
        For formats like 't20', returns 20. For plain numbers, returns the number.
        If timecode cannot be parsed as a number, returns the original string.

        Args:
            timecode (str): Timecode string (e.g., 'ctrl', 't20', '20min')

        Returns:
            int|str|None: Numerical value for sorting, or None if input is None
        """
        if timecode is None:
            return None

        if timecode == 'ctrl':
            return -1

        r = cls.timedigits_re.match(timecode)
        if r:
            return int(r.group(1))

        log.warning('Could not parse timecode: %s', timecode)
        return timecode

    @property
    def path(self) -> Path:
        """Return the file path associated with this FLIMFile."""
        return self._path

    @property
    def title(self) -> str | None:
        """Return the title associated with this FLIMFile."""
        return self._title

    @property
    def seriesTitle(self) -> str | None:
        """Return the series title associated with this FLIMFile."""
        return self._seriesTitle

    @property
    def timecode(self) -> str | None:
        """Return the timecode associated with this FLIMFile."""
        return self._timecode

    @property
    def frequency(self) -> str | None:
        """Return the laser frequency associated with this FLIMFile."""
        return self._frequency

    # TODO: Implemented lt and eq for sorting (time/channel/measure order).
    #       Does this break equality? Consider same filename in different directory!
    def __lt__(self, other) -> bool:
        if not isinstance(other, FLIMFile):
            return NotImplemented
        # Compare using tuple for lexicographical order, treating None as empty string for sorting
        self_tuple = (
            self._path.parent,
            self._seriesTitle or '',
            self.timecode_value(self._timecode) if self._timecode else '',
            self._frequency or '',
            self.path.name,
        )
        other_tuple = (
            other._path.parent,
            other._seriesTitle or '',
            other.timecode_value(other._timecode) if other._timecode else '',
            other._frequency or '',
            other.path.name,
        )
        return self_tuple < other_tuple

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
    """Represents an ASC data file containing FLIM measurement data.

    ASC files contain the actual measurement data for different FLIM parameters
    such as photon counts, chi-squared values, amplitude percentages, and lifetimes.

    Class Attributes:
        measures (list): List of valid measurement types in the files
        measures_re (Pattern): Regex pattern to extract measure type from filename
        channel_re (Pattern): Regex pattern to extract channel identifier from filename
        asc_regex (Pattern): Compiled regex for parsing ASC file paths

    """

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
        """Initialize a FLIMASCFile instance.

        Args:
            path (Path): Path object pointing to the ASC file
            setTitle (str): Base title of the file (timecode + frequency info)
            seriesTitle (str): Series identifier (title without timecode)
            timecode (str, optional): Timepoint identifier (e.g., 'ctrl', 't20')
            frequency (str, optional): Laser frequency identifier (e.g., '800')
            channel (str, optional): Channel identifier (e.g., 'Ch1', 'Ch2')
            measure (str, optional): Measurement type (e.g., 'photons', 'chi', 'a2[%]')
        """
        super().__init__(path, setTitle, seriesTitle, timecode, frequency)
        self._channel = channel
        self._measure = measure

    # TODO: Allow for overriding regex patters specific to ASC files.
    #       Consider using a subclass of FLIMFile or passing regex patterns as arguments to the constructor.
    #       How would we do that within Napari using the reader?
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
        # TODO: Check [SP_X_AXIS,I,0] in SDT file to ensure YX ordering.
        #       Check [] in SDT file to ensure shape (assuming 256x256).
        return np.loadtxt(self._path, delimiter=' ')

    @property
    def channel(self) -> str | None:
        """Return the channel identifier associated with this FLIMASCFile."""
        return self._channel

    @property
    def measure(self) -> str | None:
        """Return the measurement type associated with this FLIMASCFile."""
        return self._measure

    # TODO: Implemented lt and eq for sorting (time/channel/measure order).
    #       Does this break equality? Consider same filename in different directory!
    #       Consider pushing common comparisons to FLIMFile?
    def __lt__(self, other) -> bool:
        if not isinstance(other, FLIMFile):
            return NotImplemented

        # Compare using tuple for lexicographical order, treating None as empty string for sorting
        if (
            self._path.parent,
            self._seriesTitle or '',
            self.timecode_value(self._timecode) if self._timecode else '',
            self._frequency or '',
        ) < (
            other._path.parent,
            other._seriesTitle or '',
            other.timecode_value(other._timecode) if other._timecode else '',
            other._frequency or '',
        ):
            return True

        if isinstance(other, FLIMSDTFile):
            return False  # ASC files should come after matching SDT file

        if isinstance(other, FLIMASCFile) and (  # noqa: SIM103
            self._channel or '',
            self._measure or '',
            self.path.name,
        ) < (other._channel or '', other._measure or '', other.path.name):
            return True

        return False


class FLIMSDTFile(FLIMFile):
    """Represents an SDT data file containing FLIM measurement data.

    SDT files contain configuration for acquisition and FLIM parameters. Common to
    a set of ASC files, such as photon counts, chi-squared values, amplitude
    percentages, and lifetimes.

    """

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

    # TODO: Pick out  useful information for validation elsewhere (e.g. timestamp,
    #       acquisition parameters, image dimensions, etc.)
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

        # Compare using tuple for lexicographical order, treating None as empty string for sorting
        if (
            self._path.parent,
            self._seriesTitle or '',
            self.timecode_value(self._timecode) if self._timecode else '',
            self._frequency or '',
        ) < (
            other._path.parent,
            other._seriesTitle or '',
            other.timecode_value(other._timecode) if other._timecode else '',
            other._frequency or '',
        ):
            return True

        if isinstance(other, FLIMASCFile):  # noqa: SIM103
            return True  # ASC files should come after matching SDT file

        return False


class FLIMSet:
    """
    A set of files associated with a single SDT file.
    The files are associated with the SDT file by their title,
    which is a combination of the timecode and frequency information in the filename.
    For example, for the SDT file "a-t80_800_.sdt", the associated files would be those
    that start with "a-t80_800_" and have the same timecode and frequency information in
    their filenames.

    Following default ordering,  XYZCT.

    Structure of data for ASC files:
      {Path(file): {channel: str, measure: str}}
    """

    def __init__(
        self,
        files: list[FLIMASCFile],
        sdt: FLIMSDTFile | None = None,
        ignore_missing_files: bool = False,
        ignore_missing_sdt: bool = False,
    ):
        """
        Initialize a FLIMSet instance.

        Args:
            files (list[FLIMASCFile]): List of ASC files
            sdt (FLIMSDTFile | None): Associated SDT file
            ignore_missing_files (bool): Whether to ignore missing files
            ignore_missing_sdt (bool): Whether to ignore missing SDT file
        """
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

    # TODO: Should we return measures per channel, assume all channels have the same measures,
    #       and if so, should we check for consistency across channels?
    @property
    def measures(self):
        measures = set()
        for channel in self._channels.values():
            measures.update(channel.keys())
        return sorted(measures)

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
    # e.g. "a-t80_800_.sdt" and "a-t100_800_.sdt" would both be associated with the same series.

    def __init__(self, sets: list[FLIMSet]):
        """
        Initialize a FLIMSeries instance.

        Args:
            sets (list[FLIMSet]): A list of FLIMSet instances.
        """
        # TODO: add checks for consistency of series title and frequency across sets.
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
        """
        Create a list of FLIMSeries objects by:
            Scan directory for files, group by title, extract timecodes,
            Create FileSet objects and store them in a FileSeries object.
            Return a list of FileSeries objects containing the grouped FileSet objects
        Args:
            dirpath (str or Path): The directory path to scan for FLIM files.
            glob (str, optional): A glob pattern to filter files. Defaults to None, which means all files.
            recursive (bool, optional): Whether to scan directories recursively. Defaults to True.
            ignore_missing_files (bool, optional): Whether to ignore missing files when creating FLIMSet objects. Defaults to False.
            ignore_missing_sdt (bool, optional): Whether to ignore missing SDT files when creating FLIMSet objects. Defaults to False.

        Returns:
            dict[str, FLIMSeries]: A dictionary mapping series titles to FLIMSeries objects.
        """
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
        """
        Load FLIMSeries data into a dictionary of title -> data, where data is a dictionary containing
        the loaded data from the associated FLIMSet objects.
        """
        data = {}
        for fset in self._sets:
            data[fset.title] = fset.load()
        return data

    def loadAsTCMZYX(self) -> dict:
        """
        Load FLIMSeries data into a dictionary of data and metadata.
        The data is returned as a 6D numpy array with dimensions corresponding to time, channel, measure, z, y, x.
        The metadata is returned as a dictionary containing the time points, channel names, and measure names
        """
        if len(self._sets) == 0:
            log.warning(
                'No sets in series %s. Cannot load data.', self.seriesTitle
            )
        #     return {'data': np.array([]), 'metadata': {}}

        seriesData = self.load()

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
        channel_names = list(set1['data'].keys())
        channel1 = next(iter(set1['data'].values()))
        measure_dim = len(channel1)
        log.debug('Examining set 0 measures: %s', channel1.keys())
        # TODO: Determine shape from data
        #   Possibly from SDT file lines:
        #       #SP [SP_IMG_X,I,256]
        #       #SP [SP_IMG_Y,I,256]
        #   Or from the .asc files themselves, which should all have the same shape.
        data = np.zeros((time_dim, channel_dim, measure_dim, 1, 256, 256))
        for t, (time, timeData) in enumerate(seriesData.items()):
            log.warning('Processing time point %s with data %s', t, timeData)
            for c, (channel, channelSet) in enumerate(
                timeData['data'].items()
            ):
                log.debug('Processing channel %s with data %s', c, channelSet)
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

        # TODO: Add metadata from SDT files, e.g. image dimensions, acquisition parameters, etc.
        seriesData = {
            'time': list(seriesData.keys()),
            'channels': channel_names,
            'measures': list(channel1.keys()),
        }

        return {'data': data, 'metadata': seriesData}  # TODO: Add metadata

    @property
    def sets(self) -> list[FLIMSet]:
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
