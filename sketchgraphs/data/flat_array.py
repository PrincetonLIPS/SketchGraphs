"""This module implements helpers to read the pre-processed data files. Due to the large size of the data,
using plain pickle files is inefficient (and leads to extremely long loading times). We implement a
memory-mappable compressed format that scales to large files while offering both compression and
constant-time access to elements in the dataset.
"""

import functools
import io
import multiprocessing
import pickle

import lz4.frame
import numpy as np

try:
    # this module can be used without pytorch, some multiprocessing tools are disabled
    import torch
except ImportError:
    torch = None


_MAGIC = -2356038617
_MAGIC_BYTES = np.array([_MAGIC], dtype=np.dtype('<i8')).view(np.byte)


def human_bytes(num_bytes):
    """Formats a byte count as a human-readable byte (using KB, MB, GB units).

    Parameters
    ----------
    num_bytes: int
        An integer representing the number of bytes

    Returns
    -------
    number
        The fractional number of bytes in the given unit
    str
        The unit of the returned value
    """
    units = ('B', 'KB', 'MB', 'GB')
    power = 2 ** 10

    for unit in units:
        if num_bytes < power:
            return num_bytes, unit

        num_bytes /= power

    return num_bytes, 'TB'


def save_list_flat(data, nthreads=None):
    """Saves a list of python objects into a flat bytes format.

    Parameters
    ----------
    data: list
        a list of python objects to serialize
    nthreads: int, optional
        If not `None`, the number of processes to use.

    Returns
    -------
    np.ndarray
        A numpy array of bytes representing the serialized data.
    """
    with io.BytesIO() as temp:
        offsets = _writer_list_flat(temp, data, nthreads)
        return pack_list_flat(offsets, np.frombuffer(temp.getbuffer(), dtype=np.byte))


def _save_single(object_) -> bytes:
    """Saves a single object to its binary representation.

    The binary representation is formed by compressing the pickle'd representation
    of the object.
    """
    return lz4.frame.compress(pickle.dumps(object_, protocol=4), store_size=False, compression_level=9)


def _writer_list_flat(writer, data, nthreads=None):
    i64 = np.dtype('<i8')
    _save = functools.partial(_save_single)
    offsets = np.empty(len(data) + 1, dtype=i64)
    offsets[0] = 0

    current_offset = 0

    if nthreads is not None:
        pool = multiprocessing.Pool(nthreads)
        data_bytes = pool.imap(_save, data, chunksize=128)
    else:
        data_bytes = map(_save, data)

    for i, x_bytes in enumerate(data_bytes):
        writer.write(x_bytes)
        current_offset += len(x_bytes)
        offsets[i + 1] = current_offset

    return offsets


def pack_list_flat(offsets, data_bytes):
    """Packs the given offsets and corresponding data array into a flat array format.

    This function simply adds some metadata headers in order to create a contiguous
    packed format. See also `raw_list_flat` to obtain the raw ``offsets`` and ``data_bytes``,
    or `save_list_flat` for serializing a flat array.

    Parameters
    ----------
    offsets : array_like
        Array of offsets delineating each element in the ``data_bytes`` array
    data_bytes : array_like
        Array of bytes containing the raw serialized data

    Returns
    -------
    np.ndarray
        An array of bytes representing the raw value.
    """
    i64 = np.dtype('<i8')
    num_elements = np.array([len(offsets) - 1], dtype=i64)
    version = np.array([2], dtype=i64)

    pack = np.empty(i64.itemsize * 4 + offsets.nbytes + len(data_bytes), dtype=np.byte)
    current_offset = 0

    for arr in [_MAGIC_BYTES, version, num_elements, offsets.astype(i64, copy=False), data_bytes]:
        current_offset = _write_slice(pack, current_offset, arr)

    return pack


def raw_list_flat(data, nthreads=None):
    """Serializes the provided list into an array of bytes and offsets.

    Note that this function only provides the raw offsets and serialized bytes.
    It is advised to use `save_list_flat` or similar to encode other associated metadata.

    Parameters
    ----------
    data : iterable
        An iterable containing the data to be serialized
    nthreads : int, optional
        If not None, the number of threads to be used for performing the serialization in parallel

    Returns
    -------
    np.ndarray
        An array of offsets, of length one plus the number of elements in the data
    np.ndarray
        An array of bytes, representing the concatenated serialized data
    """
    with io.BytesIO() as temp:
        offsets = _writer_list_flat(temp, data, nthreads)
        return offsets, np.frombuffer(temp.getvalue(), dtype=np.byte)


def merge_raw_list(offset_arrays, data_arrays):
    """Merges a list of raw flat lists (as produced by `raw_list_flat`) into one single raw flat list.

    Parameters
    ----------
    offset_arrays: list of np.ndarray
        A list of arrays representing the offsets.
    data_arrays: list of np.ndarray
        A list of the same length as ``offset_arrays`` representing the data.

    Returns
    -------
    np.ndarray
        An array of offsets, of length one plus the number of elements in the data.
    np.ndarray
        An array of bytes, representing the concatenate serialized data.
    """
    i64 = np.dtype('<i8')
    total_sketches = sum(len(off) - 1 for off in offset_arrays)
    all_offsets = np.empty(total_sketches + 1, dtype=i64)
    current_offset = 0
    idx = 0

    for off in offset_arrays:
        all_offsets[idx:idx+len(off) - 1] = off[:-1] + current_offset
        current_offset += off[-1]
        idx += len(off) - 1

    all_offsets[-1] = current_offset
    all_data = np.concatenate(data_arrays)

    return all_offsets, all_data


def _next_slice(data, offset, num_elements, dt=np.dtype('<i8')):
    if num_elements < 0:
        raise ValueError('Must read a non-negative number of elements.')

    next_offset = offset + dt.itemsize * num_elements
    return np.squeeze(np.frombuffer(data[offset:next_offset], dtype=dt)), next_offset


def _write_slice(data, offset, array):
    array_bytes = array.view(np.byte)
    next_offset = offset + len(array_bytes)
    data[offset:next_offset] = array_bytes
    return next_offset


def _unpack_flat_array(byte_data):
    i64_dt = np.dtype('<i8')

    offset = 0

    magic, offset = _next_slice(byte_data, offset, 1, i64_dt)
    if magic != _MAGIC:
        raise ValueError("Did not find magic bytes in data. The data is in the wrong format or corrupted.")

    version, offset = _next_slice(byte_data, offset, 1, i64_dt)
    if version != 2:
        raise ValueError("Expected protocol version 2, but found version {0}.".format(version))

    num_items, offset = _next_slice(byte_data, offset, 1, i64_dt)
    pickle_offsets, offset = _next_slice(byte_data, offset, num_items + 1, i64_dt)
    pickle_data = byte_data[offset:]

    return pickle_offsets, pickle_data


def load_flat_array(path):
    """Loads a flat array from the given path.

    Parameters
    ----------
    path : file-like object, string, or pathlib.Path
        The file to read. Must be compatible with `np.load`.

    Returns
    -------
    FlatSerializedArray
        A FlatSerializedArray containing the data.
    """
    buffer = np.load(path, mmap_mode='r', allow_pickle=False)
    return FlatSerializedArray.from_flat_array(buffer)


class FlatSerializedArray:
    """This class implements a flat pickle-serialized array of python objects backed by a byte buffer.

    This class behaves as a standard immutable list of python objects, but materializes a new object on
    each access. This allows it to use memory-mapped byte arrays as its underlying storage, which enable
    random access and arrays larger than system memory.
    """
    def __init__(self, offsets, pickle_data):
        """Initializes a new array from offsets and pickled data.

        Note: users will typically wish to create a new instance of this class using
        the static method `from_flat_array`, which provides a view into the flat array.

        Parameters
        ----------
        offsets : array_like
            an array of offsets into the data representing each object.
        pickle_data : array_like
            an array of bytes representing the serialized data for all objects.
        """
        self._offsets = offsets
        self._pickle_data = pickle_data

    @staticmethod
    def from_flat_array(byte_data):
        """Creates a new `FlatSerializedArray` from the provided bytes.

        Parameters
        ----------
        byte_data : array_like
            A bytes-like object which provides the underlying storage for flat array of objects.
            This is usually obtained through `save_list_flat`.

        Returns
        -------
        FlatSerializedArray
            An instance of `FlatSerializedArray` providing a view into the data given by the provided bytes.
        """
        return FlatSerializedArray(*_unpack_flat_array(byte_data))

    def __len__(self):
        return len(self._offsets) - 1

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if idx.stop is not None:
                idx = slice(idx.start, idx.stop + 1, idx.step)

            return FlatSerializedArray(self._offsets[idx], self._pickle_data)

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError('index out of range')

        object_bytes = self.get_raw_bytes(idx)
        return self.decode_raw_bytes(object_bytes)

    def get_raw_bytes(self, idx: int) -> bytes:
        """Obtain the raw compressed bytes for the object at the given index.

        Parameters
        ----------
        idx : int
            The index at which to obtain the data.

        Returns
        -------
        bytes
            A bytes sequence representing the raw (compressed) object bytes.
            The user is expected to decode the data using pickle after decompressing
            the obtained data. See also `decode_raw_bytes`.
        """
        try:
            object_bytes = self._pickle_data[self._offsets[idx]:self._offsets[idx + 1]]
        except IndexError as exc:
            raise ValueError('Error when reading internal data '
                             '- offsets are likely inconsistent with underlying data.') from exc

        if torch is not None and isinstance(object_bytes, torch.Tensor):
            object_bytes = object_bytes.numpy()

        return object_bytes

    @staticmethod
    def decode_raw_bytes(object_bytes: bytes):
        return pickle.loads(lz4.frame.decompress(object_bytes))


    def share_memory_(self):
        """Moves the underlying to a `torch` shared memory location.

        Calling this method will transition the storage to torch tensors, and move
        them to a shared memory location. This will enable sharing the tensors across
        different processes without copying when using `torch.multiprocessing`.
        """
        if torch is None:
            raise ValueError('pytorch could not be loaded. It is required to share memory.')

        self._offsets = torch.as_tensor(self._offsets).share_memory_()
        self._pickle_data = torch.as_tensor(self._pickle_data).share_memory_()
        return self

    def __repr__(self):
        return 'FlatArray(len={0}, mem={1:.1f} {2})'.format(len(self), *human_bytes(len(self._pickle_data)))


_DICTIONARY_MAGIC = 87374267
_DICTIONARY_MAGIC_BYTES = np.array([_DICTIONARY_MAGIC], dtype=np.dtype('<i8')).view(np.byte)


def pack_dictionary_flat(dict_):
    """Saves a dictionary into a flat structure which is compatible
    with the structure used for `FlatSerializedArray`.

    This structure is a light extension of the `FlatSerializedArray`, and is able to encode
    the array inline so that the memory used by the array can be laid-out directly.
    This is mainly used to serialize a set of related arrays.

    Parameters
    ----------
    dict_ : dict
        An arbitrary dictionary containing the elements to be serialized.

    Returns
    -------
    np.ndarray
        An array of bytes representing the the serialized data.
    """
    i64 = np.dtype('<i8')

    header_dict = {}
    current_data_offset = 0

    element_data = []

    for k, v in dict_.items():
        if isinstance(v, np.ndarray) and v.dtype != np.byte:
            # General numpy arrays
            with io.BytesIO() as temp:
                np.save(temp, v, allow_pickle=False)
                v = np.frombuffer(temp.getvalue(), dtype=np.byte)
            data_type = 1
        elif isinstance(v, np.ndarray):
            # Assume that we have a packed flat array
            data_type = 2
        else:
            # We have a general object
            v = np.frombuffer(pickle.dumps(v, protocol=4), dtype=np.byte)
            data_type = 0

        header_dict[k] = (current_data_offset, len(v), data_type)
        current_data_offset += len(v)
        element_data.append(v)

    header_dict_data = pickle.dumps(header_dict, protocol=4)
    total_size = 3 * i64.itemsize + len(header_dict_data) + current_data_offset

    result = np.empty(total_size, dtype=np.byte)

    offset = 0
    offset = _write_slice(result, offset, _DICTIONARY_MAGIC_BYTES)
    offset = _write_slice(result, offset, np.array([1], dtype=i64)) # version
    offset = _write_slice(result, offset, np.array([len(header_dict_data)], dtype=i64))
    offset = _write_slice(result, offset, np.frombuffer(header_dict_data, dtype=np.byte))

    for element in element_data:
        offset = _write_slice(result, offset, element)

    return result


def load_dictionary_flat(data):
    """Loads a flat dictionary from the given data buffer.

    Parameters
    ----------
    data : Union[np.ndarray, str]
        ndarray of bytes representing the underlying data for the flat dictionary, or a string
        representing the filename from which to load the dictionary.

    Returns
    -------
    dict
        A dictionary containing elements serialized in the data buffer.
    """
    if not isinstance(data, np.ndarray):
        data = np.load(data, mmap_mode='r')

    offset = 0
    magic, offset = _next_slice(data, offset, 1)

    if magic != _DICTIONARY_MAGIC:
        raise ValueError('Magic bytes not found, check for data corruption')

    version, offset = _next_slice(data, offset, 1)
    if version != 1:
        raise ValueError('Unknown protocol version {0}'.format(version))

    len_header, offset = _next_slice(data, offset, 1)
    header_data, base_offset = _next_slice(data, offset, len_header, np.dtype(np.byte))

    header_dict = pickle.loads(header_data)

    result = {}

    for k, (data_offset, data_len, data_type) in header_dict.items():
        element_data = data[base_offset + data_offset:base_offset + data_offset + data_len]

        if data_type == 0:
            result[k] = pickle.loads(element_data)
        elif data_type == 1:
            with io.BytesIO(element_data) as element_io:
                result[k] = np.load(element_io, allow_pickle=False)
        elif data_type == 2:
            result[k] = FlatSerializedArray.from_flat_array(element_data)

    return result
