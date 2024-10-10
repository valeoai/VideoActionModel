'''
Tokens storage in a packed binary format.
'''
# from functools import reduce
# from operator import xor
import struct

import numpy as np


MAGIC_NUMBER = b'PTOK'
HEADER_FORMAT = f'<{len(MAGIC_NUMBER)}sIIIIIIIIII'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

TOKEN_DT = np.dtype(np.int16)
_TOKEN_DT_LITTLE_ENDIAN = TOKEN_DT.newbyteorder('<')
TOKEN_DT_LEN = 2
TOKEN_MIN = np.iinfo(TOKEN_DT).min
TOKEN_MAX = np.iinfo(TOKEN_DT).max


def encode_header(*, version=1, schedulers_n, window_size, height, width, video_id_len=11, frame_interval, records_n,
                  windows_n):
    '''
    Encode the header of a packed tokens file.
    CAVEAT: All integer fields are encoded as 32-bit unsigned integers. This puts a limitation of 4G records per file.
    Args:
        version (int): File format version
        schedulers_n (int): Number of schedulers
        window_size (int): Number of frames in the window
        height (int): Height of the token grid
        width (int): Width of the token grid
        video_id_len (int): Length of the video_id field (default: 11)
        frame_interval (int): Interval between frames in the window (usually 1 or 2)
        records_n (int): Number of records in the file
        windows_n (int): Total number of windows in the file, after reconstruction
    Returns:
        header (bytes): Encoded header (length: HEADER_SIZE)
        record_fmt (str): Format string for the records in the file, based on the header
    '''
    record_fmt, record_len = get_record_format(height=height, width=width, video_id_len=video_id_len)
    header = struct.pack(HEADER_FORMAT, MAGIC_NUMBER, version, schedulers_n, window_size, height, width,
                         video_id_len, frame_interval, records_n, windows_n, record_len)
    return header, record_fmt


def get_record_format(*, height, width, video_id_len):
    '''
    Get the format string and size of a record in a packed tokens file.
    Args:
        height: Height of the token grid
        width: Width of the token grid
        video_id_len: Length of the video_id field
    Returns:
        record_format (str): Format string for the record
        record_size (int): Size of the record in bytes
    '''
    # Record: scheduler, restart, frame_number, video_id, tokens
    record_format = f'<I?I{video_id_len}s{height * width * TOKEN_DT_LEN}s'
    return record_format, struct.calcsize(record_format)


def decode_header(header):
    '''
    Decode the header of a packed tokens file.
    Args:
        header (bytes): Header bytes (expected length: HEADER_SIZE)
    Returns:
        Dictionary with the header fields
    '''
    magic_number, version, schedulers_n, window_size, height, width, video_id_len, frame_interval, records_n, \
        windows_n, record_len = struct.unpack(HEADER_FORMAT, header)
    if magic_number != MAGIC_NUMBER:
        raise ValueError(f"Invalid magic number: {magic_number}")
    return {
        'version': version,
        'schedulers_n': schedulers_n,
        'window_size': window_size,
        'height': height,
        'width': width,
        'video_id_len': video_id_len,
        'frame_interval': frame_interval,
        'records_n': records_n,
        'windows_n': windows_n,
        'record_len': record_len,
    }


def encode_record(*, video_id, frame_number, scheduler, restart, tokens, record_fmt):
    '''
    Encode a record in a packed tokens file.
    Args:
        video_id (str): Video identifier
        frame_number (int): Frame number in the video
        scheduler (int): Scheduler index
        restart (bool): Whether the record is restarting the scheduler, clearing the window buffer
        tokens (2D array of int16): Token grid
        record_fmt (str): Record format string
    Returns:
        Encoded record bytes
    '''
    if tokens.dtype != TOKEN_DT:
        raise ValueError(f'Invalid tokens dtype: {tokens.dtype}, expected {TOKEN_DT}')
    # Get variable lengths from the record format
    video_id_len, token_len = record_fmt.strip('<?IiHhs').split('s')
    video_id_len = int(video_id_len)
    token_len = int(token_len)
    # Encode video_id and tokens and check their sizes
    video_id_bytes = video_id.encode('ascii') if isinstance(video_id, str) else video_id
    tokens_height, tokens_width = tokens.shape
    tokens_bytes = tokens.astype(_TOKEN_DT_LITTLE_ENDIAN).tobytes()
    if len(video_id_bytes) != video_id_len:
        raise ValueError(f'Invalid video_id length: {len(video_id)}, expected {video_id_len}')
    if len(tokens_bytes) != token_len:
        raise ValueError(f'Invalid tokens size: {tokens_height}x{tokens_width}, expected heigth x width == '
                         f'{token_len//TOKEN_DT_LEN}')
    # Pack the record
    record = struct.pack(record_fmt, scheduler, restart, frame_number, video_id_bytes, tokens_bytes)
    return record


def decode_record(record, *, record_fmt, header):
    '''
    Decode a record in a packed tokens file.
    Args:
        record (bytes): Record bytes
        record_fmt (str): Record format string
        header (dict): Header dictionary returned by `decode_header`
    Returns:
        Dictionary with the record fields
    '''
    if len(record) != header['record_len']:
        raise ValueError(f'Invalid record length: {len(record)}, expected {header["record_len"]}')
    # Unpack the record
    scheduler, restart, frame_number, video_id_bytes, tokens_bytes = struct.unpack(record_fmt, record)
    video_id = video_id_bytes.decode('ascii')
    tokens = np.frombuffer(tokens_bytes, dtype=_TOKEN_DT_LITTLE_ENDIAN).astype(TOKEN_DT)
    tokens = tokens.reshape(header['height'], header['width'])
    return {
        'scheduler': scheduler,
        'restart': restart,
        'frame_number': frame_number,
        'video_id': video_id,
        'tokens': tokens,
    }