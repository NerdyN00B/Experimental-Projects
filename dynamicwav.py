import numpy as np
from scipy.io.wavfile import write

def removedc(data):
    """remove DC offset from audio data.
    
    Parameters
    ----------
    data : _array_like
        audio data

    Returns
    -------
    _array_like
        processed audio data ready for WAV file
    """
    data -= np.mean(data)  # Remove DC offset    
    return data

def averagesquaremax(data, factor=2):
    squaredmean = np.mean(data**2)
    maxvalue = factor * np.sqrt(squaredmean)
    return maxvalue

def dynamic_max(data, chunksize):
    """Find local maxima per chunks of data.

    Parameters
    ----------
    data : _array_like
        audio data
    chunksize : int
        size of each chunk
    """
    maxima = np.zeros_like(data)
    for chunk in range(0, len(data), chunksize):
        end = chunk + chunksize
        if end > len(data):
            end = len(data)
        local_max = np.max(np.abs(data[chunk:end]))
        maxima[chunk:end] = local_max
    
    return maxima

def clip_and_normalize(data, dynamicmax):
    """Clip and normalize audio data based on dynamic maximum values.

    Parameters
    ----------
    data : _array_like
        audio data
    dynamicmax : _array_like
        dynamic maximum values for normalization

    Returns
    -------
    _array_like
        clipped and normalized audio data
    """
    normalized_data = data / dynamicmax
    normalized_data = np.clip(normalized_data, -1, 1)
    return normalized_data

def make_wav(data, filename):
    """Convert processed audio data to WAV format.

    Parameters
    ----------
    data : _array_like
        processed audio data
    filename : str
        output WAV file name
    """
    amplitude = np.iinfo(np.int16).max
    wav_data = (data * amplitude).astype(np.int16)
    write(filename, 44100, wav_data)


if __name__ == "__main__":
    file = r'audio\20251128135935_metklemshort_zonder_filter_dus.wav'
    data = np.load(file.replace('.wav', '.npy').replace('audio', 'data'))
    data = removedc(data)
    clipped_data_dynamic = clip_and_normalize(data, 
                                              dynamic_max(data, chunksize=44100//2)
                                              )
    clipped_data_average = clip_and_normalize(data, 
                                              averagesquaremax(data, factor=2)
                                              )
    make_wav(clipped_data_dynamic, file.replace('.wav', '_dynamic.wav'))
    make_wav(clipped_data_average, file.replace('.wav', '_average.wav'))
    