import numpy as np
import os
import librosa

def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


# Conversions
def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def normalize(S, hparams):
    return (S - hparams['min_level_db']) / -hparams['min_level_db']


def process_utterance(wav_path,
                      fft_size=1024,
                      hop_size=256,
                      win_length=1024,
                      window="hann",
                      num_mels=80,
                      fmin=80,
                      fmax=7600,
                      eps=1e-6,
                      sample_rate=22050,
                      loud_norm=False,
                      min_level_db=-100,
                      return_linear=False,
                      trim_long_sil=False, vocoder='pwg'):
    if isinstance(wav_path, str):
        wav, _ = librosa.core.load(wav_path, sr=sample_rate)
    else:
        wav = wav_path

    # get amplitude spectrogram
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc

    if vocoder == 'pwg':
        # print(mel)
        mel[np.where(mel == 0)] = 0.01
        # print(mel)
        mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)
    else:
        assert False, f'"{vocoder}" is not in ["pwg"].'

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]

    if not return_linear:
        return wav, mel
    else:
        spc = amp_to_db(spc)
        spc = normalize(spc, {'min_level_db': min_level_db})
        return wav, mel, spc
    

mel_min = np.ones(80) * float('inf')
mel_max = np.ones(80) * -float('inf')

wav_folder = './data/raw/vietsing/wavs'

count = 0
files = os.listdir(wav_folder)
for f in files:
    count += 1
    wav_path = os.path.join(wav_folder, f)
    print(count, wav_path)
    wav, mel = process_utterance(wav_path, fft_size=512,
                      hop_size=128,
                      win_length=512,
                      window="hann",
                      num_mels=80,
                      fmin=30,
                      fmax=12000,
                      eps=float(-6),
                      sample_rate=24000,
                      loud_norm=False,
                      min_level_db=-120,
                      return_linear=False,
                      trim_long_sil=False, vocoder='pwg')

    m_min, m_max = np.min(mel, axis=1), np.max(mel, axis=1)
    mel_min = np.minimum(mel_min, m_min)
    mel_max = np.maximum(mel_max, m_max)

    # if count == 10: break

print(mel_min.tolist())
print(mel_max.tolist())
print(min(mel_min))
print(max(mel_max))