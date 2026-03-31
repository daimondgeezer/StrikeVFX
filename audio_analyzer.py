"""
Audio analysis module: loads WAV, computes per-frame FFT,
extracts 4 configurable frequency bands with threshold + release logic.
"""
import numpy as np
import soundfile as sf
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BandConfig:
    """Configuration for one frequency band."""
    freq_low: float = 20.0
    freq_high: float = 200.0
    threshold: float = 0.3      # 0.0–1.0, normalized
    release: float = 0.25       # seconds to stay active after crossing threshold
    gain: float = 1.0           # pre-gain multiplier


@dataclass
class BandState:
    """Runtime state for one frequency band."""
    energy: float = 0.0         # current normalized energy (0–1)
    triggered: bool = False     # above threshold?
    active: float = 0.0         # 0–1 envelope (1 = just triggered, decays over release time)
    release_counter: float = 0.0  # remaining release time in seconds


DEFAULT_BANDS = [
    BandConfig(20,    200,   threshold=0.35, release=0.15, gain=1.0),   # Sub/Bass
    BandConfig(200,   2000,  threshold=0.30, release=0.12, gain=1.0),   # Low-Mid
    BandConfig(2000,  8000,  threshold=0.25, release=0.10, gain=1.0),   # High-Mid
    BandConfig(8000,  20000, threshold=0.20, release=0.08, gain=1.2),   # Highs
]


class AudioAnalyzer:
    def __init__(self, fps: float = 60.0):
        self.fps = fps
        self.sample_rate: int = 44100
        self.audio_data: Optional[np.ndarray] = None   # mono float32
        self.duration: float = 0.0
        self.total_frames: int = 0

        self.bands: List[BandConfig] = [BandConfig(**vars(b)) for b in DEFAULT_BANDS]
        self._band_states: List[BandState] = [BandState() for _ in self.bands]

        # Pre-computed per-frame band energies  shape: (total_frames, 4)
        self._frame_energies: Optional[np.ndarray] = None
        # Per-frame trigger/active envelopes     shape: (total_frames, 4)
        self._frame_active: Optional[np.ndarray] = None

        self._current_frame: int = 0

    # ------------------------------------------------------------------ load
    def load(self, path: str) -> None:
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        if sr != 44100:
            raise ValueError(f"Expected 44100 Hz, got {sr} Hz")
        # Mix to mono
        mono = data.mean(axis=1)
        self.audio_data = mono
        self.sample_rate = sr
        self.duration = len(mono) / sr
        self.total_frames = int(self.duration * self.fps)
        self._precompute()

    # ----------------------------------------------------------- pre-compute
    def _precompute(self) -> None:
        """Pre-compute all band energies and trigger envelopes for every video frame."""
        sr = self.sample_rate
        hop = sr // int(self.fps)          # samples per video frame
        n_fft = 4096
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        n_frames = self.total_frames
        energies = np.zeros((n_frames, 4), dtype=np.float32)

        audio = self.audio_data
        n_audio = len(audio)

        for fi in range(n_frames):
            start = fi * hop
            end = start + n_fft
            if start >= n_audio:
                break
            chunk = audio[start:min(end, n_audio)]
            if len(chunk) < n_fft:
                chunk = np.pad(chunk, (0, n_fft - len(chunk)))
            # Apply Hann window
            windowed = chunk * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(windowed)) / n_fft

            for bi, band in enumerate(self.bands):
                mask = (freqs >= band.freq_low) & (freqs < band.freq_high)
                if mask.any():
                    e = spectrum[mask].mean() * band.gain
                    energies[fi, bi] = e

        # Normalize each band 0–1 over the whole file
        for bi in range(4):
            col = energies[:, bi]
            mx = col.max()
            if mx > 0:
                energies[:, bi] = col / mx

        self._frame_energies = energies
        self._build_envelopes()

    def _build_envelopes(self) -> None:
        """Apply threshold + release to produce active envelopes."""
        n = self.total_frames
        dt = 1.0 / self.fps
        active = np.zeros((n, 4), dtype=np.float32)

        for bi, band in enumerate(self.bands):
            release_frames = max(1, int(band.release / dt))
            counter = 0
            for fi in range(n):
                e = self._frame_energies[fi, bi]
                if e >= band.threshold:
                    counter = release_frames
                if counter > 0:
                    active[fi, bi] = min(1.0, counter / release_frames * 1.5)
                    counter -= 1
                else:
                    active[fi, bi] = 0.0

        self._frame_active = active

    def rebuild_envelopes(self) -> None:
        """Call after changing band configs to recompute envelopes."""
        if self._frame_energies is not None:
            # Re-normalize energies with new gains
            sr = self.sample_rate
            hop = sr // int(self.fps)
            n_fft = 4096
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            n_frames = self.total_frames
            energies = np.zeros((n_frames, 4), dtype=np.float32)
            audio = self.audio_data
            n_audio = len(audio)
            for fi in range(n_frames):
                start = fi * hop
                end = start + n_fft
                if start >= n_audio:
                    break
                chunk = audio[start:min(end, n_audio)]
                if len(chunk) < n_fft:
                    chunk = np.pad(chunk, (0, n_fft - len(chunk)))
                windowed = chunk * np.hanning(n_fft)
                spectrum = np.abs(np.fft.rfft(windowed)) / n_fft
                for bi, band in enumerate(self.bands):
                    mask = (freqs >= band.freq_low) & (freqs < band.freq_high)
                    if mask.any():
                        e = spectrum[mask].mean() * band.gain
                        energies[fi, bi] = e
            for bi in range(4):
                col = energies[:, bi]
                mx = col.max()
                if mx > 0:
                    energies[:, bi] = col / mx
            self._frame_energies = energies
            self._build_envelopes()

    # ---------------------------------------------------------- query per frame
    def get_frame_data(self, frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (energies[4], active[4]) for the given frame index."""
        if self._frame_energies is None:
            return np.zeros(4, np.float32), np.zeros(4, np.float32)
        fi = min(frame_index, self.total_frames - 1)
        return self._frame_energies[fi], self._frame_active[fi]

    def get_realtime_data(self, time_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        fi = int(time_sec * self.fps)
        return self.get_frame_data(fi)

    def get_waveform_samples(self, frame_index: int, n_points: int = 128) -> np.ndarray:
        """Return (4, n_points) per-band bandpass waveform amplitudes, normalized ±1."""
        if self.audio_data is None:
            return np.zeros((4, n_points), np.float32)
        sr = self.sample_rate
        hop = sr // int(self.fps)
        fi = min(frame_index, self.total_frames - 1)
        start = fi * hop
        n_fft = 4096
        chunk = self.audio_data[start:start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        spectrum = np.fft.rfft(chunk * np.hanning(n_fft))
        step = n_fft // n_points
        indices = np.arange(0, n_fft, step)[:n_points]
        result = np.zeros((4, n_points), np.float32)
        for bi, band in enumerate(self.bands):
            mask = (freqs >= band.freq_low) & (freqs < band.freq_high)
            if not mask.any():
                continue
            filtered = np.zeros_like(spectrum)
            filtered[mask] = spectrum[mask]
            wave = np.fft.irfft(filtered)[:n_fft]
            samples = wave[indices].astype(np.float32)
            mx = np.abs(samples).max()
            if mx > 1e-9:
                samples /= mx
            result[bi] = samples
        return result

    @property
    def loaded(self) -> bool:
        return self.audio_data is not None
