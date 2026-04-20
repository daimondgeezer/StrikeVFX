"""
Audio analysis module: loads WAV, computes per-frame FFT,
extracts 4 configurable frequency bands with threshold + release logic.
"""
import math
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
    BandConfig(20,    200,   threshold=0.30, release=0.20, gain=1.0),   # Sub/Bass
    BandConfig(150,   350,   threshold=0.25, release=0.10, gain=1.0),   # Snare range
    BandConfig(2000,  8000,  threshold=0.25, release=0.07, gain=1.0),   # High-Mid
    BandConfig(8000,  20000, threshold=0.20, release=0.05, gain=1.2),   # Highs
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
        # Spectral flux (onset strength) per frame  shape: (total_frames, 4)
        self._frame_flux: Optional[np.ndarray] = None
        # Per-frame trigger/active envelopes     shape: (total_frames, 4)
        self._frame_active: Optional[np.ndarray] = None
        # Waveform cache: frame_index -> (4, n_points) array
        self._waveform_cache: dict = {}

        # Precomputed waveform helpers (rebuilt on load/freq-range change)
        self._waveform_win: Optional[np.ndarray] = None      # hanning window
        self._waveform_freqs: Optional[np.ndarray] = None    # rfft freq bins
        self._waveform_masks: Optional[list] = None          # per-band bool masks
        self._waveform_indices: Optional[np.ndarray] = None  # sample indices

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
    # Short window for energy/flux — 1024 samples ≈ 23ms at 44100Hz.
    # Gives 4× better temporal resolution vs the old 4096 window so transients
    # (snares, hi-hats, plucks) register as sharp spikes rather than slow ramps.
    _N_FFT_ENERGY = 1024

    # Bands whose 99th-percentile raw energy is below this are considered silent.
    # Prevents floating-point noise (p99 ~ 1e-30) from being normalized to 1.0
    # and causing false triggers across the whole file.
    _MIN_SIGNAL = 1e-5

    def _precompute(self) -> None:
        """Pre-compute band energies, spectral flux, and trigger envelopes."""
        sr = self.sample_rate
        hop = sr // int(self.fps)

        n_fft = self._N_FFT_ENERGY
        win   = np.hanning(n_fft).astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

        # Precompute band frequency masks — same for every frame
        masks = []
        for band in self.bands:
            m = (freqs >= band.freq_low) & (freqs < band.freq_high)
            masks.append(m)

        # Cache waveform helpers so get_waveform_samples avoids 6+ numpy
        # allocations per call (hanning, rfftfreq, 4 masks, arange).
        self._waveform_win     = win
        self._waveform_freqs   = freqs
        self._waveform_masks   = list(masks)
        _step                  = max(1, n_fft // 128)
        self._waveform_indices = np.arange(0, n_fft, _step, dtype=np.int64)[:128]

        n_frames = self.total_frames
        energies = np.zeros((n_frames, 4), dtype=np.float32)
        audio = self.audio_data
        n_audio = len(audio)

        for fi in range(n_frames):
            start = fi * hop
            chunk = audio[start:start + n_fft]
            if len(chunk) < n_fft:
                chunk = np.pad(chunk, (0, n_fft - len(chunk)))
            spectrum = np.abs(np.fft.rfft(chunk * win)) / n_fft
            for bi in range(4):
                if masks[bi].any():
                    energies[fi, bi] = spectrum[masks[bi]].mean()

        # Normalize per band to 99th percentile.
        # IMPORTANT: only normalize if the band has meaningful signal.
        # `if p99 > 0` is not enough — floating-point noise gives p99 ~ 1e-30,
        # which maps silence → 1.0 everywhere and causes false triggers at any threshold.
        # _MIN_SIGNAL is the floor below which a band is treated as empty.
        for bi in range(4):
            col = energies[:, bi]
            p99 = np.percentile(col, 99)
            if p99 >= self._MIN_SIGNAL:
                energies[:, bi] = np.clip(col / p99, 0.0, 1.0)
            else:
                energies[:, bi] = 0.0   # band is silent — don't amplify noise

        self._frame_energies = energies

        # Spectral flux: positive onset strength = how much energy just increased.
        # Triggers on attacks only, not sustained content.
        flux = np.diff(energies, axis=0, prepend=energies[:1])
        flux = np.clip(flux, 0.0, None).astype(np.float32)
        for bi in range(4):
            col = flux[:, bi]
            p99 = np.percentile(col, 99)
            if p99 >= self._MIN_SIGNAL:
                flux[:, bi] = np.clip(col / p99, 0.0, 1.0)
            else:
                flux[:, bi] = 0.0       # no signal → no flux
        self._frame_flux = flux

        self._build_envelopes()

    def _build_envelopes(self) -> None:
        """Build active envelopes from band energy with exponential release.

        Acts as an audio gate: when energy exceeds threshold the band is open
        (active=1.0); when energy falls below, the envelope decays with an
        exponential tail whose time-constant is band.release.

        Using strict > comparison means threshold=1.0 is truly silent — nothing
        in the normalised [0,1] energy range can exceed 1.0, so no frames fire.
        """
        n = self.total_frames
        dt = 1.0 / self.fps
        active = np.zeros((n, 4), dtype=np.float32)
        energies = self._frame_energies

        for bi, band in enumerate(self.bands):
            # Decay multiplier per frame: falls to 1/e after `release` seconds
            decay = math.exp(-dt / max(band.release, 0.005))
            val = 0.0
            thr = band.threshold
            for fi in range(n):
                if energies[fi, bi] > thr:
                    val = 1.0          # gate open: energy above threshold
                else:
                    val *= decay       # gate releasing
                active[fi, bi] = val

        self._frame_active = active

    def rebuild_envelopes_only(self) -> None:
        """Recompute only threshold/release envelopes — fast, no FFT."""
        if self._frame_energies is not None:
            self._build_envelopes()

    def rebuild_full(self) -> None:
        """Full recompute: re-run FFT for all frames then rebuild envelopes.
        Call when freq range or gain changes."""
        if self._frame_energies is None:
            return
        self._waveform_cache.clear()
        self._precompute()

    # Keep old name as alias for compat
    def rebuild_envelopes(self) -> None:
        self.rebuild_full()

    # ---------------------------------------------------------- query per frame
    def get_frame_data(self, frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (energies[4], active[4]) for the given frame index.
        active values are scaled by band.gain so the gain slider controls
        visual intensity (object scale/morph) independently of trigger threshold.
        """
        if self._frame_energies is None:
            return np.zeros(4, np.float32), np.zeros(4, np.float32)
        fi = min(frame_index, self.total_frames - 1)
        active = self._frame_active[fi].copy()
        for bi, band in enumerate(self.bands):
            active[bi] = min(1.0, active[bi] * band.gain)
        return self._frame_energies[fi], active

    def get_realtime_data(self, time_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        fi = int(time_sec * self.fps)
        return self.get_frame_data(fi)

    def get_waveform_samples(self, frame_index: int, n_points: int = 128) -> np.ndarray:
        """Return (4, n_points) per-band bandpass waveform amplitudes, normalized ±1.
        Results are cached per frame index so repeated calls are free."""
        if self.audio_data is None or self._waveform_win is None:
            return np.zeros((4, n_points), np.float32)
        fi = min(frame_index, self.total_frames - 1)
        cached = self._waveform_cache.get(fi)
        if cached is not None:
            return cached
        hop = self.sample_rate // int(self.fps)
        start = fi * hop
        n_fft = len(self._waveform_win)
        chunk = self.audio_data[start:start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        # Use precomputed window and band masks — avoids 6+ numpy allocations per call
        spectrum = np.fft.rfft(chunk * self._waveform_win)
        indices  = self._waveform_indices[:n_points]
        result   = np.zeros((4, n_points), np.float32)
        for bi, mask in enumerate(self._waveform_masks):
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
        if len(self._waveform_cache) >= 600:
            self._waveform_cache.clear()
        self._waveform_cache[fi] = result
        return result

    @property
    def loaded(self) -> bool:
        return self.audio_data is not None
