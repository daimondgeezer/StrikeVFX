"""
Video export: pipes rendered frames to FFmpeg.
Supports 4K 16:9 (YouTube) and 1080x1920 (Instagram Reels).
"""
import subprocess, shutil, os, threading
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class ExportConfig:
    output_path: str = "output.mp4"
    preset: str = "youtube_4k"   # "youtube_4k" | "instagram_reels"
    fps: float = 60.0
    crf: int = 18                 # quality: lower = better (18-23 typical)
    preset_speed: str = "slow"    # ffmpeg encoding speed


PRESETS = {
    "youtube_4k": {
        "width":  3840,
        "height": 2160,
        "vf": "",
    },
    "instagram_reels": {
        "width":  1080,
        "height": 1920,
        # Crop 16:9 centre to 9:16: scale to height=1920, crop width=1080
        "vf": "scale=-2:1920,crop=1080:1920",
    },
    "preview_720p": {
        "width":  1280,
        "height": 720,
        "vf": "",
    },
}


class Exporter:
    def __init__(self, config: ExportConfig):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._frame_count = 0
        self._cancelled = False

    def start(self) -> None:
        cfg  = self.config
        info = PRESETS[cfg.preset]
        w, h = info["width"], info["height"]
        vf   = info["vf"]

        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(cfg.fps),
            "-i", "pipe:0",
        ]

        if vf:
            cmd += ["-vf", vf]

        cmd += [
            "-c:v", "libx264",
            "-preset", cfg.preset_speed,
            "-crf", str(cfg.crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            cfg.output_path,
        ]

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._frame_count = 0
        self._cancelled = False

    def write_frame(self, frame_rgb: np.ndarray) -> None:
        """frame_rgb: H×W×3 uint8 array in the export resolution."""
        if self._proc and self._proc.stdin and not self._cancelled:
            self._proc.stdin.write(frame_rgb.tobytes())
            self._frame_count += 1

    def finish(self) -> None:
        if self._proc:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait()
            self._proc = None

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc:
            self._proc.terminate()
            self._proc = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    @staticmethod
    def mux_audio(video_path: str, audio_path: str, output_path: str) -> bool:
        """Mux audio WAV into the video."""
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        try:
            result = subprocess.run([
                ffmpeg, "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "320k",
                "-shortest",
                output_path,
            ], capture_output=True, timeout=300)
            return result.returncode == 0
        except Exception:
            return False
