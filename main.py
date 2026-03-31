"""
Atlas EP – Audio-reactive 3D visualizer
Gantz-Graf style: wireframe + solid 3D objects, glitch FX, audio-driven.

Usage:
    python main.py
"""

import sys, os, time, math
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
from typing import Optional, List
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QGridLayout,
    QFileDialog, QComboBox, QDoubleSpinBox, QSpinBox, QColorDialog,
    QProgressBar, QSizePolicy, QFrame, QScrollArea, QMessageBox,
    QSplitter,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QPalette, QFont

from audio_analyzer import AudioAnalyzer, BandConfig
from renderer import Renderer
from exporter import Exporter, ExportConfig, PRESETS


# ============================================================= colour helpers
def make_dark_palette(app: QApplication) -> QPalette:
    p = QPalette()
    dark  = QColor(18, 18, 18)
    mid   = QColor(35, 35, 35)
    light = QColor(55, 55, 55)
    text  = QColor(220, 220, 220)
    accent= QColor(0, 180, 255)
    p.setColor(QPalette.Window,          dark)
    p.setColor(QPalette.WindowText,      text)
    p.setColor(QPalette.Base,            mid)
    p.setColor(QPalette.AlternateBase,   light)
    p.setColor(QPalette.Text,            text)
    p.setColor(QPalette.Button,          mid)
    p.setColor(QPalette.ButtonText,      text)
    p.setColor(QPalette.Highlight,       accent)
    p.setColor(QPalette.HighlightedText, QColor(0,0,0))
    return p



# ============================================================= Export stepper
# Runs entirely on the main thread via a QTimer so GL calls stay legal.
class ExportStepper(QObject):
    """Steps through frames one at a time on the main thread."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)

    def __init__(self, analyzer, renderer, export_cfg, audio_path):
        super().__init__()
        self.analyzer   = analyzer
        self.renderer   = renderer
        self.cfg        = export_cfg
        self.audio_path = audio_path
        self._exporter: Optional[Exporter] = None
        self._tmp_path  = ""
        self._orig_path = ""
        self._w = self._h = 0
        self._total = 0
        self._fi    = 0
        self._dt    = 1.0 / 60.0
        self._cancelled = False
        self._timer = QTimer()
        self._timer.setInterval(0)   # fire as fast as possible
        self._timer.timeout.connect(self._step_frame)

    def start(self):
        info = PRESETS[self.cfg.preset]
        self._w, self._h = info["width"], info["height"]
        self._total = self.analyzer.total_frames
        self._dt    = 1.0 / self.cfg.fps
        self._fi    = 0
        self._cancelled = False

        self._orig_path = self.cfg.output_path
        self._tmp_path  = self.cfg.output_path + ".novid.mp4"
        self.cfg.output_path = self._tmp_path

        self._exporter = Exporter(self.cfg)
        try:
            self._exporter.start()
        except Exception as e:
            self.finished.emit(False, f"FFmpeg failed to start: {e}")
            return

        # Reset renderer for clean offline pass
        self.renderer.reset()

        self._timer.start()

    def _step_frame(self):
        if self._cancelled:
            self._exporter.cancel()
            self.finished.emit(False, "Export cancelled.")
            self._timer.stop()
            return

        if self._fi >= self._total:
            self._finish()
            return

        energies, active = self.analyzer.get_frame_data(self._fi)
        waveforms = self.analyzer.get_waveform_samples(self._fi)
        self.renderer.step(self._dt, energies, active, waveforms)
        frame = self.renderer.render_frame_to_array(self._w, self._h)
        self._exporter.write_frame(frame)
        self.progress.emit(self._fi + 1, self._total)
        self._fi += 1

    def _finish(self):
        self._timer.stop()
        self._exporter.finish()
        self.cfg.output_path = self._orig_path

        # Mux audio
        ok = False
        if os.path.exists(self.audio_path):
            ok = Exporter.mux_audio(self._tmp_path, self.audio_path, self._orig_path)
        if not ok:
            # No audio mux — just use raw video
            try:
                os.replace(self._tmp_path, self._orig_path)
                ok = True
            except Exception:
                pass
        try:
            if os.path.exists(self._tmp_path):
                os.remove(self._tmp_path)
        except Exception:
            pass

        self.finished.emit(ok, f"Saved: {self._orig_path}")

    def cancel(self):
        self._cancelled = True


# ============================================================= Band Widget
class BandWidget(QGroupBox):
    changed       = pyqtSignal()
    color_changed = pyqtSignal(list)   # emits [r, g, b] floats

    def __init__(self, band_idx: int, config: BandConfig, initial_color: list, parent=None):
        labels = ["Band 1 – Sub/Bass", "Band 2 – Low-Mid",
                  "Band 3 – High-Mid", "Band 4 – Highs"]
        super().__init__(labels[band_idx], parent)
        self.band_idx = band_idx
        self.config = config
        self._color = list(initial_color)
        self._build_ui()

    def _build_ui(self):
        grid = QGridLayout(self)
        grid.setSpacing(4)

        def lbl(text):
            l = QLabel(text)
            l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            l.setFixedWidth(70)
            return l

        def spin(lo, hi, val, decimals=1, suffix=""):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setValue(val)
            s.setFixedWidth(72)
            if suffix:
                s.setSuffix(suffix)
            return s

        # Row 0: freq range
        grid.addWidget(lbl("Lo Hz:"), 0, 0)
        self.lo = spin(20, 20000, self.config.freq_low, 0, " Hz")
        grid.addWidget(self.lo, 0, 1)
        grid.addWidget(lbl("Hi Hz:"), 0, 2)
        self.hi = spin(20, 20000, self.config.freq_high, 0, " Hz")
        grid.addWidget(self.hi, 0, 3)

        # Row 1: threshold slider + spinbox
        grid.addWidget(lbl("Threshold:"), 1, 0)
        self.thresh = QSlider(Qt.Horizontal)
        self.thresh.setRange(0, 100)
        self.thresh.setValue(int(self.config.threshold * 100))
        grid.addWidget(self.thresh, 1, 1, 1, 2)
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.0, 1.0); self.thresh_spin.setDecimals(2)
        self.thresh_spin.setSingleStep(0.01); self.thresh_spin.setFixedWidth(60)
        self.thresh_spin.setValue(self.config.threshold)
        grid.addWidget(self.thresh_spin, 1, 3)

        # Row 2: release slider + spinbox
        grid.addWidget(lbl("Release:"), 2, 0)
        self.release = QSlider(Qt.Horizontal)
        self.release.setRange(1, 200)
        self.release.setValue(int(self.config.release * 100))
        grid.addWidget(self.release, 2, 1, 1, 2)
        self.release_spin = QDoubleSpinBox()
        self.release_spin.setRange(0.01, 2.0); self.release_spin.setDecimals(2)
        self.release_spin.setSuffix(" s"); self.release_spin.setSingleStep(0.01)
        self.release_spin.setFixedWidth(66); self.release_spin.setValue(self.config.release)
        grid.addWidget(self.release_spin, 2, 3)

        # Row 3: gain slider + spinbox
        grid.addWidget(lbl("Gain:"), 3, 0)
        self.gain = QSlider(Qt.Horizontal)
        self.gain.setRange(10, 400)
        self.gain.setValue(int(self.config.gain * 100))
        grid.addWidget(self.gain, 3, 1, 1, 2)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 4.0); self.gain_spin.setDecimals(2)
        self.gain_spin.setSuffix(" ×"); self.gain_spin.setSingleStep(0.05)
        self.gain_spin.setFixedWidth(66); self.gain_spin.setValue(self.config.gain)
        grid.addWidget(self.gain_spin, 3, 3)

        # Row 4: level meter
        grid.addWidget(lbl("Level:"), 4, 0)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100); self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(10)
        self.meter.setStyleSheet("""
            QProgressBar { background: #222; border: 1px solid #444; border-radius:3px; }
            QProgressBar::chunk { background: #00b4ff; border-radius:3px; }
        """)
        grid.addWidget(self.meter, 4, 1, 1, 3)

        # Row 5: colour picker
        grid.addWidget(lbl("Colour:"), 5, 0)
        r, g, b = self._color
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(40, 22)
        self.color_btn.setStyleSheet(
            f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)}); border: 1px solid #555;")
        self.color_btn.clicked.connect(self._pick_color)
        grid.addWidget(self.color_btn, 5, 1)

        # --- Connect sliders → spinboxes (bidirectional, guard against loops)
        self._updating = False

        def thresh_slider(v):
            if self._updating: return
            self._updating = True
            self.thresh_spin.setValue(v / 100.0)
            self._updating = False
            self._on_thresh()

        def thresh_spin_changed(v):
            if self._updating: return
            self._updating = True
            self.thresh.setValue(int(v * 100))
            self._updating = False
            self._on_thresh()

        def release_slider(v):
            if self._updating: return
            self._updating = True
            self.release_spin.setValue(v / 100.0)
            self._updating = False
            self._on_release()

        def release_spin_changed(v):
            if self._updating: return
            self._updating = True
            self.release.setValue(int(v * 100))
            self._updating = False
            self._on_release()

        def gain_slider(v):
            if self._updating: return
            self._updating = True
            self.gain_spin.setValue(v / 100.0)
            self._updating = False
            self._on_gain()

        def gain_spin_changed(v):
            if self._updating: return
            self._updating = True
            self.gain.setValue(int(v * 100))
            self._updating = False
            self._on_gain()

        self.thresh.valueChanged.connect(thresh_slider)
        self.thresh_spin.valueChanged.connect(thresh_spin_changed)
        self.release.valueChanged.connect(release_slider)
        self.release_spin.valueChanged.connect(release_spin_changed)
        self.gain.valueChanged.connect(gain_slider)
        self.gain_spin.valueChanged.connect(gain_spin_changed)
        self.lo.valueChanged.connect(self._on_freq)
        self.hi.valueChanged.connect(self._on_freq)

    def _pick_color(self):
        r, g, b = self._color
        initial = QColor(int(r*255), int(g*255), int(b*255))
        c = QColorDialog.getColor(initial, self, "Pick Band Colour")
        if c.isValid():
            self._color = [c.redF(), c.greenF(), c.blueF()]
            self.color_btn.setStyleSheet(
                f"background-color: {c.name()}; border: 1px solid #555;")
            self.color_changed.emit(self._color)

    def _on_thresh(self):
        self.config.threshold = self.thresh_spin.value()
        self.changed.emit()

    def _on_release(self):
        self.config.release = self.release_spin.value()
        self.changed.emit()

    def _on_gain(self):
        self.config.gain = self.gain_spin.value()
        self.changed.emit()

    def _on_freq(self):
        self.config.freq_low  = self.lo.value()
        self.config.freq_high = self.hi.value()
        self.changed.emit()

    def update_meter(self, energy: float, active: float):
        self.meter.setValue(int(energy * 100))
        color = "#ff4400" if active > 0.01 else "#00b4ff"
        self.meter.setStyleSheet(f"""
            QProgressBar {{ background: #222; border: 1px solid #444; border-radius:3px; }}
            QProgressBar::chunk {{ background: {color}; border-radius:3px; }}
        """)



# ============================================================= Main Window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atlas EP – Audio Visualizer")
        self.resize(1600, 900)

        self.analyzer = AudioAnalyzer(fps=60.0)
        self._audio_path = ""
        self._playing    = False
        self._audio_start_wall = 0.0   # wall-clock time when Play was pressed
        self._current_frame = 0

        self._export_stepper: Optional[ExportStepper] = None

        self._build_ui()
        self._start_preview_timer()

    # ---------------------------------------------------- UI construction
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)

        # ---- LEFT: renderer preview
        left = QVBoxLayout()
        self.renderer = Renderer()
        self.renderer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.renderer, stretch=1)

        # Transport controls
        transport = QHBoxLayout()
        self.load_btn  = QPushButton("Load WAV")
        self.play_btn  = QPushButton("Play")
        self.stop_btn  = QPushButton("Stop")
        self.file_lbl  = QLabel("No file loaded")
        self.file_lbl.setStyleSheet("color: #888;")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        for w in [self.load_btn, self.play_btn, self.stop_btn, self.file_lbl]:
            transport.addWidget(w)
        transport.addStretch()
        left.addLayout(transport)

        # Seek slider + time label
        seek_row = QHBoxLayout()
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 10000)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setStyleSheet("""
            QSlider::groove:horizontal { background:#222; height:6px; border-radius:3px; }
            QSlider::handle:horizontal { background:#00b4ff; width:14px; height:14px;
                                         margin:-4px 0; border-radius:7px; }
            QSlider::sub-page:horizontal { background:#00b4ff; border-radius:3px; }
        """)
        self.time_lbl = QLabel("0:00 / 0:00")
        self.time_lbl.setFixedWidth(90)
        self.time_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        seek_row.addWidget(self.seek_slider)
        seek_row.addWidget(self.time_lbl)
        left.addLayout(seek_row)
        self._seek_dragging = False
        self.seek_slider.sliderPressed.connect(self._seek_pressed)
        self.seek_slider.sliderReleased.connect(self._seek_released)
        self.seek_slider.sliderMoved.connect(self._seek_moved)

        # Stats label (debug)
        self.stats_lbl = QLabel("Load a WAV file and press Play")
        self.stats_lbl.setStyleSheet("color:#666; font-size:10px; padding:2px;")
        left.addWidget(self.stats_lbl)

        # ---- RIGHT: controls panel
        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setFixedWidth(360)
        right.setFrameShape(QFrame.NoFrame)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(8)

        # Band controls (each has its own colour picker)
        self.band_widgets = []
        for i, cfg in enumerate(self.analyzer.bands):
            bw = BandWidget(i, cfg, self.renderer.band_colors[i])
            bw.changed.connect(self._on_band_changed)
            bw.color_changed.connect(lambda color, idx=i: self._on_band_color_changed(idx, color))
            panel_layout.addWidget(bw)
            self.band_widgets.append(bw)

        # Export group
        exp_group = QGroupBox("Export")
        exp_layout = QGridLayout(exp_group)

        exp_layout.addWidget(QLabel("Preset:"), 0, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["youtube_4k", "instagram_reels", "preview_720p"])
        exp_layout.addWidget(self.preset_combo, 0, 1)

        exp_layout.addWidget(QLabel("CRF (quality):"), 1, 0)
        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51); self.crf_spin.setValue(18)
        self.crf_spin.setToolTip("Lower = better quality, larger file (18=high, 23=medium)")
        exp_layout.addWidget(self.crf_spin, 1, 1)

        exp_layout.addWidget(QLabel("Speed:"), 2, 0)
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"])
        self.speed_combo.setCurrentText("slow")
        exp_layout.addWidget(self.speed_combo, 2, 1)

        self.export_btn = QPushButton("Export Video")
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("background: #005a9e; color: white; font-weight: bold; padding: 6px;")
        exp_layout.addWidget(self.export_btn, 3, 0, 1, 2)

        self.export_progress = QProgressBar()
        self.export_progress.setRange(0, 100)
        self.export_progress.setValue(0)
        self.export_progress.setVisible(False)
        exp_layout.addWidget(self.export_progress, 4, 0, 1, 2)

        self.cancel_btn = QPushButton("Cancel Export")
        self.cancel_btn.setVisible(False)
        exp_layout.addWidget(self.cancel_btn, 5, 0, 1, 2)

        panel_layout.addWidget(exp_group)
        panel_layout.addStretch()
        right.setWidget(panel)

        # Assemble
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left)
        splitter.addWidget(left_widget)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        # Connections
        self.load_btn.clicked.connect(self._load_wav)
        self.play_btn.clicked.connect(self._play)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn.clicked.connect(self._start_export)
        self.cancel_btn.clicked.connect(self._cancel_export)

    # ---------------------------------------------------- preview timer
    def _start_preview_timer(self):
        self._last_wall = time.perf_counter()
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(16)   # ~60 fps
        self._preview_timer.timeout.connect(self._tick)
        self._preview_timer.start()

    def _tick(self):
        now = time.perf_counter()
        dt  = now - self._last_wall
        self._last_wall = now

        if self._playing and self.analyzer.loaded:
            # Sync frame to wall clock so it stays locked to audio
            elapsed = time.perf_counter() - self._audio_start_wall
            self._current_frame = int(elapsed * self.analyzer.fps)
            if self._current_frame >= self.analyzer.total_frames:
                self._stop()
                return
            energies, active = self.analyzer.get_frame_data(self._current_frame)
            waveforms = self.analyzer.get_waveform_samples(self._current_frame)
            self.renderer.step(dt, energies, active, waveforms)
            # Update band meters
            for bi, bw in enumerate(self.band_widgets):
                bw.update_meter(float(energies[bi]), float(active[bi]))
            # Update seek slider + time label
            if not self._seek_dragging:
                pos = int(self._current_frame / max(1, self.analyzer.total_frames) * 10000)
                self.seek_slider.setValue(pos)
            elapsed = self._current_frame / max(1, self.analyzer.fps)
            self.time_lbl.setText(f"{_fmt_time(elapsed)} / {_fmt_time(self.analyzer.duration)}")
            # Stats label
            e_str = " ".join(f"{v:.2f}" for v in energies)
            a_str = " ".join(f"{v:.2f}" for v in active)
            self.stats_lbl.setText(
                f"Frame {self._current_frame} | "
                f"Energy: [{e_str}] | Active: [{a_str}] | "
                f"Objects: {len(self.renderer.objects)} | "
                f"Queue: {self.renderer.spawn_queue_size}"
            )

    # ---------------------------------------------------- actions
    def _load_wav(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open WAV file", "", "WAV files (*.wav)")
        if not path:
            return
        try:
            self.analyzer.load(path)
            self._audio_path = path
            fname = os.path.basename(path)
            dur   = self.analyzer.duration
            self.file_lbl.setText(f"{fname}  ({dur:.1f}s, {self.analyzer.total_frames} frames)")
            self.play_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            self.seek_slider.setEnabled(True)
            self.time_lbl.setText(f"0:00 / {_fmt_time(self.analyzer.duration)}")
            self._current_frame = 0
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _play(self):
        if not self.analyzer.loaded:
            return
        self._current_frame = 0
        self.renderer.reset()
        # Start audio playback
        if _SD_AVAILABLE:
            try:
                sd.stop()
                sd.play(self.analyzer.audio_data,
                        samplerate=self.analyzer.sample_rate,
                        blocking=False)
            except Exception:
                pass
        self._audio_start_wall = time.perf_counter()
        self._playing = True
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self):
        self._playing = False
        self._current_frame = 0
        if _SD_AVAILABLE:
            try: sd.stop()
            except Exception: pass
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.seek_slider.setValue(0)
        self.time_lbl.setText(f"0:00 / {_fmt_time(self.analyzer.duration)}")

    def _on_band_color_changed(self, band_idx: int, color: list):
        self.renderer.band_colors[band_idx] = list(color)

    def _on_band_changed(self):
        if self.analyzer.loaded:
            self.analyzer.rebuild_envelopes()

    def _start_export(self):
        if not self.analyzer.loaded:
            return
        if not Exporter.ffmpeg_available():
            QMessageBox.critical(self, "FFmpeg Missing",
                "FFmpeg not found in PATH.\n"
                "Install FFmpeg and make sure it's in your system PATH.\n"
                "Download: https://ffmpeg.org/download.html")
            return

        preset = self.preset_combo.currentText()
        ext_map = {"youtube_4k": "_4k.mp4", "instagram_reels": "_reels.mp4", "preview_720p": "_720p.mp4"}
        default_name = os.path.splitext(self._audio_path)[0] + ext_map.get(preset, ".mp4")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", default_name, "MP4 files (*.mp4)")
        if not path:
            return

        cfg = ExportConfig(
            output_path  = path,
            preset       = preset,
            fps          = 60.0,
            crf          = self.crf_spin.value(),
            preset_speed = self.speed_combo.currentText(),
        )

        self._preview_timer.stop()   # pause live preview during export
        self._playing = False

        self._export_stepper = ExportStepper(self.analyzer, self.renderer, cfg, self._audio_path)
        self._export_stepper.progress.connect(self._on_export_progress)
        self._export_stepper.finished.connect(self._on_export_finished)
        self._export_stepper.start()

        self.export_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.export_progress.setVisible(True)
        self.cancel_btn.setVisible(True)
        self.export_progress.setValue(0)

    def _seek_pressed(self):
        self._seek_dragging = True

    def _seek_released(self):
        self._seek_dragging = False
        self._seek_to(self.seek_slider.value())

    def _seek_moved(self, value):
        # Update time label while dragging without actually seeking
        if self.analyzer.loaded:
            t = value / 10000.0 * self.analyzer.duration
            self.time_lbl.setText(f"{_fmt_time(t)} / {_fmt_time(self.analyzer.duration)}")

    def _seek_to(self, slider_value):
        if not self.analyzer.loaded:
            return
        frame = int(slider_value / 10000.0 * self.analyzer.total_frames)
        self._current_frame = max(0, min(frame, self.analyzer.total_frames - 1))
        self.renderer.reset()
        self.renderer._time  = self._current_frame / self.analyzer.fps
        self.renderer._frame = self._current_frame
        # Restart audio from seek position and re-sync wall clock
        if self._playing and _SD_AVAILABLE:
            seek_sample = self._current_frame * (self.analyzer.sample_rate // int(self.analyzer.fps))
            try:
                sd.stop()
                sd.play(self.analyzer.audio_data[seek_sample:],
                        samplerate=self.analyzer.sample_rate, blocking=False)
            except Exception:
                pass
            self._audio_start_wall = time.perf_counter() - (self._current_frame / self.analyzer.fps)

    def _cancel_export(self):
        if self._export_stepper:
            self._export_stepper.cancel()

    def _on_export_progress(self, current, total):
        self.export_progress.setValue(int(current / total * 100))

    def _on_export_finished(self, success, message):
        self.export_progress.setVisible(False)
        self.cancel_btn.setVisible(False)
        self.export_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.play_btn.setEnabled(self.analyzer.loaded)
        self._export_stepper = None
        # Restore FBO to preview resolution and restart preview timer
        self.renderer.reset()
        self._last_wall = time.perf_counter()
        self._preview_timer.start()
        if success:
            QMessageBox.information(self, "Export Complete", message)
        else:
            QMessageBox.warning(self, "Export", message)

    def closeEvent(self, event):
        if self._export_stepper:
            self._export_stepper.cancel()
        event.accept()


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s//60}:{s%60:02d}"


# ============================================================= Entry point
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(make_dark_palette(app))
    app.setFont(QFont("Segoe UI", 9))

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
