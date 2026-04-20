"""
OpenGL renderer — PyOpenGL + PyQt5 QOpenGLWidget.
Preview: direct to screen, or preview FBO + glitch post-process on loud peaks.
Export: scene FBO → post FBO → readPixels.
ALL GL calls happen inside paintGL / initializeGL.
step() only mutates Python state and queues spawn requests.
"""
import math
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QSurfaceFormat

from OpenGL import GL
from OpenGL.GL import shaders

import geometry as geo

N_WAVE = 128   # waveform ring resolution per band
_ZERO_WAVE = np.zeros(N_WAVE, np.float32)  # fallback when no waveform data


# ============================================================ GLSL

_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4  uMVP;
uniform mat4  uModel;
uniform float uMorph;
uniform float uTime;
uniform float uWave[128];
out vec3 vNorm;
out vec3 vPos;
void main(){
    vec3 pos = aPos;
    if(uMorph > 0.001){
        // Map vertex angle (XZ plane, camera orbits Y) to a waveform sample.
        // 0.15915 = 1/(2*pi), gives a 0..1 range from atan output.
        float phi = atan(aPos.z, aPos.x) * 0.15915 + 0.5;
        int   wi  = clamp(int(phi * 128.0), 0, 127);
        float wave = uWave[wi];   // signed -1..1 audio amplitude

        // Slow sinusoidal shiver keeps wire shapes lively between transients
        float shiver = sin(aPos.x * 8.37 + uTime * 2.13)
                     + sin(aPos.y * 6.18 + uTime * 3.47)
                     + sin(aPos.z * 9.82 + uTime * 1.88);

        pos += aNorm * (wave * 0.7 + shiver * 0.1) * uMorph * 0.55;
    }
    vNorm = mat3(transpose(inverse(uModel))) * aNorm;
    vPos  = vec3(uModel * vec4(pos, 1.0));
    gl_Position = uMVP * vec4(pos, 1.0);
}
"""

_FRAG = """
#version 330 core
in vec3 vNorm;
in vec3 vPos;
uniform vec3  uColor;
uniform float uAlpha;
uniform bool  uWire;
out vec4 fragColor;
void main(){
    if(uWire){
        fragColor = vec4(uColor, uAlpha);
    } else {
        vec3 light = normalize(vec3(0.5, 1.0, 0.7));
        float diff = max(dot(normalize(vNorm), light), 0.0);
        vec3 c = uColor * (0.25 + 0.75 * diff);
        fragColor = vec4(c, uAlpha);
    }
}
"""

# Waveform ring — no lighting, just position + colour
_VERT_WAVE = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos, 1.0); }
"""

_FRAG_WAVE = """
#version 330 core
uniform vec3  uColor;
uniform float uAlpha;
out vec4 fragColor;
void main(){ fragColor = vec4(uColor, uAlpha); }
"""

# Fullscreen glitch post-process
_VERT_POST = """
#version 330 core
layout(location=0) in vec2 aPos;
out vec2 vUV;
void main(){ vUV = aPos*0.5+0.5; gl_Position = vec4(aPos,0,1); }
"""

_FRAG_POST = """
#version 330 core
in vec2 vUV;
uniform sampler2D uTex;
uniform float uGlitch;
uniform float uTime;
uniform float uSeed;
out vec4 fragColor;
float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898,78.233)))*43758.5453 + uSeed);
}
void main(){
    vec2 uv = vUV;
    if(uGlitch > 0.01){
        float row = floor(uv.y * 720.0);
        if(rand(vec2(row, uTime)) < uGlitch * 0.4)
            uv.x += (rand(vec2(row+1.0, uTime))-0.5) * uGlitch * 0.08;
        float bY = floor(uv.y * 12.0);
        if(rand(vec2(bY, floor(uTime*10.0))) < uGlitch * 0.25)
            uv.x += (rand(vec2(bY, uTime*3.0))-0.5) * uGlitch * 0.15;
    }
    float ca = uGlitch * 0.012;
    float r = texture(uTex, uv+vec2(ca,0)).r;
    float g = texture(uTex, uv).g;
    float b = texture(uTex, uv-vec2(ca,0)).b;
    vec3 col = vec3(r,g,b);
    if(uGlitch > 0.05) col += rand(uv+vec2(uTime*0.1)) * uGlitch * 0.12;
    fragColor = vec4(col, 1.0);
}
"""


# ============================================================ Data

@dataclass
class SpawnRequest:
    gtype: int; is_wire: bool
    pos: list; rot: list; scale: float
    color: list; alpha: float; rot_speed: list
    lifetime: float; band_idx: int


class SceneObject:
    __slots__ = ['vao','vbo','ibo','index_count','mode',
                 'pos','rot','scale','color','alpha',
                 'rot_speed','lifetime','age','is_wire','band_index','morph_level']
    def __init__(self, vao, vbo, ibo, index_count, mode,
                 pos, rot, scale, color, alpha,
                 rot_speed, lifetime, is_wire, band_idx):
        self.vao=vao; self.vbo=vbo; self.ibo=ibo
        self.index_count=index_count; self.mode=mode
        self.pos       = np.array(pos,       np.float32)
        self.rot       = np.array(rot,       np.float32)
        self.rot_speed = np.array(rot_speed, np.float32)
        self.color     = np.array(color,     np.float32)
        self.scale     = float(scale)
        self.alpha     = float(alpha)
        self.lifetime  = float(lifetime)
        self.age       = 0.0
        self.is_wire   = bool(is_wire)
        self.band_index = int(band_idx)
        self.morph_level = 0.0


# ============================================================ Renderer

class Renderer(QOpenGLWidget):
    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSamples(4)
        QSurfaceFormat.setDefaultFormat(fmt)
        super().__init__(parent)

        # One colour per band — user-selectable
        self.band_colors: List[list] = [
            [1.0, 1.0, 1.0],   # Band 1 Sub/Bass  – white
            [0.0, 0.8, 1.0],   # Band 2 Low-Mid   – cyan
            [1.0, 0.2, 0.0],   # Band 3 High-Mid  – orange/red
            [0.2, 1.0, 0.4],   # Band 4 Highs     – green
        ]

        self.objects:      List[SceneObject]  = []
        self._spawn_queue: List[SpawnRequest] = []
        self._dead_queue:  List[SceneObject]  = []

        self.band_energies    = np.zeros(4, np.float32)
        self.band_active      = np.zeros(4, np.float32)
        self.glitch_intensity = 0.0

        # Per-band user controls
        self.band_max_objects  = [2, 2, 2, 2]        # max simultaneous objects per band
        self.band_shapes       = [-1, -1, -1, -1]    # -1=random, 0-4=fixed shape index
        self.band_glitch_amount= [0.0, 0.0, 0.0, 0.0]  # 0-1 glitch contribution per band
        self.band_muted        = [False, False, False, False]

        self._time  = 0.0
        self._frame = 0
        self._rng   = np.random.default_rng(42)
        self._spawn_cd = np.zeros(4, np.float32)

        self._waveform_data = np.zeros((4, N_WAVE), np.float32)
        _angles = np.linspace(0, 2*math.pi, N_WAVE, endpoint=False, dtype=np.float32)
        self._wave_cos = np.cos(_angles)
        self._wave_sin = np.sin(_angles)

        self._prog      = None
        self._wave_prog = None
        self._post_prog = None

        # Export FBOs: _fbo = scene pass, _fbo2 = post-process output
        self._fbo  = self._fbo_tex  = self._fbo_rbo  = None
        self._fbo2 = self._fbo2_tex = None
        self._fbo_w = 1920; self._fbo_h = 1080

        # Preview FBO (only built/used when glitch_intensity > 0)
        self._pfbo = self._pfbo_tex = self._pfbo_rbo = None
        self._pfbo_w = 0; self._pfbo_h = 0

        self._quad_vao = self._quad_vbo = None
        self._wave_vaos: List[Optional[int]] = [None]*4
        self._wave_vbos: List[Optional[int]] = [None]*4

        self._cam_dist    = 5.0
        self._cam_azimuth = 0.0

        self.spawn_queue_size = 0   # expose for UI stats

    def minimumSizeHint(self): return QSize(640, 360)

    # -------------------------------------------------------- GL init
    def initializeGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glLineWidth(1.0)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)
        self._prog      = _compile(_VERT,      _FRAG)
        self._wave_prog = _compile(_VERT_WAVE,  _FRAG_WAVE)
        self._post_prog = _compile(_VERT_POST,  _FRAG_POST)
        self._build_quad()
        self._build_wave_vaos()

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)

    # -------------------------------------------------------- paintGL (preview)
    def paintGL(self):
        # Drain dead GL objects
        for obj in self._dead_queue:
            GL.glDeleteVertexArrays(1, [obj.vao])
            GL.glDeleteBuffers(2, [obj.vbo, obj.ibo])
        self._dead_queue.clear()

        # Materialise pending spawns
        for req in self._spawn_queue:
            self._materialise(req)
        self._spawn_queue.clear()
        self.spawn_queue_size = 0

        w, h = self.width(), self.height()
        # Only pay for the FBO round-trip when there are actually objects to glitch.
        # Rendering to an intermediate FBO costs a full extra pass on every frame;
        # skip it entirely when the scene is empty.
        use_glitch = self.glitch_intensity > 0.02 and bool(self.objects)

        self._upload_waveforms()

        if use_glitch:
            # Render the scene at half resolution to reduce intermediate FBO cost
            # (~4× fewer rasterised pixels).  The post shader bilinearly upscales
            # to fill the screen, which suits the glitch aesthetic.
            pw, ph = max(1, w // 2), max(1, h // 2)
            if self._pfbo is None or pw != self._pfbo_w or ph != self._pfbo_h:
                self._build_pfbo(pw, ph)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._pfbo)
            GL.glViewport(0, 0, pw, ph)
        else:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
            GL.glViewport(0, 0, w, h)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        if self.objects:
            pv = self._build_pv(w / max(h, 1))
            # Compute model matrices once per object — reused by both draw passes
            models = [_model(obj.pos, obj.rot, obj.scale) for obj in self.objects]
            self._draw_scene(pv, models)
            self._draw_waveforms(pv, models)

        if use_glitch:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
            GL.glViewport(0, 0, w, h)
            self._post_pass(self._pfbo_tex)

    # -------------------------------------------------------- helpers
    def _build_pv(self, aspect):
        proj = _perspective(60.0, aspect, 0.1, 100.0)
        view = _look_at(
            np.array([math.sin(self._cam_azimuth)*self._cam_dist,
                      1.5, math.cos(self._cam_azimuth)*self._cam_dist], np.float32),
            np.zeros(3, np.float32),
            np.array([0., 1., 0.], np.float32),
        )
        return proj @ view

    def _draw_scene(self, pv, models):
        GL.glUseProgram(self._prog)
        last_band = -1
        for obj, model in zip(self.objects, models):
            mvp   = pv @ model
            alpha = obj.alpha * _fade(obj)
            # GL_TRUE: OpenGL transposes row-major numpy → column-major, no copy needed
            GL.glUniformMatrix4fv(_u(self._prog,'uMVP'),   1, GL.GL_TRUE, mvp)
            GL.glUniformMatrix4fv(_u(self._prog,'uModel'), 1, GL.GL_TRUE, model)
            GL.glUniform3fv(_u(self._prog,'uColor'),  1, obj.color)
            GL.glUniform1f(_u(self._prog,'uAlpha'),   alpha)
            GL.glUniform1i(_u(self._prog,'uWire'),    1 if obj.is_wire else 0)
            GL.glUniform1f(_u(self._prog,'uMorph'),   float(obj.morph_level))
            GL.glUniform1f(_u(self._prog,'uTime'),    self._time)
            # Only re-upload waveform when band changes — same 4 waveforms repeat
            if obj.band_index != last_band:
                wave = self._waveform_data[obj.band_index] if 0 <= obj.band_index < 4 else _ZERO_WAVE
                GL.glUniform1fv(_u(self._prog,'uWave'), 128, wave)
                last_band = obj.band_index
            GL.glBindVertexArray(obj.vao)
            GL.glDrawElements(obj.mode, obj.index_count, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _draw_waveforms(self, pv, models):
        GL.glUseProgram(self._wave_prog)
        for obj, model in zip(self.objects, models):
            bi = obj.band_index
            if not (0 <= bi < 4) or self._wave_vaos[bi] is None:
                continue
            mvp = pv @ model
            GL.glUniformMatrix4fv(_u(self._wave_prog,'uMVP'), 1, GL.GL_TRUE, mvp)
            GL.glUniform3fv(_u(self._wave_prog,'uColor'), 1, obj.color)
            GL.glUniform1f(_u(self._wave_prog,'uAlpha'), obj.alpha * _fade(obj) * 0.75)
            GL.glBindVertexArray(self._wave_vaos[bi])
            GL.glDrawArrays(GL.GL_LINE_LOOP, 0, N_WAVE)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _post_pass(self, tex):
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(self._post_prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glUniform1i(_u(self._post_prog,'uTex'),    0)
        GL.glUniform1f(_u(self._post_prog,'uGlitch'), self.glitch_intensity)
        GL.glUniform1f(_u(self._post_prog,'uTime'),   self._time)
        GL.glUniform1f(_u(self._post_prog,'uSeed'),   float(self._rng.random()))
        GL.glBindVertexArray(self._quad_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def _upload_waveforms(self):
        cos_a = self._wave_cos
        sin_a = self._wave_sin
        for bi in range(4):
            if self._wave_vbos[bi] is None:
                continue
            wave = self._waveform_data[bi]
            r = 1.35 + wave * 0.4
            verts = np.ascontiguousarray(
                np.stack([cos_a * r, sin_a * r, np.zeros(N_WAVE, np.float32)], axis=1),
                np.float32)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._wave_vbos[bi])
            GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    # -------------------------------------------------------- export frame (uses FBO)
    def render_frame_to_array(self, w: int, h: int) -> np.ndarray:
        self.makeCurrent()

        if self._fbo is None or w != self._fbo_w or h != self._fbo_h:
            self._build_fbo(w, h)

        for obj in self._dead_queue:
            GL.glDeleteVertexArrays(1, [obj.vao])
            GL.glDeleteBuffers(2, [obj.vbo, obj.ibo])
        self._dead_queue.clear()
        for req in self._spawn_queue:
            self._materialise(req)
        self._spawn_queue.clear()
        self.spawn_queue_size = 0

        self._upload_waveforms()

        # Scene → _fbo
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        GL.glViewport(0, 0, w, h)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        pv = self._build_pv(w / h)
        models = [_model(obj.pos, obj.rot, obj.scale) for obj in self.objects]
        self._draw_scene(pv, models)
        self._draw_waveforms(pv, models)

        # Post-process → _fbo2
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo2)
        GL.glViewport(0, 0, w, h)
        self._post_pass(self._fbo_tex)

        # Read pixels from post output
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._fbo2)
        GL.glFinish()
        data = GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())

        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return np.flipud(arr)

    # -------------------------------------------------------- step (NO GL calls)
    def step(self, dt: float, energies: np.ndarray, active: np.ndarray,
             waveforms: Optional[np.ndarray] = None):
        self._time  += dt
        self._frame += 1
        self.band_energies = energies
        self.band_active   = active
        self._cam_azimuth += dt * 0.08

        if waveforms is not None:
            self._waveform_data = waveforms

        # Per-band glitch: each band contributes when active, scaled by user slider
        glitch = 0.0
        for bi in range(4):
            if not self.band_muted[bi] and self.band_glitch_amount[bi] > 0.01 and active[bi] > 0.01:
                glitch += self.band_glitch_amount[bi] * float(active[bi])
        self.glitch_intensity = min(1.0, glitch)

        for bi in range(4):
            self._spawn_cd[bi] = max(0.0, self._spawn_cd[bi] - dt)
            if not self.band_muted[bi] and active[bi] > 0.01 and self._spawn_cd[bi] <= 0.0:
                self._queue_spawn(bi, float(active[bi]), float(energies[bi]))
                self._spawn_cd[bi] = 0.4

        # Age + morph + reap
        keep = []
        for obj in self.objects:
            obj.age += dt
            obj.rot += obj.rot_speed * dt
            if 0 <= obj.band_index < 4:
                e = float(energies[obj.band_index])
                a = float(active[obj.band_index])
                obj.scale = min(obj.scale * (1.0 + a * 0.3 * dt), 4.0)
                # Morph tracks band energy: attack fast, release slower
                target = min(1.0, e * 1.5)
                obj.morph_level += (target - obj.morph_level) * min(1.0, dt * 8.0)
                # Band inactive: force a quick fade-out so objects disappear
                # in sync with the release setting rather than outliving it.
                if a < 0.05:
                    obj.lifetime = min(obj.lifetime, obj.age + 0.12)
            if obj.age >= obj.lifetime:
                self._dead_queue.append(obj)
            else:
                keep.append(obj)
        self.objects = keep

        while len(self.objects) > 16:
            self._dead_queue.append(self.objects.pop(0))

        self.spawn_queue_size = len(self._spawn_queue)
        self.update()

    def set_band_shape(self, bi: int, shape_idx: int):
        """Change a band's shape. Existing objects morph-out (max deform then die quickly)."""
        if self.band_shapes[bi] != shape_idx:
            for obj in self.objects:
                if obj.band_index == bi:
                    obj.morph_level = 1.0
                    obj.lifetime = min(obj.lifetime, obj.age + 0.4)
        self.band_shapes[bi] = shape_idx

    def reset(self):
        for obj in self.objects:
            self._dead_queue.append(obj)
        self.objects.clear()
        self._spawn_queue.clear()
        self.spawn_queue_size = 0
        self._time = 0.0
        self._frame = 0
        self.glitch_intensity = 0.0
        self._spawn_cd[:] = 0.0
        self._waveform_data[:] = 0.0

    # -------------------------------------------------------- spawn (no GL)
    # Each band occupies its own depth layer and scale range.
    # Band 0 = large, furthest back; Band 3 = smallest, closest front.
    _BAND_Z     = [-2.2, -0.8,  0.5,  1.8]          # world-Z per band (camera orbits Y)
    _BAND_SCALE = [(1.4, 2.6), (0.7, 1.4), (0.32, 0.72), (0.10, 0.30)]

    def _queue_spawn(self, bi: int, active_val: float, energy: float):
        rng = self._rng
        # Count materialised objects + pending queue entries for this band.
        # Without the pending check, multiple step() calls between paintGL frames
        # each pass the limit test and pile up spawns, causing a GL-allocation
        # spike when paintGL finally drains the queue — the core of the lag loop.
        existing = sum(1 for o in self.objects if o.band_index == bi)
        pending  = sum(1 for r in self._spawn_queue if r.band_idx == bi)
        if existing + pending >= self.band_max_objects[bi]:
            return

        color = list(self.band_colors[bi])
        z     = self._BAND_Z[bi] + float(rng.uniform(-0.2, 0.2))
        pos   = [float(rng.uniform(-0.35, 0.35)),
                 float(rng.uniform(-0.25, 0.25)),
                 z]
        lo, hi = self._BAND_SCALE[bi]
        scale = float(rng.uniform(lo, hi)) * (1.0 + energy * 0.25)
        rot   = [float(rng.uniform(0, 6.28)) for _ in range(3)]
        spd   = [float(rng.uniform(-1.5, 1.5)) for _ in range(3)]
        life  = float(rng.uniform(1.5, 3.5))
        alpha = float(rng.uniform(0.7, 1.0))
        bias  = [[0,0,2,4],[1,2,2,3],[3,3,4,0],[4,1,2,3]][bi]
        gtype = self.band_shapes[bi] if self.band_shapes[bi] >= 0 else int(rng.choice(bias))
        is_wire = bool(rng.random() < 0.5)
        self._spawn_queue.append(
            SpawnRequest(gtype, is_wire, pos, rot, scale, color, alpha, spd, life, bi))
        if not is_wire and rng.random() < 0.4:
            wc = [min(1.0, c + 0.3) for c in color]
            self._spawn_queue.append(
                SpawnRequest(gtype, True, pos, rot, scale * 1.02, wc, 0.7, spd, life, bi))

    # -------------------------------------------------------- materialise (GL, called from paintGL)
    def _materialise(self, req: SpawnRequest):
        if req.is_wire:
            verts, idxs = geo.WIRE_GENERATORS[req.gtype]()
            mode = GL.GL_LINES
        else:
            verts, idxs = geo.SOLID_GENERATORS[req.gtype]()
            mode = GL.GL_TRIANGLES

        vao = GL.glGenVertexArrays(1)
        bufs = GL.glGenBuffers(2)
        vbo, ibo = int(bufs[0]), int(bufs[1])

        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ibo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idxs.nbytes, idxs, GL.GL_STATIC_DRAW)
        stride = 6 * 4
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, stride,
                                 GL.ctypes.c_void_p(12))
        GL.glBindVertexArray(0)

        self.objects.append(SceneObject(
            vao, vbo, ibo, len(idxs), mode,
            req.pos, req.rot, req.scale, req.color, req.alpha,
            req.rot_speed, req.lifetime, req.is_wire, req.band_idx))

    # -------------------------------------------------------- FBO builders
    def _build_fbo(self, w, h):
        self._fbo_w, self._fbo_h = w, h
        # Scene FBO
        if self._fbo:
            GL.glDeleteFramebuffers(1, [self._fbo])
            GL.glDeleteTextures([self._fbo_tex])
            GL.glDeleteRenderbuffers(1, [self._fbo_rbo])
        self._fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo)
        self._fbo_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGB,w,h,0,GL.GL_RGB,GL.GL_UNSIGNED_BYTE,None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_S,GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_T,GL.GL_CLAMP_TO_EDGE)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D,self._fbo_tex,0)
        self._fbo_rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self._fbo_rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,GL.GL_DEPTH_COMPONENT24,w,h)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,GL.GL_DEPTH_ATTACHMENT,
                                     GL.GL_RENDERBUFFER,self._fbo_rbo)
        # Post-output FBO (no depth needed)
        if self._fbo2:
            GL.glDeleteFramebuffers(1, [self._fbo2])
            GL.glDeleteTextures([self._fbo2_tex])
        self._fbo2 = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo2)
        self._fbo2_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._fbo2_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGB,w,h,0,GL.GL_RGB,GL.GL_UNSIGNED_BYTE,None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_S,GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_T,GL.GL_CLAMP_TO_EDGE)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D,self._fbo2_tex,0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _build_pfbo(self, w, h):
        self._pfbo_w, self._pfbo_h = w, h
        if self._pfbo:
            GL.glDeleteFramebuffers(1, [self._pfbo])
            GL.glDeleteTextures([self._pfbo_tex])
            GL.glDeleteRenderbuffers(1, [self._pfbo_rbo])
        self._pfbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._pfbo)
        self._pfbo_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._pfbo_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGB,w,h,0,GL.GL_RGB,GL.GL_UNSIGNED_BYTE,None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_S,GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_WRAP_T,GL.GL_CLAMP_TO_EDGE)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D,self._pfbo_tex,0)
        self._pfbo_rbo = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER,self._pfbo_rbo)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER,GL.GL_DEPTH_COMPONENT24,w,h)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,GL.GL_DEPTH_ATTACHMENT,
                                     GL.GL_RENDERBUFFER,self._pfbo_rbo)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _build_wave_vaos(self):
        init_verts = np.zeros((N_WAVE, 3), np.float32)
        for bi in range(4):
            vao = GL.glGenVertexArrays(1)
            vbo = GL.glGenBuffers(1)
            GL.glBindVertexArray(int(vao))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, int(vbo))
            GL.glBufferData(GL.GL_ARRAY_BUFFER, init_verts.nbytes, init_verts, GL.GL_DYNAMIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, None)
            GL.glBindVertexArray(0)
            self._wave_vaos[bi] = int(vao)
            self._wave_vbos[bi] = int(vbo)

    def _build_quad(self):
        quad = np.array([-1,-1, 1,-1, -1,1, 1,1], np.float32)
        self._quad_vao = GL.glGenVertexArrays(1)
        self._quad_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._quad_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._quad_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, None)
        GL.glBindVertexArray(0)


# ============================================================ GL utils

_uloc_cache: dict = {}

def _u(prog, name):
    key = (prog, name)
    loc = _uloc_cache.get(key)
    if loc is None:
        loc = GL.glGetUniformLocation(prog, name)
        _uloc_cache[key] = loc
    return loc

def _fade(obj):
    if obj.lifetime > 0 and obj.age > obj.lifetime * 0.7:
        return max(0.0, 1.0 - (obj.age - obj.lifetime*0.7) / (obj.lifetime*0.3 + 1e-6))
    return 1.0

def _compile(vs_src, fs_src):
    vs = shaders.compileShader(vs_src, GL.GL_VERTEX_SHADER)
    fs = shaders.compileShader(fs_src, GL.GL_FRAGMENT_SHADER)
    return shaders.compileProgram(vs, fs)

def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([
        [f/aspect, 0,  0,                        0                     ],
        [0,        f,  0,                        0                     ],
        [0,        0,  (far+near)/(near-far),    2*far*near/(near-far) ],
        [0,        0, -1,                        0                     ],
    ], dtype=np.float32)

def _look_at(eye, center, up):
    f = _unit(center - eye)
    s = _unit(np.cross(f, up))
    u = np.cross(s, f)
    return np.array([
        [s[0],  s[1],  s[2],  -float(np.dot(s, eye))],
        [u[0],  u[1],  u[2],  -float(np.dot(u, eye))],
        [-f[0],-f[1], -f[2],   float(np.dot(f, eye))],
        [0,     0,     0,      1                     ],
    ], dtype=np.float32)

def _unit(v):
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-8 else v.astype(np.float32)

def _rot_x(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float32)

def _rot_y(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float32)

def _rot_z(a):
    c,s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

def _model(pos, rot, scale):
    S = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[0,3] = float(pos[0]); T[1,3] = float(pos[1]); T[2,3] = float(pos[2])
    return T @ _rot_z(float(rot[2])) @ _rot_y(float(rot[1])) @ _rot_x(float(rot[0])) @ S
