"""
Geometry generators: returns (vertices, indices) as numpy arrays.
Vertices are interleaved: position(3) + normal(3) = 6 floats per vertex.
"""
import numpy as np
from typing import Tuple


Vec = np.ndarray


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-8, 1.0, n)
    return v / n


# ------------------------------------------------------------------ cube
def cube(size=1.0) -> Tuple[np.ndarray, np.ndarray]:
    s = size / 2
    faces = [
        # pos x
        ([s,-s,-s], [s, s,-s], [s, s, s], [s,-s, s], [1,0,0]),
        # neg x
        ([-s,-s, s],[-s, s, s],[-s, s,-s],[-s,-s,-s], [-1,0,0]),
        # pos y
        ([-s, s,-s], [-s, s, s], [s, s, s], [s, s,-s], [0,1,0]),
        # neg y
        ([-s,-s, s],[-s,-s,-s],[s,-s,-s],[s,-s, s], [0,-1,0]),
        # pos z
        ([-s,-s, s],[s,-s, s],[s, s, s],[-s, s, s], [0,0,1]),
        # neg z
        ([s,-s,-s],[-s,-s,-s],[-s, s,-s],[s, s,-s], [0,0,-1]),
    ]
    verts = []
    idxs = []
    base = 0
    for p0,p1,p2,p3,n in faces:
        for p in [p0,p1,p2,p3]:
            verts.extend(p + n)
        idxs.extend([base, base+1, base+2, base, base+2, base+3])
        base += 4
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


def cube_wireframe(size=1.0) -> Tuple[np.ndarray, np.ndarray]:
    s = size / 2
    corners = [
        [-s,-s,-s],[s,-s,-s],[s, s,-s],[-s, s,-s],
        [-s,-s, s],[s,-s, s],[s, s, s],[-s, s, s],
    ]
    edges = [
        0,1, 1,2, 2,3, 3,0,  # back
        4,5, 5,6, 6,7, 7,4,  # front
        0,4, 1,5, 2,6, 3,7,  # sides
    ]
    verts = []
    for c in corners:
        verts.extend(c + [0,0,1])  # dummy normal
    return (np.array(verts, np.float32),
            np.array(edges, np.uint32))


# ------------------------------------------------------------------ sphere
def sphere(radius=1.0, slices=16, stacks=12) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    idxs = []
    for i in range(stacks+1):
        phi = np.pi * i / stacks
        for j in range(slices+1):
            theta = 2*np.pi * j / slices
            x = np.sin(phi)*np.cos(theta)
            y = np.cos(phi)
            z = np.sin(phi)*np.sin(theta)
            verts.extend([x*radius, y*radius, z*radius, x, y, z])
    for i in range(stacks):
        for j in range(slices):
            a = i*(slices+1)+j
            b = a+slices+1
            idxs.extend([a,b,a+1, b,b+1,a+1])
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


def sphere_wireframe(radius=1.0, slices=16, stacks=12) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    idxs = []
    vi = 0
    for i in range(stacks+1):
        phi = np.pi * i / stacks
        for j in range(slices+1):
            theta = 2*np.pi * j / slices
            x = np.sin(phi)*np.cos(theta)
            y = np.cos(phi)
            z = np.sin(phi)*np.sin(theta)
            verts.extend([x*radius, y*radius, z*radius, x, y, z])
    edge_set = set()
    def add_edge(a, b):
        e = (min(a,b), max(a,b))
        if e not in edge_set:
            edge_set.add(e)
            idxs.extend([a, b])
    for i in range(stacks):
        for j in range(slices):
            a = i*(slices+1)+j
            b = a+slices+1
            add_edge(a, b)
            add_edge(a, a+1)
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


# ------------------------------------------------------------------ torus
def torus(R=0.7, r=0.3, major=32, minor=16) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    idxs = []
    for i in range(major+1):
        u = 2*np.pi * i / major
        cu, su = np.cos(u), np.sin(u)
        for j in range(minor+1):
            v = 2*np.pi * j / minor
            cv, sv = np.cos(v), np.sin(v)
            x = (R + r*cv)*cu
            y = r*sv
            z = (R + r*cv)*su
            nx, ny, nz = cv*cu, sv, cv*su
            verts.extend([x, y, z, nx, ny, nz])
    for i in range(major):
        for j in range(minor):
            a = i*(minor+1)+j
            b = (i+1)*(minor+1)+j
            idxs.extend([a,b,a+1, b,b+1,a+1])
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


def torus_wireframe(R=0.7, r=0.3, major=32, minor=16) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    idxs = []
    for i in range(major+1):
        u = 2*np.pi * i / major
        cu, su = np.cos(u), np.sin(u)
        for j in range(minor+1):
            v = 2*np.pi * j / minor
            cv, sv = np.cos(v), np.sin(v)
            x = (R + r*cv)*cu
            y = r*sv
            z = (R + r*cv)*su
            verts.extend([x, y, z, cv*cu, sv, cv*su])
    edge_set = set()
    def add_edge(a, b):
        e = (min(a,b), max(a,b))
        if e not in edge_set:
            edge_set.add(e)
            idxs.extend([a, b])
    for i in range(major):
        for j in range(minor):
            a = i*(minor+1)+j
            b = (i+1)*(minor+1)+j
            add_edge(a, a+1)
            add_edge(a, b)
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


# ------------------------------------------------------------------ icosahedron
def icosahedron(radius=1.0) -> Tuple[np.ndarray, np.ndarray]:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [-1, t, 0],[1, t, 0],[-1,-t, 0],[1,-t, 0],
        [0,-1, t],[0, 1, t],[0,-1,-t],[0, 1,-t],
        [t, 0,-1],[t, 0, 1],[-t, 0,-1],[-t, 0, 1],
    ], dtype=np.float32)
    pts = _normalize(pts) * radius
    faces = [
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    ]
    verts = []
    idxs = []
    for i in range(0, len(faces), 3):
        a,b,c = faces[i], faces[i+1], faces[i+2]
        pa,pb,pc = pts[a], pts[b], pts[c]
        n = _normalize(np.cross(pb-pa, pc-pa))
        base = len(verts) // 6
        for p in [pa,pb,pc]:
            verts.extend(p.tolist() + n.tolist())
        idxs.extend([base, base+1, base+2])
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


def icosahedron_wireframe(radius=1.0) -> Tuple[np.ndarray, np.ndarray]:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [-1, t, 0],[1, t, 0],[-1,-t, 0],[1,-t, 0],
        [0,-1, t],[0, 1, t],[0,-1,-t],[0, 1,-t],
        [t, 0,-1],[t, 0, 1],[-t, 0,-1],[-t, 0, 1],
    ], dtype=np.float32)
    pts = _normalize(pts) * radius
    faces = [
        0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
        1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
        3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
        4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1,
    ]
    verts = []
    for p in pts:
        verts.extend(p.tolist() + [0,1,0])
    edge_set = set()
    idxs = []
    for i in range(0, len(faces), 3):
        a,b,c = faces[i], faces[i+1], faces[i+2]
        for e in [(a,b),(b,c),(a,c)]:
            k = (min(e), max(e))
            if k not in edge_set:
                edge_set.add(k)
                idxs.extend(list(e))
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


# ------------------------------------------------------------------ tetrahedron
def tetrahedron(radius=1.0) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.array([
        [0, 1, 0],
        [np.sqrt(8/9), -1/3, 0],
        [-np.sqrt(2/9), -1/3,  np.sqrt(2/3)],
        [-np.sqrt(2/9), -1/3, -np.sqrt(2/3)],
    ], dtype=np.float32) * radius
    faces = [(0,1,2),(0,2,3),(0,3,1),(1,3,2)]
    verts = []
    idxs = []
    for fa in faces:
        a,b,c = pts[fa[0]], pts[fa[1]], pts[fa[2]]
        n = _normalize(np.cross(b-a, c-a))
        base = len(verts) // 6
        for p in [a,b,c]:
            verts.extend(p.tolist() + n.tolist())
        idxs.extend([base, base+1, base+2])
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


def tetrahedron_wireframe(radius=1.0) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.array([
        [0, 1, 0],
        [np.sqrt(8/9), -1/3, 0],
        [-np.sqrt(2/9), -1/3,  np.sqrt(2/3)],
        [-np.sqrt(2/9), -1/3, -np.sqrt(2/3)],
    ], dtype=np.float32) * radius
    verts = []
    for p in pts:
        verts.extend(p.tolist() + [0,1,0])
    edges = [0,1,1,2,2,0,0,3,1,3,2,3]
    return (np.array(verts, np.float32),
            np.array(edges, np.uint32))


# ------------------------------------------------------------------ grid plane
def grid(size=2.0, divs=10) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    idxs = []
    step = size / divs
    half = size / 2
    for i in range(divs+1):
        x = -half + i*step
        for j in range(divs+1):
            z = -half + j*step
            verts.extend([x, 0, z, 0, 1, 0])
    for i in range(divs):
        for j in range(divs):
            a = i*(divs+1)+j
            b = a + divs+1
            idxs.extend([a,b,a+1, b,b+1,a+1])
    return (np.array(verts, np.float32),
            np.array(idxs, np.uint32))


SOLID_GENERATORS = [cube, sphere, torus, icosahedron, tetrahedron]
WIRE_GENERATORS  = [cube_wireframe, sphere_wireframe, torus_wireframe,
                    icosahedron_wireframe, tetrahedron_wireframe]
