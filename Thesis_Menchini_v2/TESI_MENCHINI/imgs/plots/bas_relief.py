#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import cv2
import trimesh

def image_to_heightmap(
    img_path: str,
    out_w: int,
    out_h: int,
    threshold: int,
    invert: bool,
    blur: float,
) -> np.ndarray:
    # Read as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Immagine non trovata o non leggibile: {img_path}")

    # Resize to working resolution (controls mesh density)
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # Optional blur to smooth jagged edges
    if blur and blur > 0:
        k = int(max(3, round(blur) * 2 + 1))  # odd kernel size
        img = cv2.GaussianBlur(img, (k, k), sigmaX=blur, sigmaY=blur)

    # Threshold to binary mask
    _, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # We want "black" raised by default => black pixels -> 1, white -> 0
    # bin_img: white=255, black=0
    raised = (bin_img == 0).astype(np.float32)

    # If user wants the opposite
    if invert:
        raised = 1.0 - raised

    return raised  # values in {0,1}

def heightmap_to_mesh(
    hmap: np.ndarray,
    relief_height_mm: float,
    base_thickness_mm: float,
    pixel_size_mm: float,
) -> trimesh.Trimesh:
    """
    Create a closed watertight bas-relief mesh:
    - base: constant thickness
    - relief: extra height on top where hmap=1
    """
    H, W = hmap.shape

    # Grid coordinates in mm
    xs = np.arange(W, dtype=np.float32) * pixel_size_mm
    ys = np.arange(H, dtype=np.float32) * pixel_size_mm
    X, Y = np.meshgrid(xs, ys)

    # Top surface Z: base + relief*hmap
    Z_top = base_thickness_mm + relief_height_mm * hmap

    # Bottom surface Z=0
    Z_bot = np.zeros_like(Z_top, dtype=np.float32)

    # Build vertices: top then bottom
    # vertex index: vid = y*W + x
    top_verts = np.stack([X, Y, Z_top], axis=-1).reshape(-1, 3)
    bot_verts = np.stack([X, Y, Z_bot], axis=-1).reshape(-1, 3)
    verts = np.vstack([top_verts, bot_verts])

    def vid(x, y):
        return y * W + x

    top_offset = 0
    bot_offset = W * H

    faces = []

    # Triangulate top and bottom as a regular grid
    for y in range(H - 1):
        for x in range(W - 1):
            v00 = vid(x, y)
            v10 = vid(x + 1, y)
            v01 = vid(x, y + 1)
            v11 = vid(x + 1, y + 1)

            # Top (counter-clockwise looking from above)
            faces.append([top_offset + v00, top_offset + v10, top_offset + v11])
            faces.append([top_offset + v00, top_offset + v11, top_offset + v01])

            # Bottom (reverse winding to face outward downward)
            faces.append([bot_offset + v00, bot_offset + v11, bot_offset + v10])
            faces.append([bot_offset + v00, bot_offset + v01, bot_offset + v11])

    # Side walls around perimeter to close the mesh
    # Top edge: y=0
    y = 0
    for x in range(W - 1):
        vt0 = top_offset + vid(x, y)
        vt1 = top_offset + vid(x + 1, y)
        vb0 = bot_offset + vid(x, y)
        vb1 = bot_offset + vid(x + 1, y)
        # outward normal is -Y, ensure winding accordingly
        faces.append([vt0, vb1, vb0])
        faces.append([vt0, vt1, vb1])

    # Bottom edge: y=H-1
    y = H - 1
    for x in range(W - 1):
        vt0 = top_offset + vid(x, y)
        vt1 = top_offset + vid(x + 1, y)
        vb0 = bot_offset + vid(x, y)
        vb1 = bot_offset + vid(x + 1, y)
        # outward normal is +Y
        faces.append([vt0, vb0, vb1])
        faces.append([vt0, vb1, vt1])

    # Left edge: x=0
    x = 0
    for y in range(H - 1):
        vt0 = top_offset + vid(x, y)
        vt1 = top_offset + vid(x, y + 1)
        vb0 = bot_offset + vid(x, y)
        vb1 = bot_offset + vid(x, y + 1)
        # outward normal is -X
        faces.append([vt0, vb0, vb1])
        faces.append([vt0, vb1, vt1])

    # Right edge: x=W-1
    x = W - 1
    for y in range(H - 1):
        vt0 = top_offset + vid(x, y)
        vt1 = top_offset + vid(x, y + 1)
        vb0 = bot_offset + vid(x, y)
        vb1 = bot_offset + vid(x, y + 1)
        # outward normal is +X
        faces.append([vt0, vb1, vb0])
        faces.append([vt0, vt1, vb1])

    faces = np.asarray(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    # Small cleanup: ensure it’s watertight if possible
    if not mesh.is_watertight:
        mesh = mesh.fill_holes()

    return mesh

def preview_mesh(mesh: trimesh.Trimesh, show_edges: bool = False):
    # Interactive real-time viewer via PyVista
    import pyvista as pv

    pv_mesh = pv.wrap(mesh)

    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, show_edges=show_edges)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

def main():
    ap = argparse.ArgumentParser(description="Crea un bassorilievo 3D da un’immagine B/N (nero in rilievo).")
    ap.add_argument("--input", "-i", required=True, help="Path immagine (png/jpg/...)")
    ap.add_argument("--output", "-o", default="bas_relief.stl", help="Output mesh (.stl o .obj)")
    ap.add_argument("--w", type=int, default=300, help="Risoluzione mesh in larghezza (più alto = più dettagli)")
    ap.add_argument("--h", type=int, default=300, help="Risoluzione mesh in altezza")
    ap.add_argument("--threshold", type=int, default=200, help="Soglia binaria 0-255 (default 200)")
    ap.add_argument("--invert", action="store_true", help="Inverti: bianco in rilievo invece del nero")
    ap.add_argument("--blur", type=float, default=0.0, help="Gaussian blur sigma (0 = off). Utile per smussare")
    ap.add_argument("--relief_height", type=float, default=2.0, help="Altezza rilievo (mm)")
    ap.add_argument("--base_thickness", type=float, default=1.0, help="Spessore base (mm)")
    ap.add_argument("--pixel_size", type=float, default=0.2, help="Dimensione pixel in mm (scala XY)")
    ap.add_argument("--preview", action="store_true", help="Apri viewer interattivo (debug)")
    ap.add_argument("--edges", action="store_true", help="Mostra edges nel viewer")
    args = ap.parse_args()

    hmap = image_to_heightmap(
        img_path=args.input,
        out_w=args.w,
        out_h=args.h,
        threshold=args.threshold,
        invert=args.invert,
        blur=args.blur,
    )

    mesh = heightmap_to_mesh(
        hmap=hmap,
        relief_height_mm=args.relief_height,
        base_thickness_mm=args.base_thickness,
        pixel_size_mm=args.pixel_size,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export
    suffix = out_path.suffix.lower()
    if suffix == ".stl":
        mesh.export(out_path.as_posix())
    elif suffix == ".obj":
        mesh.export(out_path.as_posix())
    else:
        raise ValueError("Formato output non supportato. Usa .stl o .obj")

    print(f"[OK] Salvato: {out_path} | watertight={mesh.is_watertight} | faces={len(mesh.faces)}")

    if args.preview:
        preview_mesh(mesh, show_edges=args.edges)

if __name__ == "__main__":
    main()
