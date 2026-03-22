import argparse
import numpy as np

import pyvista as pv
from Bio.PDB import MMCIFParser


def load_structure(cif_path: str):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure("prot", cif_path)


def extract_ca_coords_and_plddt(structure):
    coords = []
    plddt = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if "CA" not in residue:
                    continue
                ca = residue["CA"]
                coords.append(ca.coord.astype(float))
                plddt.append(float(ca.bfactor))
        break

    if not coords:
        raise RuntimeError("No C-alpha atoms found in the structure.")

    return np.asarray(coords), np.asarray(plddt)


def build_backbone_polyline(coords: np.ndarray) -> pv.PolyData:
    n = coords.shape[0]
    poly = pv.PolyData(coords)

    lines = np.empty(n + 1, dtype=np.int64)
    lines[0] = n
    lines[1:] = np.arange(n, dtype=np.int64)
    poly.lines = lines
    return poly


def main():
    ap = argparse.ArgumentParser(description="Visualize AlphaFold mmCIF in PyVista (no HTML, no Jupyter).")
    ap.add_argument("--cif", required=True, help="Path to AlphaFold .cif file")
    ap.add_argument("--tube_radius", type=float, default=0.6, help="Backbone tube radius")
    ap.add_argument("--point_size", type=float, default=8.0, help="CA point size")
    ap.add_argument("--show_points", action="store_true", help="Show CA points in addition to tube")
    ap.add_argument("--offscreen_png", default=None, help="Render offscreen and save PNG to this path")
    args = ap.parse_args()

    structure = load_structure(args.cif)
    coords, plddt = extract_ca_coords_and_plddt(structure)

    print(f"CA residues: {len(plddt)}")
    print(f"Mean pLDDT: {plddt.mean():.2f}")

    backbone = build_backbone_polyline(coords)
    backbone["pLDDT"] = plddt  
    tube = backbone.tube(radius=args.tube_radius)

    offscreen = args.offscreen_png is not None
    plotter = pv.Plotter(off_screen=offscreen)
    plotter.set_background("white")

    plotter.add_mesh(
        tube,
        scalars="pLDDT",
        clim=[0, 100],
        cmap="viridis",
        show_scalar_bar=True,
        scalar_bar_args={"title": "pLDDT"},
        smooth_shading=True,
    )

    if args.show_points:
        pts = pv.PolyData(coords)
        pts["pLDDT"] = plddt
        plotter.add_mesh(
            pts,
            scalars="pLDDT",
            clim=[0, 100],
            cmap="viridis",
            render_points_as_spheres=True,
            point_size=args.point_size,
            opacity=0.9,
        )

    plotter.add_axes()
    plotter.camera_position = "iso"
    plotter.reset_camera()

    if offscreen:
        plotter.show(screenshot=args.offscreen_png, auto_close=True)
        print(f"Wrote screenshot: {args.offscreen_png}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()