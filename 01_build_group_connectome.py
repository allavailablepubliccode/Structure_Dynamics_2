#!/usr/bin/env python3
"""Build the 68-region group connectome and group coordinates from BrainGraph GraphML files.

The input can be either the downloaded zip archive or a directory containing the
1064 `*_repeated10_scale33.graphml` files. Edges are aggregated as expected
streamline count = prevalence × mean streamline count when present. No 50%
prevalence threshold is applied.
"""
import argparse, io, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from model_utils import spectral_radius

NS = "{http://graphml.graphdrawing.org/xmlns}"
SEED = 20260907


def graphml_bytes(input_path):
    p = Path(input_path)
    if p.is_dir():
        for f in sorted(p.rglob("*_repeated10_scale33.graphml")):
            yield f.name, f.read_bytes()
    else:
        with zipfile.ZipFile(p) as zf:
            names = sorted(n for n in zf.namelist() if n.endswith(".graphml") and "__MACOSX" not in n)
            for n in names:
                yield n, zf.read(n)


def parse_graph(data):
    root = ET.fromstring(data)
    keys = {k.attrib["id"]: k.attrib.get("attr.name") for k in root.findall(NS+"key")}
    graph = root.find(NS+"graph")
    nodes = {}
    for node in graph.findall(NS+"node"):
        d = {keys[x.attrib["key"]]: x.text for x in node.findall(NS+"data")}
        nodes[node.attrib["id"]] = d
    edges = []
    for edge in graph.findall(NS+"edge"):
        d = {keys[x.attrib["key"]]: x.text for x in edge.findall(NS+"data")}
        edges.append((edge.attrib["source"], edge.attrib["target"], d))
    return nodes, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="BrainGraph zip archive or directory of GraphML files")
    ap.add_argument("--output-dir", default="data/derived")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    items = list(graphml_bytes(args.input))
    if not items:
        raise RuntimeError("No GraphML files found")

    first_nodes, _ = parse_graph(items[0][1])
    labels = [d["dn_name"] for _,d in first_nodes.items() if d.get("dn_region") == "cortical"]
    # preserve the atlas order from the files
    idx = {lab:i for i,lab in enumerate(labels)}
    n = len(labels)
    if n != 68:
        raise RuntimeError(f"Expected 68 cortical regions, found {n}")

    present = np.zeros((n,n), dtype=int)
    fiber_sum = np.zeros((n,n), dtype=float)
    coord_sum = np.zeros((n,3), dtype=float)
    coord_sq = np.zeros((n,3), dtype=float)
    coord_n = np.zeros(n, dtype=int)

    for _,data in items:
        nodes, edges = parse_graph(data)
        node_to_index = {}
        for nid,d in nodes.items():
            lab = d.get("dn_name")
            if lab not in idx:
                continue
            i = idx[lab]; node_to_index[nid] = i
            try:
                xyz = np.array([float(d["dn_position_x"]), float(d["dn_position_y"]), float(d["dn_position_z"])])
            except (KeyError, TypeError, ValueError):
                continue
            if np.all(np.isfinite(xyz)):
                coord_sum[i] += xyz; coord_sq[i] += xyz*xyz; coord_n[i] += 1

        for s,t,d in edges:
            if s not in node_to_index or t not in node_to_index:
                continue
            w = float(d.get("number_of_fibers", 0.0))
            if w <= 0:
                continue
            i,j = node_to_index[s], node_to_index[t]
            present[i,j] += 1; present[j,i] += 1
            fiber_sum[i,j] += w; fiber_sum[j,i] += w

    n_subjects = len(items)
    prevalence = present / n_subjects
    mean_present = np.divide(fiber_sum, present, out=np.zeros_like(fiber_sum), where=present>0)
    expected_fibers = np.where(prevalence>0, prevalence*mean_present, 0.0)
    W = np.log1p(expected_fibers); np.fill_diagonal(W, 0.0)
    W /= spectral_radius(W)

    rng = np.random.default_rng(SEED)
    gain = rng.uniform(0.85,1.15,n)
    rM = np.max(np.real(np.linalg.eigvals(np.diag(gain) @ W)))
    coupling = 0.72/rM
    noise_sd = rng.uniform(0.85,1.15,n)

    coords = coord_sum/coord_n[:,None]
    sd = np.sqrt(np.maximum(coord_sq/coord_n[:,None]-coords*coords, 0.0))

    np.savez(out/"hcp86_empirical_consensus.npz", W=W, labels=np.asarray(labels), prevalence=prevalence,
             mean_fibers_present=mean_present, gain=gain, noise_sd=noise_sd, global_coupling=coupling)
    pd.DataFrame({"label":labels,"x":coords[:,0],"y":coords[:,1],"z":coords[:,2],
                  "sd_x":sd[:,0],"sd_y":sd[:,1],"sd_z":sd[:,2],"n_valid":coord_n}).to_csv(
                      out/"hcp68_group_coordinates.csv", index=False)
    print(f"subjects={n_subjects}; cortical regions={n}; undirected edges={np.count_nonzero(np.triu(W,1))}")
    print(f"global coupling={coupling:.16f}")

if __name__ == "__main__":
    main()
