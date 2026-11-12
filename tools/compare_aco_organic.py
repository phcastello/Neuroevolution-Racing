"""Render every ant in the paired Rust battery; never select only successful ants."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
root = args.directory
rows = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
valid = [r for r in rows if r.get("id")]
for row in valid:
    source = root / f'{row["variant"]}-{row["seed"]}' / f'{row["id"]}.json'
    points = np.array(json.loads(source.read_text(encoding="utf-8"))["definition"]["control_points"])
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    split = np.searchsorted(distances, row["explored_length_fraction"] * row["length"])
    # Both generators use R50 and the same world scale. Also report physical
    # occupancy; normalized occupancy alone can hide a larger actual excursion.
    row["occupied_cells_200"] = len(np.unique(np.floor(points/200), axis=0))
    row["explored_cells_200"] = len(np.unique(np.floor(points[:split+1]/200), axis=0))
    ab = points - np.roll(points, 1, axis=0)
    bc = np.roll(points, -1, axis=0) - points
    curvature = 2*np.abs(ab[:, 0]*bc[:, 1]-ab[:, 1]*bc[:, 0])/np.maximum(
        np.linalg.norm(ab, axis=1)*np.linalg.norm(bc, axis=1)*np.linalg.norm(ab+bc, axis=1), 1e-8)
    row["min_radius"] = float(1/np.max(curvature))
plt.rcParams.update({"figure.facecolor": "#101d26", "axes.facecolor": "#101d26",
                     "text.color": "white", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white", "axes.edgecolor": "#778899"})


def draw(ax, row):
    if not row or not row.get("id"):
        ax.text(.5, .5, "Candidata inválida", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return
    source = root / f'{row["variant"]}-{row["seed"]}' / f'{row["id"]}.json'
    p = np.array(json.loads(source.read_text(encoding="utf-8"))["definition"]["control_points"])
    p = np.vstack([p, p[0]])
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    d = np.r_[0, np.cumsum(lengths)]
    split = np.searchsorted(d, row["explored_length_fraction"] * row["length"])
    ax.plot(p[:split+1, 0], p[:split+1, 1], color="#49ddba", lw=2)
    ax.plot(p[split:, 0], p[split:, 1], color="#ffb75a", lw=2)
    ax.scatter(*p[0], color="white", s=14, zorder=5)
    o = row["organic"]
    ax.set_title(f's{row["seed"]} i{row["iteration"]} a{row["ant"]} · {row["id"][-8:]}\n'
                 f'R={o["rectangularity"]:.2f} H={o["heading_entropy"]:.2f} '
                 f'E={o["geometric_events"]} · livre={100*row["explored_length_fraction"]:.0f}%', fontsize=9)
    ax.set_aspect("equal")
    ax.margins(.1)
    ax.axis("off")


for variant in ("before", "after"):
    for seed in (17, 47):
        subset = [r for r in rows if r["variant"] == variant and r["seed"] == seed]
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        for ax, row in zip(axes.flat, subset):
            draw(ax, row)
        fig.suptitle(f'{"Antes" if variant == "before" else "Depois"} · seed {seed} · todas as 12 formigas\n'
                     'Verde: exploração livre/orientada · laranja: fechamento · branco: início', fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(root / f"{variant}-{seed}.png", dpi=130)
        plt.close(fig)

worst = sorted([r for r in valid if r["variant"] == "before"], key=lambda r: r["organic"]["rectangularity"], reverse=True)[:4]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, row in enumerate(worst):
    draw(axes[0, i], row)
    other = next((r for r in rows if r["variant"] == "after" and
                  all(r[k] == row[k] for k in ("seed", "iteration", "ant"))), None)
    draw(axes[1, i], other)
fig.suptitle("Quatro maiores retangularidades anteriores (acima) e mesmas posições da bateria nova (abaixo)\n"
             "Mesmas seeds e orçamento de candidatas; os grafos e orçamentos de decisões diferem", fontsize=15)
fig.tight_layout(rect=(0, 0, 1, .92))
fig.savefig(root / "comparison.png", dpi=140)
plt.close(fig)

fields = ["length", "longest_straight", "straight_fraction", "compactness", "close_decision", "explored_length_fraction", "extra_strolls", "occupied_cells_200", "explored_cells_200", "min_radius"]
organic_fields = ["longest_straight_ratio", "accumulated_turn", "heading_entropy", "axis_concentration",
                  "bounding_box_fill", "bounding_box_aspect", "rectangularity", "orthogonal_turn_fraction",
                  "geometric_events", "element_families", "spatial_cells", "complexity"]
summary = {}
for variant in ("before", "after"):
    tracks = [r for r in valid if r["variant"] == variant]
    stats = {key: float(np.mean([r[key] for r in tracks])) for key in fields}
    stats.update({key: float(np.mean([r["organic"][key] for r in tracks])) for key in organic_fields})
    stats.update(valid=len(tracks), attempted=sum(r["variant"] == variant for r in rows),
                 accepted=sum(r["accepted"] for r in tracks), completed=sum(r["reference_completed"] for r in tracks),
                 rectangular_over_030=sum(r["organic"]["rectangularity"] > .30 for r in tracks),
                 straight_over_025=sum(r["organic"]["longest_straight_ratio"] > .25 for r in tracks),
                 hairpin_tracks=sum(r["hairpins"] > 0 for r in tracks), r50=sum(r["min_radius"] < 50.5 for r in tracks), chicanes=sum(r["chicanes"] > 0 for r in tracks),
                 s_bends=sum(r["s_bends"] > 0 for r in tracks),
                 straight_over_700=sum(r["longest_straight"] > 700 for r in tracks))
    summary[variant] = stats
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
before, after = summary["before"], summary["after"]
assert before["attempted"] == after["attempted"] == 24
assert before["valid"] == after["valid"] == 24
assert after["completed"] == 24
assert after["rectangularity"] < .65 * before["rectangularity"]
assert after["axis_concentration"] < .65 * before["axis_concentration"]
assert after["longest_straight_ratio"] < .65 * before["longest_straight_ratio"]
assert after["heading_entropy"] > before["heading_entropy"] + .08
assert after["explored_length_fraction"] > before["explored_length_fraction"] + .08
assert after["geometric_events"] > 1.4 * before["geometric_events"]
with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as out:
    columns = ["variant", "seed", "iteration", "ant", "id", *fields, *organic_fields, "accepted", "reference_completed"]
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    for r in valid:
        flat = {**r, **r["organic"]}
        writer.writerow({k: flat[k] for k in columns})

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for variant, color, label in (("before", "#ffb75a", "Antes"), ("after", "#49ddba", "Depois")):
    tracks = [r for r in valid if r["variant"] == variant]
    heading = np.mean([r["organic"]["heading_histogram"] for r in tracks], axis=0)
    axes[0].plot(np.arange(24)*15, heading, color=color, label=label)
    lengths = np.sort([x for r in tracks for x in r["organic"]["straight_lengths"]])
    axes[1].step(lengths, np.arange(1, len(lengths)+1)/len(lengths), color=color, label=label)
    axes[2].scatter([r["organic"]["rectangularity"] for r in tracks],
                    [r["organic"]["complexity"] for r in tracks], color=color, label=label, alpha=.8)
axes[0].set(xlabel="Heading (graus, referencial de partida)", ylabel="Fração de comprimento", title="Distribuição de headings")
axes[1].set(xlabel="Comprimento da reta", ylabel="Fração acumulada de retas", title="Todas as retas ≥ 20 unidades")
axes[2].set(xlabel="Retangularidade (menor é melhor)", ylabel="Complexidade", title="Todas as candidatas válidas")
for ax in axes:
    ax.legend()
    ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(root / "distributions.png", dpi=140)
print(json.dumps(summary, indent=2))
