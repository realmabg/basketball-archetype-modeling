from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "abstract_feature_space_examples"

FEATURES = [
    "mins_per_game",
    "FTR",
    "three_share",
    "three_pa_per_100_team_poss",
    "usg",
    "tov_pct",
    "ast_pct",
    "orb_pct",
    "drb_pct",
    "blk_pct",
    "stl_pct",
    "ast_tov",
]

ARCHETYPE_NAMES = {
    1: "Low Usage\nConnector",
    2: "Rim Protecting\nBig",
    3: "Lead\nGuard",
    4: "Defensive\nSpacer",
    5: "Scoring\nBig",
    6: "Pure\nShooter",
}

PALETTE = {
    1: (33, 112, 181),   # blue
    2: (117, 107, 177),  # violet
    3: (65, 171, 93),    # green
    4: (217, 95, 14),    # orange
    5: (31, 150, 139),   # teal
    6: (189, 60, 88),    # rose
}

BG = (249, 250, 252)
INK = (31, 41, 55)
MUTED = (100, 116, 139)
GRID = (226, 232, 240)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size, index=1 if bold and candidate.endswith(".ttc") else 0)
        except Exception:
            continue
    return ImageFont.load_default()


def pca_projection(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    use = df.dropna(subset=FEATURES + ["k6_no_minutes_drop5_7_archetype_id", "division"]).copy()
    use = use[use["k6_no_minutes_drop5_7_archetype_id"].between(1, 6)]

    x = use[FEATURES].astype(float).to_numpy()
    lo = np.nanpercentile(x, 1, axis=0)
    hi = np.nanpercentile(x, 99, axis=0)
    x = np.clip(x, lo, hi)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0] = 1.0
    z = (x - mu) / sigma
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    components = vt[:2].T
    coords = z @ components

    use["x"] = coords[:, 0]
    use["y"] = coords[:, 1]

    # Stable orientation for interpretability: creators/perimeter to the right,
    # interior size/paint pressure upward.
    ast_idx = FEATURES.index("ast_pct")
    three_idx = FEATURES.index("three_pa_per_100_team_poss")
    if components[ast_idx, 0] + components[three_idx, 0] < 0:
        use["x"] *= -1
        components[:, 0] *= -1
    blk_idx = FEATURES.index("blk_pct")
    orb_idx = FEATURES.index("orb_pct")
    if components[blk_idx, 1] + components[orb_idx, 1] < 0:
        use["y"] *= -1
        components[:, 1] *= -1

    return use, components, mu, sigma, np.vstack([lo, hi])


def project_transfer_rows(transfer: pd.DataFrame, components: np.ndarray, mu: np.ndarray, sigma: np.ndarray, clip: np.ndarray) -> pd.DataFrame:
    mapping = {
        "mins_per_game": "d2_MPG",
        "FTR": "d2_FTR",
        "three_share": "d2_three_share",
        "three_pa_per_100_team_poss": "d2_three_pa_per_100_team_poss_model",
        "usg": "d2_usg",
        "tov_pct": "d2_TOV_pct",
        "ast_pct": "d2_AST_pct",
        "orb_pct": "d2_ORB_pct",
        "drb_pct": "d2_DRB_pct",
        "blk_pct": "d2_Blk_pct",
        "stl_pct": "d2_Stl_pct",
        "ast_tov": "d2_AST_TOV",
    }
    archetype_cols = [
        "d2_k6_no_minutes_drop5_7_archetype_1_source_k8_1_low_usage_connector_weight",
        "d2_k6_no_minutes_drop5_7_archetype_2_source_k8_2_rim_protecting_big_weight",
        "d2_k6_no_minutes_drop5_7_archetype_3_source_k8_3_lead_guard_weight",
        "d2_k6_no_minutes_drop5_7_archetype_4_source_k8_4_defensive_spacer_weight",
        "d2_k6_no_minutes_drop5_7_archetype_5_source_k8_6_scoring_big_weight",
        "d2_k6_no_minutes_drop5_7_archetype_6_source_k8_8_pure_shooter_weight",
    ]
    needed = list(mapping.values()) + archetype_cols
    t = transfer.dropna(subset=needed).copy()
    x = t[[mapping[f] for f in FEATURES]].astype(float).to_numpy()
    x = np.clip(x, clip[0], clip[1])
    z = (x - mu) / sigma
    coords = z @ components
    t["x"] = coords[:, 0]
    t["y"] = coords[:, 1]
    weights = t[archetype_cols].astype(float).to_numpy()
    t["dominant_arch"] = np.argmax(weights, axis=1) + 1
    return t


class Canvas:
    def __init__(self, width: int = 1800, height: int = 1200):
        self.scale = 2
        self.w = width * self.scale
        self.h = height * self.scale
        self.im = Image.new("RGB", (self.w, self.h), BG)
        self.draw = ImageDraw.Draw(self.im, "RGBA")

    def xy(self, x: float, y: float) -> tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    def text(self, xy, text, size=28, fill=INK, bold=False, anchor=None, align="left", spacing=6):
        self.draw.text(self.xy(*xy), text, font=font(size * self.scale, bold), fill=fill, anchor=anchor, align=align, spacing=spacing * self.scale)

    def line(self, xy, fill=GRID, width=2):
        pts = [self.xy(x, y) for x, y in xy]
        self.draw.line(pts, fill=fill, width=width * self.scale)

    def ellipse(self, bbox, fill=None, outline=None, width=2):
        box = tuple(v * self.scale for v in bbox)
        self.draw.ellipse(box, fill=fill, outline=outline, width=width * self.scale)

    def rectangle(self, bbox, fill=None, outline=None, width=2):
        box = tuple(v * self.scale for v in bbox)
        self.draw.rectangle(box, fill=fill, outline=outline, width=width * self.scale)

    def rounded_rectangle(self, bbox, radius=8, fill=None, outline=None, width=2):
        box = tuple(v * self.scale for v in bbox)
        self.draw.rounded_rectangle(box, radius=radius * self.scale, fill=fill, outline=outline, width=width * self.scale)

    def save(self, path: Path):
        out = self.im.resize((self.w // self.scale, self.h // self.scale), Image.Resampling.LANCZOS)
        out.save(path, quality=95)


def scaler(points: pd.DataFrame, margin: tuple[int, int, int, int], width=1800, height=1200):
    left, top, right, bottom = margin
    xs = points["x"].to_numpy()
    ys = points["y"].to_numpy()
    xlo, xhi = np.percentile(xs, [0.5, 99.5])
    ylo, yhi = np.percentile(ys, [0.5, 99.5])
    dx = xhi - xlo
    dy = yhi - ylo
    xlo -= dx * 0.08
    xhi += dx * 0.08
    ylo -= dy * 0.08
    yhi += dy * 0.08

    def map_xy(x, y):
        px = left + (x - xlo) / (xhi - xlo) * (width - left - right)
        py = height - bottom - (y - ylo) / (yhi - ylo) * (height - top - bottom)
        return px, py

    return map_xy


def draw_frame(c: Canvas, title: str, subtitle: str, margin=(155, 170, 360, 145)):
    c.text((90, 64), title, size=42, bold=True)
    c.text((90, 116), subtitle, size=25, fill=MUTED)
    left, top, right, bottom = margin
    plot = (left, top, 1800 - right, 1200 - bottom)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = plot[0] + frac * (plot[2] - plot[0])
        y = plot[1] + frac * (plot[3] - plot[1])
        c.line([(x, plot[1]), (x, plot[3])], fill=GRID, width=1)
        c.line([(plot[0], y), (plot[2], y)], fill=GRID, width=1)
    c.rectangle(plot, outline=(203, 213, 225), width=2)
    c.text(((plot[0] + plot[2]) / 2, 1110), "PC1: perimeter creation and spacing", size=24, fill=MUTED, anchor="mm")
    c.text((32, 505), "PC2:\ninterior pressure\nrebounding\nrim protection", size=19, fill=MUTED, spacing=4)
    return plot


def draw_legend(c: Canvas, x: int, y: int, include_division: bool = True):
    c.text((x, y), "Archetype", size=24, bold=True)
    yy = y + 42
    for aid in range(1, 7):
        color = PALETTE[aid]
        c.ellipse((x, yy + 4, x + 18, yy + 22), fill=color + (235,))
        c.text((x + 32, yy), ARCHETYPE_NAMES[aid].replace("\n", " "), size=20, fill=INK)
        yy += 38
    if include_division:
        yy += 18
        c.text((x, yy), "Division", size=24, bold=True)
        yy += 42
        c.ellipse((x, yy + 5, x + 19, yy + 24), fill=(28, 45, 64, 180))
        c.text((x + 32, yy), "D1 player-season", size=20, fill=INK)
        yy += 38
        c.rectangle((x, yy + 6, x + 19, yy + 25), fill=(28, 45, 64, 90))
        c.text((x + 32, yy), "D2 player-season", size=20, fill=INK)


def draw_points(c: Canvas, df: pd.DataFrame, map_xy, mode: str):
    rng = np.random.default_rng(7)
    order = rng.permutation(len(df))
    for idx in order:
        row = df.iloc[idx]
        x, y = map_xy(row["x"], row["y"])
        if not (125 < x < 1500 and 130 < y < 1080):
            continue
        aid = int(row["k6_no_minutes_drop5_7_archetype_id"])
        color = PALETTE[aid]
        if mode == "division":
            alpha = 34 if str(row["division"]).lower() == "d2" else 46
            r = 2.0
            if str(row["division"]).lower() == "d2":
                c.rectangle((x - r, y - r, x + r, y + r), fill=color + (alpha,))
            else:
                c.ellipse((x - r, y - r, x + r, y + r), fill=color + (alpha,))
        elif mode == "muted":
            c.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(87, 99, 115, 24))
        else:
            c.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color + (44,))


def draw_density_cells(c: Canvas, df: pd.DataFrame, map_xy, plot, mode: str = "archetype", bins: int = 86):
    left, top, right, bottom = plot
    counts = {}
    max_count = 1
    for _, row in df.iterrows():
        x, y = map_xy(row["x"], row["y"])
        if not (left <= x <= right and top <= y <= bottom):
            continue
        ix = int((x - left) / (right - left) * bins)
        iy = int((y - top) / (bottom - top) * bins)
        ix = max(0, min(bins - 1, ix))
        iy = max(0, min(bins - 1, iy))
        aid = int(row["k6_no_minutes_drop5_7_archetype_id"])
        key = (ix, iy)
        if key not in counts:
            counts[key] = {i: 0 for i in range(1, 7)}
        counts[key][aid] += 1
        max_count = max(max_count, sum(counts[key].values()))

    cw = (right - left) / bins
    ch = (bottom - top) / bins
    for (ix, iy), by_arch in counts.items():
        total = sum(by_arch.values())
        if total < 2:
            continue
        x0 = left + ix * cw
        y0 = top + iy * ch
        x1 = x0 + cw * 1.05
        y1 = y0 + ch * 1.05
        alpha = int(35 + 170 * (math.log1p(total) / math.log1p(max_count)))
        if mode == "muted":
            color = (79, 91, 109)
        else:
            aid = max(by_arch, key=by_arch.get)
            color = PALETTE[aid]
        c.rectangle((x0, y0, x1, y1), fill=color + (alpha,))


def draw_centroid_labels(c: Canvas, df: pd.DataFrame, map_xy, with_boxes=True):
    offsets = {
        1: (36, -44),
        2: (-120, -34),
        3: (28, 52),
        4: (-138, 24),
        5: (42, -18),
        6: (34, -52),
    }
    for aid in range(1, 7):
        sub = df[df["k6_no_minutes_drop5_7_archetype_id"] == aid]
        cx, cy = map_xy(sub["x"].median(), sub["y"].median())
        color = PALETTE[aid]
        c.ellipse((cx - 13, cy - 13, cx + 13, cy + 13), fill=color + (250,), outline=(255, 255, 255, 245), width=3)
        ox, oy = offsets[aid]
        label = ARCHETYPE_NAMES[aid]
        tx, ty = cx + ox, cy + oy
        if with_boxes:
            lines = label.split("\n")
            w = max(c.draw.textlength(line, font=font(20 * c.scale, True)) for line in lines) / c.scale
            h = 52
            c.rounded_rectangle((tx - 10, ty - 8, tx + w + 12, ty + h), radius=6, fill=(255, 255, 255, 225), outline=color + (120,), width=1)
        c.text((tx, ty), label, size=20, bold=True, fill=INK, spacing=3)


def fig_role_space(df: pd.DataFrame, path: Path):
    c = Canvas()
    plot = draw_frame(
        c,
        "Shared D1/D2 Player Role Space",
        "Aggregated player-season density; colors show learned k6 archetype regions.",
    )
    map_xy = scaler(df, (155, 170, 360, 145))
    draw_density_cells(c, df, map_xy, plot, mode="archetype", bins=88)
    draw_centroid_labels(c, df, map_xy)
    draw_legend(c, 1490, 215, include_division=False)
    c.text((155, 1127), "Projection: PCA on standardized role/usage/rate features. Grid cells aggregate nearby player-seasons and are colored by dominant archetype.", size=18, fill=MUTED)
    c.save(path)


def fig_transfer_overlay(df: pd.DataFrame, transfers: pd.DataFrame, path: Path):
    c = Canvas()
    draw_frame(
        c,
        "Where D2-to-D1 Transfers Sit in the Role Space",
        "Transfer source seasons are highlighted against the full D1/D2 background.",
    )
    map_xy = scaler(df, (155, 170, 360, 145))
    draw_density_cells(c, df, map_xy, (155, 170, 1440, 1055), mode="muted", bins=88)
    draw_centroid_labels(c, df, map_xy, with_boxes=False)

    for _, row in transfers.iterrows():
        x, y = map_xy(row["x"], row["y"])
        if not (125 < x < 1500 and 130 < y < 1080):
            continue
        aid = int(row["dominant_arch"])
        color = PALETTE[aid]
        r = 5.5
        c.ellipse((x - r, y - r, x + r, y + r), fill=color + (185,), outline=(17, 24, 39, 210), width=1)

    c.text((1490, 215), "Overlay", size=24, bold=True)
    c.ellipse((1492, 269, 1512, 289), fill=(87, 99, 115, 45))
    c.text((1524, 262), "All player-seasons", size=20, fill=INK)
    c.ellipse((1492, 319, 1516, 343), fill=PALETTE[1] + (190,), outline=(17, 24, 39, 230), width=2)
    c.text((1524, 313), "D2-to-D1 transfer\nsource season", size=20, fill=INK, spacing=4)
    c.text((1490, 405), "Use case", size=24, bold=True)
    c.text((1490, 447), "Shows whether transfer\ncandidates are clustered in\nparticular basketball roles,\ninstead of treating D2\nproduction as one pool.", size=20, fill=MUTED, spacing=7)
    c.save(path)


def draw_profile_bars(c: Canvas, x: int, y: int, values: dict[str, float], color):
    labels = ["3PA", "FTR", "AST", "REB", "BLK", "USG"]
    for i, label in enumerate(labels):
        yy = y + i * 24
        c.text((x, yy - 3), label, size=15, fill=MUTED)
        w = 82 * max(0.05, min(1.0, values[label]))
        c.rounded_rectangle((x + 46, yy, x + 46 + w, yy + 10), radius=2, fill=color + (210,))


def fig_centroid_map(df: pd.DataFrame, profile: pd.DataFrame, path: Path):
    c = Canvas()
    plot = draw_frame(
        c,
        "Labeled Archetype Centers in Common Feature Space",
        "A cleaner abstract version: density background plus readable role labels.",
    )
    map_xy = scaler(df, (155, 170, 360, 145))
    draw_density_cells(c, df, map_xy, plot, mode="muted", bins=76)
    draw_centroid_labels(c, df, map_xy, with_boxes=True)

    c.text((1490, 215), "Best for abstracts", size=24, bold=True)
    c.text((1490, 258), "Use this when the figure\nneeds to be understood in\none glance. The full cloud is\naggregated, while the role\nnames stay prominent.", size=20, fill=MUTED, spacing=7)
    c.text((155, 1127), "Projection: PCA on standardized archetype-model features. Background cells show total player-season density.", size=18, fill=MUTED)
    c.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_df = pd.read_csv(ROOT / "data" / "archetype_model_features.csv", low_memory=False)
    join_keys = ["division", "season", "team", "player_name", "player_uid"]
    labels = pd.read_csv(
        ROOT / "data" / "enriched_d1_d2_player_stats_with_archetypes.csv",
        usecols=join_keys
        + ["k6_no_minutes_drop5_7_archetype_id", "k6_no_minutes_drop5_7_archetype_confidence"],
        low_memory=False,
    )
    df = feature_df.merge(labels, on=join_keys, how="left")
    projected, components, mu, sigma, clip = pca_projection(df)

    transfer = pd.read_csv(ROOT / "data" / "transfer_model_ready_with_archetypes.csv", low_memory=False)
    transfer_projected = project_transfer_rows(transfer, components, mu, sigma, clip)

    profile = pd.read_csv(ROOT / "models" / "player_archetypes" / "k6_actual_game_stats.csv")
    profile = profile.rename(
        columns={
            "3PA_per_100": "3PA_per_100",
            "BLK_pct": "BLK_pct",
        }
    )

    fig_role_space(projected, OUT_DIR / "example_1_shared_d1_d2_role_space.png")
    fig_transfer_overlay(projected, transfer_projected, OUT_DIR / "example_2_transfer_overlay.png")
    fig_centroid_map(projected, profile, OUT_DIR / "example_3_abstract_centroid_map.png")
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
