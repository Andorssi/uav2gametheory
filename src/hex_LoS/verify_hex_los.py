# 正六角領域内のLoS確率をAl-Houraniモデルで検証する．

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ====== 現在の六角領域モデルの幾何学的条件 ======
HEX_RADIUS = 50.0       # m：正六角形の一辺の長さ＝中心から頂点までの距離
UAV_HEIGHT = 100.0      # m：今回の検証で仮定するUAV高度（現行モデルの確定値ではない）
GRID_SIZE = 501         # 平均値を近似するための格子分割数

# Al-Hourani型モデルの環境依存パラメータ (a, b)
ENVIRONMENTS = {
    "Suburban": (4.88, 0.43),
    "Urban": (9.61, 0.16),
    "Dense urban": (12.08, 0.11),
    "High-rise urban": (27.23, 0.08),
}

# 図示の対象．都市環境を基準として検証する．
ENVIRONMENT = "Urban"
OUTPUT_DIR = Path(__file__).resolve().parent


def elevation_angle_deg(h, r):
    """UAV高度hと水平距離rから仰角[deg]を求める．r=0にも対応する．"""
    return np.degrees(np.arctan2(h, r))


def los_probability(h, r, a, b):
    """Al-Hourani型LoS確率．返り値は0～1の確率．"""
    theta = elevation_angle_deg(h, r)
    return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))


def hex_grid(radius, grid_size):
    """中心(0,0)，左右に頂点を持つ正六角形内部の地上座標を作る．"""
    xy = np.linspace(-radius, radius, grid_size)
    x, y = np.meshgrid(xy, xy)

    # |y| <= sqrt(3)/2*R，|x| <= R-|y|/sqrt(3) を満たせば六角形内部．
    inside = ((np.abs(y) <= np.sqrt(3) * radius / 2)
            & (np.abs(x) <= radius - np.abs(y) / np.sqrt(3)))
    r = np.where(inside, np.hypot(x, y), np.nan)
    return x, y, r


def plot_hex_map(x, y, r, height, radius, env_name, a, b):
    """六角領域内のLoS確率分布を描画する．"""
    p = np.ma.masked_invalid(100.0 * los_probability(height, r, a, b))
    p_min = 100.0 * los_probability(height, radius, a, b)

    angles = np.deg2rad(np.arange(0, 361, 60))
    vx = radius * np.cos(angles)
    vy = radius * np.sin(angles)

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(x, y, p, shading="nearest", cmap="viridis",
                        vmin=p_min, vmax=100.0, rasterized=True)
    fig.colorbar(mesh, ax=ax, label="LoS probability [%]")
    ax.plot(vx, vy, "k-", linewidth=1.5)
    ax.plot(0.0, 0.0, "k+", markersize=10, markeredgewidth=2,
            label="UAV ground projection")
    ax.set_aspect("equal")
    ax.set_xlabel("Ground X [m]")
    ax.set_ylabel("Ground Y [m]")
    ax.set_title(f"{env_name}: LoS probability inside hexagon\n"
                f"h = {height:g} m, R = {radius:g} m, min = {p_min:.4f}%")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "los_hex_map.png", dpi=180)
    plt.close(fig)


def plot_height_sensitivity(radius, height):
    """環境・高度を変えたときの六角領域内の最小LoS確率．"""
    heights = np.linspace(20.0, 200.0, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (a, b) in ENVIRONMENTS.items():
        p_min = 100.0 * los_probability(heights, radius, a, b)
        ax.plot(heights, p_min, label=name)

    a, b = ENVIRONMENTS[ENVIRONMENT]
    p_current = 100.0 * los_probability(height, radius, a, b)
    ax.plot(height, p_current, "ko", markersize=6,
            label=f"{ENVIRONMENT} at h={height:g} m ({p_current:.4f}%)")
    ax.axhline(99.0, color="gray", linewidth=1, linestyle="--", label="99%")
    ax.set_xlabel("UAV altitude [m]")
    ax.set_ylabel("Minimum LoS probability inside hexagon [%]")
    ax.set_title(f"Worst-case LoS probability vs. altitude (R = {radius:g} m)")
    ax.set_xlim(20, 200)
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "los_altitude_sensitivity.png", dpi=180)
    plt.close(fig)


def main():
    if HEX_RADIUS <= 0 or UAV_HEIGHT <= 0 or GRID_SIZE < 3:
        raise ValueError("HEX_RADIUSとUAV_HEIGHTは正，GRID_SIZEは3以上にすること．")

    x, y, r = hex_grid(HEX_RADIUS, GRID_SIZE)
    theta_min = float(elevation_angle_deg(UAV_HEIGHT, HEX_RADIUS))
    rows = []

    for name, (a, b) in ENVIRONMENTS.items():
        # 六角形で最も遠い点が頂点 (r=R) なので，ここでの値が厳密な下限．
        p_min = 100.0 * float(los_probability(UAV_HEIGHT, HEX_RADIUS, a, b))
        # 中心直下 (r=0) が上限．
        p_max = 100.0 * float(los_probability(UAV_HEIGHT, 0.0, a, b))
        # 均一なUE分布を仮定した面積平均の数値近似．下限の証明には使わない．
        p_mean = 100.0 * float(np.nanmean(los_probability(UAV_HEIGHT, r, a, b)))
        rows.append([name, f"{UAV_HEIGHT:.2f}", f"{HEX_RADIUS:.2f}",
                    f"{theta_min:.6f}", f"{p_min:.6f}",
                    f"{p_mean:.6f}", f"{p_max:.6f}"])

    print(f"Hexagon radius R = {HEX_RADIUS:.1f} m")
    print(f"UAV altitude     h = {UAV_HEIGHT:.1f} m (assumption)")
    print(f"Worst-case elevation angle = {theta_min:.6f} degrees\n")
    print(f"{'Environment':<17} {'Min [%]':>12} {'Mean [%]':>12} {'Max [%]':>12}")
    for row in rows:
        print(f"{row[0]:<17} {float(row[4]):12.6f} {float(row[5]):12.6f} {float(row[6]):12.6f}")

    with (OUTPUT_DIR / "los_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["environment", "height_m", "hex_radius_m", "min_elevation_deg",
                        "min_los_percent", "mean_los_percent", "max_los_percent"])
        writer.writerows(rows)

    a, b = ENVIRONMENTS[ENVIRONMENT]
    plot_hex_map(x, y, r, UAV_HEIGHT, HEX_RADIUS, ENVIRONMENT, a, b)
    plot_height_sensitivity(HEX_RADIUS, UAV_HEIGHT)
    print("\nSaved: los_summary.csv, los_hex_map.png, los_altitude_sensitivity.png")


if __name__ == "__main__":
    main()
