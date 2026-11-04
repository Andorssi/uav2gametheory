import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np
from collections import defaultdict

width = 7
height = 7
B = 24


def area_center(area_id):
    r = area_id // width
    c = area_id % width

    x = c * 1.5
    y = -r * np.sqrt(3)

    if c % 2 == 1:
        y -= np.sqrt(3) / 2

    return x, y


def count_client(filename):
    area_count = {i: 0 for i in range(width * height)}

    with open(filename, "r") as f:
        for line in f:
            u, area = map(int, line.split())
            area_count[area] += 1

    return area_count


def draw_hex_field(filename="client_area.txt", output="hex_field.png"):
    area_count = count_client(filename)

    fig, ax = plt.subplots(figsize=(10, 10))

    for area_id in range(width * height):
        x, y = area_center(area_id)

        facecolor = "pink" if area_id == B else "white"

        hexagon = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=1.0,
            orientation=np.radians(30),
            facecolor=facecolor,
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(hexagon)

        # 領域番号：左上
        ax.text(
            x - 0.45,
            y + 0.45,
            str(area_id),
            ha="center",
            va="center",
            fontsize=10,
            color="red",
            fontweight="bold"
        )

        # UE数：右下
        ax.text(
            x + 0.45,
            y - 0.45,
            f"[{area_count[area_id]}]",
            ha="center",
            va="center",
            fontsize=10,
            color="black"
        )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    draw_hex_field()