"""
UEの初期座標を独立に生成し，TXTファイルだけを出力する．
出力形式：UE番号 x座標[m] y座標[m]（小数第1位まで）
"""
import argparse
import math
import random
from pathlib import Path

# UE配置を生成する範囲の設定（TXTを読み込む側の対象領域に合わせる）
U = list(range(400))
width = 7
height = 7
HEX_RADIUS = 50.0
SEED = 1

BASE_DIR = Path(__file__).resolve().parent


class Field:
    """生成プログラム内だけで使う六角格子の座標情報．"""
    def __init__(self):
        self.areas = list(range(width * height))

    def area_center(self, area_id):
        r = area_id // width
        c = area_id % width
        x = c * 1.5 * HEX_RADIUS
        y = -r * math.sqrt(3) * HEX_RADIUS
        if c % 2 == 1:
            y -= math.sqrt(3) * HEX_RADIUS / 2
        return x, y


def random_position_in_area(field, area_id, rng):
    # 小数第1位に丸めた後も六角領域内に収まる位置を生成する（棄却法）．
    cx, cy = field.area_center(area_id)
    R = HEX_RADIUS
    while True:
        dx = rng.uniform(-R, R)
        dy = rng.uniform(-math.sqrt(3) * R / 2, math.sqrt(3) * R / 2)
        x, y = round(cx + dx, 1), round(cy + dy, 1)
        dx, dy = x - cx, y - cy
        if abs(dy) <= math.sqrt(3) * R / 2 and math.sqrt(3) * abs(dx) + abs(dy) <= math.sqrt(3) * R:
            return x, y


def generate_positions(output_file, seed, overwrite=False):
    field = Field()
    rng = random.Random(seed)
    area_ids = list(field.areas)
    output_file = Path(output_file)

    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} は既に存在します．再生成する場合は --overwrite を指定してください．")

    rows = []
    for client_id in U:
        area_id = rng.choice(area_ids)
        x, y = random_position_in_area(field, area_id, rng)
        rows.append(f"{client_id} {x:.1f} {y:.1f}\n")

    with open(output_file, "w", encoding="utf-8", newline="\n") as f:
        f.write("# UE番号 x[m] y[m]\n")
        f.writelines(rows)

    print(f"saved: {output_file} ({len(rows)} UEs, seed={seed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UEの初期座標を独立に生成してTXTに保存します．")
    parser.add_argument("--output", type=Path, default=BASE_DIR / f"ue_positions_{len(U)}.txt", help="保存先TXTファイル")
    parser.add_argument("--seed", type=int, default=SEED, help="座標生成用乱数シード")
    parser.add_argument("--overwrite", action="store_true", help="既存の座標ファイルを上書きする")
    args = parser.parse_args()
    generate_positions(args.output, args.seed, args.overwrite)
