import itertools
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.patheffects as pe
import numpy as np
import random
import math
from pathlib import Path

# データ設定
U = list(range(200))    # クライアント生成
D = list(range(5))      # UAV生成
I = list(range(49))     # 領域生成
M = 5                   # 一度に通信できるクライアント数
width = 7               # 横
height = 7              # 縦
move_k = 2              # UAVはmove_kマス移動可能
MOVE_p = 0.3            # クライアントはMOVE_pの確率で移動する
B = 24                  # サーバ設定
HEX_RADIUS = 50.0       # 六角領域の中心から頂点までの距離[m]（仮設定）
SEED = 1                # UEの初期座標生成用乱数シード
CLIENT_POSITION_FILE = Path(__file__).resolve().parent / "ue_positions.txt"  # 固定したUE初期配置

# パラメータ
alpha = 0.1
beta = 0.3

"""
add_client(u): uを追加する
"""
class Area:
    def __init__(self, area_id):
        self.id = area_id
        self.clients = set()
        self.neighbors = []

    def add_client(self, u):
        self.clients.add(u)

"""
set_rinsetu(): 隣接領域をareasに設定する
get_rinsetu(i): 領域iの隣接領域を取得する
get_k_rinsetu(i): kマス以内の隣接領域を取得する
field_distance(start_area, goal_area): startからgoalまでの距離を取得する
area_center(area_id): 六角領域の中心座標を返す
"""
class Field:
    def __init__(self):
        self.areas = {i: Area(i) for i in I}
        self.set_rinsetu()

    def set_rinsetu(self):
        for area_id in self.areas:
            self.areas[area_id].neighbors = self.get_rinsetu(area_id)

    def get_rinsetu(self, i):
        x = i // width
        y = i % width

        if y % 2 == 0:
            directions = [(-1,-1), (0,-1), (-1,0), (1,0), (-1,1), (0,1)]
        else:
            directions = [(0,-1), (1,-1), (-1,0), (1,0), (0,1), (1,1)]

        neighbor = []
        for dr, dc in directions:
            r, c = x + dr, y + dc
            if 0 <= r < height and 0 <= c < width:
                neighbor.append(r * width + c)

        return neighbor

    def get_k_rinsetu(self, i, k):
        visited = {i}
        queue = deque([(i, 0)])

        while queue:
            current, dist = queue.popleft()

            if dist == k:
                continue

            for nb in self.areas[current].neighbors:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

        return list(visited)

    def field_distance(self, start_area, goal_area):
        if start_area == goal_area:
            return 0

        visited = {start_area}
        queue = deque([(start_area, 0)])

        while queue:
            current, dist = queue.popleft()

            for nb in self.areas[current].neighbors:
                if nb == goal_area:
                    return dist + 1

                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))

        return 10**9

    def area_center(self, area_id):
        r = area_id // width
        c = area_id % width

        x = c * 1.5 * HEX_RADIUS
        y = -r * np.sqrt(3) * HEX_RADIUS

        if c % 2 == 1:
            y -= np.sqrt(3) * HEX_RADIUS / 2

        return x, y

    def find_area(self, x, y):
        # UEの連続座標(x, y)が属する六角領域を返す．
        # 六角格子の外側であればNoneを返す．
        R = HEX_RADIUS
        tol = 1e-6
        for area_id in self.areas:
            cx, cy = self.area_center(area_id)
            dx, dy = x - cx, y - cy
            if abs(dy) <= math.sqrt(3) * R / 2 + tol and math.sqrt(3) * abs(dx) + abs(dy) <= math.sqrt(3) * R + tol:
                return area_id
        return None

"""
move(dx, dy): UEの位置を変更する
position(): UEの現在位置を取得する
"""
class UE:
    def __init__(self, client_id, x, y, priority):
        self.id = client_id
        self.x = x
        self.y = y
        self.priority = priority

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def position(self):
        return (self.x, self.y)


"""
load_client_file(filename): UEの初期座標ファイルを読み込む
random_position_in_area(area_id): 六角領域内の一様ランダム座標を生成する
set_clients(): UEオブジェクトを生成し所属領域を登録する
move_clients(remaining_true): 未収集UEを隣接領域へ確率移動させる
plot_client_positions(output): 六角格子およびUE位置を描画する
"""
class UEField:
    def __init__(self, field):
        self.field = field
        self.clients = {}       # {UE番号: UEオブジェクト}
        self.client_area = {}   # {UE番号: 所属領域番号}
        self.rng = random.Random(SEED)
        self.set_clients()

    def load_client_file(self, filename):
        # ファイル形式: UE番号 x[m] y[m]（領域番号は座標から判定する）
        cu = {}
        with open(filename, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                values = line.split()
                if len(values) != 3:
                    raise ValueError(f"{filename}:{line_no}: UE番号 x y の3列が必要です．")
                client_id = int(values[0])
                x, y = map(float, values[1:])
                if client_id in cu:
                    raise ValueError(f"{filename}:{line_no}: UE{client_id} が重複しています．")
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f"{filename}:{line_no}: 座標は有限値である必要があります．")
                area_id = self.field.find_area(x, y)
                if area_id is None:
                    raise ValueError(f"{filename}:{line_no}: UE{client_id} の座標が六角格子の外側です．")
                cu[client_id] = (area_id, x, y)

        missing = set(U) - set(cu)
        extra = set(cu) - set(U)
        if missing or extra:
            raise ValueError(f"UE番号がUと一致しません．不足: {sorted(missing)}, 余分: {sorted(extra)}")
        return cu

    def random_position_in_area(self, area_id):
        # 六角領域内に一様に位置する座標を生成する（棄却法）
        cx, cy = self.field.area_center(area_id)
        R = HEX_RADIUS
        while True:
            dx = self.rng.uniform(-R, R)
            dy = self.rng.uniform(-math.sqrt(3) * R / 2, math.sqrt(3) * R / 2)
            if math.sqrt(3) * abs(dx) + abs(dy) <= math.sqrt(3) * R:
                return cx + dx, cy + dy

    def set_clients(self):
        # 初期配置は毎回抽選せず，別プログラムで生成した座標ファイルから読み込む．
        if not CLIENT_POSITION_FILE.exists():
            raise FileNotFoundError(f"UE初期座標ファイルが見つかりません: {CLIENT_POSITION_FILE}")
        cu = self.load_client_file(CLIENT_POSITION_FILE)
        for client_id, (area_id, x, y) in cu.items():
            priority = 1  # 優先度は仮値．既存の優先度データとの連携は次段階
            self.clients[client_id] = UE(client_id, x, y, priority)
            self.client_area[client_id] = area_id
            self.field.areas[area_id].add_client(client_id)

    """
    今後移動モデルは改善していく必要有
    """
    def move_clients(self, remaining_true):
        # 既存の「隣接領域へ確率移動する」仕様を維持する．
        # UEの連続的な少量移動への変更は次の段階で行う．
        for client_id in remaining_true:
            if random.random() >= MOVE_p:
                continue
            current_area = self.client_area[client_id]
            candidates = self.field.areas[current_area].neighbors
            if len(candidates) == 0:
                continue
            next_area = random.choice(candidates)
            # 現在の領域から削除
            self.field.areas[current_area].clients.discard(client_id)
            # 新しい領域へ追加
            self.field.areas[next_area].clients.add(client_id)
            # 現在位置更新
            self.client_area[client_id] = next_area
            x, y = self.random_position_in_area(next_area)
            client = self.clients[client_id]
            dx, dy = x - client.x, y - client.y
            client.move(dx, dy)

    def plot_client_positions(self, output="hex_ue_positions.png"):
        # 領域とUE座標の確認用（初期計画の処理とは独立）
        fig, ax = plt.subplots(figsize=(10, 10))
        for area_id in I:
            x, y = self.field.area_center(area_id)
            ax.add_patch(RegularPolygon((x, y), numVertices=6, radius=HEX_RADIUS, orientation=np.pi / 6, facecolor="white", edgecolor="gray", linewidth=0.8))
            ax.text(x, y, str(area_id), ha="center", va="center", color="gray", fontsize=8)
        xs = [u.x for u in self.clients.values()]
        ys = [u.y for u in self.clients.values()]
        ax.scatter(xs, ys, s=12, c="tab:green", label="UE", zorder=4)
        bx, by = self.field.area_center(B)
        ax.scatter([bx], [by], s=120, marker="*", c="tab:red", label="Server / UAV start", zorder=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output, dpi=200)
        plt.close()
        print(f"saved: {output}")

"""
move: 次に移動する
"""
class UAV:
    def __init__(self, uav_id, start):
        self.id = uav_id
        self.area = start
        self.plan = []

    def move(self, next_area):
        self.area = next_area

"""
social_score_for_plan(assignment, game, virtual_pos, virtual_remaining): 目的関数の計算(assignmentで収集可能クライアント数を計算している)
social_optimize_for_plan(uavs, game, virtual_pos, virtual_remaining): 収集可能クライアント数を計算
create_initial_plan(uavs, game): 社会最適軌道を生成
"""
class Server:
    def __init__(self, remaining_clients):
        self.remaining_clients = set(remaining_clients)

    def social_score_for_plan(self, assignment, game, virtual_pos, virtual_remaining):
        covered = set()
        total_distance = 0
        area_count = defaultdict(int)

        for uav, area in assignment.items():
            covered |= game.field.areas[area].clients & virtual_remaining
            total_distance += game.field.field_distance(virtual_pos[uav], area)
            area_count[area] += 1

        overlap = sum(max(0, count - 1) for count in area_count.values())

        return len(covered) - alpha * total_distance - beta * overlap

    def social_optimize_for_plan(self, uavs, game, virtual_pos, virtual_remaining):
        candidate_lists = []

        for uav in uavs:
            candidates = game.field.get_k_rinsetu(virtual_pos[uav], move_k)
            candidate_lists.append(candidates)

        best_assignment = None
        best_score = -10**18

        for combo in itertools.product(*candidate_lists):
            assignment = {
                uav: area
                for uav, area in zip(uavs, combo)
            }

            score = self.social_score_for_plan(
                assignment,
                game,
                virtual_pos,
                virtual_remaining
            )

            if score > best_score:
                best_score = score
                best_assignment = assignment

        return best_assignment

    def nearest_remaining_distance(self, area, game, virtual_remaining):
        """areaから未収集クライアントが残っている最寄り領域までの距離。"""
        remaining_areas = {
            game.ue_field.client_area[u]
            for u in virtual_remaining
        }

        if not remaining_areas:
            return 0

        return min(
            game.field.field_distance(area, target_area)
            for target_area in remaining_areas
        )

    def selfish_score_for_plan(self, uav, candidate_area, game, virtual_pos, virtual_remaining):
        """自己最適の目的関数。

        自分が候補領域で収集できるクライアント数と移動距離だけを見る。
        他UAVとの重複は考慮しない。
        候補範囲内に収集可能クライアントがいない場合に停止し続けないよう，
        最寄りの未収集クライアント領域へ近づく弱い項を入れる。
        """
        C = len(game.field.areas[candidate_area].clients & virtual_remaining)
        L = game.field.field_distance(virtual_pos[uav], candidate_area)
        H = self.nearest_remaining_distance(candidate_area, game, virtual_remaining)

        return C - alpha * L - 0.01 * H

    def selfish_optimize_for_plan(self, uavs, game, virtual_pos, virtual_remaining):
        """各UAVが独立に自己最適な移動先を選ぶ。

        社会最適と異なり，UAV間の重複配置ペナルティは入れない。
        """
        assignment = {}

        for uav in uavs:
            candidates = game.field.get_k_rinsetu(virtual_pos[uav], move_k)

            best_area = max(
                candidates,
                key=lambda area: self.selfish_score_for_plan(
                    uav,
                    area,
                    game,
                    virtual_pos,
                    virtual_remaining
                )
            )

            assignment[uav] = best_area

        return assignment

    def create_initial_plan(self, uavs, game, plan_mode):
        plans = {uav: [] for uav in uavs}

        virtual_pos = { uav: uav.area for uav in uavs }

        virtual_remaining = set(self.remaining_clients)

        while len(virtual_remaining) > 0:
            if plan_mode == "social":
                assignment = self.social_optimize_for_plan(
                    uavs,
                    game,
                    virtual_pos,
                    virtual_remaining
                )
            elif plan_mode == "selfish":
                assignment = self.selfish_optimize_for_plan(
                    uavs,
                    game,
                    virtual_pos,
                    virtual_remaining
                )
            else:
                raise ValueError("plan_mode must be 'social' or 'selfish'")

            collected_this_step = set()

            for uav, area in assignment.items():
                plans[uav].append(area)

                candidates = sorted(
                    game.field.areas[area].clients & virtual_remaining
                )

                collected = set(candidates[:M])
                collected_this_step |= collected
                virtual_remaining -= collected
                virtual_pos[uav] = area

            # 念のため，どのUAVも収集できない状態が続くことを防ぐ。
            if len(collected_this_step) == 0:
                print("Warning: no clients collected while creating plan.")
                print("remaining:", len(virtual_remaining))
                break

        return plans

"""
コンストラクタ生成: 社会最適となる軌道が計算される

"""
class Game:
    def __init__(self, plan_mode):
        self.plan_mode = plan_mode
        self.field = Field()
        self.ue_field = UEField(self.field)
        self.uavs = [UAV(uav_id=d, start=B) for d in D]
        self.remaining_true = set(U)

        self.server = Server(remaining_clients=self.remaining_true)

        self.ue_field.plot_client_positions()
        self.create_initial_plan()

    def move_uavs(self, placement):
        for uav, area in placement.items():
            uav.move(area)

    def collect_clients(self, placement):
        collected_all = set()

        for uav in sorted(placement.keys(), key=lambda x: x.id):
            area_id = uav.area

            candidates = sorted(
                self.field.areas[area_id].clients & self.remaining_true
            )

            collected = set(candidates[:M])
            self.remaining_true -= collected
            collected_all |= collected

        return collected_all

    def step(self, t):
        placement = {}

        for uav in self.uavs:
            if t < len(uav.plan):
                placement[uav] = uav.plan[t]

        self.move_uavs(placement)

        collected_all = self.collect_clients(placement)

        self.ue_field.move_clients(self.remaining_true)

        self.print_step(t, collected_all)

    def run(self):
        max_plan_length = max(len(uav.plan) for uav in self.uavs)

        for t in range(max_plan_length):
            self.step(t)

            if len(self.remaining_true) == 0:
                break

    def print_step(self, t, collected_all):
        print()
        print(f"t = {t}")

        print("placement:")
        for uav in self.uavs:
            print(f"  UAV{uav.id} -> area {uav.area}")

        print("収集済UE数:", len(collected_all))
        print("未収集UE数:", len(self.remaining_true))

    def create_initial_plan(self):
        plans = self.server.create_initial_plan(self.uavs, self, self.plan_mode)

        for uav in self.uavs:
            uav.plan = plans[uav]

        print(f"initial plan created: {self.plan_mode}")
        for uav in self.uavs:
            print(f"UAV{uav.id} plan:", uav.plan)

        self.plot_initial_plan(f"initial_plan_{self.plan_mode}.png")

    def plot_initial_plan(self, output="initial_plan.png"):
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

        # 各領域をどのUAVが通るか記録
        passed_by = defaultdict(list)

        for uav in self.uavs:
            route = [B] + uav.plan
            for t, area_id in enumerate(route):
                passed_by[area_id].append((uav.id, t))

        fig, ax = plt.subplots(figsize=(11, 10))

        for area_id, area in self.field.areas.items():
            x, y = self.field.area_center(area_id)

            # 通過UAVがいる領域は最初に通ったUAVの色で塗る
            if area_id in passed_by:
                first_uav_id = passed_by[area_id][0][0]
                facecolor = colors[first_uav_id % len(colors)]
                alpha_value = 0.35
            elif area_id == B:
                facecolor = "pink"
                alpha_value = 0.8
            else:
                facecolor = "white"
                alpha_value = 1.0

            hexagon = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=HEX_RADIUS,
                orientation=np.pi / 6,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.2,
                alpha=alpha_value
            )
            ax.add_patch(hexagon)

            # 領域番号：左上
            ax.text(
                x - 0.45 * HEX_RADIUS,
                y + 0.45 * HEX_RADIUS,
                str(area_id),
                ha="center",
                va="center",
                fontsize=9,
                color="red",
                fontweight="bold"
            )

            # UE数：右下
            ax.text(
                x + 0.45 * HEX_RADIUS,
                y - 0.45 * HEX_RADIUS,
                f"[{len(area.clients)}]",
                ha="center",
                va="center",
                fontsize=9,
                color="black"
            )

        # UEを連続座標で描画
        ax.scatter([p[0] for p in (u.position() for u in self.ue_field.clients.values())], [p[1] for p in (u.position() for u in self.ue_field.clients.values())], s=10, c="black", alpha=0.65, label="UE", zorder=3)

        # UAVごとの軌道線と時刻ラベル
        offsets = {
            0: (0.00,  0.00),
            1: (0.14,  0.14),
            2: (-0.14, 0.14),
            3: (0.14, -0.14),
            4: (-0.14, -0.14),
        }

        # 同じ領域に複数の時刻ラベルが来たときの追加ずらし
        label_offsets = [
            (0.00,  0.00),
            (0.22,  0.00),
            (-0.22, 0.00),
            (0.00,  0.22),
            (0.00, -0.22),
            (0.18,  0.18),
            (-0.18, 0.18),
            (0.18, -0.18),
            (-0.18, -0.18),
        ]

        label_count = defaultdict(int)

        for uav in self.uavs:
            color = colors[uav.id % len(colors)]
            dx, dy = offsets.get(uav.id, (0.0, 0.0))
            dx *= HEX_RADIUS
            dy *= HEX_RADIUS

            route = [B] + uav.plan

            xs = []
            ys = []

            for area_id in route:
                x, y = self.field.area_center(area_id)
                xs.append(x + dx)
                ys.append(y + dy)

            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4,
                linewidth=2,
                color=color,
                label=f"UAV{uav.id}",
                alpha=0.85
            )

            for t, area_id in enumerate(route):
                x, y = self.field.area_center(area_id)

                k = label_count[area_id]
                ldx, ldy = label_offsets[k % len(label_offsets)]
                label_count[area_id] += 1

                ax.text(
                    x + dx + ldx * HEX_RADIUS,
                    y + dy + ldy * HEX_RADIUS,
                    f"{t}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
                )

        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"saved: {output}")


if __name__ == "__main__":
    print("計算方法を選択してください")
    print("1: 社会最適")
    print("2: 自己最適")

    mode_input = input("番号を入力: ").strip()

    if mode_input == "1":
        plan_mode = "social"
    elif mode_input == "2":
        plan_mode = "selfish"
    else:
        print("入力が不正なので，社会最適で実行します。")
        plan_mode = "social"

    game = Game(plan_mode)
    game.run()
