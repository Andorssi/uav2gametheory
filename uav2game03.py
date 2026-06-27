import itertools
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.patheffects as pe
import numpy as np
import random

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
load_client_file: 
set_client(): client_areaとareasにクライアント情報を設定する
area_center(area_id): plot時に用いる関数
move_clients(remaining_true): 未収集クライアントが動く
"""
class Field:
    def __init__(self):
        self.areas = {i: Area(i) for i in I}
        self.client_area = {}
        self.set_rinsetu()
        self.set_clients()

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

    def load_client_file(self, filename):
        cu = {}

        with open(filename, "r") as f:
            for line in f:
                u, area = map(int, line.split())
                cu[u] = area

        return cu

    def set_clients(self):
        cu = self.load_client_file("client_area.txt")

        for client_id, area_id in cu.items():
            self.client_area[client_id] = area_id
            self.areas[area_id].add_client(client_id)
    
    def area_center(self, area_id):
        r = area_id // width
        c = area_id % width

        x = c * 1.5
        y = -r * np.sqrt(3)

        if c % 2 == 1:
            y -= np.sqrt(3) / 2

        return x, y

    def move_clients(self, remaining_true):
        for client_id in remaining_true:
            if random.random() >= MOVE_p:
                continue
            current_area = self.client_area[client_id]
            candidates = self.areas[current_area].neighbors
            if len(candidates) == 0:
                continue
            next_area = random.choice(candidates)
            # 現在の領域から削除
            self.areas[current_area].clients.discard(client_id)
            # 新しい領域へ追加
            self.areas[next_area].clients.add(client_id)
            # 現在位置更新
            self.client_area[client_id] = next_area

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

    def create_initial_plan(self, uavs, game):
        plans = {uav: [] for uav in uavs}

        virtual_pos = { uav: uav.area for uav in uavs }

        virtual_remaining = set(self.remaining_clients)

        while len(virtual_remaining) > 0:
            assignment = self.social_optimize_for_plan(
                uavs,
                game,
                virtual_pos,
                virtual_remaining
            )

            for uav, area in assignment.items():
                plans[uav].append(area)

                candidates = sorted(
                    game.field.areas[area].clients & virtual_remaining
                )

                collected = set(candidates[:M])
                virtual_remaining -= collected
                virtual_pos[uav] = area

        return plans

"""
コンストラクタ生成: 社会最適となる軌道が計算される

"""
class Game:
    def __init__(self):
        self.field = Field()
        self.uavs = [UAV(uav_id=d, start=B) for d in D]
        self.remaining_true = set(U)

        self.server = Server(remaining_clients=self.remaining_true)

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

        self.field.move_clients(self.remaining_true)

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
        plans = self.server.create_initial_plan(self.uavs, self)

        for uav in self.uavs:
            uav.plan = plans[uav]

        print("initial plan created")
        for uav in self.uavs:
            print(f"UAV{uav.id} plan:", uav.plan)

        self.plot_initial_plan("initial_plan.png")

    def plot_initial_plan(self, output="./figure/initial_plan.png"):
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
                radius=1.0,
                orientation=np.radians(30),
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.2,
                alpha=alpha_value
            )
            ax.add_patch(hexagon)

            # 領域番号：左上
            ax.text(
                x - 0.45,
                y + 0.45,
                str(area_id),
                ha="center",
                va="center",
                fontsize=9,
                color="red",
                fontweight="bold"
            )

            # UE数：右下
            ax.text(
                x + 0.45,
                y - 0.45,
                f"[{len(area.clients)}]",
                ha="center",
                va="center",
                fontsize=9,
                color="black"
            )

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
                    x + dx + ldx,
                    y + dy + ldy,
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
    game = Game()
    game.run()