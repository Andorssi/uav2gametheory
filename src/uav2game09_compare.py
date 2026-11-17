from collections import defaultdict, deque
import csv
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib.patheffects as pe
import numpy as np
import random
import math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

# データ設定
U = list(range(200))    # クライアント生成
D = list(range(5))      # UAV生成
I = list(range(49))     # 領域生成
M = 5                   # 一度に通信できるクライアント数
width = 7               # 横
height = 7              # 縦
move_k = 1              # UAVはmove_kマス移動可能

# CTCRW（Continuous-Time Correlated Random Walk）のパラメータ（暫定値）
STEP_DURATION = 60.0      # UAV移動・通信を含む1ステップの時間[s]
CTCRW_DT = STEP_DURATION  # UEを1ステップ分移動させるCTCRWの更新間隔[s]
CTCRW_BETA = 0.12        # 速度相関の減衰率[1/s]
CTCRW_SIGMA = 0.30       # 速度の拡散係数[m/s^(3/2)]

B = 24                  # サーバ設定
HEX_RADIUS = 50.0       # 六角領域の中心から頂点までの距離[m]（仮設定）
SEED = random.SystemRandom().randrange(2**63)  # 実行ごとに変更する乱数シード（初期座標はTXTから取得）
EPSILON = 0.1            # ε-greedyのランダム移動確率
T_MAX = 500              # t=0で生成する初期軌道の最大計画ステップ数（無限ループ防止）
OPERATION_MAX_STEPS = 25  # 実際に移動・収集する最大ステップ数（60秒×25=25分）
SELF_GAIN_THRESHOLD = 2     # 自己最適の切り替えに必要な収集数増加（仮置き，1にも変更可能）
SELF_MAX_PER_UAV = 3         # 自己最適の濫用防止：1機あたりの切り替え回数上限（仮置き）
SELF_MAX_RECOVERY_HOPS = 2  # 社会最適の次の訪問先への復帰を最大2移動に制限
# 実行結果は result/<プログラムファイル名>-YYYYMMDD/ に保存する．
# 同じ日に複数回実行した場合は -02，-03... を付与し，前回の結果を上書きしない．
RESULT_ROOT = Path(__file__).resolve().parent / "result"
RESULT_NAME = f"{Path(__file__).stem}-{datetime.now(ZoneInfo('Asia/Tokyo')):%Y%m%d}"
FIGURE_DIR = RESULT_ROOT / RESULT_NAME  # 画像・CSVなど実行結果の共通保存先
result_index = 2
while FIGURE_DIR.exists():
    FIGURE_DIR = RESULT_ROOT / f"{RESULT_NAME}-{result_index:02d}"
    result_index += 1
CLIENT_POSITION_FILE = Path(__file__).resolve().parent / f"ue_positions_{len(U)}.txt"  # 固定したUE初期配置

# パラメータ
alpha = 0.1

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
        self.prepare_recovery_table()  # サーバ側で事前計算して配布する想定の距離表

    def prepare_recovery_table(self):
        """全49領域間の最短ホップ数を事前計算する．運用中にはBFSを実行しない．"""
        self.recovery_distance = {}
        for start in I:
            distance = {start: 0}
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in self.areas[current].neighbors:
                    if neighbor not in distance:
                        distance[neighbor] = distance[current] + 1
                        queue.append(neighbor)
            self.recovery_distance[start] = distance

    def get_recovery_candidates(self, current, target):
        """復帰先への距離を1以上減らす，次の1マス移動候補を返す．"""
        current_distance = self.recovery_distance[current][target]
        candidates = self.get_k_rinsetu(current, move_k)
        return sorted(
            (area for area in candidates if self.recovery_distance[area][target] < current_distance),
            key=lambda area: (self.recovery_distance[area][target], area)
        )

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
        self.vx = 0.0     # x方向の移動速度[m/s]
        self.vy = 0.0     # y方向の移動速度[m/s]

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def position(self):
        return (self.x, self.y)


"""
load_client_file(filename): UEの初期座標ファイルを読み込む
random_position_in_area(area_id): 六角領域内の一様ランダム座標を生成する
set_clients(): UEオブジェクトを生成し所属領域を登録する
move_clients(remaining_true): 未収集UEをCTCRWに従って小規模移動させる
plot_client_positions(output): 六角格子およびUE位置を描画する
"""
class UEField:
    def __init__(self, field):
        self.field = field
        self.clients = {}       # {UE番号: UEオブジェクト}
        self.client_area = {}   # {UE番号: 所属領域番号}
        self.rng = random.Random(SEED)
        self.mobility_rng = {}  # UEごとに独立した乱数系列を管理する．
        if CTCRW_DT <= 0 or CTCRW_BETA <= 0 or CTCRW_SIGMA <= 0:
            raise ValueError("CTCRW_DT, CTCRW_BETA, CTCRW_SIGMAは正の値にしてください．")
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
            # 定常速度分布から初速度を設定する．UEごとに再現可能な乱数系列を使う．
            ue_rng = random.Random(SEED + 1000003 * client_id)
            self.mobility_rng[client_id] = ue_rng
            initial_std = CTCRW_SIGMA / math.sqrt(2 * CTCRW_BETA)
            self.clients[client_id].vx = ue_rng.gauss(0, initial_std)
            self.clients[client_id].vy = ue_rng.gauss(0, initial_std)
            self.client_area[client_id] = area_id
            self.field.areas[area_id].add_client(client_id)

    def move_clients(self, remaining_true):
        # CTCRWの位置と速度を同時に更新する．移動確率MOVE_pは使用しない．
        # 位置と速度の確率変動を独立に生成せず，共分散を考慮する（厳密な遷移分布）．
        dt = CTCRW_DT
        beta_ctcrw = CTCRW_BETA
        sigma = CTCRW_SIGMA
        a = math.exp(-beta_ctcrw * dt)
        b = (1 - a) / beta_ctcrw

        var_v = sigma**2 * (1 - a**2) / (2 * beta_ctcrw)
        var_x = (sigma / beta_ctcrw)**2 * (dt - 2 * b + (1 - a**2) / (2 * beta_ctcrw))
        cov_xv = sigma**2 * (1 - a)**2 / (2 * beta_ctcrw**2)
        std_v = math.sqrt(var_v)
        shared_x = cov_xv / std_v
        residual_x = math.sqrt(max(0.0, var_x - shared_x**2))

        moved = 0
        area_changed = 0
        boundary_reflections = 0
        total_distance = 0.0

        for client_id in sorted(remaining_true):
            client = self.clients[client_id]
            ue_rng = self.mobility_rng[client_id]

            # CTCRWの1次元遷移をx軸とy軸でそれぞれ計算する．
            # zeta（速度ノイズ）とxi（位置ノイズ）には相関を持たせる．
            v_noise_x = ue_rng.gauss(0, 1)
            x_noise_x = ue_rng.gauss(0, 1)
            v_noise_y = ue_rng.gauss(0, 1)
            x_noise_y = ue_rng.gauss(0, 1)
            dx = b * client.vx + shared_x * v_noise_x + residual_x * x_noise_x
            dy = b * client.vy + shared_x * v_noise_y + residual_x * x_noise_y
            next_vx = a * client.vx + std_v * v_noise_x
            next_vy = a * client.vy + std_v * v_noise_y

            # 領域内外の移動は許容し，フィールド全体から外に出るときだけ反射する．
            next_area = self.field.find_area(client.x + dx, client.y + dy)
            if next_area is None:
                boundary_reflections += 1
                # 境界での簡易反射．対称位置もフィールド外なら，その場にとどまる．
                reflected_area = self.field.find_area(client.x - dx, client.y - dy)
                next_vx, next_vy = -next_vx, -next_vy
                if reflected_area is not None:
                    dx, dy = -dx, -dy
                    next_area = reflected_area
                else:
                    dx, dy = 0.0, 0.0
                    next_area = self.client_area[client_id]

            current_area = self.client_area[client_id]
            client.move(dx, dy)
            client.vx, client.vy = next_vx, next_vy
            total_distance += math.hypot(dx, dy)
            if dx != 0 or dy != 0:
                moved += 1

            if next_area != current_area:
                # 六角境界を越えた場合だけ所属情報を変更する．
                self.field.areas[current_area].clients.discard(client_id)
                self.field.areas[next_area].clients.add(client_id)
                self.client_area[client_id] = next_area
                area_changed += 1

        return {
            "moved": moved,
            "area_changed": area_changed,
            "boundary_reflections": boundary_reflections,
            "mean_distance_m": total_distance / len(remaining_true) if remaining_true else 0.0,
        }

    def plot_client_positions(self, output="hex_ue_positions.png"):
        # 領域とUE座標の確認用（初期計画の処理とは独立）
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        output = FIGURE_DIR / Path(output).name
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
        self.plan_index = 0         # 次に実行すべき社会最適リストの要素
        self.mode = "SOCIAL"       # SOCIAL / RECOVERY
        self.recovery_target = None  # 自己最適前に訪問予定だった領域
        self.selfish_count = 0
        self.recovery_steps = 0

    def move(self, next_area):
        self.area = next_area

"""
social_score_for_plan(assignment, game, virtual_pos, virtual_remaining): 社会評価関数C - αLを計算
social_optimize_for_plan(uavs, game, virtual_pos, virtual_remaining): ε-greedyで1機ずつ協調配置を決める
create_initial_plan(uavs, game, plan_mode): t=0のUE分布のみで全UAVの初期軌道を生成する
"""
class Server:
    def __init__(self, remaining_clients):
        self.remaining_clients = set(remaining_clients)
        self.rng = random.Random(SEED + 1)  # UEのCTCRW用乱数と独立した選択用乱数
        self.last_selection = {}
        self.selection_history = []  # t=0で計画した各時刻のGreedy/ランダム選択履歴
        self.planned_remaining_count = 0

    def social_score_for_plan(self, assignment, game, virtual_pos, virtual_remaining):
        total_distance = 0
        area_count = defaultdict(int)

        for uav, area in assignment.items():
            total_distance += game.field.field_distance(virtual_pos[uav], area)
            area_count[area] += 1

        # 同一領域に複数機を配置した場合，機数×M台まで収集可能．
        # 同じUEを二重に数えず，重複配置自体へのペナルティは設けない．
        collected_count = sum(
            min(len(game.field.areas[area].clients & virtual_remaining), M * count)
            for area, count in area_count.items()
        )
        return collected_count - alpha * total_distance

    def social_optimize_for_plan(self, uavs, game, virtual_pos, virtual_remaining):
        # UAVを1機ずつ決めることで，移動先候補の全組合せ列挙を避ける．
        assignment = {}
        self.last_selection = {}

        for uav in sorted(uavs, key=lambda x: x.id):
            candidates = sorted(game.field.get_k_rinsetu(virtual_pos[uav], move_k))

            if self.rng.random() < EPSILON:
                # εの確率で移動可能領域から一様ランダムに選択する．
                best_area = self.rng.choice(candidates)
                self.last_selection[uav] = "random"
            else:
                # 1-εの確率で社会評価関数の増分が最大の領域を選択する．
                best_score = -float("inf")
                best_area = None
                for area in candidates:
                    candidate_assignment = dict(assignment)
                    candidate_assignment[uav] = area
                    score = self.social_score_for_plan(candidate_assignment, game, virtual_pos, virtual_remaining)
                    if score > best_score:
                        best_score = score
                        best_area = area

                # 収集可能なUEが候補内に残っていない場合は，最寄りの未収集UEに接近する．
                # 先読み探索ではなく，現在の未収集UE分布から距離を計算する．
                area_count = defaultdict(int)
                for area in assignment.values():
                    area_count[area] += 1
                marginal_exists = any(
                    len(game.field.areas[area].clients & virtual_remaining) > M * area_count[area]
                    for area in candidates
                )
                if not marginal_exists:
                    remaining_areas = {
                        game.ue_field.client_area[client_id]
                        for client_id in virtual_remaining
                    }
                    target_areas = {
                        area for area in remaining_areas
                        if len(game.field.areas[area].clients & virtual_remaining) > M * area_count[area]
                    }
                    if target_areas:
                        best_area = min(
                            candidates,
                            key=lambda area: (
                                min(game.field.field_distance(area, target) for target in target_areas),
                                game.field.field_distance(virtual_pos[uav], area),
                                area
                            )
                        )
                        self.last_selection[uav] = "greedy (approach)"
                    else:
                        self.last_selection[uav] = "greedy"
                else:
                    self.last_selection[uav] = "greedy"

            assignment[uav] = best_area

        return assignment

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
        C = min(len(game.field.areas[candidate_area].clients & virtual_remaining), M)
        L = game.field.field_distance(virtual_pos[uav], candidate_area)
        H = self.nearest_remaining_distance(candidate_area, game, virtual_remaining)

        return C - alpha * L - 0.01 * H

    def selfish_optimize_for_plan(self, uavs, game, virtual_pos, virtual_remaining):
        """各UAVが独立に自己最適な移動先を選ぶ。

        社会最適と異なり，UAV間の重複配置ペナルティは入れない。
        """
        assignment = {}
        self.last_selection = {}

        for uav in sorted(uavs, key=lambda x: x.id):
            candidates = sorted(game.field.get_k_rinsetu(virtual_pos[uav], move_k))

            if self.rng.random() < EPSILON:
                best_area = self.rng.choice(candidates)
                self.last_selection[uav] = "random"
            else:
                best_area = max(candidates, key=lambda area: self.selfish_score_for_plan(uav, area, game, virtual_pos, virtual_remaining))
                # 収集可能な領域がなければ最寄りの未収集UEへ近づく．
                if all(len(game.field.areas[area].clients & virtual_remaining) == 0 for area in candidates):
                    best_area = min(
                        candidates,
                        key=lambda area: (
                            self.nearest_remaining_distance(area, game, virtual_remaining),
                            game.field.field_distance(virtual_pos[uav], area),
                            area
                        )
                    )
                    self.last_selection[uav] = "greedy (approach)"
                else:
                    self.last_selection[uav] = "greedy"

            assignment[uav] = best_area

        return assignment

    def create_initial_plan(self, uavs, game, plan_mode):
        # t=0で把握したUE分布だけを用いて，全時刻の初期軌道を一括計算する．
        # 計画中にUEを移動させない．現実の収集結果・移動後のUE分布も参照しない．
        plans = {uav: [] for uav in uavs}
        virtual_pos = {uav: uav.area for uav in uavs}
        virtual_remaining = set(self.remaining_clients)
        self.selection_history = []

        for t in range(T_MAX):
            if not virtual_remaining:
                break

            if plan_mode == "social":
                assignment = self.social_optimize_for_plan(uavs, game, virtual_pos, virtual_remaining)
            elif plan_mode == "selfish":
                assignment = self.selfish_optimize_for_plan(uavs, game, virtual_pos, virtual_remaining)
            else:
                raise ValueError("plan_mode must be 'social' or 'selfish'")

            self.selection_history.append(dict(self.last_selection))

            for uav, area in assignment.items():
                plans[uav].append(area)

                # 仮想収集：初期分布に対して各UAVが最大M台を収集する想定．
                # 実際の未収集UE集合（game.remaining_true）は変更しない．
                candidates = sorted(game.field.areas[area].clients & virtual_remaining)
                collected = set(candidates[:M])
                virtual_remaining -= collected
                virtual_pos[uav] = area

            # 収集数が0のステップでも，遠方の未収集UEへ移動する途中の可能性がある．
            # したがって0件収集を理由に初期計画を終了しない．

        self.planned_remaining_count = len(virtual_remaining)
        if virtual_remaining:
            print(f"Warning: 初期軌道計算がT_MAX={T_MAX}に到達．仮想未収集UE数: {len(virtual_remaining)}")

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
        self.hybrid_actions = {}
        self.hybrid_log = []
        self.hybrid_paths = {uav.id: [B] for uav in self.uavs}
        self.step_history = []  # 社会最適のみ・ハイブリッドの比較用：ステップ別収集履歴

        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        self.ue_field.plot_client_positions()
        self.create_initial_plan()

    def move_uavs(self, placement):
        for uav, area in placement.items():
            uav.move(area)

    def collect_clients(self, placement):
        collected_all = set()
        self.last_collected_by_uav = {}

        for uav in sorted(placement.keys(), key=lambda x: x.id):
            area_id = uav.area

            candidates = sorted(
                self.field.areas[area_id].clients & self.remaining_true
            )

            collected = set(candidates[:M])
            self.remaining_true -= collected
            collected_all |= collected
            self.last_collected_by_uav[uav.id] = len(collected)

        return collected_all

    def expected_collections(self, placement):
        """現在のUE配置に対し，UAV番号順に収集した場合の各機の収集数を見積もる．"""
        virtual_remaining = set(self.remaining_true)
        result = {}
        for uav in sorted(self.uavs, key=lambda x: x.id):
            candidates = sorted(self.field.areas[placement[uav]].clients & virtual_remaining)
            collected = set(candidates[:M])
            result[uav] = len(collected)
            virtual_remaining -= collected
        return result

    def decide_hybrid_placement(self, t):
        """社会最適の訪問順序を優先し，周囲1マスの単純比較でのみ自己最適を許可する．

        社会最適・復帰中の行動を先に確定し，自己最適による新たな配置重複を避ける．
        元の社会最適が許可する同一領域への複数機配置は，ここでは変更しない．
        """
        placement = {}
        actions = {}
        social_reservations = set()

        # まずSOCIALの予定地を予約する．既存の社会最適の重複配置は維持する．
        for uav in sorted(self.uavs, key=lambda x: x.id):
            if uav.mode != "RECOVERY":
                target = uav.plan[uav.plan_index] if uav.plan_index < len(uav.plan) else uav.area
                if self.field.recovery_distance[uav.area][target] > move_k:
                    raise RuntimeError(f"UAV{uav.id}: SOCIALの次の予定領域まで1マスで移動できません ({uav.area} -> {target})")
                placement[uav] = target
                actions[uav] = "SOCIAL" if uav.plan_index < len(uav.plan) else "DONE"
                social_reservations.add(target)

        # RECOVERY中は自己最適を禁止し，事前計算表で元の次の予定地へ近づく．
        recovery_reservations = set()
        for uav in sorted(self.uavs, key=lambda x: x.id):
            if uav.mode != "RECOVERY":
                continue
            target = uav.recovery_target
            if uav.area == target:
                next_area = target
            else:
                choices = self.field.get_recovery_candidates(uav.area, target)
                next_area = next((area for area in choices if area not in social_reservations and area not in recovery_reservations), None)
                if next_area is None:
                    # 既存の社会最適では同一領域への複数機配置が可能．復帰が永久に止まることを避けるため，
                    # 最短復帰経路が社会最適の予約領域と重なる場合に限り重複を許容する．
                    next_area = next((area for area in choices if area not in recovery_reservations), uav.area)
            placement[uav] = next_area
            actions[uav] = "RECOVERY"
            recovery_reservations.add(next_area)

        # 社会最適の予定に全機が従った場合の収集可能数を比較基準とする．
        social_gains = self.expected_collections(placement)
        reserved_areas = set(placement.values())
        selfish_reservations = set()
        selfish_gains = {}

        for uav in sorted(self.uavs, key=lambda x: x.id):
            if uav.mode == "RECOVERY" or uav.plan_index >= len(uav.plan) or uav.selfish_count >= SELF_MAX_PER_UAV:
                continue

            target = uav.plan[uav.plan_index]
            candidates = sorted(self.field.get_k_rinsetu(uav.area, move_k))
            best_area = None
            best_gain = -1

            for area in candidates:
                if area == target or area in reserved_areas or area in selfish_reservations:
                    continue

                recovery_hops = self.field.recovery_distance[area][target]
                if recovery_hops > SELF_MAX_RECOVERY_HOPS:
                    continue
                # 復帰を完了する時間がない逸脱は許容しない．
                remaining_steps = OPERATION_MAX_STEPS - (t + 1)
                if remaining_steps < recovery_hops:
                    continue

                gain = min(M, len(self.field.areas[area].clients & self.remaining_true))
                # 収集数を最大化し，同数なら復帰の近い領域を選択する．
                if gain > best_gain or (gain == best_gain and best_area is not None and (recovery_hops, area) < (self.field.recovery_distance[best_area][target], best_area)):
                    best_area, best_gain = area, gain

            if best_area is not None and best_gain - social_gains[uav] >= SELF_GAIN_THRESHOLD:
                placement[uav] = best_area
                actions[uav] = "SELF"
                selfish_reservations.add(best_area)
                selfish_gains[uav] = best_gain
                # 元の予定地は引き続き予約したままにし，保守的に競合を回避する．

        return placement, actions, social_gains, selfish_gains

    def step_hybrid(self, t):
        # UE移動 -> 局所的な切り替え判定 -> UAV移動 -> 通信 の順に処理する．
        self.last_movement_stats = self.ue_field.move_clients(self.remaining_true)
        placement, actions, social_gains, selfish_gains = self.decide_hybrid_placement(t)

        for uav, next_area in placement.items():
            if self.field.recovery_distance[uav.area][next_area] > move_k:
                raise RuntimeError(f"UAV{uav.id}: 移動制限違反")

        self.move_uavs(placement)
        collected_all = self.collect_clients(placement)
        for uav in self.uavs:
            self.hybrid_paths[uav.id].append(uav.area)

        for uav in self.uavs:
            action = actions[uav]
            if action == "SELF":
                # 自己最適は1ステップで終了し，次から元の訪問予定領域へ復帰する．
                uav.recovery_target = uav.plan[uav.plan_index]
                uav.mode = "RECOVERY"
                uav.selfish_count += 1
            elif action == "RECOVERY":
                uav.recovery_steps += 1
                if uav.area == uav.recovery_target:
                    uav.plan_index += 1
                    uav.recovery_target = None
                    uav.mode = "SOCIAL"
            elif action == "SOCIAL":
                # 社会最適の訪問予定地に到達したら，訪問順序リストを1つ進める．
                uav.plan_index += 1

        self.hybrid_actions = actions
        self.step_history.append({
            "step": t + 1,
            "collected": len(collected_all),
            "total_collected": len(U) - len(self.remaining_true),
            "per_uav": dict(self.last_collected_by_uav),
        })
        self.hybrid_log.append({
            "t": t,
            "placement": {uav.id: placement[uav] for uav in self.uavs},
            "plan_indices": {uav.id: uav.plan_index for uav in self.uavs},
            "actions": {uav.id: actions[uav] for uav in self.uavs},
            "social_gains": {uav.id: social_gains[uav] for uav in self.uavs},
            "selfish_gains": {uav.id: selfish_gains[uav] for uav in selfish_gains},
            "collected_per_uav": dict(self.last_collected_by_uav),
            "collected": len(collected_all),
        })
        self.print_step(t, collected_all)

    def step(self, t):
        # t=0で確定した初期軌道に従って移動し，配置の再計算は行わない．
        placement = {}
        for uav in self.uavs:
            if t < len(uav.plan):
                placement[uav] = uav.plan[t]

        # 1ステップ（60秒）のUE移動を先に反映し，更新後の所属領域で通信する．
        # 初期軌道は更新後のUE分布を参照せず，生成時のまま維持する．
        self.last_movement_stats = self.ue_field.move_clients(self.remaining_true)
        self.move_uavs(placement)
        collected_all = self.collect_clients(placement)
        self.step_history.append({
            "step": t + 1,
            "collected": len(collected_all),
            "total_collected": len(U) - len(self.remaining_true),
            "per_uav": dict(self.last_collected_by_uav),
        })
        self.print_step(t, collected_all)

    def run(self):
        max_plan_length = max((len(uav.plan) for uav in self.uavs), default=0)
        executed_steps = 0

        # 初期軌道はt=0で全て生成するが，実際の稼働は最大25ステップに制限する．
        # tは0始まりなので，25ステップ実行する場合の最終ステップはt=24となる．
        steps = OPERATION_MAX_STEPS if self.plan_mode == "hybrid" else min(max_plan_length, OPERATION_MAX_STEPS)
        for t in range(steps):
            if self.plan_mode == "hybrid":
                if all(uav.plan_index >= len(uav.plan) and uav.mode == "SOCIAL" for uav in self.uavs):
                    break
                self.step_hybrid(t)
            else:
                self.step(t)
            executed_steps += 1
            if not self.remaining_true:
                break

        self.executed_steps = executed_steps
        print()
        print(f"初期計画の実行終了: {executed_steps}ステップ（最大{OPERATION_MAX_STEPS}ステップ）")
        print(f"実行時間: {executed_steps * STEP_DURATION / 60:.1f}分")
        print(f"収集済UE総数: {len(U) - len(self.remaining_true)}")
        print(f"t=0の仮想計画での未収集UE数: {self.server.planned_remaining_count}")
        print(f"実際の未収集UE数: {len(self.remaining_true)}")
        if self.plan_mode == "hybrid":
            print("自己最適回数:", {uav.id: uav.selfish_count for uav in self.uavs})
            print("復帰状態で移動したステップ数:", {uav.id: uav.recovery_steps for uav in self.uavs})
            self.plot_hybrid_execution()
            self.export_hybrid_log()
        if self.remaining_true:
            if executed_steps == OPERATION_MAX_STEPS and executed_steps < max_plan_length:
                print("最大稼働時間に達したため終了しました．未実行の初期軌道があります．")
            if self.plan_mode == "hybrid":
                print("未収集UEが残りました．社会最適の訪問順序と許可された局所的逸脱による収集を終了します．")
            else:
                print("未収集UEが残りました．初期軌道に基づく収集を終了します（自動再計画・自己最適への切り替えは行いません）．")
        else:
            print("全UEを収集しました．")

    def print_step(self, t, collected_all):
        print()
        print(f"t = {t}")

        print("placement:")
        selections = self.server.selection_history[t] if t < len(self.server.selection_history) else {}
        for uav in self.uavs:
            selection = selections.get(uav)
            suffix = f" ({selection})" if selection else ""
            if self.plan_mode == "hybrid":
                action = self.hybrid_actions.get(uav, "SOCIAL")
                suffix = f" ({action}, next_plan_index={uav.plan_index})"
            print(f"  UAV{uav.id} -> area {uav.area}{suffix}")

        print("収集済UE数:", len(collected_all))
        print("未収集UE数:", len(self.remaining_true))
        print("UE移動（領域変更数）:", self.last_movement_stats["area_changed"])
        print("UE移動（平均距離[m]）:", f'{self.last_movement_stats["mean_distance_m"]:.2f}')

    def create_initial_plan(self):
        # 実際の移動・収集を始める前に，固定されたt=0の分布から全軌道を計算する．
        initial_plan_mode = "social" if self.plan_mode == "hybrid" else self.plan_mode
        plans = self.server.create_initial_plan(self.uavs, self, initial_plan_mode)

        for uav in self.uavs:
            uav.plan = plans[uav]

        print(f"initial plan created at t=0: {self.plan_mode}, epsilon={EPSILON}, move_k={move_k}")
        print(f"計画終了時の仮想未収集UE数: {self.server.planned_remaining_count}")
        for uav in self.uavs:
            print(f"UAV{uav.id} plan:", uav.plan)

        # 初期UE分布と計画軌道の図を実行前に保存する．
        self.plot_initial_plan(f"initial_plan_{self.plan_mode}.png")

    def export_hybrid_log(self, output="hybrid_log.csv"):
        """実行時の切り替え判定と各UAVの収集数をCSVに残す．"""
        output = FIGURE_DIR / output
        with open(output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "uav_id", "area", "action", "next_plan_index", "social_gain", "selfish_gain", "actual_collected"])
            for record in self.hybrid_log:
                for uav in self.uavs:
                    uav_id = uav.id
                    writer.writerow([
                        record["t"] + 1, uav_id, record["placement"][uav_id],
                        record["actions"][uav_id], record["plan_indices"][uav_id],
                        record["social_gains"][uav_id], record["selfish_gains"].get(uav_id, ""),
                        record["collected_per_uav"].get(uav_id, 0)
                    ])
        print(f"saved: {output}")

    def plot_hybrid_execution(self, output="hybrid_executed_route.png"):
        """初期計画ではなく，実際に実行した経路とSELF・RECOVERYの発生場所を表示する．"""
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        offsets = [(0.00, 0.00), (0.14, 0.14), (-0.14, 0.14), (0.14, -0.14), (-0.14, -0.14)]
        fig, ax = plt.subplots(figsize=(11, 10))
        for area in I:
            x, y = self.field.area_center(area)
            ax.add_patch(RegularPolygon((x, y), numVertices=6, radius=HEX_RADIUS, orientation=np.pi / 6, facecolor="white", edgecolor="0.75", linewidth=0.8))
            ax.text(x - 0.43 * HEX_RADIUS, y + 0.40 * HEX_RADIUS, str(area), fontsize=8, color="0.45", ha="center", va="center")

        for uav in self.uavs:
            color = colors[uav.id % len(colors)]
            offx, offy = offsets[uav.id % len(offsets)]
            path = self.hybrid_paths[uav.id]
            xy = [self.field.area_center(area) for area in path]
            xs = [x + offx * HEX_RADIUS for x, y in xy]
            ys = [y + offy * HEX_RADIUS for x, y in xy]
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.8, color=color, label=f"UAV{uav.id}")
            for i, record in enumerate(self.hybrid_log, start=1):
                action = record["actions"][uav.id]
                if action in ("SELF", "RECOVERY"):
                    ax.scatter([xs[i]], [ys[i]], marker="*" if action == "SELF" else "s", s=80 if action == "SELF" else 35, facecolors=color if action == "SELF" else "white", edgecolors=color, zorder=6)
                    if action == "SELF":
                        ax.annotate(str(i), (xs[i], ys[i]), xytext=(5, 6), textcoords="offset points", color=color, fontsize=7)

        ax.set_title("Actual hybrid route (*: SELF, square: RECOVERY)")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="upper right")
        plt.tight_layout()
        output = FIGURE_DIR / output
        plt.savefig(output, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"saved: {output}")

    def plot_initial_plan(self, output="initial_plan.png"):
        # t=0の初期UE分布と，収集実行前に作成した全軌道を描画する．
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        output = FIGURE_DIR / Path(output).name
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


def compare_results(social_game, hybrid_game):
    """同一初期条件で実行した社会最適のみとハイブリッドの収集結果を比較する．"""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    social_history = {record["step"]: record for record in social_game.step_history}
    hybrid_history = {record["step"]: record for record in hybrid_game.step_history}

    # 途中で計画が終了した場合は，終了後の累積収集数を一定として比較する．
    social_total = 0
    hybrid_total = 0
    comparison_rows = []

    for step in range(1, OPERATION_MAX_STEPS + 1):
        social_record = social_history.get(step)
        hybrid_record = hybrid_history.get(step)
        social_collected = social_record["collected"] if social_record else 0
        hybrid_collected = hybrid_record["collected"] if hybrid_record else 0
        social_total += social_collected
        hybrid_total += hybrid_collected

        comparison_rows.append({
            "step": step,
            "social_collected": social_collected,
            "hybrid_collected": hybrid_collected,
            "social_cumulative": social_total,
            "hybrid_cumulative": hybrid_total,
            "difference": hybrid_total - social_total,
        })

    step_output = FIGURE_DIR / "comparison_steps.csv"
    with open(step_output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary_output = FIGURE_DIR / "comparison_summary.csv"
    summary = [
        {
            "mode": "social",
            "seed": SEED,
            "executed_steps": social_game.executed_steps,
            "collected": len(U) - len(social_game.remaining_true),
            "remaining": len(social_game.remaining_true),
            "selfish_count": 0,
            "recovery_steps": 0,
        },
        {
            "mode": "hybrid",
            "seed": SEED,
            "executed_steps": hybrid_game.executed_steps,
            "collected": len(U) - len(hybrid_game.remaining_true),
            "remaining": len(hybrid_game.remaining_true),
            "selfish_count": sum(uav.selfish_count for uav in hybrid_game.uavs),
            "recovery_steps": sum(uav.recovery_steps for uav in hybrid_game.uavs),
        },
    ]
    with open(summary_output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # 収集UE数の累積推移を同一グラフに表示する．
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = [row["step"] for row in comparison_rows]
    ax.plot(steps, [row["social_cumulative"] for row in comparison_rows], marker="o", markersize=3, label="Social only")
    ax.plot(steps, [row["hybrid_cumulative"] for row in comparison_rows], marker="s", markersize=3, label="Social + Selfish")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative collected UEs")
    ax.set_title("Data collection comparison (same initial conditions)")
    ax.set_xlim(1, OPERATION_MAX_STEPS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plot_output = FIGURE_DIR / "comparison_collected.png"
    plt.savefig(plot_output, dpi=220)
    plt.close()

    print()
    print("=" * 60)
    print("社会最適のみ vs 社会最適＋自己最適（同一初期UE配置・乱数シード）")
    print(f"乱数シード: {SEED}")
    for record in summary:
        print(f"{record['mode']}: 収集済UE={record['collected']}, 未収集UE={record['remaining']}, 実行ステップ={record['executed_steps']}")
    print(f"収集数の差（hybrid - social）: {summary[1]['collected'] - summary[0]['collected']}")
    print(f"自己最適回数: {summary[1]['selfish_count']}, 復帰ステップ数: {summary[1]['recovery_steps']}")
    print(f"saved: {step_output}")
    print(f"saved: {summary_output}")
    print(f"saved: {plot_output}")


class Tee:
    """ターミナルと結果ファイルの両方に同じ出力を書き込む．"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


if __name__ == "__main__":
    FIGURE_DIR.mkdir(parents=True, exist_ok=False)
    # 実行ログを結果ディレクトリに保存し，ターミナルにも同じ内容を表示する．
    with (FIGURE_DIR / "result.txt").open("w", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            try:
                print(f"結果保存先: {FIGURE_DIR}")
                print(f"乱数シード: {SEED}")
                print("社会最適のみと社会最適＋自己最適を，同じ初期条件で続けて実行します．")

                # 両方式で同じSEEDを使用し，初期UE配置・初期社会最適経路・CTCRWの乱数系列を揃える．
                # 2つのGameは別インスタンスとして生成し，収集済みUE集合やUAV位置を独立に管理する．
                social_game = Game("social")
                social_game.run()

                hybrid_game = Game("hybrid")
                for social_uav, hybrid_uav in zip(social_game.uavs, hybrid_game.uavs):
                    if social_uav.plan != hybrid_uav.plan:
                        raise RuntimeError(f"UAV{social_uav.id}: 2方式の初期社会最適経路が異なります．")
                hybrid_game.run()

                compare_results(social_game, hybrid_game)
            except Exception:
                # 実行途中のエラーも結果ファイルに記録する．
                traceback.print_exc()
                sys.exit(1)
