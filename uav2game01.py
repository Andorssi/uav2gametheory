import random
from collections import defaultdict
from collections import deque
from math import hypot
import numpy as np
import os
import csv
import datetime
import copy 
import time
import itertools

# データ設定
U = list(range(200))     # UEインデックス
D = list(range(5))       # UAV番号
I = list(range(49))      # 領域番号
M = 5                    # UAVが同時に通信可能な最大UE数
width = 7; height = 7    # 格子領域(7x7)
T_MAX = 30               # 規定ステップ
move_k = 2               # UAV移動可能マス
SERVER = 1               # サーバ台数
OUTSIDE_MODE = "selfish" # サーバ範囲外のUAVの挙動方針

# サーバ設定
B = 24                   # BS
RANGE_OF_OBSERVE = 2     # 観測可能範囲

# パラメータ
alpha = 0.1
beta = 0.3
gamma = 0.3


"""
add_client: UEを追加する
true_remaining_clients: 実際に残っているUE数
"""
class Area:
    def __init__(self, area_id):
        self.id = area_id
        self.clients = set()
        self.neighbors = []
    
    def add_client(self, u):
        self.clients.add(u)
    
    def true_remaining_clients(self, remaining_true):
        return self.clients & remaining_true


"""
move: 次の領域に移動する
own_optimize: 自己最適の目的関数
choose_selfish_area: selfishモードでの領域選択候補
choose_return_area: returnモードでの領域選択候補
"""
class UAV:
    def __init__(self, uav_id, start):
        self.id = uav_id
        self.area = start
    
    def move(self, next_area):
        self.area = next_area
    
    def own_optimize(self, candidate_area, game):
        C = len(game.areas[candidate_area].true_remaining_clients(game.remaining_true))
        L = game.field_distance(self.area, candidate_area)
        S = 1 if candidate_area in game.server.observable_area else 0
        return C - alpha*L + gamma*S
    
    def choose_selfish_area(self, game):
        candidates = game.get_k_rinsetu(self.area, move_k)
        return max( candidates, key=lambda area: self.own_optimize(area, game) )

    def choose_return_area(self, game):
        candidates = game.get_k_rinsetu(self.area, move_k)
        return min( candidates, key=lambda area: game.field_distance(area, game.server.area) )

"""
is_observable: 領域areaがサーバ観測範囲内か判定
clients_in_area: サーバが認識する領域areaの残存UE数(集合)
update_belief: サーバ範囲内で収集されたUEだけサーバ認識から削除
social_score: 社会最適の目的関数
social_optimize: サーバ内UAVの配置決定
"""
class Server:
    def __init__(self, area, observable_area, remaining_clients):
        self.area = area
        self.observable_area = observable_area
        self.remaining_clients = set(remaining_clients)

    def is_observable(self, area):
        return area in self.observable_area

    def clients_in_area(self, area):
        return area.clients & self.remaining_clients

    def update_belief(self, collected_users, collected_area):
        if self.is_observable(collected_area):
            self.remaining_clients -= collected_users

    def social_score(self, assignment, game):
        covered = set()
        total_distance = 0
        area_count = defaultdict(int)
        server_bonus = 0

        for uav, area in assignment.items():
            covered |= self.clients_in_area(game.areas[area])
            total_distance += game.field_distance(uav.area, area)
            area_count[area] += 1

            if self.is_observable(area):
                server_bonus += 1

        overlap = sum(max(0, count - 1) for count in area_count.values())

        return len(covered) - alpha*total_distance - beta*overlap + gamma*server_bonus

    def social_optimize(self, uavs, game):
        if not uavs:
            return {}

        candidate_lists = []

        for uav in uavs:
            candidates = game.get_k_rinsetu(uav.area, move_k)
            candidate_lists.append(candidates)

        best_assignment = None
        best_score = -10**18

        for combo in itertools.product(*candidate_lists):
            assignment = {
                uav: area
                for uav, area in zip(uavs, combo)
            }

            score = self.social_score(assignment, game)

            if score > best_score:
                best_score = score
                best_assignment = assignment

        return best_assignment


"""
set_rinsetu: 各領域の隣接領域を格納
get_rinsetu: 隣接領域を取得
get_k_rinsetu: kマス以内の領域を取得
field_distance: 六角領域上の最短距離計算
load_client_file: UE配置ファイルを読み込む
set_clients: UEをランダムに配置
obserable_uavs: UAVがサーバ観測範囲内・外にいるか
decide_placement: UAVの移動先を決定
move_uavs: 移動
collect_clients: 通信
step: 1ステップ
run: 実行
print_step: 標準出力
"""
class Game:
    def __init__(self):
        self.areas = {i: Area(i) for i in I}
        self.set_rinsetu()
        self.set_clients()

        self.uavs = [UAV(uav_id=d, start=B) for d in D]
        self.remaining_true = set(U)
        observable_area = set(self.get_k_rinsetu(B, RANGE_OF_OBSERVE))
        self.server = Server(area=B, observable_area=observable_area, remaining_clients=self.remaining_true)

    def set_rinsetu(self):
        for area_id in self.areas:
            self.areas[area_id].neighbors = self.get_rinsetu(area_id)

    def get_rinsetu(self, i):
        x = i // width
        y = i % width

        if y%2 == 0:
            directions = [(-1,-1), (0,-1), (-1,0), (1,0), (-1,1), (0,1)]
        else:
            directions = [(0,-1), (1,-1), (-1,0), (1,0), (0,1), (1,1)]
        neighbor = []
        for dr, dc in directions:
            r, c = x+dr, y+dc
            if 0<=r<height and 0<=c<width:
                neighbor.append(r*width + c)
        return neighbor

    def get_k_rinsetu(self, i, k):
        if k < 0:
            raise ValueError("k must be >= 0")

        visited = {i}
        queue = deque([(i, 0)])

        while queue:
            current, dist = queue.popleft()

            # これ以上広げない
            if dist == k:
                continue

            for nb in self.get_rinsetu(current):
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
            self.areas[area_id].add_client(client_id)

    def observable_uavs(self):
        inside = []
        outside = []

        for uav in self.uavs:
            if self.server.is_observable(uav.area):
                inside.append(uav)
            else:
                outside.append(uav)

        return inside, outside

    def decide_placement(self):
        inside_uavs, outside_uavs = self.observable_uavs()

        placement = {}

        # サーバ範囲内UAV：社会最適
        placement.update(self.server.social_optimize(inside_uavs, self))

        # サーバ範囲外UAV：自己最適 or 帰還
        for uav in outside_uavs:
            if OUTSIDE_MODE == "selfish":
                placement[uav] = uav.choose_selfish_area(self)
            elif OUTSIDE_MODE == "return":
                placement[uav] = uav.choose_return_area(self)
            else:
                raise ValueError("outside_mode must be 'selfish' or 'return'")

        return placement, inside_uavs, outside_uavs

    def move_uavs(self, placement):
        for uav, area in placement.items():
            uav.move(area)

    def collect_clients(self, placement):
        collected_all = set()
        collected_each = {}

        # 同じ領域に複数UAVが来た場合，UAV ID順に収集する
        for uav in sorted(placement.keys(), key=lambda x: x.id):
            area_id = uav.area

            candidates = sorted(self.areas[area_id].true_remaining_clients(self.remaining_true))

            collected = set(candidates[:M])
            collected_each[uav.id] = collected

            # 真の状態を更新
            self.remaining_true -= collected

            # サーバの認識を更新
            self.server.update_belief(collected_users=collected, collected_area=area_id)

            collected_all |= collected

        return collected_all, collected_each

    def step(self, t):
        placement, inside_uavs, outside_uavs = self.decide_placement()

        self.move_uavs(placement)

        collected_all, collected_each = self.collect_clients(placement)

        stale = len(self.server.remaining_clients - self.remaining_true)

        self.print_step(t=t, placement=placement, inside_uavs=inside_uavs, outside_uavs=outside_uavs, collected_all=collected_all, stale=stale)

    def run(self):
        print("server_area:", sorted(self.server.observable_area))
        print("outside_mode:", OUTSIDE_MODE)

        for t in range(T_MAX):
            self.step(t)

            if len(self.remaining_true) == 0:
                break

    def print_step(self, t, placement, inside_uavs, outside_uavs, collected_all, stale):
        print()
        print(f"t = {t}")
        print("D_in :", [uav.id for uav in inside_uavs])
        print("D_out:", [uav.id for uav in outside_uavs])

        print("placement:")
        for uav in self.uavs:
            print(f"  UAV{uav.id} -> area {uav.area}")

        print("収集済UE数:", len(collected_all))
        print("未収集UE数:", len(self.remaining_true))
        print("サーバ上での未収集UE数:", len(self.server.remaining_clients))
        print("サーバ誤認差:", stale)



if __name__ == "__main__":
    game = Game()
    game.run()