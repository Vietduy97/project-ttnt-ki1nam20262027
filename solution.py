"""
Thuật toán giải Sokoban Đa Robot bằng Anytime Weighted A*
Triển khai độc lập cho bài tập lớn Trí tuệ Nhân tạo (BTL AI TTNT).
Tuân thủ 100% quy chuẩn đề bài: Chỉ dùng thư viện chuẩn Python, không sửa framework.
"""

from collections import deque
from heapq import heapify, heappop, heappush
from math import inf
from time import perf_counter

from sokoban import SokobanState, sokoban_goal_state


# =============================================================================
# 1. CẤU TRÚC DỮ LIỆU NODE TÌM KIẾM & KHỬ ĐỐI XỨNG ROBOT
# =============================================================================

class _Node:
    """
    Node tìm kiếm chuẩn theo lý thuyết môn AI:
    Lưu vị trí robot, vị trí thùng, chi phí thực tế g, con trỏ parent,
    và bước di chuyển move để lần ngược tìm lại đường đi khi về đích.
    """
    def __init__(self, robots, boxes, g, parent=None, move=None):
        self.robots = robots      # tuple các tọa độ robot dạng số nguyên 1D
        self.boxes = boxes        # frozenset các tọa độ thùng dạng số nguyên 1D
        self.g = g                # Tổng số bước đi thực tế từ trạng thái đầu (g-value)
        self.parent = parent      # Trỏ về node cha để truy vết đường đi
        self.move = move          # Tuple lưu bước di chuyển: (chỉ_số_robot, hướng_đi)

        # ---------------------------------------------------------------------
        # KHỬ ĐỐI XỨNG HOÁN VỊ (ROBOT SYMMETRY REDUCTION):
        # Mọi robot đều có chức năng như nhau và không phân biệt vai trò:
        # [Robot 0 ở 5, Robot 1 ở 9] hoàn toàn tương đương với [Robot 0 ở 9, Robot 1 ở 5].
        # Sắp xếp tuple(sorted(robots)) làm key tra cứu giúp thu nhỏ không gian tìm kiếm
        # đi K! lần (ví dụ: 3 robot giảm 6 lần, 4 robot giảm 24 lần).
        # ---------------------------------------------------------------------
        self.key = (tuple(sorted(robots)), boxes)


class _Timeout(Exception):
    """Ngoại lệ kích hoạt khi hết thời gian chạy cho phép (timebound)."""
    pass


# =============================================================================
# 2. THUẬT TOÁN GÁN CẶP HUNGARY CHỮ NHẬT (HUNGARIAN ASSIGNMENT)
# =============================================================================

def _matching(cost, check_time):
    """
    Thuật toán Hungary dạng chữ nhật giải bài toán gán cặp cực tiểu O(N^2 * M)
    với N thùng (boxes) <= M ô đích (goals).
    
    ƯU ĐIỂM:
    - Chạy trong thời gian đa thức, hoàn toàn không bị bùng nổ như Bitmask DP O(N * 2^M).
    - Tự động bỏ qua các trường hợp không thể gán (cạnh có chi phí bằng inf).
    """
    n = len(cost)
    if not n:
        return 0
    m = len(cost[0])
    if n > m:
        return inf

    u, v = [0] * (n + 1), [0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)

    for i in range(1, n + 1):
        check_time()
        p[0] = i
        j0 = 0
        minimum = [inf] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            if delta == inf:
                return inf
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return -v[0]


# =============================================================================
# 3. HÀM CHÍNH: SOLVER TÌM ĐƯỜNG ĐI CHO SOKOBAN
# =============================================================================

def solve(initial_state, timebound=120):
    """
    Hàm giải chính đáp ứng chuẩn đề bài.
    Trả về trạng thái đích cuối cùng (kèm chuỗi parent/action/gval hợp lệ),
    hoặc trả về False nếu không tìm được đường đi trong thời gian cho phép.
    """
    started = perf_counter()
    if timebound <= 0:
        return False
    if sokoban_goal_state(initial_state):
        return initial_state

    # Dành ra một khoảng thời gian dự phòng (tối đa 1.5s) để tái tạo chuỗi SokobanState
    reserve = min(1.5, timebound * 0.15)
    deadline = started + timebound - reserve

    def check_time():
        """Ngắt thuật toán an toàn nếu chạm ngưỡng thời gian quy định."""
        if perf_counter() >= deadline:
            raise _Timeout

    width, height = initial_state.width, initial_state.height
    directions = ((0, -1), (1, 0), (0, 1), (-1, 0))  # 4 hướng: Lên, Phải, Xuống, Trái
    names = ("up", "right", "down", "left")
    incumbent = None  # Lời giải tốt nhất hiện tại (chứa node đích có g nhỏ nhất)

    # -------------------------------------------------------------------------
    # BẢNG GHI NHỚ (MEMOIZATION) DÙNG DICTIONARY CƠ BẢN:
    # Trong Sokoban, robot thường phải đi bộ nhiều bước qua phòng trống để đến chỗ thùng.
    # Trong các bước đi bộ đó, vị trí các thùng (boxes) HOÀN TOÀN KHÔNG ĐỔI.
    # Việc ghi nhớ kết quả giúp tránh tính lại Deadlock và Hungary hàng chục nghìn lần.
    # -------------------------------------------------------------------------
    deadlock_memo = {}   # Lưu kết quả kiểm tra bế tắc của cấu hình thùng: {boxes: True/False}
    box_h_memo = {}      # Lưu chi phí đẩy Hungary của cấu hình thùng: {boxes: chi_phi}
    approach_memo = {}   # Lưu khoảng cách tiếp cận của robot tới cấu hình thùng: {boxes: dist_dict}

    # =========================================================================
    # 4. TIỀN TÍNH TOÁN (PRECOMPUTATION) TRÊN MẶT BÀN 1D
    # =========================================================================
    def precompute():
        """
        Chuyển đổi toàn bộ tọa độ (x, y) sang chỉ số 1 chiều: cell = y * width + x.
        Tính trước bảng láng giềng, khoảng cách đẩy lùi (Reverse-Push BFS) và bẫy tường biên.
        """
        floor = set()
        for y in range(height):
            check_time()
            for x in range(width):
                if (x, y) not in initial_state.obstacles:
                    floor.add(y * width + x)

        goals = frozenset(y * width + x for x, y in initial_state.storage)

        # Bảng láng giềng 4 hướng cho từng ô sàn (-1 nếu là tường hoặc vượt biên)
        neighbors = {}
        for cell in floor:
            check_time()
            x, y = cell % width, cell // width
            neighbors[cell] = tuple(
                (y + dy) * width + x + dx
                if 0 <= x + dx < width and 0 <= y + dy < height
                and (y + dy) * width + x + dx in floor else -1
                for dx, dy in directions
            )

        # REVERSE-PUSH BFS TỪ TỪNG Ô ĐÍCH:
        # Ta lần ngược lại động tác đẩy: Nếu thùng đang ở cell, nó có thể được đẩy từ
        # ô previous sang cell nếu previous là sàn và ô đằng sau previous (nơi robot đứng) hợp lệ.
        # Từ đó tính được chi phí đẩy tối thiểu chính xác từ mọi ô sàn về từng ô đích.
        reverse_distances = []
        for goal in sorted(goals):
            check_time()
            dist, queue = {goal: 0}, deque([goal])
            while queue:
                check_time()
                cell = queue.popleft()
                for d in range(4):
                    previous = neighbors[cell][d]
                    # previous là vị trí trước đó của thùng, neighbors[previous][d] là nơi robot đứng đẩy
                    if previous != -1 and neighbors[previous][d] != -1:
                        if previous not in dist:
                            dist[previous] = dist[cell] + 1
                            queue.append(previous)
            reverse_distances.append(dist)

        # Ma trận chi phí đẩy từ mỗi ô đến từng ô đích
        costs = {}
        for cell in floor:
            check_time()
            costs[cell] = tuple(dist.get(cell, inf) for dist in reverse_distances)

        # Ô CHẾT TĨNH (DEAD SQUARES): Ô sàn mà thùng không bao giờ có thể về bất kỳ đích nào
        dead = frozenset(cell for cell in floor if all(v == inf for v in costs[cell]))

        # BẪY TƯỜNG BIÊN CÓ ĐẦU CỤT (CAPPED WALL LINES):
        # Dải tường thẳng bị chặn 2 đầu. Nếu số thùng đẩy vào > số đích trên dải -> Bế tắc!
        wall_lines = set()
        for wall_side in range(4):
            check_time()
            along = 1 if wall_side % 2 == 0 else 2
            opposite = (along + 2) % 4
            candidates = {c for c in floor if neighbors[c][wall_side] == -1}
            while candidates:
                check_time()
                seed = candidates.pop()
                strip = {seed}
                for d in (along, opposite):
                    c = neighbors[seed][d]
                    while c in candidates:
                        candidates.remove(c)
                        strip.add(c)
                        c = neighbors[c][d]
                # Chỉ coi là dải bẫy nếu dải này bị khóa kín cả 2 đầu
                if all(neighbors[c][d] == -1 or neighbors[c][d] in strip
                       for c in strip for d in (along, opposite)):
                    wall_lines.add(frozenset(strip))

        return goals, neighbors, costs, dead, tuple(
            (line, len(line & goals)) for line in wall_lines)

    # =========================================================================
    # 5. CÁC TẦNG NHẬN DIỆN DEADLOCK THÔNG MINH
    # =========================================================================
    def has_deadlock(boxes):
        """
        Kiểm tra trạng thái bế tắc (deadlock) của tập hợp thùng.
        Dùng bảng ghi nhớ deadlock_memo để lấy ngay kết quả nếu đã kiểm tra cấu hình này.
        """
        if boxes in deadlock_memo:
            return deadlock_memo[boxes]

        # Tầng 1: Có thùng nằm trên ô chết vĩnh viễn (dead square)
        if boxes & dead_squares:
            deadlock_memo[boxes] = True
            return True

        # Tầng 2: Tràn số lượng thùng trên dải tường biên cụt
        for line, capacity in wall_lines:
            if len(boxes & line) > capacity:
                deadlock_memo[boxes] = True
                return True

        # Tầng 3: Khối 2x2 gồm toàn tường hoặc thùng (không thể tách rời)
        for b in boxes - goals:
            x, y = b % width, b // width
            for left in (x - 1, x):
                for top in (y - 1, y):
                    if all(not (0 <= xx < width and 0 <= yy < height)
                           or yy * width + xx not in neighbors
                           or yy * width + xx in boxes
                           for xx, yy in ((left, top), (left + 1, top),
                                          (left, top + 1), (left + 1, top + 1))):
                        deadlock_memo[boxes] = True
                        return True

        # Tầng 4: KIỂM TRA ĐÔNG CỨNG BẢO TOÀN (CONSERVATIVE FREEZE DEADLOCK):
        # Thuật toán bóc tách dần các thùng còn trục tự do (cả ô đón và ô đẩy đều trống).
        # Nếu sau khi bóc tách mà vẫn còn thùng chưa về đích bị kẹt lại -> Deadlock vĩnh viễn!
        frozen = set(boxes)
        changed = True
        while changed:
            changed = False
            for b in tuple(frozen):
                ns = neighbors[b]
                # Kiểm tra 2 trục: Trục dọc (0, 2) hoặc Trục ngang (1, 3)
                if any(all(ns[d] != -1 and ns[d] not in frozen for d in axis)
                       for axis in ((0, 2), (1, 3))):
                    frozen.remove(b)
                    changed = True

        is_dead = bool(frozen - goals)
        deadlock_memo[boxes] = is_dead
        return is_dead

    # =========================================================================
    # 6. HEURISTIC KẾT HỢP: ĐẨY THÙNG TỐI ƯU + TIẾP CẬN CỦA ROBOT
    # =========================================================================
    def box_heuristic(boxes):
        """Tính tổng chi phí đẩy thùng tối ưu bằng thuật toán Hungary (kèm ghi nhớ)."""
        if boxes in box_h_memo:
            return box_h_memo[boxes]

        if has_deadlock(boxes):
            result = inf
        else:
            result = _matching([costs[b] for b in sorted(boxes)], check_time)

        box_h_memo[boxes] = result
        return result

    def heuristic(robots, boxes):
        """
        Heuristic hợp lệ và chặt chẽ (Admissible):
        h = (Tổng chi phí đẩy thùng theo Hungary) + (Khoảng cách ngắn nhất từ robot đến vị trí đẩy)
        """
        push_cost = box_heuristic(boxes)
        if push_cost == inf or push_cost == 0:
            return push_cost

        # Nếu cấu hình thùng này chưa được tính khoảng cách tiếp cận: Chạy BFS đa nguồn
        if boxes not in approach_memo:
            dist, queue = {}, deque()
            for b in boxes:
                ns = neighbors[b]
                for d in range(4):
                    target, support = ns[d], ns[(d + 2) % 4]
                    if (target != -1 and support != -1
                            and target not in boxes and support not in boxes
                            and target not in dead_squares):
                        if support not in dist:
                            dist[support] = 0
                            queue.append(support)
            while queue:
                check_time()
                cell = queue.popleft()
                for nxt in neighbors[cell]:
                    if nxt != -1 and nxt not in boxes and nxt not in dist:
                        dist[nxt] = dist[cell] + 1
                        queue.append(nxt)
            approach_memo[boxes] = dist

        approach = approach_memo[boxes]
        # Khoảng cách từ robot gần nhất đến một vị trí đẩy khả thi
        return push_cost + min((approach.get(r, inf) for r in robots), default=inf)

    # =========================================================================
    # 7. CHIẾN LƯỢC ANYTIME WEIGHTED A* (SUY GIẢM TRỌNG SỐ)
    # =========================================================================
    def weighted_astar(root):
        """
        Tìm kiếm với trọng số w suy giảm dần: w = 5.0 -> 3.0 -> 2.0 -> 1.5 -> 1.2 -> 1.0.
        Pha 1 (w = 5.0) tìm ra lời giải đầu tiên siêu nhanh (< 0.1s), lập kỷ lục cận trên upper_bound.
        Các pha sau tái sử dụng tập OPEN và best_g để rút ngắn số bước đi về tối ưu.
        """
        nonlocal incumbent
        weights = (5.0, 3.0, 2.0, 1.5, 1.2, 1.0)
        phase, serial = 0, 0

        root_h = heuristic(root.robots, root.boxes)
        if root_h == inf:
            return

        open_heap = [(weights[phase] * root_h, root_h, serial, root)]
        best_g = {root.key: 0}
        upper_bound = inf

        while open_heap:
            check_time()
            _, h, _, node = heappop(open_heap)

            # Cắt tỉa: Nếu đã có đường đi tốt hơn tới trạng thái này hoặc vượt quá cận trên
            if node.g != best_g.get(node.key) or node.g + h >= upper_bound:
                continue

            improved = False
            occupied = set(node.robots)

            # Mở rộng trạng thái cho từng robot di chuyển sang 4 hướng
            for robot, position in enumerate(node.robots):
                for d, target in enumerate(neighbors[position]):
                    if target == -1 or target in occupied:
                        continue

                    boxes = node.boxes
                    # Nếu ô đích đến có thùng -> Hành động ĐẨY THÙNG
                    if target in boxes:
                        beyond = neighbors[target][d]
                        if (beyond == -1 or beyond in boxes or beyond in occupied
                                or beyond in dead_squares):
                            continue
                        boxes = (boxes - {target}) | {beyond}

                    # Cập nhật vị trí robot
                    robots = node.robots[:robot] + (target,) + node.robots[robot + 1:]
                    g = node.g + 1

                    if g >= upper_bound:
                        continue

                    key = (tuple(sorted(robots)), boxes)
                    if g >= best_g.get(key, inf):
                        continue

                    h_next = heuristic(robots, boxes)
                    if g + h_next >= upper_bound:
                        continue

                    child = _Node(robots, boxes, g, node, (robot, d))
                    best_g[key] = g

                    # Nếu tất cả thùng đã nằm trong các ô đích
                    if boxes <= goals:
                        incumbent, upper_bound = child, g
                        improved = True
                    else:
                        serial += 1
                        heappush(open_heap, (g + weights[phase] * h_next,
                                             h_next, serial, child))

            # Khi tìm thấy lời giải tốt hơn: Chuyển sang trọng số nhỏ hơn để tối ưu độ dài
            if improved:
                phase = min(phase + 1, len(weights) - 1)
                next_heap = []
                for _, hh, tie, n in open_heap:
                    check_time()
                    if n.g == best_g.get(n.key) and n.g + hh < upper_bound:
                        next_heap.append((n.g + weights[phase] * hh, hh, tie, n))
                open_heap = next_heap
                heapify(open_heap)

    # =========================================================================
    # 8. KHỞI CHẠY VÀ XÂY DỰNG CHUỖI KẾT QUẢ SOKOBANSTATE HỢP LỆ
    # =========================================================================
    try:
        check_time()
        # Kiểm tra sơ bộ: Số thùng nhiều hơn số đích hoặc không có robot
        if len(initial_state.boxes) > len(initial_state.storage) or not initial_state.robots:
            return False

        goals, neighbors, costs, dead_squares, wall_lines = precompute()
        robots = tuple(y * width + x for x, y in initial_state.robots)
        boxes = frozenset(y * width + x for x, y in initial_state.boxes)

        weighted_astar(_Node(robots, boxes, 0))
    except _Timeout:
        pass

    if incumbent is None:
        return False

    # Lần ngược từ node đích về node gốc để thu thập các bước đi
    moves = []
    node = incumbent
    while node.parent is not None:
        moves.append(node.move)
        node = node.parent

    # Tái tạo lại chuỗi SokobanState chính quy của đề bài
    state = initial_state
    for robot, d in reversed(moves):
        dx, dy = directions[d]
        robots = list(state.robots)
        x, y = robots[robot]
        target = (x + dx, y + dy)
        boxes = state.boxes
        if target in boxes:
            boxes = (boxes - {target}) | {(target[0] + dx, target[1] + dy)}
        robots[robot] = target
        state = SokobanState(
            action=str(robot) + " " + names[d],
            gval=state.gval + 1,
            parent=state,
            width=width,
            height=height,
            robots=tuple(robots),
            boxes=boxes,
            storage=initial_state.storage,
            obstacles=initial_state.obstacles
        )

    return state