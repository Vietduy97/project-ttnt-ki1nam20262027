"""Student entry point. See README.md for the API and scoring rules."""

import heapq
import time
from sokoban import sokoban_goal_state


def solve(initial_state, timebound=120):

    # Rút các chỉ số ra cho dễ dùng
    width = initial_state.width
    height = initial_state.height
    robots = initial_state.robots
    boxes = initial_state.boxes
    storage = initial_state.storage
    obstacles = initial_state.obstacles

    # Hàm kiểm tra xem (x, y) có phải tường không
    def wall_checker(x, y):
        return (
            x < 0
            or y < 0
            or x >= width
            or y >= height
            or (x, y) in obstacles
        )

    # Hàm kiểm tra xem (x, y) có phải deadlock không
    def deadlock_checker(x, y):

        # Ô tường hoặc ô đích không tính là deadlock
        if wall_checker(x, y) or (x, y) in storage:
            return False

        # Kiểm tra 4 phía xung quanh
        wall_tren = wall_checker(x, y - 1)
        wall_duoi = wall_checker(x, y + 1)
        wall_trai = wall_checker(x - 1, y)
        wall_phai = wall_checker(x + 1, y)

        # TRƯỜNG HỢP 1: Deadlock góc (Bị chặn 1 chiều dọc VÀ 1 chiều ngang)
        blocked_doc = wall_tren or wall_duoi
        blocked_ngang = wall_trai or wall_phai

        if blocked_doc and blocked_ngang:
            return True

        # TRƯỜNG HỢP 2a: Mép tường biên trên/dưới ngoài cùng không có đích
        if (y == 0 and not any((nx, 0) in storage for nx in range(width))) or \
           (y == height - 1 and not any((nx, height - 1) in storage for nx in range(width))):
            return True

        # TRƯỜNG HỢP 2b: Mép tường biên trái/phải ngoài cùng không có đích
        if (x == 0 and not any((0, ny) in storage for ny in range(height))) or \
           (x == width - 1 and not any((width - 1, ny) in storage for ny in range(height))):
            return True

        return False

    # TẠO TẬP DEADLOCK Ở NGOÀI deadlock_checker()
    dead_locks = set()

    for i in range(height):
        for j in range(width):
            # j là x (cột), i là y (hàng)
            if deadlock_checker(j, i):
                dead_locks.add((j, i))

    # Hàm ước lượng khoảng cách
    def heuristics(state):
        total_distance = 0

        # 1. Khoảng cách từ các thùng đến storage gần nhất
        for bx, by in state.boxes:
            min_distance = 10**9

            for sx, sy in storage:
                d = abs(bx - sx) + abs(by - sy)

                if d < min_distance:
                    min_distance = d

            total_distance += min_distance

        # 2. Khoảng cách từ robot gần nhất đến thùng gần nhất
        min_robot_distance = 10**9

        for rx, ry in state.robots:
            for bx, by in state.boxes:
                d = abs(rx - bx) + abs(ry - by)

                if d < min_robot_distance:
                    min_robot_distance = d

        # Nếu robot chưa đứng sát cạnh thùng
        if min_robot_distance > 1:
            total_distance += min_robot_distance - 1

        return total_distance

    # Kiểm tra giới hạn thời gian
    if timebound <= 0:
        return False

    start = time.time()

    # Kiểm tra trạng thái ban đầu đã thắng chưa
    if sokoban_goal_state(initial_state):
        return initial_state

    count = 0
    frontier = []

    h_start = heuristics(initial_state)
    f_start = initial_state.gval + 2 * h_start

    # Weighted A*: f(n) = g(n) + 2 * h(n)
    heapq.heappush(frontier, (f_start, count, initial_state))

    visited = {
        initial_state.hashable_state(): initial_state.gval
    }

    while frontier:

        # Dừng nếu hết thời gian
        if time.time() - start > timebound - 0.5:
            return False

        f, counter, current_state = heapq.heappop(frontier)

        # Kiểm tra trạng thái thắng
        if sokoban_goal_state(current_state):
            return current_state

        current_key = current_state.hashable_state()

        # Bỏ qua trạng thái đã có đường đi tốt hơn
        if current_state.gval > visited.get(current_key, 10**9):
            continue

        # Sinh các trạng thái kế tiếp
        for next_state in current_state.successors():

            # Kiểm tra deadlock của các thùng
            is_deadlock = False

            for box in next_state.boxes:
                if box in dead_locks:
                    is_deadlock = True
                    break

            if is_deadlock:
                continue

            new_gval = next_state.gval
            state_key = next_state.hashable_state()

            # Chỉ cập nhật nếu tìm được đường đi tốt hơn
            if (
                state_key not in visited
                or new_gval < visited[state_key]
            ):
                visited[state_key] = new_gval

                h = heuristics(next_state)
                f = new_gval + 2 * h

                count += 1

                heapq.heappush(
                    frontier,
                    (f, count, next_state)
                )

    return False