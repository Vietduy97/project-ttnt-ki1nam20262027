"""Student entry point. See README.md for the API and scoring rules."""


# Import đáp án và các thư viện cần sử dụng
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

# Hàm kiểm tra xem (x,y) có phải tường không?
    def wall_checker(x, y):
        return (x < 0 or y < 0 or x >= width or y >= height or (x, y) in obstacles)

# Hàm kiểm tra xem (x,y) co phai deadlock không?
    def deadlock_checker(x, y):
        if wall_checker(x, y) or (x, y) in storage:
            return False
        
        blocked_doc = wall_checker(x, y - 1) or wall_checker(x, y + 1)
        blocked_ngang = wall_checker(x - 1, y) or wall_checker(x + 1, y)

        return (blocked_doc and blocked_ngang)

    dead_locks = set()

    for i in range(height):
        for j in range(width):
            # j là x (cột), i là y (hàng)
            if deadlock_checker(j, i):
                dead_locks.add((j, i))

# hàm ước lượng khoảng cách nhỏ nhất
    def heuristics(state):
        total_distance = 0
        # 1. Khoảng cách từ các thùng đến đích gần nhất:
        for bx, by in state.boxes:
            min_distance = 10**9
            for sx, sy in storage:
                d = abs(bx - sx) + abs(by - sy)
                if d < min_distance:
                    min_distance = d
            total_distance += min_distance

        # 2. Khoảng cách từ Robot gần nhất đến Thùng gần nhất:
        min_robot_distance = 10**9
        for rx, ry in state.robots:
            for bx, by in state.boxes:
                d = abs(rx - bx) + abs(ry - by)
                if d < min_robot_distance:
                    min_robot_distance = d
        
        # Nếu robot chưa đứng sát cạnh thùng, cộng thêm khoảng cách để kéo robot lại gần:
        if min_robot_distance > 1:
            total_distance += (min_robot_distance - 1)

        return total_distance

    if timebound <= 0:
        return False
    start = time.time()

    # Kiểm tra thắng sẵn từ đầu: phải truyền (initial_state) và return initial_state
    if sokoban_goal_state(initial_state):
        return initial_state

    count = 0
    frontier = []

    h_start = heuristics(initial_state)
    heapq.heappush(frontier, (h_start, count, initial_state))

    visited = {initial_state.hashable_state(): 0}

    while frontier:
        if time.time() - start > timebound - 0.5:
            return False
        f, counter, current_state = heapq.heappop(frontier)

        # Kiểm tra đã đến đích chưa
        if sokoban_goal_state(current_state):
            return current_state

        if current_state.gval > visited.get(current_state.hashable_state(), 10**9):
            continue

        for next_state in current_state.successors():
            is_deadlock = False
            for box in next_state.boxes:
                if box in dead_locks:
                    is_deadlock = True
                    break
            
            # Thụt lề vào bên trong vòng lặp for next_state:
            if is_deadlock == True:
                continue

            new_gval = next_state.gval
            state_key = next_state.hashable_state()

            if state_key not in visited or new_gval < visited[state_key]:
                visited[state_key] = new_gval
                h = heuristics(next_state)
                f = new_gval + 2 * h  # Weighted A* với trọng số w = 2
                count += 1
                heapq.heappush(frontier, (f, count, next_state))
    
    return False