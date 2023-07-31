from collections import defaultdict, deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import auto, Enum, IntFlag
from functools import wraps
from queue import PriorityQueue
from typing import Any, Callable, TypeAlias


MazeType: TypeAlias = "list[list[Direction]]"
HeuristicType: TypeAlias = Callable[
    [tuple[float, float], tuple[float, float]],
    float
]

MazePosition = namedtuple('MazePosition', ['x', 'y'])


class Direction(IntFlag):
    DOWN = 0b0100
    LEFT = 0b1000
    NONE = 0b0000
    RIGHT = 0b0010
    UP = 0b0001


class MoveList(Enum):
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()


class MazeSymbols:
    H_WALL = '--'
    V_WALL = '|'


def get_available_directions(maze: MazeType,
                             position: MazePosition) -> list[MoveList]:
    directions = maze[position.x][position.y]

    return (
        [move for direction, move in {
            Direction.DOWN: MoveList.DOWN,
            Direction.LEFT: MoveList.LEFT,
            Direction.RIGHT: MoveList.RIGHT,
            Direction.UP: MoveList.UP,
        }.items()
            if directions & direction
        ]
    )


# MoveList != Direction, cannot use it for q2
def get_move_as_str(move: MoveList) -> str:
    if move == MoveList.DOWN:
        return 'D'
    elif move == MoveList.LEFT:
        return 'L'
    elif move == MoveList.RIGHT:
        return 'R'
    elif move == MoveList.UP:
        return 'U'
    else:
        raise ValueError(f'Invalid move encountered! {move=!r}')


class Search:
    class Heuristics:
        @staticmethod
        def euclidean(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
            return sum(((pos1[0] - pos2[0]) ** 2, (pos1[1] - pos2[1]) ** 2)) ** 0.5

        @staticmethod
        def manhattan(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
            return sum((abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])))

    @staticmethod
    def _compute_next_position(pos, move):
        if move == MoveList.DOWN:
            return MazePosition(pos.x + 1, pos.y)
        elif move == MoveList.LEFT:
            return MazePosition(pos.x, pos.y - 1)
        elif move == MoveList.RIGHT:
            return MazePosition(pos.x, pos.y + 1)
        elif move == MoveList.UP:
            return MazePosition(pos.x - 1, pos.y)
        else:
            raise ValueError(f'Invalid move encountered! {move=!r} ({move})')

    @staticmethod
    def A_search(maze: MazeType,
                 start_position: MazePosition,
                 end_position: MazePosition,
                 *args,
                 return_visited: bool = False,
                 **kwargs) -> list[MoveList] | tuple[list[MoveList], set[MazePosition]]:

        @dataclass(order=True)
        class PriortizedPosition:
            score: float
            position: Any = field(compare=False)
            path: list = field(default_factory=list, compare=False)

            def __eq__(self, other):
                return self.position == other.position

        heuristic: HeuristicType = kwargs.get(
            'heuristic',
            Search.Heuristics.euclidean
        )

        g_score: dict[MazePosition, float] = defaultdict(lambda: float('inf'))
        g_score[start_position] = 0

        queue: PriorityQueue = PriorityQueue(maxsize=len(maze) * len(maze[0]))
        queue.put(
            PriortizedPosition(
                heuristic(start_position, end_position),
                start_position,
                []
            )
        )

        if return_visited:
            visited: set = set()

        while not queue.empty():
            pos: PriortizedPosition = queue.get()

            if return_visited and pos.position not in visited:
                visited.add(pos.position)

            if pos.position == end_position:
                return (pos.path, visited) if return_visited else pos.path

            for move in get_available_directions(maze, pos.position):
                # we add one, because distances to the next spot is 1
                # in all directions, and we only choose neighbors that are
                # available to go to, so blocked neighbors are not considered
                tentative_g_score = g_score[pos.position] + 1.

                new_pos = Search._compute_next_position(pos.position, move)
                if tentative_g_score >= g_score[new_pos]:
                    continue

                g_score[new_pos] = tentative_g_score
                f_score = tentative_g_score + heuristic(new_pos, end_position)
                next_ = PriortizedPosition(f_score, new_pos, pos.path + [move])
                if next_ not in queue.queue:
                    queue.put(next_)

        return ([], visited) if return_visited else []

    @staticmethod
    def BFS(maze: MazeType,
            start_position: MazePosition,
            end_position: MazePosition,
            *args,
            return_visited: bool = False,
            **kwargs) -> list[MoveList] | tuple[list[MoveList], set[MazePosition]]:
        queue: deque[tuple[MazePosition, list[MoveList]]
                     ] = deque([(start_position, [])])
        visited = set()

        while queue:
            pos, path = queue.popleft()
            if pos in visited:
                continue
            visited.add(pos)

            if pos == end_position:
                return (path, visited) if return_visited else path

            for move in get_available_directions(maze, pos):
                new_pos = Search._compute_next_position(pos, move)

                queue.append((new_pos, path + [move]))

        return ([], visited) if return_visited else []

    @ staticmethod
    def DFS(maze: MazeType,
            start_position: MazePosition,
            end_position: MazePosition,
            *args,
            return_visited: bool = False,
            **kwargs) -> list[MoveList] | tuple[list[MoveList], set[MazePosition]]:
        stack: list[tuple[MazePosition, list[MoveList]]] = [
            (start_position, [])]

        visited: set[MazePosition] = set()

        while stack:
            pos, path = stack.pop()

            if pos in visited:
                continue

            visited.add(pos)

            if pos == end_position:
                # type: ignore[return-value]
                return (path, visited) if return_visited else path

            for move in get_available_directions(maze, pos):
                new_pos = Search._compute_next_position(pos, move)
                if not all((0 <= new_pos.x < len(maze), 0 <= new_pos.y < len(maze[0]))):
                    continue
                stack.append((new_pos, path + [move]))

        # stack is an empty list, rather return that than create new list
        # type: ignore[return-value]
        return ([], visited) if return_visited else []

    algorithms = {
        'A*': A_search,
        'bfs': BFS,
        'dfs': DFS,
    }

    @ staticmethod
    def solve_maze(maze: MazeType,
                   start_position: MazePosition,
                   end_position: MazePosition,
                   algorithm: str = 'A*',
                   heuristic: HeuristicType | None = None,
                   return_visited: bool = False
                   ) -> list[MoveList] | tuple[list[MoveList], set[MazePosition]]:
        func = Search.algorithms.get(algorithm)

        if func is None:
            raise ValueError(f'{algorithm} is an unknown algorithm.')

        return func(
            maze,
            start_position,
            end_position,
            heuristic=heuristic,
            return_visited=return_visited
        )


def parse_line(args: tuple[int, list[str]]) -> list[int]:
    line_idx, maze = args
    maze_line: list[int] = []

    # We use len(...) - 1 because there is one more character used
    # to completely draw borders
    for i in range(0, len(maze[line_idx]) - 1, 3):
        bounds = Direction.NONE

        if maze[line_idx][i] == MazeSymbols.V_WALL:
            bounds = Direction.LEFT

        if maze[line_idx][i + 3] == MazeSymbols.V_WALL:
            bounds = (bounds | Direction.RIGHT) if bounds else Direction.RIGHT

        if maze[line_idx - 1][i + 1:i + 3] == MazeSymbols.H_WALL:
            bounds = (bounds | Direction.UP) if bounds else Direction.UP

        if maze[line_idx + 1][i + 1:i + 3] == MazeSymbols.H_WALL:
            bounds = (bounds | Direction.DOWN) if bounds else Direction.DOWN

        maze_line.append(~bounds)

    return maze_line


def parse_maze(maze: list[str]):
    relevant_maze = map(lambda x: (x, maze), range(1, len(maze), 2))
    with ThreadPoolExecutor() as executor:
        return [_ for _ in executor.map(parse_line, relevant_maze)]


def get_maze(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        maze = [x for line in f if (x := line.strip())]

    return maze


def find_starting_positions(maze: MazeType) -> tuple[MazePosition, MazePosition]:
    start_col = max(enumerate(maze[0]),
                    key=lambda x: x[1] & Direction.UP)[0]
    end_col = max(enumerate(maze[-1]),
                  key=lambda x: x[1] & Direction.DOWN)[0]

    start_position = MazePosition(0, start_col)
    end_position = MazePosition(len(maze) - 1, end_col)

    return (start_position, end_position)


def _():
    pass


FuncType: type = type(_)
del _


def save_result(func_or_filename):
    if s := isinstance(func_or_filename, str):
        file = func_or_filename
    elif isinstance(func_or_filename, FuncType):
        file = f'{func_or_filename.__name__}.txt'
    else:
        raise TypeError(f'save_result only accepts types {str} and {FuncType}')

    def decorator(func):
        @ wraps(func)
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            with open(file, 'w') as f:
                f.write(result)
            return result
        return inner

    return decorator if s else decorator(func_or_filename)


@ save_result
def q1(txt_maze: list[str]) -> str:
    return '\n'.join(txt_maze)


@ save_result
def q2(maze: MazeType) -> str:
    def convert(direction: Direction):
        # This is such a scuffed way of doing this
        direction_repr = repr(direction)
        succession_str = ''
        for d in ['DOWN', 'LEFT', 'RIGHT', 'UP', ]:
            if d in direction_repr:
                succession_str += d[0]
        return succession_str

    def line_convert(line):
        return map(convert, line)

    return '\n'.join(','.join(line_convert(line)) for line in maze)


@save_result
def q3(maze: MazeType,
       start_position: MazePosition,
       end_position: MazePosition) -> str:
    moves = Search.solve_maze(maze, start_position,
                              end_position, algorithm='dfs')
    # MyPy isn't great at inferring map args
    return ''.join(map(get_move_as_str, moves))  # type: ignore[arg-type]


@save_result
def q4(txt_maze: list[str],
       start_position: MazePosition,
       end_position: MazePosition) -> str:
    txt_maze = txt_maze.copy()
    parsed_maze = parse_maze(txt_maze)
    moves = Search.solve_maze(parsed_maze, start_position,
                              end_position, algorithm='dfs')

    # Using list as an argument to map is okay, mypy can't infer that
    maze = list(map(list, txt_maze[1::2]))  # type: ignore[arg-type]

    pos = start_position
    maze[pos.x][3 * pos.y + 1: 3 * pos.y + 3] = ['@', '@']

    for move in moves:
        if move == MoveList.DOWN:
            pos = MazePosition(pos.x + 1, pos.y)
        elif move == MoveList.LEFT:
            pos = MazePosition(pos.x, pos.y - 1)
        elif move == MoveList.RIGHT:
            pos = MazePosition(pos.x, pos.y + 1)
        elif move == MoveList.UP:
            pos = MazePosition(pos.x - 1, pos.y)

        maze[pos.x][3 * pos.y + 1: 3 * pos.y + 3] = ['@', '@']

    maze = list(map(''.join, maze))  # type: ignore[arg-type]
    txt_maze[1::2] = maze  # type:ignore[assignment]
    return '\n'.join(txt_maze)


@save_result
def q5(maze: MazeType,
       start_position: MazePosition,
       end_position: MazePosition) -> str:
    _, visited = Search.solve_maze(
        maze, start_position, end_position, algorithm='bfs', return_visited=True)

    matrix = [['0' for _ in range(len(maze[0]))] for _ in range(len(maze))]

    for pos in visited:  # type: ignore[union-attr]
        matrix[pos.x][pos.y] = '1'

    return '\n'.join(map(','.join, matrix))


@save_result
def q6(maze: MazeType,
       start_position: MazePosition,
       end_position: MazePosition) -> str:
    _, visited = Search.solve_maze(
        maze, start_position, end_position, algorithm='dfs', return_visited=True)

    matrix = [['0' for _ in range(len(maze[0]))] for _ in range(len(maze))]

    for pos in visited:  # type: ignore[union-attr]
        matrix[pos.x][pos.y] = '1'

    return '\n'.join(map(','.join, matrix))


@save_result
def q7(maze: MazeType, end_position: MazePosition) -> str:
    height, width = len(maze), len(maze[0])

    dist_matrix = [[str(Search.Heuristics.manhattan(end_position, (h, w)))
                    for w in range(width)] for h in range(height)]

    return '\n'.join(map(','.join, dist_matrix))


@save_result
def q8(maze: MazeType, start_position: MazePosition, end_position: MazePosition) -> str:
    _, visited = Search.solve_maze(
        maze, start_position, end_position,
        algorithm='A*', heuristic=Search.Heuristics.manhattan, return_visited=True)

    matrix = [['0' for _ in range(len(maze[0]))] for _ in range(len(maze))]

    for pos in visited:  # type: ignore[union-attr]
        matrix[pos.x][pos.y] = '1'

    return '\n'.join(map(','.join, matrix))


@save_result
def q9(maze: MazeType, start_position: MazePosition, end_position: MazePosition) -> str:
    _, visited = Search.solve_maze(
        maze, start_position, end_position,
        algorithm='A*', heuristic=Search.Heuristics.euclidean, return_visited=True)

    matrix = [['0' for _ in range(len(maze[0]))] for _ in range(len(maze))]

    for pos in visited:  # type: ignore[union-attr]
        matrix[pos.x][pos.y] = '1'

    return '\n'.join(map(','.join, matrix))


if __name__ == '__main__':
    txt_maze = get_maze('maze.txt')
    parsed_maze = parse_maze(txt_maze)
    start_position, end_position = find_starting_positions(parsed_maze)

    q1(txt_maze)
    q2(parsed_maze)
    q3(parsed_maze, start_position, end_position)
    q4(txt_maze, start_position, end_position)
    q5(parsed_maze, start_position, end_position)
    q6(parsed_maze, start_position, end_position)
    q7(parsed_maze, end_position)
    q8(parsed_maze, start_position, end_position)
    q9(parsed_maze, start_position, end_position)
