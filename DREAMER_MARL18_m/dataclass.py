from dataclasses import dataclass
import multiprocessing as mp
import queue  # for Empty

@dataclass
class TransitionMsg:
    worker_id: int
    state: np.ndarray
    robot_state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    next_robot_state: np.ndarray
    done: bool

@dataclass
class EpisodeStatMsg:
    worker_id: int
    episode_idx: int
    total_reward: float
    evac_time_80: int
    evac_time_100: int
    total_lifetime: float
    map_num: int
    abnormal: int
