"""2D座標系のワールドとオブジェクト管理"""

from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path

import numpy as np


@dataclass
class Obstacle:
    """障害物（矩形）

    Attributes:
        x: 中心X座標
        y: 中心Y座標
        width: 幅
        height: 高さ
    """

    x: float
    y: float
    width: float
    height: float

    def get_corners(self) -> np.ndarray:
        """障害物の4隅の座標を取得

        Returns:
            4x2のnumpy配列 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        hw, hh = self.width / 2, self.height / 2
        return np.array([
            [self.x - hw, self.y - hh],
            [self.x + hw, self.y - hh],
            [self.x + hw, self.y + hh],
            [self.x - hw, self.y + hh],
        ])

    def get_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """障害物の4辺を取得（線分リスト）

        Returns:
            [(start_point, end_point), ...] の形式
        """
        corners = self.get_corners()
        edges = []
        for i in range(4):
            edges.append((corners[i], corners[(i + 1) % 4]))
        return edges


@dataclass
class World:
    """2D座標系のワールド

    Attributes:
        width: ワールドの幅
        height: ワールドの高さ
        obstacles: 障害物リスト
    """

    width: float = 100.0
    height: float = 100.0
    obstacles: list[Obstacle] = field(default_factory=list)

    def add_obstacle(
        self, x: float, y: float, width: float, height: float
    ) -> Obstacle:
        """障害物を追加"""
        obstacle = Obstacle(x, y, width, height)
        self.obstacles.append(obstacle)
        return obstacle

    def is_collision(self, x: float, y: float, radius: float = 0.0) -> bool:
        """指定した位置が障害物または境界と衝突するかを判定"""
        # 境界チェック
        if x - radius < 0 or x + radius > self.width:
            return True
        if y - radius < 0 or y + radius > self.height:
            return True

        # 障害物チェック
        for obs in self.obstacles:
            if self._circle_rect_collision(x, y, radius, obs):
                return True

        return False

    def _circle_rect_collision(
        self, cx: float, cy: float, r: float, obs: Obstacle
    ) -> bool:
        """円と矩形の衝突判定"""
        hw, hh = obs.width / 2, obs.height / 2
        closest_x = max(obs.x - hw, min(cx, obs.x + hw))
        closest_y = max(obs.y - hh, min(cy, obs.y + hh))
        dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
        return dist_sq <= r**2

    def get_all_edges(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """全障害物と境界の辺を取得（LiDAR用）"""
        edges = []

        # 境界の辺
        boundary = [
            (np.array([0, 0]), np.array([self.width, 0])),
            (np.array([self.width, 0]), np.array([self.width, self.height])),
            (np.array([self.width, self.height]), np.array([0, self.height])),
            (np.array([0, self.height]), np.array([0, 0])),
        ]
        edges.extend(boundary)

        # 障害物の辺
        for obs in self.obstacles:
            edges.extend(obs.get_edges())

        return edges

    @classmethod
    def load(cls, filepath: str | Path) -> "World":
        """外部ファイルからワールド設定をロード

        Args:
            filepath: 設定ファイルのパス（JSON形式）

        Returns:
            Worldインスタンス
        """
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        world = cls(
            width=data.get("width", 100.0),
            height=data.get("height", 100.0),
        )

        for obs_data in data.get("obstacles", []):
            world.add_obstacle(
                x=obs_data["x"],
                y=obs_data["y"],
                width=obs_data["width"],
                height=obs_data["height"],
            )

        return world
