"""2D LiDARセンサーモデル"""

import numpy as np


class Lidar2D:
    """2D LiDARセンサー

    360度を指定された分解能で分割し、各方向への障害物までの距離を計測。

    Attributes:
        num_rays: レイの数（デフォルト1081）
        max_range: 最大検出距離
        min_range: 最小検出距離
        angles: 各レイの角度（車両座標系）
    """

    def __init__(
        self,
        num_rays: int = 1081,
        max_range: float = 30.0,
        min_range: float = 0.1,
        fov: float = 2 * np.pi,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ):
        self.num_rays = num_rays
        self.max_range = max_range
        self.min_range = min_range
        self.fov = fov
        self.offset_x = offset_x
        self.offset_y = offset_y

        self.angles = np.linspace(-fov / 2, fov / 2, num_rays, endpoint=False)
        # レイ方向のcos/sinを事前計算（ローカル座標系）
        self._local_cos = np.cos(self.angles)
        self._local_sin = np.sin(self.angles)

    def scan(
        self,
        sensor_x: float,
        sensor_y: float,
        sensor_yaw: float,
        edges: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """LiDARスキャンを実行（ベクトル化版）

        Args:
            sensor_x: センサーのX座標
            sensor_y: センサーのY座標
            sensor_yaw: センサーの向き（ラジアン）
            edges: 障害物の辺のリスト [(start_point, end_point), ...]

        Returns:
            各レイの距離値（1D numpy配列）
        """
        if not edges:
            return np.full(self.num_rays, self.max_range)

        cos_yaw = np.cos(sensor_yaw)
        sin_yaw = np.sin(sensor_yaw)

        # センサー位置
        sx = sensor_x + self.offset_x * cos_yaw - self.offset_y * sin_yaw
        sy = sensor_y + self.offset_x * sin_yaw + self.offset_y * cos_yaw

        # 全レイの方向ベクトルを一括計算 (num_rays, 2)
        # ワールド座標系でのレイ方向 = 回転行列 × ローカル方向
        ray_dirs_x = cos_yaw * self._local_cos - sin_yaw * self._local_sin
        ray_dirs_y = sin_yaw * self._local_cos + cos_yaw * self._local_sin

        # 辺データを配列に変換 (num_edges, 2)
        num_edges = len(edges)
        seg_starts = np.array([e[0] for e in edges])  # (num_edges, 2)
        seg_ends = np.array([e[1] for e in edges])    # (num_edges, 2)
        seg_vecs = seg_ends - seg_starts              # (num_edges, 2)

        # ブロードキャスト用に形状を調整
        # ray_dirs: (num_rays, 1, 2) -> 各レイに対して全辺をチェック
        # seg_vecs: (1, num_edges, 2) -> 全レイに対して各辺をチェック
        ray_dx = ray_dirs_x[:, np.newaxis]  # (num_rays, 1)
        ray_dy = ray_dirs_y[:, np.newaxis]  # (num_rays, 1)
        seg_vx = seg_vecs[:, 0]             # (num_edges,)
        seg_vy = seg_vecs[:, 1]             # (num_edges,)

        # クロス積: ray_dir × seg_vec (num_rays, num_edges)
        cross = ray_dx * seg_vy - ray_dy * seg_vx

        # diff = seg_start - ray_origin (num_edges, 2)
        diff_x = seg_starts[:, 0] - sx  # (num_edges,)
        diff_y = seg_starts[:, 1] - sy  # (num_edges,)

        # t = (diff × seg_vec) / cross
        # t: レイ上のパラメータ（距離に相当）
        t_numer = diff_x * seg_vy - diff_y * seg_vx  # (num_edges,)
        # u = (diff × ray_dir) / cross
        # u: 線分上のパラメータ（0〜1で線分内）
        u_numer = diff_x[np.newaxis, :] * \
            ray_dy - diff_y[np.newaxis, :] * ray_dx

        # ゼロ除算を回避
        cross_safe = np.where(np.abs(cross) < 1e-10, 1e-10, cross)
        t = t_numer / cross_safe  # (num_rays, num_edges)
        u = u_numer / cross_safe  # (num_rays, num_edges)

        # 有効な交差判定: t >= 0 かつ 0 <= u <= 1 かつ cross != 0
        valid = (
            (t >= self.min_range) &
            (u >= 0) & (u <= 1) &
            (np.abs(cross) >= 1e-10)
        )

        # 無効な交差点はmax_rangeに設定
        t_valid = np.where(valid, t, self.max_range)

        # 各レイについて最小距離を取得
        ranges = np.min(t_valid, axis=1)
        ranges = np.clip(ranges, self.min_range, self.max_range)

        return ranges

    def _ray_segment_intersection(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> float | None:
        """レイと線分の交差判定（後方互換性のため残す）"""
        seg_vec = seg_end - seg_start
        cross = ray_dir[0] * seg_vec[1] - ray_dir[1] * seg_vec[0]

        if abs(cross) < 1e-10:
            return None

        diff = seg_start - ray_origin
        t = (diff[0] * seg_vec[1] - diff[1] * seg_vec[0]) / cross
        u = (diff[0] * ray_dir[1] - diff[1] * ray_dir[0]) / cross

        if t >= 0 and 0 <= u <= 1:
            return t

        return None

    def scan_to_points(
        self,
        sensor_x: float,
        sensor_y: float,
        sensor_yaw: float,
        ranges: np.ndarray,
    ) -> np.ndarray:
        """スキャン結果を点群に変換（可視化用）"""
        cos_yaw = np.cos(sensor_yaw)
        sin_yaw = np.sin(sensor_yaw)

        sx = sensor_x + self.offset_x * cos_yaw - self.offset_y * sin_yaw
        sy = sensor_y + self.offset_x * sin_yaw + self.offset_y * cos_yaw

        valid_mask = ranges < self.max_range
        valid_ranges = ranges[valid_mask]
        valid_angles = self.angles[valid_mask]

        world_angles = sensor_yaw + valid_angles
        points_x = sx + valid_ranges * np.cos(world_angles)
        points_y = sy + valid_ranges * np.sin(world_angles)

        return np.column_stack([points_x, points_y])
