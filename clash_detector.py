"""BIM 构件增量碰撞检测库（Python 3.11 标准库）。

构件几何为项目局部坐标系下的轴对齐包围盒（AABB），单位米。
碰撞判定：每轴重叠长度 = min(上界) - max(下界)，三个长度均严格大于 0
才算碰撞；贴面、贴边、贴点（任一轴重叠为 0）不算碰撞，不使用容差。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

# 包围盒：(min_x, min_y, min_z, max_x, max_y, max_z)，下界严格小于上界。
BBox = Tuple[float, float, float, float, float, float]

# 构件对，id 小的在前，保证每对唯一表示。
Pair = Tuple[int, int]


@dataclass(frozen=True)
class Clash:
    """一条碰撞记录：构件对及三个轴的重叠尺寸（均为严格正数，单位米）。"""

    id1: int
    id2: int
    overlap_x: float
    overlap_y: float
    overlap_z: float

    @property
    def pair(self) -> Pair:
        return (self.id1, self.id2)


@dataclass(frozen=True)
class ApplyResult:
    """Apply 的增量结果：新增 / 消失 / 尺寸变化三类碰撞记录。"""

    added: Tuple[Clash, ...]
    removed: Tuple[Clash, ...]
    updated: Tuple[Clash, ...]


def _make_clash(id_a: int, box_a: BBox, id_b: int, box_b: BBox) -> Optional[Clash]:
    """计算两个包围盒的碰撞记录；不碰撞（含贴面/贴边/贴点）返回 None。"""
    ox = min(box_a[3], box_b[3]) - max(box_a[0], box_b[0])
    if ox <= 0.0:
        return None
    oy = min(box_a[4], box_b[4]) - max(box_a[1], box_b[1])
    if oy <= 0.0:
        return None
    oz = min(box_a[5], box_b[5]) - max(box_a[2], box_b[2])
    if oz <= 0.0:
        return None
    id1, id2 = (id_a, id_b) if id_a < id_b else (id_b, id_a)
    return Clash(id1, id2, ox, oy, oz)


class ClashDetector:
    """增量碰撞检测器。数据全部驻留内存。"""

    def __init__(self) -> None:
        self._boxes: Dict[int, BBox] = {}
        self._clashes: Dict[Pair, Clash] = {}

    def build(self, elements: Mapping[int, BBox]) -> None:
        """建立场景及初始碰撞状态。"""
        self._boxes = dict(elements)
        self._clashes = {}
        ids = sorted(self._boxes)
        for i, id_a in enumerate(ids):
            box_a = self._boxes[id_a]
            for id_b in ids[i + 1:]:
                clash = _make_clash(id_a, box_a, id_b, self._boxes[id_b])
                if clash is not None:
                    self._clashes[clash.pair] = clash

    def apply(
        self,
        upserts: Mapping[int, BBox],
        deletes: Iterable[int],
    ) -> ApplyResult:
        """批量新增、替换和删除构件，返回碰撞增量。

        增量恰好等于整批执行前后两个场景的碰撞差异：
        added 为新出现的对，removed 为消失的对（携带旧尺寸），
        updated 为仍存在但重叠尺寸变化的对（携带新尺寸）。
        """
        upserts = dict(upserts)
        delete_ids = set(deletes)
        affected = set(upserts) | delete_ids

        # 批次前：受影响的旧碰撞对。
        old_pairs = {
            pair: clash
            for pair, clash in self._clashes.items()
            if pair[0] in affected or pair[1] in affected
        }

        # 整体应用几何变更。
        for element_id in delete_ids:
            del self._boxes[element_id]
        self._boxes.update(upserts)

        # 批次后：基于最终几何，重算所有触及受影响构件的碰撞对。
        # 不触及受影响构件的对不可能变化，直接保留。
        remaining = affected & self._boxes.keys()
        new_pairs: Dict[Pair, Clash] = {}
        for id_a in remaining:
            box_a = self._boxes[id_a]
            for id_b, box_b in self._boxes.items():
                if id_b == id_a:
                    continue
                if id_b in remaining and id_b < id_a:
                    continue  # 双方均受影响的对只算一次
                clash = _make_clash(id_a, box_a, id_b, box_b)
                if clash is not None:
                    new_pairs[clash.pair] = clash

        # 更新维护的碰撞状态。
        for pair in old_pairs:
            del self._clashes[pair]
        self._clashes.update(new_pairs)

        # 求前后差异。
        added = [new_pairs[p] for p in new_pairs.keys() - old_pairs.keys()]
        removed = [old_pairs[p] for p in old_pairs.keys() - new_pairs.keys()]
        updated = [
            new_pairs[p]
            for p in new_pairs.keys() & old_pairs.keys()
            if new_pairs[p] != old_pairs[p]
        ]
        key = lambda c: c.pair  # noqa: E731
        return ApplyResult(
            added=tuple(sorted(added, key=key)),
            removed=tuple(sorted(removed, key=key)),
            updated=tuple(sorted(updated, key=key)),
        )

    def clashes(self) -> List[Clash]:
        """返回当前全部碰撞记录，每对恰好一次。"""
        return [self._clashes[p] for p in sorted(self._clashes)]
