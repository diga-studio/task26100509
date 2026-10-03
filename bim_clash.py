"""BIM 构件增量碰撞检测库（轴对齐包围盒，项目局部坐标系，单位米）。

仅使用 Python 标准库，兼容 Python 3.11+。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Element:
    """构件：唯一整数 id + 三轴下界/上界（每轴下界严格小于上界）。"""

    id: int
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or isinstance(self.id, bool):
            raise TypeError("id 必须是整数")
        bounds = (
            self.min_x, self.min_y, self.min_z,
            self.max_x, self.max_y, self.max_z,
        )
        for value in bounds:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError("坐标必须是实数")
            if not math.isfinite(value):
                raise ValueError("坐标必须是有限浮点数")
        if not (self.min_x < self.max_x
                and self.min_y < self.max_y
                and self.min_z < self.max_z):
            raise ValueError("每轴下界必须严格小于上界")


@dataclass(frozen=True)
class Clash:
    """碰撞记录：构件对（id 小者在前）及三轴重叠尺寸（均严格大于 0）。"""

    id_a: int
    id_b: int
    overlap_x: float
    overlap_y: float
    overlap_z: float


@dataclass(frozen=True)
class ApplyResult:
    """Apply 的增量结果：新增 / 消失 / 尺寸变化的碰撞。"""

    added: Tuple[Clash, ...]
    removed: Tuple[Clash, ...]
    updated: Tuple[Clash, ...]


def _overlap(a: Element, b: Element) -> Clash | None:
    """两构件三轴重叠长度均严格大于 0 才算碰撞；贴面/贴边/贴点不算。"""
    ox = min(a.max_x, b.max_x) - max(a.min_x, b.min_x)
    if ox <= 0.0:
        return None
    oy = min(a.max_y, b.max_y) - max(a.min_y, b.min_y)
    if oy <= 0.0:
        return None
    oz = min(a.max_z, b.max_z) - max(a.min_z, b.min_z)
    if oz <= 0.0:
        return None
    lo, hi = (a.id, b.id) if a.id < b.id else (b.id, a.id)
    return Clash(lo, hi, ox, oy, oz)


def _sweep_clashes(elements: Iterable[Element]) -> Dict[Tuple[int, int], Clash]:
    """沿 x 轴扫掠求全场景碰撞，键为 (小 id, 大 id)。"""
    ordered = sorted(elements, key=lambda e: (e.min_x, e.max_x, e.id))
    active: List[Element] = []
    result: Dict[Tuple[int, int], Clash] = {}
    for elem in ordered:
        active = [e for e in active if e.max_x > elem.min_x]
        for other in active:
            clash = _overlap(other, elem)
            if clash is not None:
                result[(clash.id_a, clash.id_b)] = clash
        active.append(elem)
    return result


class Scene:
    """碰撞场景：Build / Apply / Clashes。"""

    def __init__(self) -> None:
        self._elements: Dict[int, Element] = {}
        self._clashes: Dict[Tuple[int, int], Clash] = {}

    def build(self, elements: Iterable[Element]) -> None:
        items = list(elements)
        ids = [e.id for e in items]
        if len(set(ids)) != len(ids):
            raise ValueError("构件 id 必须唯一")
        self._elements = {e.id: e for e in items}
        self._clashes = _sweep_clashes(items)

    def apply(
        self,
        upserts: Iterable[Element],
        deletes: Iterable[int],
    ) -> ApplyResult:
        upsert_list = list(upserts)
        delete_list = list(deletes)
        upsert_ids = [e.id for e in upsert_list]
        if len(set(upsert_ids)) != len(upsert_ids):
            raise ValueError("upserts 内 id 重复")
        if len(set(delete_list)) != len(delete_list):
            raise ValueError("deletes 内 id 重复")
        if set(upsert_ids) & set(delete_list):
            raise ValueError("upserts 与 deletes 的 id 必须互不相交")
        missing = [i for i in delete_list if i not in self._elements]
        if missing:
            raise KeyError(f"删除目标不存在: {missing}")

        affected = set(upsert_ids) | set(delete_list)

        # 1) 记录受影响的旧碰撞（携带旧尺寸）
        old_pairs = {
            pair: clash
            for pair, clash in self._clashes.items()
            if pair[0] in affected or pair[1] in affected
        }

        # 2) 应用整批修改
        for element_id in delete_list:
            del self._elements[element_id]
        for element in upsert_list:
            self._elements[element.id] = element

        # 3) 只对受影响的构件重新探测碰撞（沿 x 轴扫掠裁剪候选）
        survivors = [self._elements[i] for i in affected if i in self._elements]
        ordered = sorted(self._elements.values(),
                         key=lambda e: (e.min_x, e.max_x, e.id))
        min_xs = [e.min_x for e in ordered]
        new_pairs: Dict[Tuple[int, int], Clash] = {}
        for elem in survivors:
            # 候选：min_x < elem.max_x（x 轴可能相交的全部构件）
            hi = bisect.bisect_left(min_xs, elem.max_x)
            for other in ordered[:hi]:
                if other.id == elem.id:
                    continue
                if other.max_x <= elem.min_x:
                    continue
                pair = (elem.id, other.id) if elem.id < other.id \
                    else (other.id, elem.id)
                if pair in new_pairs:
                    continue
                clash = _overlap(elem, other)
                if clash is not None:
                    new_pairs[pair] = clash

        # 4) 更新全量碰撞表
        for pair in old_pairs:
            del self._clashes[pair]
        self._clashes.update(new_pairs)

        # 5) 前后差异即增量；尺寸相同的不报
        added = tuple(clash for pair, clash in sorted(new_pairs.items())
                      if pair not in old_pairs)
        removed = tuple(clash for pair, clash in sorted(old_pairs.items())
                        if pair not in new_pairs)
        updated = tuple(
            new_pairs[pair] for pair in sorted(new_pairs)
            if pair in old_pairs and new_pairs[pair] != old_pairs[pair]
        )
        return ApplyResult(added=added, removed=removed, updated=updated)

    def clashes(self) -> List[Clash]:
        return sorted(self._clashes.values(), key=lambda c: (c.id_a, c.id_b))
