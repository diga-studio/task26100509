"""bim_clash 的测试：固定种子随机场景 + 独立穷举实现交叉核对。"""

import random
import unittest
from typing import Dict, Iterable, List, Tuple

from bim_clash import ApplyResult, Clash, Element, Scene


def brute_force(elements: Iterable[Element]) -> Dict[Tuple[int, int], Clash]:
    """独立穷举参考实现：全部两两组合，严格正厚度判定。"""
    items = list(elements)
    result: Dict[Tuple[int, int], Clash] = {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            ox = min(a.max_x, b.max_x) - max(a.min_x, b.min_x)
            oy = min(a.max_y, b.max_y) - max(a.min_y, b.min_y)
            oz = min(a.max_z, b.max_z) - max(a.min_z, b.min_z)
            if ox > 0.0 and oy > 0.0 and oz > 0.0:
                lo, hi = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                result[(lo, hi)] = Clash(lo, hi, ox, oy, oz)
    return result


def diff_clashes(
    before: Dict[Tuple[int, int], Clash],
    after: Dict[Tuple[int, int], Clash],
) -> ApplyResult:
    added = tuple(clash for pair, clash in sorted(after.items())
                  if pair not in before)
    removed = tuple(clash for pair, clash in sorted(before.items())
                    if pair not in after)
    updated = tuple(
        after[pair] for pair in sorted(after)
        if pair in before and after[pair] != before[pair]
    )
    return ApplyResult(added=added, removed=removed, updated=updated)


def random_element(rng: random.Random, element_id: int) -> Element:
    cx = rng.uniform(0.0, 5.0)
    cy = rng.uniform(0.0, 5.0)
    cz = rng.uniform(0.0, 5.0)
    sx = rng.uniform(0.2, 1.0)
    sy = rng.uniform(0.2, 1.0)
    sz = rng.uniform(0.2, 1.0)
    return Element(element_id,
                   cx - sx / 2, cy - sy / 2, cz - sz / 2,
                   cx + sx / 2, cy + sy / 2, cz + sz / 2)


class RandomBatchTest(unittest.TestCase):
    def test_random_mixed_batches(self) -> None:
        rng = random.Random(20261005)
        scene = Scene()

        # 初始场景
        elements: Dict[int, Element] = {}
        for element_id in range(1, 41):
            elements[element_id] = random_element(rng, element_id)
        scene.build(list(elements.values()))
        self._check_full(scene, elements)

        next_id = 41
        for batch_no in range(30):
            existing = sorted(elements)
            upserts: List[Element] = []
            deletes: List[int] = []
            upsert_ids = set()

            # 替换若干已有构件（含重复提交相同几何、仅尺寸微变）
            for element_id in rng.sample(existing, k=min(6, len(existing))):
                old = elements[element_id]
                roll = rng.random()
                if roll < 0.2:
                    new = old  # 重复提交相同几何
                elif roll < 0.5:
                    # 仅轻微移动：碰撞对不变但重叠尺寸变化
                    dx = rng.uniform(-0.05, 0.05)
                    new = Element(element_id,
                                  old.min_x + dx, old.min_y, old.min_z,
                                  old.max_x + dx, old.max_y, old.max_z)
                else:
                    new = random_element(rng, element_id)
                upserts.append(new)
                upsert_ids.add(element_id)

            # 删除若干未被替换的构件
            deletable = [i for i in existing if i not in upsert_ids]
            for element_id in rng.sample(deletable,
                                         k=min(4, len(deletable))):
                deletes.append(element_id)

            # 新增若干构件
            for _ in range(rng.randint(0, 5)):
                upserts.append(random_element(rng, next_id))
                next_id += 1

            before = brute_force(elements.values())
            result = scene.apply(upserts, deletes)

            for element_id in deletes:
                del elements[element_id]
            for element in upserts:
                elements[element.id] = element
            after = brute_force(elements.values())

            expected = diff_clashes(before, after)
            self.assertEqual(expected.added, result.added,
                             f"batch {batch_no}: added 不匹配")
            self.assertEqual(expected.removed, result.removed,
                             f"batch {batch_no}: removed 不匹配")
            self.assertEqual(expected.updated, result.updated,
                             f"batch {batch_no}: updated 不匹配")
            self._check_no_overlap_between_sets(result, batch_no)
            self._check_full(scene, elements)

    def _check_full(self, scene: Scene,
                    elements: Dict[int, Element]) -> None:
        expected = brute_force(elements.values())
        actual = {(c.id_a, c.id_b): c for c in scene.clashes()}
        self.assertEqual(expected, actual)

    def _check_no_overlap_between_sets(self, result: ApplyResult,
                                       batch_no: int) -> None:
        def keys(clashes: Iterable[Clash]) -> set:
            return {(c.id_a, c.id_b) for c in clashes}

        a, r, u = (keys(result.added), keys(result.removed),
                   keys(result.updated))
        self.assertFalse(a & r, f"batch {batch_no}: added/removed 交叉")
        self.assertFalse(a & u, f"batch {batch_no}: added/updated 交叉")
        self.assertFalse(r & u, f"batch {batch_no}: removed/updated 交叉")


class SwapTest(unittest.TestCase):
    def test_swap_disjoint_elements_no_transient_clashes(self) -> None:
        # a、b 原本不相交；交换位置后也不相交，但各自途中会撞上 c。
        a = Element(1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        b = Element(2, 5.0, 0.0, 0.0, 6.0, 1.0, 1.0)
        c = Element(3, 2.5, 0.0, 0.0, 3.5, 1.0, 1.0)
        scene = Scene()
        scene.build([a, b, c])
        self.assertEqual([], scene.clashes())

        swapped_a = Element(1, 5.0, 0.0, 0.0, 6.0, 1.0, 1.0)
        swapped_b = Element(2, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        result = scene.apply([swapped_a, swapped_b], [])
        self.assertEqual((), result.added)
        self.assertEqual((), result.removed)
        self.assertEqual((), result.updated)
        self.assertEqual([], scene.clashes())


class BoundaryTest(unittest.TestCase):
    def test_touching_faces_not_clash(self) -> None:
        scene = Scene()
        scene.build([
            Element(1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            Element(2, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0),  # 贴面
            Element(3, 1.0, 1.0, 0.0, 2.0, 2.0, 1.0),  # 贴边
            Element(4, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0),  # 贴点
        ])
        self.assertEqual([], scene.clashes())

    def test_thin_positive_overlap_is_clash(self) -> None:
        eps = 1e-9
        scene = Scene()
        scene.build([
            Element(1, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            Element(2, 1.0 - eps, 0.0, 0.0, 2.0, 1.0, 1.0),
        ])
        clashes = scene.clashes()
        self.assertEqual(1, len(clashes))
        self.assertEqual(1.0 - (1.0 - eps), clashes[0].overlap_x)
        self.assertEqual(1.0, clashes[0].overlap_y)
        self.assertEqual(1.0, clashes[0].overlap_z)


if __name__ == "__main__":
    unittest.main()
