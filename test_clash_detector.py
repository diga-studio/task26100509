"""clash_detector 的测试。

包含：
- 固定随机种子的小场景 + 连续混合批次，用独立穷举实现核对每批
  added/removed/updated、重叠尺寸及完整当前碰撞集；
- 同批交换两个不相交构件的位置（不得产生临时增量）；
- 贴面不算碰撞、极薄正厚度交叠算碰撞。
"""

import random
import unittest

from clash_detector import Clash, ClashDetector


def brute_force_clashes(boxes):
    """独立穷举实现：直接对全部构件对计算碰撞，返回 {pair: Clash}。"""
    result = {}
    ids = sorted(boxes)
    for i, id_a in enumerate(ids):
        ba = boxes[id_a]
        for id_b in ids[i + 1:]:
            bb = boxes[id_b]
            ox = min(ba[3], bb[3]) - max(ba[0], bb[0])
            oy = min(ba[4], bb[4]) - max(ba[1], bb[1])
            oz = min(ba[5], bb[5]) - max(ba[2], bb[2])
            if ox > 0.0 and oy > 0.0 and oz > 0.0:
                result[(id_a, id_b)] = Clash(id_a, id_b, ox, oy, oz)
    return result


def random_box(rng):
    cx = rng.uniform(-5.0, 5.0)
    cy = rng.uniform(-5.0, 5.0)
    cz = rng.uniform(-5.0, 5.0)
    sx = rng.uniform(0.5, 3.0)
    sy = rng.uniform(0.5, 3.0)
    sz = rng.uniform(0.5, 3.0)
    return (cx, cy, cz, cx + sx, cy + sy, cz + sz)


def as_map(clashes):
    return {c.pair: c for c in clashes}


class TestRandomBatches(unittest.TestCase):
    def test_random_mixed_batches_against_brute_force(self):
        rng = random.Random(20261005)
        detector = ClashDetector()

        # 初始小场景。
        scene = {i: random_box(rng) for i in range(12)}
        detector.build(scene)
        self.assertEqual(as_map(detector.clashes()), brute_force_clashes(scene))

        prev = brute_force_clashes(scene)
        next_id = max(scene) + 1

        for step in range(60):
            upserts = {}
            deletes = set()

            # 随机新增 0~3 个构件。
            for _ in range(rng.randint(0, 3)):
                upserts[next_id] = random_box(rng)
                next_id += 1

            live_ids = sorted(set(scene) - deletes)
            # 随机替换 0~3 个已有构件（整体替换包围盒）。
            for element_id in rng.sample(live_ids, min(len(live_ids), rng.randint(0, 3))):
                upserts[element_id] = random_box(rng)

            # 随机删除 0~3 个未被本批替换的构件。
            deletable = sorted(set(scene) - set(upserts))
            deletes = set(rng.sample(deletable, min(len(deletable), rng.randint(0, 3))))

            # 每 10 步强制一次“重复提交相同几何”。
            if step % 10 == 5 and scene:
                same_id = rng.choice(sorted(set(scene) - set(upserts) - deletes))
                upserts[same_id] = scene[same_id]

            result = detector.apply(upserts, deletes)

            # 维护参考场景。
            for element_id in deletes:
                del scene[element_id]
            scene.update(upserts)

            now = brute_force_clashes(scene)

            # 期望增量 = 前后两个场景穷举结果的差异。
            exp_added = {p: now[p] for p in now.keys() - prev.keys()}
            exp_removed = {p: prev[p] for p in prev.keys() - now.keys()}
            exp_updated = {
                p: now[p] for p in now.keys() & prev.keys() if now[p] != prev[p]
            }

            self.assertEqual(as_map(result.added), exp_added, f"step {step} added")
            self.assertEqual(as_map(result.removed), exp_removed, f"step {step} removed")
            self.assertEqual(as_map(result.updated), exp_updated, f"step {step} updated")

            # 三类集合互不交叉。
            groups = [set(as_map(g)) for g in (result.added, result.removed, result.updated)]
            self.assertEqual(groups[0] & groups[1], set())
            self.assertEqual(groups[0] & groups[2], set())
            self.assertEqual(groups[1] & groups[2], set())

            # Apply 后的完整碰撞集与直接检测最终几何完全一致。
            self.assertEqual(as_map(detector.clashes()), now, f"step {step} clashes")

            prev = now

    def test_overlap_dims_only_change_is_updated(self):
        detector = ClashDetector()
        scene = {
            1: (0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
            2: (1.0, 1.0, 1.0, 3.0, 3.0, 3.0),
        }
        detector.build(scene)
        # 构件 2 移动后仍与 1 碰撞，但重叠尺寸变化。
        result = detector.apply({2: (1.5, 1.0, 1.0, 3.5, 3.0, 3.0)}, [])
        self.assertEqual(result.added, ())
        self.assertEqual(result.removed, ())
        self.assertEqual(
            as_map(result.updated),
            {(1, 2): Clash(1, 2, 0.5, 1.0, 1.0)},
        )

    def test_resubmit_identical_geometry_yields_empty_delta(self):
        detector = ClashDetector()
        scene = {
            1: (0.0, 0.0, 0.0, 2.0, 2.0, 2.0),
            2: (1.0, 1.0, 1.0, 3.0, 3.0, 3.0),
        }
        detector.build(scene)
        result = detector.apply({1: scene[1], 2: scene[2]}, [])
        self.assertEqual(result.added, ())
        self.assertEqual(result.removed, ())
        self.assertEqual(result.updated, ())
        self.assertEqual(as_map(detector.clashes()), brute_force_clashes(scene))


class TestBatchAtomicity(unittest.TestCase):
    def test_swap_disjoint_boxes_in_one_batch_has_no_delta(self):
        detector = ClashDetector()
        box_a = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        box_b = (5.0, 5.0, 5.0, 6.0, 6.0, 6.0)
        detector.build({1: box_a, 2: box_b})
        # 同一批次交换两个原本不相交构件的位置。
        result = detector.apply({1: box_b, 2: box_a}, [])
        self.assertEqual(result.added, ())
        self.assertEqual(result.removed, ())
        self.assertEqual(result.updated, ())
        self.assertEqual(detector.clashes(), [])


class TestBoundarySemantics(unittest.TestCase):
    def test_face_touching_is_not_a_clash(self):
        detector = ClashDetector()
        detector.build({
            1: (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            2: (1.0, 0.0, 0.0, 2.0, 1.0, 1.0),  # 与 1 贴面
        })
        self.assertEqual(detector.clashes(), [])

    def test_thin_positive_overlap_is_a_clash(self):
        detector = ClashDetector()
        eps = 1e-9
        detector.build({
            1: (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            2: (1.0 - eps, 0.0, 0.0, 2.0, 1.0, 1.0),  # 极薄正厚度交叠
        })
        clashes = detector.clashes()
        self.assertEqual(len(clashes), 1)
        self.assertEqual(clashes[0].pair, (1, 2))
        self.assertAlmostEqual(clashes[0].overlap_x, eps)
        self.assertEqual(clashes[0].overlap_y, 1.0)
        self.assertEqual(clashes[0].overlap_z, 1.0)


if __name__ == "__main__":
    unittest.main()
