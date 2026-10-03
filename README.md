# BIM 构件增量碰撞检测库

纯 Python 3.11 标准库实现，无第三方依赖，数据全部驻留内存。

## 接口

```python
from clash_detector import ClashDetector

detector = ClashDetector()
detector.build({1: (0, 0, 0, 2, 2, 2), 2: (1, 1, 1, 3, 3, 3)})
result = detector.apply(upserts={3: (0.5, 0.5, 0.5, 1.5, 1.5, 1.5)}, deletes=[2])
# result.added / result.removed / result.updated
all_clashes = detector.clashes()
```

- 包围盒格式：`(min_x, min_y, min_z, max_x, max_y, max_z)`，单位米，下界严格小于上界。
- 碰撞判定：每轴重叠长度 = `min(上界) - max(下界)`，三个长度均严格大于 0 才算碰撞；贴面/贴边/贴点不算，不使用容差。
- `apply` 的增量恰好等于整批执行前后两个场景的碰撞差异，不暴露批内临时状态；`removed` 携带旧尺寸，`added`/`updated` 携带新尺寸。

## 成本

设构件总数为 `n`，单批修改（upsert + delete）数为 `k`，场景真实碰撞数为 `c`：

- `build`：穷举全部构件对，`O(n²)` 时间，`O(n + c)` 空间。
- `apply`：只对触及被修改构件的配对做重算与差异比较，时间为 `O(k·n + c_k)`（`c_k` 为触及修改构件的碰撞对数），与未受影响的 `n - k` 个构件之间的碰撞无关；额外空间 `O(c_k)`。
- `clashes`：返回当前维护的碰撞表，`O(c log c)`（按构件对排序输出）。

即增量成本随修改数 `k` 和真实碰撞数线性增长，与场景总规模 `n` 仅在重算候选对时线性相关，避免了对整场景重新做 `O(n²)` 检测。

## 运行测试

```bash
python3 -m unittest test_clash_detector -v
```

测试使用固定随机种子生成小场景和连续混合批次（含新增、删除、替换、重复提交相同几何、仅重叠尺寸变化），以独立穷举实现逐批核对三类增量、重叠尺寸与完整碰撞集；另含同批交换不相交构件位置的原子性用例，以及贴面与极薄正厚度交叠的边界用例。
