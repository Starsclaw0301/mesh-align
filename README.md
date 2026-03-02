# mesh-align

将高精度 OBJ 模型对齐到低精度 STL 模型的坐标系。

**方案：PCA 粗配准 + ICP 精配准**  
**目标精度：0.1mm**

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python align.py <低精度.stl> <高精度.obj> -o aligned.obj
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `stl` | 低精度 STL（目标坐标系） | 必填 |
| `obj` | 高精度 OBJ（待对齐） | 必填 |
| `-o / --output` | 输出 OBJ 路径 | `aligned.obj` |
| `-n / --n-points` | 采样点数（越多越精确，越慢） | `50000` |
| `--save-matrix` | 保存 4×4 变换矩阵（.npy） | `transform.npy` |

### 示例

```bash
python align.py model_low.stl model_high.obj -o model_aligned.obj -n 100000
```

## 输出

- `aligned.obj`：对齐后的高精度模型
- `transform.npy`：4×4 变换矩阵，可用 `np.load("transform.npy")` 读取复用

## 流程说明

1. **采样**：对两个网格均匀面积采样，低精度 STL 也能得到足够密集的点云
2. **PCA 粗配准**：用主成分分析对齐主轴，生成 4 个候选变换（处理轴对称歧义）
3. **ICP 精配准**：两阶段 ICP（5mm → 0.1mm 容差），取 RMSE 最小的候选
4. **验证**：输出最终 RMSE，目标 ≤ 0.1mm

## 注意事项

- 模型必须**形状一致**（同一物体，不同面数），ICP 不适用于形状差异大的模型
- 如果 RMSE 超过 0.1mm，尝试增大 `-n`（采样点数）
- 对高度对称的模型，4 个候选中可能有多个接近，建议人工验证输出结果
