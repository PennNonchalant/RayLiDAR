# RayLiDAR

Hybrid Mesh–Gaussian LiDAR Simulation — GPU Programming final project.

用 RGB 重建室外静态场景（mesh-only / gaussian-only / hybrid），沿 GT LiDAR 射线求交生成仿真点云，并在 GPU 上优化推理。

## Documentation

- 项目记忆与完整设计文档：[docs/README.md](docs/README.md)
- 方案 V2（当前）：[PDF](docs/proposal_V2.pdf) · [LaTeX](docs/proposal_V2.tex)
- 方案 V1（原始）：[PDF](docs/proposal_V1.pdf) · [LaTeX](docs/proposal_V1.tex)

## Roadmap (V2)

1. **P1** — 单场景双基线：mesh-only（2DGS depth → TSDF → Marching Cubes）与 gaussian-only（3DGS / 2DGS），用 GT-defined rays 做 LiDAR cast 与评测。
2. **P2** — naive hybrid casting：mesh-first + 廉价 fallback 的逐 ray 路由，并做 GPU 加速（逐 ray BVH、虚拟球面 tile、选择性 GS splat）。
3. **P3（可选）** — 真混合表示 `M + residual 2DGS`，替换路由并优化重建精度。

详见 [docs/README.md §0.5](docs/README.md)。
