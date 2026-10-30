# Hybrid Mesh–Gaussian LiDAR Simulation

> 可迁移项目记忆（research memo + 工程约定）。换机器 / 换会话时先读本文件。
> 课程定位：GPU Programming Final Project。目标不是发论文级创新，而是在**可实现的范围内**做出一个有明确 GPU 优化叙事、可量化、可复现的 LiDAR 仿真系统。
>
> **文档版本：V2（2026-10）**。V2 执行计划以 **§0.5** 为准；§1–§12 为 V1 研究背景（文献、方法设想、数据集），仍然有效，但任务优先级与实施顺序以 §0.5 覆盖。对应 PDF：`proposal_V1.pdf`（原方案）、`proposal_V2.pdf`（本版）。

---

## 0. TL;DR

- **V2 路线（当前）**：**P1 单场景双基线** → **P2 路由级 hybrid casting + GPU 加速** → **P3（可选）真混合表示 + 重建优化**。详见 §0.5。
- **输入**：一段室外场景的多视角 RGB + 相机位姿（+ 标定外参）。
- **表示**：先得到 `mesh-only` 与 `gaussian-only`（3DGS、2DGS 各一条）；hybrid 先以**逐 ray 路由**实现（P2），P3 再升级为单一 `M + residual 2DGS`。
- **推理**：从相机光心出发，沿 LiDAR GT 点定义的方向发射射线，与重建场景求交，交点即 LiDAR 仿真点。
- **评测**：仿真点 vs. 真实 LiDAR GT 的几何误差（range / CD / F-score / recall）。
- **核心论点**：LiDAR 是 *surface-first* 传感器，几何应交给 mesh；Gaussian 只负责薄结构/非表面/高频残差。这样能把 BVH/光栅的图元数降一个数量级，从而在保真度不降的前提下显著提升推理速度。
- **主要工作量**：P1/P2 的推理侧 CUDA（逐 ray BVH + 虚拟球面 tile + 选择性 GS splat）；P3 才转向重建侧的表示与精度优化。

---

## 0.5 执行计划（V2，2026-10 更新，当前以此为准）

> §1–§12 是 V1 的完整研究备忘（背景、文献、方法设想）；V1 的 §5 任务表、§7 实验、§9 计划仍**有效**，但**优先级与顺序以本节为准**。
> V2 把"直接做表示级 hybrid"改为"先基线、再路由级 hybrid、最后（可选）表示级 hybrid"的三阶段路线：降低风险，保证每阶段都能交付可量化结果。

### 0.5.1 与 V1 的核心差异

| 维度 | V1 | V2 |
|---|---|---|
| P2 的本质 | 单一混合表示 `M + G_res`，单 kernel 融合 | **两套完整场景 + 逐 ray 路由**（adaptive dispatcher），**不是**混合表示 |
| 目标 | 直接冲表示级 hybrid | 先用路由拿到 hybrid casting 的 Pareto 与 GPU 加速，再决定是否升级 |
| 术语 | "adaptive 射线发射器" | **adaptive ray dispatcher / router**：GT 射线方向固定，自适应的是**后端分派**，不是射线方向 |
| 工期风险 | 高（依赖残差 GS 放置/联合优化） | 低（P1/P2 均有独立可交付） |

**关键认知**：P2 是"mixture-of-experts per ray"，衡量它必须报告 **fallback rate** 与 **oracle gap**；若两侧都算完再挑，就没有加速，退化为 ensemble。P2→P3 才是"推理优化 → 重建优化"的转向。

### 0.5.2 阶段 P1 — 单场景双基线 + GT-ray 仿真（不关注重建优化）

1. **选场景**：KITTI-360 一个低动态序列；用官方 pose/calib（不跑 COLMAP）；语义 mask 剔动态；剔除 ego 自身点；评测截断 0–50 m、按 20 m 分桶。
2. **重建**：
   - 训练 **3DGS** 与 **2DGS** 各一份（同场景）。
   - **gaussian-only 基线** = 3DGS、2DGS 本身（两条）。
   - **mesh-only 基线** = 由 **2DGS depth → TSDF 融合 → Marching Cubes** 得 watertight mesh（共享 2DGS 训练；3DGS-derived mesh 仅作消融，避免重复训练）。
3. **GT-ray cast（主协议不变）**：`p_C = T_CL p_L`，`d = p_C/\|p_C\|`，`r_gt = \|p_C\|`，从相机原点沿 `d` 求交；未命中记 `inf`，计入 recall / miss-rate。
   - 后端：**逐 ray BVH**。mesh 用 Möller–Trumbore；2DGS 用 **ray–disk 解析交**；3DGS 用椭球解析或 proxy。
4. **eval harness**：range MAE/RMSE/MedAE、CD(L1/L2)、F@5/20/50 cm、P/R、miss、分桶、FPS。

**产出**：三条 baseline 曲线（mesh-only / 3DGS / 2DGS）+ 正确可复现的评测脚本。
**验收**：三种 cast 都能复现合理的 GT 误差；mesh 与 gaussian 的误差差异符合预期（薄结构 gaussian 更好、平面 mesh 更好）。

### 0.5.3 阶段 P2 — naive hybrid casting（路由）+ GPU 加速

**定义**：per-ray 选择"走 mesh 场景"还是"走 gaussian 场景"，采用 **mesh-first + 廉价 fallback**，**绝不两侧都算再挑**。

- **廉价线索（严禁用 GT）**：mesh 未命中（洞）、TSDF/三角置信度低、掠射角 `|n·d|` 小、深度/法向不连续、薄结构语义（杆/线/植被）、距离超出 LOD 可信范围。
- **两种 router 都实现并对比，择优采用**：
  - **H（启发式规则）**：上述线索 + 可调阈值，无训练、可解释、主线。
  - **L（学习式）**：小 MLP，特征同 H 再加局部深度方差；**用一条序列的 GT 训练、另一条序列评测**，避免泄漏。若跨序列泛化差则弃用。
- **评测**：相对两基线的精度、**fallback rate**、**oracle gap**（GT 逐 ray 取最优作上界，仅分析用）、分派耗时占比、mesh 服务 ray 占比。
- **GPU 加速（文档中的 I 类）**：
  - **I5** 逐 ray BVH 遍历（Möller–Trumbore）——基线正确性与兜底。
  - **任意 GT 射线非规则**：把射线与三角形装箱到**虚拟球面角度 tile（如 64×2650）**，再做 **I3** tile binning + shared-memory 深度缓冲；**I1** 的 range-image 光栅化在此角度域上实现。
  - **I2** 重解释为"同一 tile 内 mesh z-buffer + **仅对 fallback ray** 做 GS splat 的选择性融合"，而非全量 GS splat。
  - **I4** 占用栅格 / DDA 空域跳过；**I7** Morton 重排 ray/tile；**E1** GPU CD/F-score/recall。
  - 规则 beam-pattern 次协议可选（用于展示原生 range-image 光栅优势），主协议仍是 GT rays。

**产出**：fallback rate–精度–FPS 的 Pareto；H vs L router 对比；融合/分派耗时拆解；oracle gap 分析。
**验收**：相对 gaussian-only 有明确加速且精度不显著退化（目标 FPS 2–4×、CD 退化 <5–10%，以实测为准）；相对 mesh-only 薄结构 recall 明显提升。

### 0.5.4 阶段 P3 —（可选）真混合表示 + 重建优化

- 用单一 `M + residual 2DGS` 场景**替换 P2 的路由**，回到 §4.1 的残差驱动放置（R3），配合 2DGS 几何（R1）、depth prior（R2）、Mesh LOD（R4）。
- **对比命题**：P2 routed vs P3 true-hybrid —— 精度、#primitives、显存、速度、fallback 消失后的收益。
- 诚实定位：P2 是**路由基线**，P3 才是**表示级 hybrid**；若时间不足，P2 自成完整成果。

### 0.5.5 技术决策记录（V2）

| 项 | 决定 |
|---|---|
| Router | 启发式 H 与学习式 L **都做**，按验证集择优，另一条作消融 |
| Gaussian 基线 | **3DGS 与 2DGS 都跑** |
| Mesh 来源 | **2DGS depth → TSDF → Marching Cubes**（共享训练；3DGS-derived 仅消融） |
| 求交后端 | **先逐 ray BVH**；GPU 优化阶段加**虚拟球面 tile**；range-image 光栅化视需要再上 |
| 评测协议 | GT-defined rays（不变） |
| 场景规模 | **单场景先行**，跑通后再扩展序列/数据集 |
| 术语 | adaptive **ray dispatcher**（非射线方向自适应） |

### 0.5.6 与 V1 任务表的映射

| V2 阶段 | 对应 V1 任务 |
|---|---|
| P1 基线 | 基础重建 + E1（先 CPU 对拍）+ I5 |
| P2 路由 + 加速 | H/L router + I1/I2/I3/I4/I5/I7 + E1 |
| P3 真 hybrid（可选） | R1/R2/R3/R4（+ 可选 I 类继续调优） |

**不做（不变）**：可微 mesh 联合优化、扩散先验补洞、动态 scene graph、BRDF/intensity 反演。

### 0.5.7 时间线（V2，8–10 周）

| 周 | 任务 | 里程碑 |
|---|---|---|
| 1 | KITTI-360 预处理 + GT-ray 协议 + eval harness（先 CPU） | 评测管线跑通 |
| 2 | 3DGS / 2DGS 训练 + TSDF mesh | 三类表示就绪 |
| 3 | 逐 ray BVH cast（mesh / 2DGS / 3DGS） | 三条 baseline 曲线 |
| 4–5 | 启发式 router + 虚拟球面 tile（I3/I5） | fallback 率–FPS 初版 |
| 6 | 选择性融合（I2）+ I4/I7 + 学习式 router | Pareto + H/L 对比 |
| 7 | 实验矩阵 + 消融 + 三张必交图 | 报告主体 |
| 8–10 | （可选）P3 真 hybrid + R1–R4、Waymo 泛化、demo | 加分项 |

**降级预案（若只有 8 周）**：砍学习式 router、I4/I7 进阶版；P3 只做"概念验证 + 一组对比"，不追 SOTA。

### 0.5.8 风险与对策（V2 专属）

| 风险 | 对策 |
|---|---|
| 路由无加速（两侧都算） | 强制 mesh-first 廉价 fallback；报告 fallback rate；oracle gap 仅作分析 |
| 任意射线 vs range-image 不兼容 | 先逐 ray BVH；再做角度域装箱；规则 beam 次协议可选 |
| 两套表示双倍显存 | P2 报告显存；P3 合并为单一 hybrid 解决 |
| 学习式 router 用 GT 泄漏 | 跨序列训练/评测；泛化差则弃用，保留启发式 |
| 3DGS ray 求交慢/不准 | 几何主基线用 2DGS；3DGS 作对照与消融 |
| RGB 深度噪声 / TSDF 空洞（影响 mesh 基线） | 2DGS depth、depth prior、confidence 加权 TSDF；必要时 residual 填洞（P3） |

---

## 1. 项目定义

### 1.1 输入 / 输出

| 项 | 内容 |
|---|---|
| 输入 | 多视角 RGB 图像、相机内参/外参、相机位姿；同一轨迹的真实 LiDAR 扫描（仅用于评测，不参与重建） |
| 中间表示 | ① mesh-only ② Gaussian-only ③ hybrid（水面/地面/建筑用 mesh，薄结构/植被/杆件用 residual 2DGS） |
| 输出 | 与 LiDAR GT 同角度、同原点语义的仿真点云 / range image |
| 评测 | 与 GT 的 range 误差、Chamfer Distance、F-score、precision/recall、分距离桶指标、FPS/显存 |

### 1.2 任务边界（重要）

- **只做静态室外场景**。动态物体（车、人）通过“选序列 + 语义 mask + 只评测静态 GT 点”绕开，不做动态 scene graph。
- **只做几何**。不要求 intensity / ray-drop / 多回波的物理建模（可作为可选加分项）。
- **重建只用 RGB**。LiDAR GT 只进评测，不进训练（保证“从 RGB 重建”的设定）。
- **距离限制 0–50m 为主**，60m 外单列，避免用远距离噪声淹没方法差异。

### 1.3 评测协议：GT-defined rays（推荐主协议）

直接使用固定 LiDAR beam pattern 会引入视差问题：LiDAR 与相机光心不重合（外参平移 0.1–0.8m），从相机原点沿 LiDAR 角度打射线会与 GT 产生系统性偏差。因此主协议用 **GT 点定义射线**：

对每个 LiDAR GT 点 `p_L`（LiDAR 系）：

```
p_C = T_CL · p_L                 # 变换到目标相机系
d   = p_C / ||p_C||              # 射线方向（相机系）
r_gt = ||p_C||                   # GT range
r_pred = raycast(o=0, d)         # 与重建场景求交
err = |r_pred - r_gt|
p_pred = T_CL^{-1} · (r_pred · d)  # 映射回 LiDAR 系，用于 CD / F-score
```

优点：
- 无视差偏差；
- 无 “GT 没回波但预测有” 的歧义；
- ray 数 = GT 点数（约 1e5/帧），天然并行，非常适合 CUDA。

约定：
- `r_pred = ∞`（未命中）记为 **miss**，不计入 range 误差，但计入 recall / miss-rate。
- 预测点与 GT 点的最近邻搜索在一个统一坐标系（建议 LiDAR 系）里做。
- 多相机时按相机 FOV 分配 GT 点，逐相机评测后再合并；重叠区去重。

**次协议（可选）**：LiDAR 原点 + 完整 beam pattern，只用于报告 coverage / no-return，不作为主指标。

### 1.4 指标

| 类别 | 指标 | 说明 |
|---|---|---|
| Range | MAE / RMSE / MedAE | 逐 ray 的 `|r_pred − r_gt|`，只统计命中 |
| 点云 | Chamfer Distance (L1/L2) | 双向最近邻均值 |
| 点云 | F-score@τ | τ = 0.05 / 0.2 / 0.5 m 都报 |
| 点云 | Precision / Recall | 预测点与 GT 点在 τ 内的匹配率；recall 反映漏检（mesh 洞、薄结构丢失） |
| 覆盖 | Miss rate | 未命中 ray 占比 |
| 分桶 | 0–20 / 20–40 / 40–60 m | 避免远距离主导均值 |
| 效率 | FPS、单帧延迟、rays/s、峰值显存、#primitives、BVH build time | 固定分辨率（如 64×2650）报告 |

> 注意：RGB 重建的几何精度通常达不到 5cm，F-score@5cm 可能很低；主表建议用 τ=0.2m，同时报 5cm 作为严格指标。

---

## 2. 核心洞察（写进报告 Introduction）

1. **表示与物理不匹配**：LiDAR 首回波是硬表面命中，3DGS 的 volumetric alpha blending 是 radiance 近似。用 mesh 表达表面更符合物理，且能提供可靠法向（intensity/入射角建模需要）。
2. **LiDAR 的 ray 很少**：64×2650 ≈ 0.17M ray，而 8 路相机是 16M pixel。**瓶颈不是 ray 数，而是每条 ray 要遍历的图元数**。
3. **现有 GS-LiDAR 的 BVH 代理很贵**：LiDAR-RT 为每个 Gaussian 建 icosahedron 代理，图元数 1e6–1e7；mesh 三角形可少 1–2 个数量级，且三角形求交比椭球便宜。
4. **hybrid 的 Pareto 优势**：平坦区域（路面、墙、建筑）用纹理 mesh，薄结构（杆、线、栏杆、植被）用少量 Gaussian。已有 camera 域证据：Hybrid Mesh-Gaussian (IJCAI'25) 把 1.667M Gaussians 降到 0.295M，FPS 122→446，PSNR 反而略升。
5. **融合渲染**：把 mesh z-buffer 与 GS splat 放进同一张球面 range image、同一个 tile，可省掉一次 global memory 往返，并天然做深度竞争。

---

## 3. 文献调研（截至 2026-10）

### 3.1 LiDAR 专用 NeRF（历史基线）

| 方法 | 年/会 | 表示 | 关键点 | 速度 |
|---|---|---|---|---|
| LiDARsim | CVPR'20 | surfel mesh + 神经 raydrop | reconstruct-then-simulate 代表；mesh 几何 + NN 学 raydrop | — |
| NFL | ICCV'23 | NeRF + 物理 LiDAR 模型 | **beam divergence / secondary return / ray-drop** 建模 | 慢 |
| LiDAR-NeRF | 2023 | Instant-NGP | 密度 + 期望距离 + intensity/drop | ≈1.8 FPS |
| LiDAR4D | CVPR'24 | 4D hybrid (multi-planar + grid) | 动态时空 NVS | 426min / 1.7 FPS |
| AlignMiF | CVPR'24 | 多模态隐式场 | LiDAR+相机几何对齐 | 110min / 0.1 FPS |
| DyNFL | 2023 | 动态 NeRF | 动态场景 NFL | 464min / 0.15 FPS |
| PBNLiDAR | ITSC'25 | NeRF + 传感器物理 | **rolling shutter / 激光功率变化 / 强度衰减** | — |

### 3.2 LiDAR 专用 Gaussian（当前 SOTA，重点对标）

| 方法 | 年/会 | 表示 | 渲染 | 数据集 | 速度 | 备注 |
|---|---|---|---|---|---|---|
| **LiDAR-GS** | arXiv:2410.05111 | 3DGS + range-view | 光栅化（laser beam splatting） | Waymo / KITTI-360 | **18min / 15 FPS** | micro cross-section 投影消除局部仿射误差；神经高斯编码视角相关 intensity/drop；Waymo CD 0.14、intensity PSNR 24.52 |
| **GS-LiDAR** | ICLR'25 | 全景 **2DGS** + ray-splat 显式求交 | 光栅化 | KITTI-360 / nuScenes | 实时 | periodic vibration 处理动态；SH 编码 intensity/ray-drop；KITTI-360 静态 CD −5.3%、depth RMSE −10.7% |
| **LiDAR-RT** | CVPR'25 | GS + **OptiX BVH 光线追踪** + scene graph | ray tracing | Waymo / KITTI-360 | **2h / 30 FPS@64×2650** | 每个 Gaussian 建代理几何做 BVH；支持插删物体、改轨迹、换传感器 |
| SplatAD | arXiv:2411.16816 | 3DGS | 光栅化 | 3 个 AD 数据集 | 比 NeRF 快一个量级 | rolling shutter + intensity + dropout |
| **SimULi** | ICLR'26 | 3DGUT + factorized Gaussian | 光栅化 + ray culling | 2 个 AD 数据集 | **比 ray tracing 快 10–20×，比 raster 快 1.5–10×** | 任意相机模型 + 旋转式 LiDAR；自动 tiling |
| **LiDAR-GS++** | AAAI'26 | neural **2DGS** + 扩散先验 | 光栅化 | Waymo / KITTI-360 | 实时 | 单趟扫描外推 CD 0.417→0.102 |
| **FGGS-LiDAR** | arXiv:2509.17390 | 3DGS→TSDF **watertight mesh** | GPU ray casting | indoor + outdoor | **>500 FPS，4096 环境并行** | 任意 3DGS 资产转 mesh；Outdoor/VLP-32 CD 0.025 / F 0.979 |

**对本项目的意义**：LiDAR-GS / GS-LiDAR 提供重建与 range-view 渲染的强基线；LiDAR-RT 提供 ray-tracing 对照；FGGS-LiDAR 证明 “先转 mesh 再 ray cast” 在速度和精度上都可行（但它**没有** hybrid，也没有用 mesh 做重建主干）。

### 3.3 Hybrid Mesh + Gaussian（方法来源）

| 方法 | 年/会 | 做法 | 效果 |
|---|---|---|---|
| **Hybrid Mesh-Gaussian** | IJCAI'25 | 纹理 mesh 管平坦高纹理区，Gaussian 管复杂几何；mesh 剪枝 + **transmittance-aware** 联合优化 | 1.667M→0.295M Gaussians，122→446 FPS |
| OMeGa | arXiv:2509.24308 | 显式三角 mesh + 2DGS 联合优化，Gaussian 绑定 mesh 坐标系 | Chamfer-L1 比 2DGS 降 47.3% |
| **CityGo** | ACM MM'25 | 航拍城市：proxy building mesh + **residual / surrounding Gaussians** | 训练 1.4× 加速，移动端实时 |
| MeshGS / Direct Mesh-Gaussian | arXiv:2405.06945 | Gaussian 以重心坐标绑定到三角形 | 显式 mesh + GS 联合模型 |
| Hybrid Rendering for AD | arXiv:2503.09464 | NeRF2GS 蒸馏 + 点云 depth 正则；相机 raster、LiDAR **ray-traced backend** | 动态 agent 任意摆放 |

### 3.4 Mesh 提取 / 表面重建工具

| 方法 | 年/会 | 特点 | 参考耗时 |
|---|---|---|---|
| SuGaR | CVPR'24 | Gaussian 对齐表面 + 提 mesh | 中 |
| 2DGS | SIGGRAPH'24 | 2D oriented disk，几何更准 | 10.9 min |
| GOF | SIGGRAPH'24 | opacity field + 自适应 marching tetrahedra | 中 |
| MILo | SIGGRAPH'25 | mesh-in-the-loop，最准 | 35 min |
| TSDF + Marching Cubes | 经典 | 最快、最稳；FGGS-LiDAR 用的路线 | 秒–分钟 |

### 3.5 综述 / 2026 前沿

- *Data-driven Camera and Lidar Simulation Models for AD: A Review* — arXiv:2402.10079
- *Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis* — CVPR'26
- *LiDAR-EVS: Enhance Extrapolated View Synthesis for 3DGS with Pseudo-LiDAR Supervision* — arXiv:2603.14763
- *PIN-SLAM* (T-RO'24, arXiv:2401.09101)、*GS-LIVM* (ICCV'25, arXiv:2410.17084)：LiDAR 侧在线建图/重建，可借鉴 SDF/TSDF 与位姿处理。

### 3.6 调研结论（gap）

1. **没有** mesh+Gaussian hybrid 的 LiDAR 仿真工作；现有 hybrid 全在相机/室内/航拍。
2. 现有 LiDAR GS 方法**全部**用 Gaussian 表达整个场景（含平坦表面），图元冗余。
3. 现有 LiDAR ray tracing 方法为每个 Gaussian 建代理 BVH，**图元数**是主要开销。
4. 目前评测协议不统一（有的用固定 beam，有的用 GT 点），本项目应显式定义并固定协议。

---

## 4. 方法设计

### 4.1 重建 pipeline（两阶段，低风险）

```
多视角 RGB + poses
      │
      ▼
[1] 3DGS / 2DGS 训练（gsplat 基线）
      │  渲染 depth（训练视角 + 少量新视角）
      ▼
[2] TSDF 融合 → Marching Cubes → watertight mesh
      │  去噪 / decimate / Laplacian 平滑
      ▼
[3] Mesh LOD 金字塔（0–50m 5cm；50–120m 中；>120m 粗）
      │
      ├──► 逐 ray 计算 mesh-only 误差 e(r)
      │
      ▼
[4] 残差驱动 Gaussian 放置：e(r) > τ 或语义=植被/杆/线 → 生成 2DGS
      │  只优化 G_res（mesh 冻结），预算 = 全量 GS 的 5–20%
      ▼
[5]（可选）transmittance-aware 联合微调
      │
      ▼
Hybrid scene = (M, G_res)
```

关键选择：
- **不做端到端可微 mesh 联合优化**（课程项目性价比低）。用 “GS 训练 → 提 mesh → 拟合残差” 的后处理式 hybrid，仍然满足 “hybrid reconstruction”。
- 若 mesh 洞多：用 confidence 加权 TSDF + 少量 Gaussian 填洞；必要时加单目深度先验（Depth Anything V2）做 depth 正则。
- 训练 loss：depth L1 + median、normal/Laplacian 平滑、SSIM/PSNR、（可选）depth prior。
- **尺度**：优先用数据集提供的米制位姿，避免 COLMAP 尺度对齐。

### 4.2 表示细节

**M（mesh 主干）**
- 三角网格 + 每面/每 texel：reflectance（可选）、semantic class、confidence。
- LOD 由屏幕空间/距离阈值选择；远处用 impostor 或粗 mesh。

**G_res（残差 2DGS）**
- 2D oriented Gaussian（surfel），不是 3D 椭球。
- 两类：
  - *surface-attached*：重心坐标绑定到 mesh 面片，随 LOD/编辑移动（SuGaR/OMeGa 思路）。
  - *free*：植被、栅栏、电线、玻璃等非表面/多层结构。
- 参数：μ, R, s, opacity；不需要 SH（纯几何评测）。

**动态**：不做。用语义 mask 从训练图像 loss 与评测 GT 中剔除。

### 4.3 推理 pipeline

**Backend A（主线）：球面 range-image 光栅化**

```
GT 点 → 相机系 ray (o=0, d, r_gt)
      │
      ▼
[1] 三角形按球面 tile binning（每 tile 维护三角形列表）
      │
      ▼
[2] shared-memory z-buffer：atomicMin(packed(depth, tri_id, bary))
      │
      ▼
[3] residual G_res 在同一 tile 内 splat + 深度测试
      │  （必要时 alpha 合成，模拟薄结构部分遮挡）
      ▼
[4] 每像素 resolve → r_pred
      │
      ▼
[5] p_pred = T_CL^{-1}(r_pred · d) → 与 GT 比较
```

成本：`O(#pixels + #visible_G_res)`，与场景面积无关；用 LOD 后与距离近似无关。

**Backend B（对照）：混合 BVH 光线追踪**

- 对 LOD 选中的三角形建 LBVH（Morton + CUB radix sort）。
- per-thread 栈式遍历 + Möller–Trumbore 求交。
- residual Gaussian 只在其所在体素建小 BVH，沿 ray 的剩余线段做 alpha 合成。
- 支持任意传感器配置、二次回波；用于证明 “range raster 更适合 LiDAR”。

### 4.4 复杂度对比

```
LiDAR-GS / SplatAD : O(HW) + O(N_gauss_visible)        N_g ~ 1e6–1e7
LiDAR-RT           : O(N_ray · log N_proxy)            N_proxy ~ N_gauss
Hybrid (Ours)      : O(HW) + O(N_g,res_visible)        N_g,res ~ 1e5, N_tri ~ 1e5–1e6
```

### 4.5 参考伪代码

```python
# reconstruction
gs = train_3dgs(images, poses, iters=30k)
depth = render_depth(gs, train_poses)
mesh = tsdf_marching_cubes(depth, poses, voxel=0.05)
mesh = decimate_and_lod(mesh)
err = mesh_only_error(mesh, train_rays)          # per-ray residual
seeds = sample_where(err > tau or semantic in THIN_CLASSES)
gres = fit_residual_2dgs(seeds, images, poses, freeze=mesh, budget=0.1 * len(gs))

# inference
rays = gt_points_to_camera_rays(gt, T_CL)
depth_img, tri_id, bary = spherical_raster(mesh, rays)
depth_img = splat_residual(depth_img, gres, rays)
pred = range_to_points(depth_img, rays, T_CL)
metrics = evaluate(pred, gt)
```

---

## 5. 待实现优化任务（按性价比排序）

难度 / 价值：1–5。**Part**：R = 重建，I = 推理，E = 评测。

| ID | Part | 任务 | 难度 | 价值 | GPU | 参考 |
|---|---|---|---|---|---|---|
| I1 | I | 球面 range-image mesh 光栅化（z-buffer） | 2 | 5 | 高 | LiDAR-GS、GS-LiDAR |
| I2 | I | 单 kernel 融合 mesh z-buffer + residual GS splat | 4 | 5 | 高 | Hybrid Mesh-Gaussian、CityGo |
| I3 | I | Tile binning + shared-memory 深度缓冲 | 3 | 4 | 高 | 3DGS rasterizer |
| I4 | I | 占用栅格 DDA 空域跳过 | 2 | 3 | 高 | 经典 |
| I5 | I | 逐线程 LBVH 遍历（Möller–Trumbore） | 2 | 3 | 高 | LiDAR-RT（对照后端） |
| I6 | I | Warp-cooperative / persistent-thread BVH（进阶） | 4 | 4 | 高 | 经典 |
| I7 | I | Morton 重排 ray / tile 排序 | 3 | 3 | 高 | 经典 |
| I8 | I | 自适应 beam-divergence 子采样 | 3 | 3 | 中 | NFL |
| I10 | I | 解析 no-return 预测 | 1 | 3 | 低 | NFL、LiDARsim |
| I11 | I | 多 scan / 多环境 batch | 2 | 3 | 中 | FGGS-LiDAR |
| I13 | E | GPU CD / F-score / recall（栅格哈希） | 2 | 4 | 高 | 经典 |
| R1 | R | 2DGS / surfel 替代 3DGS 提升几何 | 2 | 4 | 中 | 2DGS、GOF |
| R2 | R | 单目深度/法向先验正则 | 2 | 4 | 中 | Depth Anything V2 |
| R3 | R | 残差驱动 Gaussian 增密/剪枝 + 预算 | 3 | 5 | 中 | Hybrid Mesh-Gaussian、CityGo |
| R4 | R | Mesh LOD + 自适应细分 | 3 | 4 | 中 | — |
| R5 | R | GPU TSDF 融合 + 并行 Marching Cubes | 3 | 3 | 高 | — |
| R6 | R | 并行 LBVH 构建（Morton + radix sort） | 3 | 3 | 高 | — |
| R7 | R | 混合精度 / fused optimizer | 2 | 2 | 中 | — |

**不做**：可微 mesh 联合优化、扩散先验补洞、动态 scene graph、BRDF/intensity 反演。

**建议投入**：I1→I3→I2→I4→I13 占 60–70%；R1/R2→R3→R4 占 20–30%；I8/I10 占 10%。

---

## 6. 数据集

| 数据集 | 相机 | LiDAR | 动态 | 位姿 | 语义 | 推荐度 |
|---|---|---|---|---|---|---|
| **KITTI-360** | 4 针孔 + 2 鱼眼 | HDL-64E | 中，可按序列筛 | 提供（米制） | 2D/3D | ★★★★★ |
| **Waymo Open** | 5 | 64/32 线 | 高 | 提供 | 2D/3D | ★★★☆（筛静止帧 + mask） |
| **Boreas** | 6 | 128 线 | 低–中（重复路线/四季） | 提供 | 部分 | ★★★★ |
| nuScenes | 6 | 32 线 | 高 | 提供 | 2D/3D | ★★★（32 线泛化） |
| Argoverse 2 | 7 | 2×32 | 高 | 提供 | 2D/3D | ★★★ |
| Newer College | 少 | 多种 | 低（手持校园） | 提供 | 无 | ★★★（相机覆盖稀疏，慎用） |
| ZOD | 1 | 1 | 中 | 提供 | 部分 | ★★ |
| M2DGR | 多 | 多 | 低（校园机器人） | 提供 | 无 | ★★★ |

**主推 KITTI-360**：米制位姿（免 COLMAP/免 Umeyama）、4 路相机 + 64 线、语义标注可做动态 mask、且 LiDAR-GS / GS-LiDAR 都用它，baseline 数字可直接对标。

### 6.1 KITTI-360 预处理协议

1. 选 2–3 个低动态序列（居民区/郊区优先）。
2. 用提供的 pose + 标定，不跑 COLMAP。
3. 训练：动态类语义 mask 从 loss 中剔除（或选无动态帧）。
4. 评测：只保留静态 GT 点；去掉 ego 车辆自身点（自遮挡）。
5. 距离截断 0–50m，按 20m 分桶。
6. 多相机按 FOV 分配 GT 点；重叠区去重。

### 6.2 目录建议

```
/data-20T/hybrid-lidar-sim/
├── raw/kitti360/...
├── processed/kitti360/seq_0000/
│   ├── images/  poses.json  calib.json
│   ├── lidar/   lidar_static.npz   # 过滤后 GT
│   └── masks/   dynamic/
└── runs/        # 训练输出、mesh、G_res
```

---

## 7. 实验设计

**Baselines**：mesh-only、Gaussian-only（3DGS / 2DGS）、hybrid；外部对照 LiDAR-GS、GS-LiDAR、LiDAR-RT、FGGS-LiDAR（能用官方权重/代码就跑）。

**主矩阵**：3 表示 × 2 后端（raster / BVH-RT）× 2 数据集 × 3 距离桶。

**消融**：
1. Gaussian 预算 5 / 10 / 20 / 40% → fidelity–FPS Pareto；
2. 放置策略：随机 vs 残差驱动 vs 语义驱动；
3. 融合 kernel vs 串行 kernel；
4. LOD on/off；占用栅格 on/off；FP16 on/off；
5. 3DGS vs 2DGS；depth prior on/off；
6. 多相机合并 vs 逐相机。

**必交三张图**：
1. CD vs FPS 的 Pareto（mesh / GS / hybrid 三条线）；
2. 推理耗时拆解（ray gen / mesh / GS / resolve）与融合前后对比；
3. 分距离桶 range RMSE + recall（证明不是靠多打点刷分）。

**预期目标（量级估计，需实测）**：
- hybrid 相对 GS-only：FPS 2–4×，CD 退化 < 5–10%；
- hybrid 相对 mesh-only：薄结构区域 recall 明显提升；
- 融合 kernel 相对串行：1.3–2×；
- 占用栅格 + LOD：再 1.2–2×。

---

## 8. 工程结构建议

```
hybrid-lidar-sim/
├── README.md
├── docs/
│   ├── proposal_V1.tex / .pdf   # V1 原始方案（旧）
│   ├── proposal_V2.tex / .pdf   # V2 当前方案
│   └── figures/
├── configs/
│   ├── kitti360.yaml
│   └── waymo.yaml
├── src/
│   ├── recon/
│   │   ├── train_gs.py        # 基于 gsplat
│   │   ├── render_depth.py
│   │   ├── tsdf_mesh.py
│   │   ├── lod.py
│   │   └── residual_gs.py
│   ├── infer/
│   │   ├── rays.py
│   │   ├── mesh_raster.cu     # I1/I3
│   │   ├── gs_splat.cu        # GS tile splat
│   │   ├── fused_raster.cu    # I2
│   │   ├── bvh.cu             # I5/I6
│   │   └── pipeline.py
│   ├── eval/
│   │   ├── metrics.cu         # I13
│   │   └── report.py
│   └── common/
│       ├── camera.py
│       ├── lidar.py
│       └── mesh.py
├── scripts/
│   ├── prepare_kitti360.sh
│   ├── run_baseline.sh
│   └── run_all.sh
└── results/
```

### 8.1 CUDA 实现要点

- **级联求交**：先用球面 tile / 占用栅格粗筛，再三角形精求交。
- **深度打包**：`uint64 = (float_bits(depth) << 32) | tri_id`，一次 `atomicMin` 同时解决深度与索引。
- **共享内存**：每 tile 一份 z-buffer + 三角形索引列表；tile 大小按 16×16 / 32×8 调优。
- **排序**：Morton 码 + `cub::DeviceRadixSort` 做 ray/triangle binning。
- **访存**：`__ldg`、`float4` 向量化、结构体数组改数组结构体（SoA）。
- **负载均衡**：persistent threads + 全局工作队列，避免 ray 命中率差异导致 warp 空转。
- **性能分析**：Nsight Compute 看 occupancy、shared memory bank conflict、L2 命中；Nsight Systems 看 kernel 间空隙。
- **精度**：FP32 深度、FP16 顶点/法向；注意 range image 的球面非线性，三角形需在球面空间裁剪。

### 8.2 依赖

- Python 3.11、PyTorch 2.x、CUDA 12.x
- `gsplat`（CUDA 光栅化基线，易改）
- Open3D（TSDF、mesh IO；仅离线用）
- CUB / Thrust（排序、扫描）
- 可选：OptiX（仅作对照，不强制）

---

## 9. 计划（8–10 周）

> **V1 原时间线，保留参考；当前请以 §0.5.7 的 V2 时间线为准。**

| 周 | 任务 | 里程碑 |
|---|---|---|
| 1 | KITTI-360 预处理、GT-ray 协议、评测脚本 | eval harness 跑通 |
| 2 | 3DGS 基线 + TSDF mesh-only 基线 | 两条 baseline 曲线 |
| 3–4 | I1 + I3：球面 mesh 光栅化 | mesh 渲染 FPS/正确性 |
| 5 | R1/R2 + R3：几何改善 + 残差 Gaussian | hybrid CD 提升 |
| 6 | I2：融合 kernel | 融合 vs 串行加速比 |
| 7 | I4 + I7：占用栅格、Morton | 再挤 1.2–2× |
| 8 | 实验矩阵 + Pareto + 消融 | 报告主体 |
| 9–10 | 可选：I8/I10、Waymo 泛化、demo | 加分项 |

**降级预案（若只有 8 周）**：砍 I6/I8/I11，R5/R6 只做“能用即可”。

---

## 10. 风险与对策

| 风险 | 对策 |
|---|---|
| RGB 重建深度噪声大、TSDF 有洞 | 2DGS/surfel、depth prior、confidence 加权 TSDF、residual GS 填洞 |
| 动态物体污染 | 选序列 + 语义 mask + 只评测静态 GT |
| 尺度/位姿不一致 | 用数据集提供的米制 pose；若必须 COLMAP，则 Umeyama 对齐 |
| 球面投影下三角形光栅化误差 | range-space 裁剪；退化时回落到 BVH 后端 |
| 远处几何不可靠 | 0–50m 主评测；分桶报告 |
| 公平性 | 固定 ray 集合、GT、分辨率、硬件；报告多次运行均值 |
| 显存 | LOD、FP16、分块流式、G_res 预算 |
| 融合 kernel 同步复杂 | 先做两 kernel 串行版本，再逐步融合（保留可回退开关） |

---

## 11. 待确认

1. **评测协议**：确认采用 “GT 点定义射线 + 相机原点 + 只比 range/3D 距离”；还是作业要求固定 beam pattern、从 LiDAR 原点发射。
2. **算力/周期**：单卡 24GB？8 周还是 16 周？决定是否保留 I6/I8/I11。
3. **是否需要 intensity / ray-drop**：当前按“纯几何”设计；若需要，优先做解析模型（I10），不做学习式 BRDF。

---

## 12. 参考文献

**LiDAR 仿真 / NVS**
1. Manivasagam et al. *LiDARsim*. CVPR 2020. arXiv:2006.09348
2. Huang et al. *Neural LiDAR Fields for Novel View Synthesis (NFL)*. ICCV 2023. arXiv:2305.01643
3. Tao et al. *LiDAR-NeRF*. arXiv:2304.10406
4. Zheng et al. *LiDAR4D*. CVPR 2024. arXiv:2404.02742
5. Tao et al. *AlignMiF*. CVPR 2024. arXiv:2402.17483
6. Marcus & Stamminger. *Physically Based Neural LiDAR Resimulation*. ITSC 2025. arXiv:2507.12489

**LiDAR Gaussian**
7. Chen et al. *LiDAR-GS*. arXiv:2410.05111
8. Jiang et al. *GS-LiDAR*. ICLR 2025. arXiv:2501.13971
9. Zhou et al. *LiDAR-RT*. CVPR 2025. arXiv:2412.15199
10. Hess et al. *SplatAD*. arXiv:2411.16816
11. Turki et al. *SimULi*. ICLR 2026. arXiv:2510.12901
12. Chen et al. *LiDAR-GS++*. AAAI 2026. arXiv:2511.12304
13. Wu et al. *FGGS-LiDAR*. arXiv:2509.17390

**Hybrid Mesh–Gaussian**
14. Huang et al. *Hybrid Mesh-Gaussian Representation*. IJCAI 2025. arXiv:2506.06988
15. Cao et al. *OMeGa*. arXiv:2509.24308
16. Liu et al. *CityGo*. ACM MM 2025. arXiv:2505.21041
17. *Direct and Adaptable Mesh-Gaussian Scene Reconstruction*. arXiv:2405.06945
18. Tóth et al. *Hybrid Rendering for Multimodal AD*. arXiv:2503.09464

**表面重建 / 工具**
19. Kerbl et al. *3D Gaussian Splatting*. SIGGRAPH 2023. arXiv:2308.04079
20. Huang et al. *2D Gaussian Splatting*. SIGGRAPH 2024. arXiv:2403.17888
21. Guédon & Lepetit. *SuGaR*. CVPR 2024. arXiv:2311.12775
22. Yu et al. *Gaussian Opacity Fields*. SIGGRAPH 2024. arXiv:2404.10772
23. Guédon et al. *MILo*. SIGGRAPH 2025. arXiv:2506.24096

**LiDAR 建图 / 综述**
24. Pan et al. *PIN-SLAM*. T-RO 2024. arXiv:2401.09101
25. Xie et al. *GS-LIVM*. ICCV 2025. arXiv:2410.17084
26. *Data-driven Camera and Lidar Simulation Models for AD: A Review*. arXiv:2402.10079
27. *LiDAR-EVS*. arXiv:2603.14763

**数据集**
28. Liao et al. *KITTI-360*. TPAMI 2022.
29. Sun et al. *Waymo Open Dataset*. CVPR 2020.
30. Caesar et al. *nuScenes*. CVPR 2020.
31. Wilson et al. *Argoverse 2*. NeurIPS 2023.
32. Burnett et al. *Boreas*. IJRR 2023.
33. Ramezani et al. *Newer College Dataset*. 2020.
