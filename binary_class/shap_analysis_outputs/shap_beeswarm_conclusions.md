# 三个 SHAP beeswarm 图的分析结论

## 图像来源

- `binary_class/shap_analysis_outputs/BM_dataset3_predmask/BM_test/LightGBM_BAG_L1_beeswarm_BM.png`
- `binary_class/shap_analysis_outputs/FTCPTC_FangDai/LightGBMXT_BAG_L1_beeswarm_FTCPTC.png`
- `binary_class/shap_analysis_outputs/LymphUs_fake_predmask/LNM_CN01/LightGBM_BAG_L1_beeswarm_LNM_CN01.png`

说明：图中红色表示特征值高，蓝色表示特征值低；SHAP 值大于 0 说明该特征推动模型输出朝“正类”方向变化，SHAP 值小于 0 则相反。

---

## 1. 甲状腺超声图像良恶性分类（BM）

### 主要结论

1. **形态特征是该任务的核心判别依据。**
   - `SHAPE: Sphericity`（球形度）和 `SHAPE: Elongation`（伸长率）位于最前，说明模型最依赖病灶外形信息来区分良恶性。
   - 从散点分布看，这两个特征的高值整体更倾向于推动模型输出正类。

2. **纹理与强度统计特征提供了重要补充信息。**
   - 重要特征包括：
     - `RLM: LRHGLE`（Long Run High Gray Level Emphasis，长游程高灰度强调）
     - `INT: Skewness`（偏度）
     - `GLCM: Corr`（Correlation，相关性）
     - `INT: Median`（中位数）
     - `TEX(GLCM): Cluster Shade`（簇阴影）
     - `RLM: SRLGLE`（Short Run Low Gray Level Emphasis，短游程低灰度强调）
     - `TEX(RLM): Run Length NUN`（Run Length Non-Uniformity，游程长度非均匀性）
     - `INT: 10th pctl`（第 10 百分位）
   - 这些特征整体上也表现出较明显的高低值分离，说明模型不仅关注形状，也关注内部灰度分布与纹理变化。

### 可归纳为一句话

- **BM 任务主要依赖病灶形态学特征，纹理和灰度统计特征起辅助作用。**

---

## 2. 甲状腺超声图像病理亚型二分类（FTC/PTC）

### 主要结论

1. **该任务最依赖纹理特征，而不是单纯形态特征。**
   - 排名前列的特征主要来自纹理描述：
     - `RLM: GLNU`（Gray Level Non-Uniformity，灰度非均匀性）
     - `TEX(GLCM): Imc2`（Informational Measure of Correlation 2，相关信息度量 2）
     - `TEX(NGTDM): Strength`（Strength，强度）
     - `RLM: RunEnt`（Run Entropy，游程熵）
     - `RLM: SRLGLE`（短游程低灰度强调）
     - `TEX(GLCM): Cluster Shade`（簇阴影）
     - `NGTDM: Coarse`（Coarseness，粗糙度）
   - 这说明 FTC 与 PTC 的差异更多体现在**内部纹理异质性、灰度分布规则性以及局部结构复杂度**上。

2. **形态特征仍然重要，但重要性低于纹理特征。**
   - `SHAPE: Sphericity`（球形度）
   - `SHAPE: Elongation`（伸长率）
   - 这些形态指标依然具有区分能力，但不是该任务最主导的证据。

3. **高值特征整体更倾向于推动正类。**
   - 例如较高的 `GLNU`、`Imc2`、`Strength`、`RunEnt`、`Cluster Shade`、`Coarse` 往往对应正 SHAP 值。

### 可归纳为一句话

- **FTC/PTC 二分类主要由纹理异质性驱动，形态特征作为补充。**

---

## 3. 甲状腺超声图像淋巴结转移二分类（LNM）

### 主要结论

1. **强度峰值和能量特征是最关键的判别依据。**
   - 最重要的特征是：
     - `INT: Maximum`（最大值）
     - `INT: Energy`（能量）
   - 高 `Maximum` 和高 `Energy` 明显更倾向于推动模型输出正类，说明转移相关病灶往往具有更突出的强回声/高能量特征。

2. **形态大小相关特征也占据重要地位。**
   - 主要包括：
     - `SHAPE: Elongation`（伸长率）
     - `SHAPE: Perim/Area`（Perimeter/Area Ratio，周长-面积比）
     - `SHAPE: Perimeter`（周长）
     - `SHAPE: Minor axis`（短轴长度）
     - `SHAPE: Pixel Surface`（像素表面）
   - 这些特征说明模型对病灶大小、边界轮廓和外形结构很敏感。

3. **纹理特征同样提供了有效补充。**
   - `TEX(GLCM): Cluster Shade`（簇阴影）
   - `TEX(GLCM): Cluster Tendency`（簇趋势）
   - `NGTDM: Contrast`（对比度）
   - 这些特征表明，转移相关病例往往伴随更复杂的局部灰度变化。

### 可归纳为一句话

- **LNM 任务主要依赖强度峰值、能量以及形态大小信息，纹理特征用于进一步增强区分。**

---

## 三个任务的横向对比

- **BM（良恶性分类）**：以**形态特征**为主，纹理和强度统计为辅。
- **FTC/PTC（二分类）**：以**纹理异质性特征**为主，形态特征次之。
- **LNM（淋巴结转移二分类）**：以**强度峰值 + 能量 + 形态大小**为主，纹理特征辅助判别。

换句话说：

- **BM：看“形状”**
- **FTC/PTC：看“纹理”**
- **LNM：看“强度和大小”**

---

## 缩写与全称对照

| 图中缩写 | 全称 |
|---|---|
| `SHAPE` | Shape features，形状特征 |
| `INT` | Intensity / First-order features，强度/一阶统计特征 |
| `TEX` | Texture features，纹理特征 |
| `GLCM` | Gray Level Co-occurrence Matrix，灰度共生矩阵 |
| `RLM` | Run Length Matrix，游程矩阵 |
| `NGTDM` | Neighboring Gray Tone Difference Matrix，相邻灰度差矩阵 |
| `LRHGLE` | Long Run High Gray Level Emphasis，长游程高灰度强调 |
| `SRLGLE` | Short Run Low Gray Level Emphasis，短游程低灰度强调 |
| `GLNU` | Gray Level Non-Uniformity，灰度非均匀性 |
| `RLNU` | Run Length Non-Uniformity，游程长度非均匀性 |
| `RunEnt` | Run Entropy，游程熵 |
| `Corr` | Correlation，相关性 |
| `Imc2` | Informational Measure of Correlation 2，相关信息度量 2 |
| `Coarse` | Coarseness，粗糙度 |
| `Perim/Area` | Perimeter/Area Ratio，周长-面积比 |
| `Maximum` | Maximum，最大值 |
| `Energy` | Energy，能量 |
| `Median` | Median，中位数 |
| `Skewness` | Skewness，偏度 |
| `Sphericity` | Sphericity，球形度 |
| `Elongation` | Elongation，伸长率 |
| `Perimeter` | Perimeter，周长 |
| `Minor axis` | Minor Axis Length，短轴长度 |
| `Cluster Shade` | Cluster Shade，簇阴影 |
| `Cluster Tendency` | Cluster Tendency，簇趋势 |
| `Contrast` | Contrast，对比度 |
| `Strength` | Strength，强度 |
| `Pixel Surface` | Pixel Surface，像素表面 |

---

## 结论

这三张 beeswarm 图共同说明：模型并不是依赖单一类型特征，而是根据不同任务自动学习到不同的判别模式。
- 良恶性分类更关注**轮廓与整体形态**；
- FTC/PTC 更关注**内部纹理与灰度组织方式**；
- 淋巴结转移分类则更关注**强度峰值、能量和结构尺度**。
