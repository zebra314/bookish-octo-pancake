---
marp: true
theme: default
paginate: true
style: |
  /* ── Base: dark background ── */
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 22px;
    background: #0d1117;
    color: #e6edf3;
    padding: 40px 60px;
  }
  h1 {
    color: #58c4dc;
    font-size: 36px;
    border-bottom: 2px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 18px;
  }
  h2 {
    color: #79c0ff;
    font-size: 28px;
    margin-top: 0;
  }
  h3 {
    color: #58c4dc;
    font-size: 22px;
    margin-bottom: 6px;
  }
  /* ── Cover slide ── */
  section.cover {
    background: radial-gradient(ellipse at 30% 40%, #0c2340 0%, #050d1a 55%, #0d1117 100%);
    color: #e6edf3;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.cover h1 {
    font-size: 40px;
    border: none;
    color: #58c4dc;
    text-shadow: 0 0 24px rgba(88,196,220,0.4);
  }
  section.cover h2 {
    color: #79c0ff;
    font-size: 20px;
    font-weight: 400;
    border: none;
    margin-top: 4px;
  }
  section.cover p {
    color: #8b949e;
  }
  /* ── Tables ── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 17px;
  }
  th {
    background: #161b22;
    color: #58c4dc;
    padding: 8px 12px;
    border-bottom: 2px solid #30363d;
  }
  td {
    padding: 6px 12px;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
  }
  tr:nth-child(even) td { background: #161b22; }
  /* ── Callout boxes ── */
  .highlight {
    background: #1c2a1a;
    border-left: 4px solid #3fb950;
    padding: 10px 16px;
    border-radius: 4px;
    margin: 10px 0;
    color: #aff5b4;
  }
  .box {
    background: #0c2340;
    border-left: 4px solid #58c4dc;
    padding: 10px 16px;
    border-radius: 4px;
    margin: 8px 0;
    color: #cae8ff;
  }
  .redbox {
    background: #2a0e14;
    border-left: 4px solid #f85149;
    padding: 10px 16px;
    border-radius: 4px;
    margin: 8px 0;
    color: #ffa198;
  }
  /* ── Code blocks ── */
  pre {
    background: #161b22 !important;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 15px;
    color: #c9d1d9;
  }
  code { color: #79c0ff; }
  /* ── Layout ── */
  ul { margin: 8px 0; padding-left: 24px; }
  li { margin-bottom: 6px; line-height: 1.5; color: #c9d1d9; }
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
  footer { font-size: 14px; color: #484f58; }
  /* ── Section divider ── */
  section.section-divider {
    background: radial-gradient(ellipse at center, #0c2340 0%, #0d1117 70%);
    color: #58c4dc;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  section.section-divider h1 {
    color: #58c4dc;
    border: none;
    font-size: 44px;
    text-shadow: 0 0 30px rgba(88,196,220,0.5);
  }
  section.section-divider p {
    color: #8b949e;
    font-size: 20px;
  }
  /* Paginate color */
  section::after { color: #484f58; }
---

<!-- _class: cover -->

# Differentially Private Relational Learning with Entity-level Privacy Guarantees

## NeurIPS 2025

<br>

**Yinan Huang\*, Haoteng Yin\*, Eli Chien, Rongzhe Wei, Pan Li**

Georgia Institute of Technology · Purdue University

<br>

> \* Equal contributions

---

# 目錄

1. **研究動機與問題背景**
2. **核心挑戰：高敏感度 & 耦合採樣**
3. **貢獻一：自適應梯度裁剪（FREQ-CLIP）**
4. **貢獻二：耦合採樣下的隱私放大分析**
5. **完整演算法（Algorithm 3）**
6. **實驗結果**
7. **延伸方向與可改進之處**

---

<!-- _class: section-divider -->

# 01
# 研究動機

---

# 為什麼需要 Entity-level DP？

<div class="two-col">
<div>

### 應用場景
- **醫療**：病患診斷紀錄 + 治療關係
- **金融**：交易網路、異常偵測
- **推薦系統**：使用者-物品互動圖
- **知識圖譜**：多關係實體預測

### 隱私風險
- ML 模型容易洩漏訓練資料
- **Membership Inference** 攻擊
- 模型記憶（Memorization）問題

</div>
<div>

### 什麼是 Relational Learning？

> 透過觀測到的邊（關係）訓練節點屬性編碼器，使模型能對**未見過的實體**做零樣本關係預測

```
節點 u, v → 屬性編碼器 fΘ
hu = fΘ(xu), hv = fΘ(xv)
score(u,v) → 預測邊存在機率
```

訓練完成後可遷移至**新的實體集合**（零樣本場景）

</div>
</div>

---

# DP-SGD 標準做法

<div class="box">

**DP-SGD** 透過兩個機制保護隱私：
1. **Per-sample 梯度裁剪**：控制單一樣本對梯度的影響上限（sensitivity）
2. **Gaussian 雜訊注入**：遮蔽個別貢獻

</div>

### 成功條件（標準設定）

| 條件 | 標準 DP-SGD | Relational Learning |
|------|------------|---------------------|
| 資料結構 | 獨立樣本集合 | 圖（節點 + 邊） |
| 梯度分解 | 每樣本獨立 | **單一節點影響多條邊** |
| 採樣方式 | 單步隨機子採樣 | **多階段耦合採樣** |
| 隱私放大 | ✅ 直接適用 | ❌ 無法直接套用 |

---

<!-- _class: section-divider -->

# 02
# 核心挑戰

---

# 挑戰一：高敏感度（High Sensitivity）

### 問題根源

在 Relational Learning 中，Mini-batch $\mathcal{B} = \{T_1, ..., T_b\}$，其中每個 edge tuple：

$$T_i = (e_i^+, e_{i,1}^-, ..., e_{i,k_\text{neg}}^-)$$

<div class="redbox">

**一個節點可同時出現在多個 $T_i$ 中** → 影響多個梯度項 → 敏感度暴增

</div>

### 敏感度分析（移除節點 $u^*$）

$$\|g(\mathcal{B}) - g(\mathcal{B}')\| \leq \underbrace{\sum_{T_i \in \mathcal{B}^+(u^*)} \|g(T_i)\|}_{\text{正邊貢獻}} + \underbrace{\sum_{T_i \in \mathcal{B}^-(u^*)} \|g(T_i)\|}_{\text{負邊貢獻}} + \underbrace{\sum_{T_i' \in \mathcal{B}_-'(u^*)} \|g(T_i')\|}_{\text{替換後貢獻}}$$

**最壞情況**：標準裁剪導致全域敏感度高達 $2 \cdot |\mathcal{B}| \cdot C$

---

# 挑戰二：耦合採樣（Coupled Sampling）

<div class="two-col">
<div>

### 標準採樣（可放大隱私）
```
Mini-batch ← 從獨立樣本集 D 
             隨機子採樣
```
隱私放大：子採樣使得有效隱私預算變小 ✅

</div>
<div>

### Relational Learning 採樣
```
Step 1: 正邊採樣 E⁺ ← Poisson(E)
Step 2: 負邊採樣 E⁻ ← f(E⁺, V)
         (依賴 Step 1 的結果)
```
兩個採樣步驟相互依賴 → **標準隱私放大分析失效** ❌

</div>
</div>

<div class="highlight">

**本文核心問題**：如何在這兩個挑戰同時存在的情況下，為 Relational Learning 建立嚴格的 entity-level DP 保證？

</div>

---

<!-- _class: section-divider -->

# 03
# 貢獻一
# 自適應梯度裁剪

---

# FREQ-CLIP：頻率感知自適應裁剪

### 核心觀察

<div class="box">

不同節點在 Mini-batch 中的出現頻率（敏感度）不同 → 應針對每個邊 tuple **動態調整**裁剪閾值

</div>

### Algorithm 1：FREQ-CLIP

```
對每個節點 v，計算在 batch 中的出現頻率 freq(v)
對每個 tuple Ti：
  max-freq(Ti) ← max_{v ∈ Ti} freq(v)
  g̃(Ti) ← g(Ti) / max{1, 2·max-freq(Ti)·‖g(Ti)‖/C}
輸出：g̃ = Σᵢ g̃(Ti)   ← 敏感度上界為 C
```

---

# 自適應裁剪的敏感度分析

### Proposition 4.1

採用 FREQ-CLIP 後，對任意相鄰 Mini-batch $\mathcal{B} \sim \mathcal{B}'$（差異在節點 $u^*$）：

$$\|\bar{g}(\mathcal{B}) - \bar{g}(\mathcal{B}')\| \leq \big(1 + |\mathcal{B}^-(u^*)\big|) \cdot C$$

<div class="two-col">
<div>

### 關鍵設計
**限制 $|\mathcal{B}^-(u)| \leq 1$**（透過負採樣演算法設計）

→ 全域敏感度被常數 $2C$ 界定！

</div>
<div>

### 對比標準裁剪
| 方法 | 全域敏感度 |
|------|-----------|
| 標準裁剪 | $2 \cdot \|\mathcal{B}\| \cdot C$（無界） |
| **FREQ-CLIP** | $2C$（常數！） |

</div>
</div>

---

# 自適應裁剪的附加益處

<div class="highlight">

**高度節點的梯度被降權**，但這在 Relational Learning 中實際上有益：

</div>

- 正邊以均勻方式採樣 → 高度節點出現更頻繁
- 高度節點已有充分訓練訊號，降權有助於防止過擬合
- 改善對低度節點的泛化能力
- **實驗驗證**：自適應裁剪在 utility-privacy trade-off 上優於標準裁剪

> 與非圖領域的自適應裁剪（如基於梯度分位數）不同，本方法基於**節點出現次數**做正規化，針對 relational learning 的獨特挑戰設計

---

<!-- _class: section-divider -->

# 04
# 貢獻二
# 耦合採樣的隱私放大

---

# 耦合採樣（Coupled Sampling）形式化

### Definition 4.1

給定複合資料集 $D = (D^{(1)}, D^{(2)})$，耦合採樣包含兩步驟：

1. $B^{(1)} \sim p(\cdot | D^{(1)})$（正邊採樣）
2. $B^{(2)} \sim q(\cdot | D^{(2)}, B^{(1)})$（負邊採樣，**依賴** $B^{(1)}$）

**耦合性（Coupling）** = $B^{(2)}$ 的分布依賴 $B^{(1)}$ 的**具體內容**

### 可處理的子類：基數相依採樣（Cardinality-dependent）

$$q(B^{(2)} | D^{(2)}, B^{(1)}) = q(B^{(2)} | D^{(2)}, |B^{(1)}|)$$

**只依賴樣本數量，不依賴具體內容** → 可進行分析！

---

# Algorithm 2：負採樣設計（NEG-SAMPLE-WOR）

<div class="box">

**設計目標**：同時滿足以下兩個關鍵性質

1. 解耦負採樣至**基數相依**（支援隱私放大分析）
2. 確保任意節點 $u$ 滿足 $|\mathcal{B}^-(u)| \leq 1$（控制敏感度）

</div>

```
輸入：正邊集合 E⁺，每正邊負樣本數 k_neg，節點集合 V

1. 從 V 中不重複抽取 b·k_neg 個節點 {v₁,₁, ..., vb,k_neg}
2. 對每個 vi,j：隨機選取 e⁺ᵢ 的一端 w，令 e⁻ᵢ,ⱼ = (w, vi,j)
3. 返回 B = {T₁, T₂, ..., Tb}
```

> 注意：配對過程可能產生「假負邊」（實際上是正邊），但在真實稀疏圖中發生機率極低

---

# Theorem 4.1：耦合採樣的隱私放大界

採用 Poisson 正採樣 + WOR 負採樣，Gaussian 機制達到 $(\alpha, \varepsilon(\alpha))$-RDP：

$$\varepsilon(\alpha) = \frac{1}{\alpha-1} \log \mathbb{E}_{\ell \sim \text{Bin}(m,\gamma)} \Psi_\alpha\!\left((1-\Gamma_\ell)\mathcal{N}(0,\sigma^2) + \Gamma_\ell \mathcal{N}(1,\sigma^2) \,\|\, \mathcal{N}(0,\sigma^2)\right)$$

其中有效採樣率：
$$\Gamma_\ell = 1 - (1-\gamma)^K \left(1 - \frac{\ell \cdot k_\text{neg}}{n}\right)$$

<div class="highlight">

**關鍵洞見**：有效採樣率 $\Gamma_\ell$ 同時考慮了**正邊採樣率** $\gamma$、**最大節點度** $K$，以及**負採樣比例** $\ell \cdot k_\text{neg}/n$

</div>

---

<!-- _class: section-divider -->

# 05
# 完整演算法

---

# Algorithm 3：DP-SGD for Relational Learning

```
輸入：編碼器 fΘ, 圖 G=(V,E,X), 損失函數 L, 最大節點度 K,
      學習率 ηt, batch size b, k_neg, 裁剪閾值 C, 雜訊倍率 σ

初始化：隨機刪邊使最大節點度 ≤ K；令 γ ← b/|Ē|

for t = 1 to T:
  E⁺ ← Poisson 採樣，每條正邊以機率 γ 獨立入選
  Bt ← NEG-SAMPLE-WOR(E⁺, k_neg, V)        # 負採樣
  gt(Ti) ← ∂L(Θt, Ti)/∂Θt                   # 計算梯度
  g̃t ← FREQ-CLIP(Bt, {gt(Ti)}, C)           # 自適應裁剪
  g̃t ← (1/b)(g̃t + N(0, σ²C²I))            # 注入雜訊
  Θt+1 ← Θt − ηt·g̃t                        # 更新參數
```

**Corollary 4.1**：Algorithm 3 達到 $(\alpha, \varepsilon(\alpha))$-RDP，$\varepsilon(\alpha)$ 由 Theorem 4.1 定義

---

<!-- _class: section-divider -->

# 06
# 實驗結果

---

# 實驗設定

<div class="two-col">
<div>

### 資料集（文字屬性圖）
| 資料集 | 節點數 | 邊數 |
|--------|--------|------|
| AMAZ-Cloth | 960,613 | 4,626,125 |
| AMAZ-Sports | 357,936 | 2,024,691 |
| MAG-USA | 132,558 | 702,482 |
| MAG-CHN | 101,952 | 285,991 |

跨子領域評估：訓練在一個子圖，測試在另一個

</div>
<div>

### 模型與設定
- **編碼器**：BERT-base, BERT-large, Llama2-7B
- **任務**：零樣本關係預測（Link Prediction）
- **指標**：PREC@1, MRR
- **隱私預算**：$\varepsilon \in \{4, 10\}$，$\delta = 1/|E_\text{train}|$
- 最大節點度 $K = 5$（度數截斷）

### Baselines
1. Base model（不微調）
2. Non-private 微調（$\varepsilon = \infty$）
3. 標準裁剪 DP-SGD

</div>
</div>

---

# 主要實驗結果（Table 1 節錄）

| 方法 | MAG CHN→USA MRR | MAG USA→CHN MRR | AMAZ S→C MRR | AMAZ C→S MRR |
|------|:-:|:-:|:-:|:-:|
| Base（無微調）| 9.94 | 12.69 | 22.41 | 14.04 |
| Non-private | 39.11 | 53.91 | 47.07 | 39.61 |
| 標準裁剪 ε=10 | 21.80 | 40.29 | 33.31 | 29.59 |
| **Ours ε=10** | **27.51** | **42.01** | **37.80** | **32.71** |
| 標準裁剪 ε=4 | 20.05 | 38.53 | 30.63 | 27.10 |
| **Ours ε=4** | **25.07** | **39.86** | **36.03** | **31.52** |

<div class="highlight">

✅ 在所有資料集上，相同隱私預算下 **Ours 均優於標準裁剪**，且大幅超越不微調的 base model

</div>

---

# 隱私界的量化比較

<div class="two-col">
<div>

### 三種方法的比較
1. **Naive bound**：Gaussian RDP 界 $\varepsilon(\alpha) = \alpha/\sigma^2$（無放大）
2. **標準裁剪 + 耦合採樣放大**
3. **Ours：自適應裁剪 + 耦合採樣放大**

</div>
<div>

### 觀察結果
- 我們的方法 RDP 界**遠小於** Naive bound
- 自適應裁剪比標準裁剪有更緊的界
- 在高 $K$（高節點度）時差異更顯著
- 更緊的隱私界 → 相同 $\varepsilon$ 預算下可使用更少雜訊 → 更好的模型效用

</div>
</div>

<div class="box">

**核心結論**：本方法在隱私界分析和實際效用上均有顯著改進，兩者相互印證

</div>

---

# 消融實驗

<div class="two-col">
<div>

### 負樣本數 $k_\text{neg}$ 的影響
- 增加 $k_\text{neg}$ → **模型效用提升**
- 但同時影響有效採樣率（隱私界輕微增加）
- 實踐建議：$k_\text{neg} = 4$ 為好的平衡點

</div>
<div>

### 節點度截斷 $K$ 的影響
- 增加 $K$ → 效用提升
- 但更大的 $K$ 使隱私界更鬆
- **有趣發現**：帶截斷的 non-private 微調表現**優於**不截斷！（防止高度節點過擬合）

</div>
</div>

### 雜訊倍率 $\sigma$ 的 Privacy-Utility Trade-off

隨 $\sigma$ 增大，$\varepsilon$ 減小但效用也下降——實驗清楚展示此消長關係，在 $\sigma \approx 0.3$–$0.5$ 之間為較佳選擇

---

<!-- _class: section-divider -->

# 07
# 延伸方向與可改進之處

---

# 現有限制

<div class="redbox">

**隱私放大界的緊緻性（Tightness）尚未探索**

Theorem 4.1 是否為緊界，或仍有進一步改進空間，是一個開放問題

</div>

<div class="redbox">

**自適應裁剪策略空間受限**

目前只比較「固定閾值」vs「基於出現次數的閾值」，更一般的自適應策略（如基於節點度分布的函數）尚未研究

</div>

<div class="redbox">

**僅考慮二元關係（Binary Relations）**

論文聚焦於邊存在/不存在的設定，多關係場景（知識圖譜）需要進一步擴展

</div>

---

# 可改進的方向

### 理論層面

- 📐 **更緊的隱私放大界**：探索耦合採樣放大界的下界，確認現有界是否最優
- 📐 **更一般的耦合採樣**：本文只處理基數相依子類，一般耦合採樣（依賴具體內容）的分析仍是開放問題
- 📐 **局部差分隱私（LDP）版本**：針對去中心化場景的延伸

### 演算法層面

- ⚙️ **更精緻的自適應裁剪**：結合梯度分布資訊（如分位數、分布矩）與節點出現頻率
- ⚙️ **異質節點類型**：不同類型節點的敏感度可能差異很大，需要差異化處理
- ⚙️ **動態圖設定**：時序圖中新增/刪除節點的隱私保護

---

# 可改進的方向（續）

### 應用層面

- 🔗 **多關係知識圖譜**（如 FB15k、Wikidata）的驗證
- 🔗 **聯邦學習結合**：多方持有部分圖，如何在不集中資料的情況下進行 private relational learning
- 🔗 **差分隱私 GNN 結合**：本方法針對 loss 函數的隱私保護，與 GNN 中訊息傳遞的隱私保護如何結合仍是開放問題

### 評估層面

- 📊 **成員推論攻擊的實證驗證**：量化本方法對實際攻擊的防護效果
- 📊 **更多真實應用場景**：醫療知識圖譜、金融交易網路等高敏感度資料

---

# 總結

<div class="two-col">
<div>

### 三大核心貢獻

**① 敏感度分析 + 自適應裁剪**
- 識別正/負邊對敏感度的不同影響
- FREQ-CLIP 將全域敏感度控制為常數 $2C$

**② 耦合採樣隱私放大**
- 形式化基數相依耦合採樣
- 導出嚴格 RDP 放大界（Theorem 4.1）

**③ 整合演算法（Algorithm 3）**
- 可直接用於微調預訓練文字編碼器
- 具備完整隱私保證

</div>
<div>

### 實驗驗證

- 在 4 個真實文字屬性圖上驗證
- BERT-base/large + Llama2-7B
- 相同隱私預算下**一致優於標準裁剪**
- 隱私界比 Naive bound 小數個數量級

<br>

### 開放問題

- 放大界緊緻性
- 一般耦合採樣分析
- 更豐富的自適應裁剪策略

</div>
</div>

---

<!-- _class: cover -->

# 謝謝聆聽

<br>

**Code**: https://github.com/Graph-COM/Node_DP

<br>

> Yinan Huang, Haoteng Yin, Eli Chien, Rongzhe Wei, Pan Li
> *NeurIPS 2025*

<br>

**Q & A**
