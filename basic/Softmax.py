import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# pick CJK-capable font when available (best-effort)
preferred_fonts = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
available = {f.name for f in fm.fontManager.ttflist}
for name in preferred_fonts:
    if name in available:
        mpl.rcParams["font.sans-serif"] = [name]
        break
mpl.rcParams["axes.unicode_minus"] = False

def softmax(x, temperature=1.0):
    x = np.array(x, dtype=float)
    z = x / float(temperature)
    # numerical stability
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    return e, p

def render_compact_table(table_dict):
    headers = list(table_dict.keys())
    rows = list(zip(*[table_dict[h] for h in headers]))
    html = """
    <style>
    table.compact { border-collapse: collapse; font-size:12px; }
    table.compact th, table.compact td { padding: 3px 6px; height: 18px; line-height:14px; border: 1px solid #eee; }
    table.compact th { background:#f8f9fa; font-weight:600; }
    </style>
    <table class='compact'>
      <thead>
        <tr>
    """
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr>\n      </thead>\n      <tbody>\n"
    for r in rows:
        html += "<tr>"
        for c in r:
            html += f"<td>{c}</td>"
        html += "</tr>\n"
    html += "</tbody>\n    </table>"
    return html

def main():
    st.title("Softmax 可视化演示")

    st.markdown("""
    **Softmax** 将一组实数转换为概率分布：

    $$\mathrm{softmax}(x_i)=\frac{e^{x_i/T}}{\sum_j e^{x_j/T}}$$

    其中 $T$ 为温度（temperature），控制概率分布的平滑程度。
    """)

    st.markdown("**输入：** 编辑下面的数值（按索引排列），调整 `Temperature` 观察概率变化。")

    # editable inputs in one row
    default_vals = [3.0, 1.0, 0.2, 5.0]
    cols = st.columns(len(default_vals))
    inputs = []
    for i, v in enumerate(default_vals):
        inputs.append(cols[i].number_input(f"x[{i}]", value=v, key=f"soft_x_{i}", format="%f", step=0.1))

    temperature = st.slider("Temperature (T)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)

    e_vals, probs = softmax(inputs, temperature)

    # prepare compact tables: show x, e^{x/T}, softmax
    pos_table = {"x": [f"x[{i}]" for i in range(len(inputs))],
                 "value": [float(v) for v in inputs],
                 "exp(x/T)": [float(round(v, 6)) for v in e_vals],
                 "softmax": [float(round(p, 6)) for p in probs]}

    st.markdown("**结果表（数值）**")
    # render compact via components.html to keep layout tight
    st.components.v1.html(render_compact_table(pos_table), height=140, scrolling=False)

    # plot probabilities
    fig, ax = plt.subplots(figsize=(6, 3.5))
    indices = np.arange(len(inputs))
    ax.bar(indices, probs, color='#1f77b4')
    ax.set_xticks(indices)
    ax.set_xticklabels([f"x[{i}]" for i in indices])
    ax.set_ylim(0, 1)
    ax.set_ylabel('P(softmax)')
    ax.set_title(f'Softmax (T={temperature})')
    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p:.3f}", ha='center')
    ax.grid(axis='y', linestyle=':')
    st.pyplot(fig)
    st.markdown("""
        - `Temperature` 越小，分布越尖锐（接近 one-hot）；越大，分布越平滑。

        **在 Transformer 中的使用位置与原因**
        - 注意力机制（Scaled Dot-Product Attention）：Softmax 用在查询-键相似度（logits）上，将这些相似度归一化为权重，用于对值向量进行加权平均。通常先除以一个缩放因子（sqrt(d_k)），这个操作类似于在 logits 上应用了温度（temperature）缩放，以稳定梯度并防止数值溢出。
        - 输出层（分类/语言模型）：将模型输出的 logits 转为概率分布以计算交叉熵损失或做概率采样。

        **为什么使用 Softmax**
        - 将任意实数向量转换为概率分布，便于解释与优化（与交叉熵损失紧密配合）；可微，支持反向传播；温度参数允许控制分布的尖锐/平滑程度。

        **优点**
        - 简单、有效、可微且实现成本低。
        - 提供自然的概率解释，方便训练时与交叉熵联合使用。
        - 在 Transformer 中配合缩放（scaled）能稳定训练。

        **缺点与局限**
        - 输出通常不稀疏：会给很多选项分配非零概率，难以直接产生稀疏注意力或稀疏预测。
        - 对极端 logits 敏感：大 logits 会主导分布，可能导致数值稳定性问题（需做数值稳定化处理）。
        - 在某些任务上概率校准（calibration）或稀疏性不是最优，出现替代方法的需求。

        **是否是“现在解决此类问题的主要函数”？**
        - Softmax 是 Transformer 中生成注意力权重与将 logits 转概率的默认、最常见方法，也是工业与学术界的主流选择。但并非唯一选择：为了解决稀疏性与可解释性问题，研究中出现了 `sparsemax`、`entmax`、Top-k/Top-p 等变种，工程上也会结合温度、掩码或归一化技巧来调整行为。

        总结：Softmax 是标准且广泛使用的方法，适用于绝大多数 Transformer 场景；在需要稀疏注意力或特定概率特性的任务中，可以考虑替代或改进方案。
        """)

        # 术语解释与目的说明
    st.markdown("""
        **术语解释（名词解释）**

        - **稀疏注意力（Sparse Attention）**：指注意力权重矩阵中大部分元素为零或接近零的机制或方法。例如使用 `sparsemax`、`entmax`、Top-k 注意力或通过掩码限制注意力范围。稀疏注意力会让每个查询仅关注有限的一小部分键，从而得到更稀疏的权重分布。

        - **交叉熵损失（Cross-Entropy Loss）**：用于分类和语言建模的常见损失函数。对于单个样本与目标分布 y（通常为 one-hot），交叉熵为
            $$L=-\sum_i y_i\log p_i$$
            其中 $p$ 为模型通过 Softmax 输出的概率分布。最小化交叉熵等价于最大化训练数据的对数似然（最大似然估计），因而是训练分类模型的标准目标。

        - **概率采样（Probability Sampling）**：指根据 Softmax 输出的概率分布随机选择一个或多个项来作为模型的输出（用于生成任务）。常见策略包括：贪心（取最大概率）、温度采样（调节概率形状）、Top-k、Top-p（nucleus）等，目的是在确定性与多样性之间做折中。

        **为什么希望注意力稀疏？目的是什么**
        - **计算与存储效率**：稀疏注意力使得每个查询只与少量键交互，可以降低时间与内存复杂度（对长序列尤为重要）。
        - **聚焦与可解释性**：稀疏权重更容易解读，能反映出模型关注的少数关键位置，有助于诊断和解释模型行为。
        - **减少噪声与提高泛化**：忽略不相关的位置可减少噪声干扰，可能提升下游任务的鲁棒性与泛化能力。
        - **硬性结构需求**：某些任务需要局部或稀疏的交互（如长文本、图结构），稀疏注意力便于引入先验结构。

        **为什么要计算交叉熵损？目的是什么**
        - **概率意义上的监督目标**：交叉熵直接对应于模型预测分布与真实分布间的差距，最小化交叉熵就是使模型估计的分布更接近数据真实分布（最大似然原理）。
        - **可导且梯度合理**：交叉熵与 Softmax 结合可以产生良好的梯度信号，便于通过反向传播进行有效训练。
        - **与评价指标对齐**：对于分类任务（包括语言建模中的逐字预测），交叉熵/对数似然与最终评估指标（困惑度、准确率）有直接关系。

        **概率采样的目的**n+    - **生成多样性**：纯贪心会导致重复和缺乏多样性，概率采样（结合温度、Top-k/Top-p）能生成更丰富、更自然的样本。
        - **平衡确定性与随机性**：通过温度或截断策略可以控制随机程度，满足不同应用对创造性或稳定性的需求。

        以上术语与目的都与 Softmax 密切相关：Softmax 提供了可微且可解释的概率分布，既是注意力权重的基础，也是交叉熵训练与概率采样生成的核心组成部分。
        """)

if __name__ == '__main__':
    main()
