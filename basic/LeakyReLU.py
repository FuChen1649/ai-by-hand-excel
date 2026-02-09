
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import matplotlib.font_manager as fm

# Try to pick a CJK-capable font (Windows common fallback: Microsoft YaHei / SimHei)
preferred_fonts = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]
available = {f.name for f in fm.fontManager.ttflist}
chosen = None
for name in preferred_fonts:
  if name in available:
    chosen = name
    break
if chosen:
  mpl.rcParams["font.sans-serif"] = [chosen]
else:
  # leave default but ensure minus sign renders
  pass
mpl.rcParams["axes.unicode_minus"] = False

def main():
  st.title("LeakyReLU 可视化演示")

  st.markdown("""
  **引言 — 为什么使用 ReLU / LeakyReLU，以及在 Transformer 中的应用**

  - 激活函数用于在两层线性变换之间引入非线性，使模型能拟合复杂函数。ReLU（Rectified Linear Unit）通过将负值截断为 0 提供简单高效的非线性，而 LeakyReLU 在负半轴保留小斜率以传递梯度，减少“神经元死亡”的风险。
  - 优缺点概览：ReLU 计算简单、稀疏激活、收敛快，但负区间恒为 0 可能导致部分单元长期不更新；LeakyReLU 在负区间保留小梯度，训练更稳定，但会减少稀疏性，需要选择合适的 alpha（常见 0.01 或 0.1）。
  - 在 Transformer 中，激活函数通常用于每个层内的前馈网络（FFN）：FFN(x) = Linear2( activation( Linear1(x) ) )。Vaswani et al.（2017）原始实现使用 ReLU，后续许多模型（如 BERT、GPT）采用更平滑的 GELU，但 ReLU/LeakyReLU 在某些场景仍为有效且高效的选择。
  - 实践建议：复现基线或资源受限时可用 ReLU；若训练不稳定或出现“死亡”神经元，考虑 LeakyReLU（或可学习的 PReLU）；若追求最佳精度，可尝试 GELU。

  下面的交互式演示可帮助你直观比较 ReLU 与 LeakyReLU（支持修改输入、alpha，并观察表格与图形变化）。
  """)

  st.markdown("""
  **LeakyReLU** 是一种常用的激活函数，定义如下：

  $$
  	ext{LeakyReLU}(x) = \begin{cases}
    x, & x \geq 0 \\
    \alpha x, & x < 0
  \end{cases}
  $$

  - 右侧可调整 $\alpha$（负半轴斜率）参数，观察输出变化。
  """)



  # ReLU 详细介绍
  st.markdown("""
  ### ReLU（Rectified Linear Unit）
  **功能**：ReLU 是深度学习中最常用的激活函数之一，能够有效缓解梯度消失问题，加速神经网络收敛。

  **出处**：首次大规模应用于深度神经网络是在论文：
  - [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)（AlexNet，2012，Krizhevsky et al.）

  **计算公式**：
  $$
  \mathrm{ReLU}(x) = \max(0, x)
  $$
    
  **特点**：
  - 计算简单，收敛快
  - 只保留正数部分，负数直接置零
  - 可能导致“神经元死亡”问题（部分神经元长期输出0）
  """)

  st.subheader("ReLU")
  # 允许用户自定义x取值

  st.markdown("**自定义输入 x 的取值（可编辑，按索引0-9排序）**")
  # ------------------ ReLU 输入（独立） ------------------
  # x 顺序为 9,8,...,0,...,-9（正数在上，表格显示顺序）
  x_values_relu = list(range(9, -10, -1))
  default_x_relu = x_values_relu.copy()
  x_list_relu = []
  n = len(x_values_relu)
  n1 = n // 2 + n % 2
  n2 = n - n1
  cols1 = st.columns(n1)
  cols2 = st.columns(n2)
  st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
  for i in range(n1):
    x_val = cols1[i].number_input(f"ReLU x={x_values_relu[i]}", value=default_x_relu[i], key=f"relu_x_{x_values_relu[i]}", step=1, format="%d")
    x_list_relu.append(x_val)
  for i in range(n2):
    x_val = cols2[i].number_input(f"ReLU x={x_values_relu[n1+i]}", value=default_x_relu[n1+i], key=f"relu_x_{x_values_relu[n1+i]}", step=1, format="%d")
    x_list_relu.append(x_val)
  x_relu = np.array(x_list_relu)

  relu = np.maximum(0, x_relu)
  # 表格按 x=9..-9 展示（正数在上），但图的 x 轴从 -9..9 （升序）以符合常见坐标系
  relu_table = {"x": x_values_relu, "ReLU(x)": relu}
  # 左侧缩小为原来一半宽度，右侧图表更宽
  col1, col2 = st.columns([1, 2])
  # 生成紧凑型 HTML 表格，减小单元格高度并去掉多余间距
  def render_compact_table(table_dict):
    headers = list(table_dict.keys())
    rows = list(zip(*[table_dict[h] for h in headers]))
    html = """
    <style>
    table.compact { border-collapse: collapse; font-size:12px; }
    table.compact th, table.compact td { padding: 2px 6px; height: 16px; line-height:14px; border: 1px solid #eee; }
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

  # 将左侧表格分为两组：0..9 和 -9..-1，分别展示，避免滚动条（ReLU）
  pos_vals = list(range(0, 10))
  neg_vals = list(range(-9, 0))
  mapping_relu = {label: val for label, val in zip(x_values_relu, x_relu)}
  pos_table = {"x": pos_vals, "ReLU(x)": [max(0.0, mapping_relu.get(v, 0)) for v in pos_vals]}
  neg_table = {"x": neg_vals, "ReLU(x)": [max(0.0, mapping_relu.get(v, 0)) for v in neg_vals]}
  with col1:
    cols_left = st.columns(2)
    with cols_left[0]:
      components.html(render_compact_table(pos_table), height=220, scrolling=False)
    with cols_left[1]:
      components.html(render_compact_table(neg_table), height=220, scrolling=False)
  with col2:
    # 为了让图表从 -9 到 9 左到右显示，需要对输入顺序做映射（ReLU）
    mapping = mapping_relu
    x_axis = sorted(x_values_relu)  # -9..9
    # y 对应输入值的 ReLU
    y_vals = [max(0.0, mapping[label]) for label in x_axis]
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x_axis, y_vals, 'o', label="ReLU(x)", markersize=6)
    ax1.plot(x_axis, y_vals, '-', color='#1f77b4', alpha=0.6)
    ax1.set_xticks(x_axis)
    ax1.set_xlim(min(x_axis) - 0.5, max(x_axis) + 0.5)
    ax1.set_xlabel("x")
    ax1.set_ylabel("ReLU(x)")
    ax1.set_title("ReLU 激活函数")
    ax1.grid(True, linestyle=':')
    st.pyplot(fig1)


  # LeakyReLU 详细介绍
  st.markdown("""
  ### LeakyReLU（带泄漏的线性整流单元）
  **功能**：LeakyReLU 是对 ReLU 的改进，解决了 ReLU 在负区间恒为0导致神经元“死亡”的问题。LeakyReLU 在 $x<0$ 时给一个很小的斜率，使负区间也能传递梯度。

  **出处**：
  - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)（Maas et al., 2013）

  **计算公式**：
  $$
  \mathrm{LeakyReLU}(x) = \begin{cases}
    x, & x \geq 0 \\
    \alpha x, & x < 0
  \end{cases}
  $$
  其中 $\alpha$ 通常取 0.01 或 0.1，可调。

  **特点**：
  - 负区间也有非零输出，缓解神经元死亡
  - $\alpha$ 可调，灵活性更高
  """)

  st.subheader("LeakyReLU")
  # ------------------ LeakyReLU 输入（独立） ------------------
  alpha = st.number_input("alpha (负半轴斜率)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key='alpha_leaky')
  x_values_leaky = list(range(9, -10, -1))
  default_x_leaky = x_values_leaky.copy()
  x_list_leaky = []
  n = len(x_values_leaky)
  n1 = n // 2 + n % 2
  n2 = n - n1
  cols1_l = st.columns(n1)
  cols2_l = st.columns(n2)
  st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
  for i in range(n1):
    x_val = cols1_l[i].number_input(f"Leaky x={x_values_leaky[i]}", value=default_x_leaky[i], key=f"leaky_x_{x_values_leaky[i]}", step=1, format="%d")
    x_list_leaky.append(x_val)
  for i in range(n2):
    x_val = cols2_l[i].number_input(f"Leaky x={x_values_leaky[n1+i]}", value=default_x_leaky[n1+i], key=f"leaky_x_{x_values_leaky[n1+i]}", step=1, format="%d")
    x_list_leaky.append(x_val)
  x_leaky = np.array(x_list_leaky)
  leaky_relu = np.where(np.array(x_values_leaky) >= 0, x_leaky, np.round(alpha * x_leaky, 6))
  # make alpha column strings to avoid mixed-type conversion errors
  leaky_table = {"x": x_values_leaky, "LeakyReLU(x)": leaky_relu, "alpha": ["" if v >= 0 else f"{alpha:.6f}" for v in x_values_leaky]}
  col3, col4 = st.columns([1, 2])
  # LeakyReLU 左侧也分为两组展示（0..9 与 -9..-1）
  mapping_leaky = {label: val for label, val in zip(x_values_leaky, x_leaky)}
  pos_table_leaky = {"x": pos_vals, "LeakyReLU(x)": [mapping_leaky.get(v, 0) for v in pos_vals], "alpha": ['' for _ in pos_vals]}
  neg_table_leaky = {"x": neg_vals, "LeakyReLU(x)": [round(alpha * mapping_leaky.get(v, 0), 6) for v in neg_vals], "alpha": [f"{alpha:.6f}" for _ in neg_vals]}
  with col3:
    cols_left2 = st.columns(2)
    with cols_left2[0]:
      components.html(render_compact_table(pos_table_leaky), height=220, scrolling=False)
    with cols_left2[1]:
      components.html(render_compact_table(neg_table_leaky), height=220, scrolling=False)
  with col4:
    mapping = mapping_leaky
    x_axis = sorted(x_values_leaky)
    y_vals_leaky = [mapping[label] if label >= 0 else round(alpha * mapping[label], 6) for label in x_axis]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x_axis, y_vals_leaky, 'o', label="LeakyReLU(x)", markersize=6)
    ax2.plot(x_axis, y_vals_leaky, '-', color='#1f77b4', alpha=0.6)
    ax2.set_xticks(x_axis)
    ax2.set_xlim(min(x_axis) - 0.5, max(x_axis) + 0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("LeakyReLU(x)")
    ax2.set_title("LeakyReLU 激活函数")
    ax2.grid(True, linestyle=':')
    st.pyplot(fig2)

  st.markdown("""
  - **ReLU**: $f(x) = \max(0, x)$
  - **LeakyReLU**: $f(x) = x$ if $x \geq 0$ else $\alpha x$

  可通过调整 $\alpha$，观察负半轴输出的变化。
  """)

  # ------------------ GELU 扩展 ------------------
  st.markdown("""
  ### GELU（Gaussian Error Linear Unit）
  GELU 是一种常见的平滑激活函数，经典形式为：
  $$
  \mathrm{GELU}(x)=x\cdot\Phi(x)=\frac{x}{2}\left[1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
  $$
  它在 Transformer 系列模型（如 BERT、GPT）中广泛使用，因其平滑性在某些任务上能稍微提升性能。
  """)

  st.subheader("GELU")
  # GELU 使用独立的一组 x 输入（与 ReLU/LeakyReLU 分离）
  x_values_gelu = list(range(9, -10, -1))
  default_x_gelu = x_values_gelu.copy()
  x_list_gelu = []
  n = len(x_values_gelu)
  n1 = n // 2 + n % 2
  n2 = n - n1
  cols1_g = st.columns(n1)
  cols2_g = st.columns(n2)
  st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
  for i in range(n1):
    x_val = cols1_g[i].number_input(f"GELU x={x_values_gelu[i]}", value=default_x_gelu[i], key=f"gelu_x_{x_values_gelu[i]}", step=1, format="%d")
    x_list_gelu.append(x_val)
  for i in range(n2):
    x_val = cols2_g[i].number_input(f"GELU x={x_values_gelu[n1+i]}", value=default_x_gelu[n1+i], key=f"gelu_x_{x_values_gelu[n1+i]}", step=1, format="%d")
    x_list_gelu.append(x_val)
  x_gelu = np.array(x_list_gelu)

  # 计算 GELU
  def gelu_fn(arr):
    return 0.5 * arr * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in arr]))

  y_gelu = gelu_fn(x_gelu)
  # 左侧两组表格（0..9 与 -9..-1）
  pos_table_gelu = {"x": list(range(0, 10)), "GELU(x)": [float(y_gelu[list(x_values_gelu).index(v)]) if v in x_values_gelu else 0.0 for v in range(0, 10)]}
  neg_table_gelu = {"x": list(range(-9, 0)), "GELU(x)": [float(y_gelu[list(x_values_gelu).index(v)]) if v in x_values_gelu else 0.0 for v in range(-9, 0)]}
  col_g1, col_g2 = st.columns([1, 2])
  with col_g1:
    ccols = st.columns(2)
    with ccols[0]:
      components.html(render_compact_table(pos_table_gelu), height=220, scrolling=False)
    with ccols[1]:
      components.html(render_compact_table(neg_table_gelu), height=220, scrolling=False)
  with col_g2:
    mapping_gelu = {label: val for label, val in zip(x_values_gelu, x_gelu)}
    x_axis_g = sorted(x_values_gelu)
    y_axis_g = [float(gelu_fn(np.array([mapping_gelu[label]]))[0]) for label in x_axis_g]
    figg, axg = plt.subplots(figsize=(6, 4))
    axg.plot(x_axis_g, y_axis_g, 'o', markersize=6)
    axg.plot(x_axis_g, y_axis_g, '-', color='#2ca02c', alpha=0.7)
    axg.set_xticks(x_axis_g)
    axg.set_xlim(min(x_axis_g) - 0.5, max(x_axis_g) + 0.5)
    axg.set_xlabel('x')
    axg.set_ylabel('GELU(x)')
    axg.set_title('GELU 激活函数')
    axg.grid(True, linestyle=':')
    st.pyplot(figg)

  # 补充解释：为何在两层线性变换之间加入非线性激活？
  st.markdown("""
  **补充 — 为什么需要在线性层之间加入非线性激活函数**

  - 核心原因：如果网络只由线性层堆叠组成，那么任意多个线性变换的组合仍然等价于一个线性变换。例如：
    $W_2(W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$，因此堆叠线性层并不能提高模型的表达能力。

  - 引入非线性激活后，网络可以表示复杂的非线性函数，具备更强的表达能力（基于通用逼近定理，在足够宽/深的情况下可以逼近任意连续函数）。

  - 作用与好处：
    - 提升表示能力，使模型学习层次化特征（低层学习简单模式，深层组合成复杂模式）。
    - 某些激活（如 ReLU）带来稀疏性，有利于效率和正则化；某些平滑激活（如 GELU）在优化上更稳定并能带来精度提升。

  - 实践权衡：
    - 缺点：某些激活会导致梯度消失（如 sigmoid/tanh 在极端区间），或出现“死神经元”（ReLU）；需要配合合适的初始化、归一化（LayerNorm/BatchNorm）和优化器来稳定训练。为了解决稀疏性或可解释性问题，也可采用 `sparsemax` / `entmax` 等替代方案。

  总结：在每两个仿射（线性）变换之间加入非线性激活，是让深度神经网络学到复杂、层次化映射的关键做法，也是 Transformer 与大多数现代神经网络的标准设计。
  """)

if __name__ == "__main__":
  main()
