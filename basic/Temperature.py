import streamlit as st
import numpy as np
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


def render_compact_table(table_dict):
    headers = list(table_dict.keys())
    rows = list(zip(*[table_dict[h] for h in headers]))
    html = """
    <style>
    table.compact { border-collapse: collapse; font-size:12px; }
    table.compact th, table.compact td { padding: 4px 8px; height: 20px; line-height:14px; border: 1px solid #ddd; }
    table.compact th { background:#f8f9fa; font-weight:600; }
    .sampled { background:#dfefff; padding:6px 10px; border-radius:4px; display:inline-block; }
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


def softmax_probs(logits, T=1.0):
    a = np.array(logits, dtype=float) / float(T)
    a = a - np.max(a)
    e = np.exp(a)
    s = np.sum(e)
    return e, e / s


def main():
    st.title("Temperature: Softmax Sampling 区间演示")

    st.markdown("""
    这个页面按表格还原 Excel 的展示：给定若干词与对应 logits，
    计算 $z=logit/T$, 显示 $e^{z}$、归一化分母与最终的 Softmax 概率；
    同时按概率排序并计算累计区间，给定随机数 `r` 可直接看到采样结果。
    """)

    vocab = ["who", "where", "why", "what", "how"]
    st.markdown("**输入（词表与 logits）**")
    st.markdown("""
    **Logit（未归一化得分）定义**

    - `logit` 指模型在 Softmax 之前输出的实数分数（unnormalized score），不是概率，取值可以为任意实数。
    - Softmax 将一组 logits 转换为概率分布；更大的 logit 对应更高的概率。
    - 在二分类中 “logit” 有时指对数几率（log-odds），但在多分类或语言模型中通常泛指未归一化的模型输出。
    """)
    cols = st.columns(len(vocab))
    default_logits = {"who": 2.50, "where": 1.00, "why": 5.20, "what": 3.10, "how": 2.15}
    logits = []
    for i, w in enumerate(vocab):
        logits.append(cols[i].number_input(f"{w}", value=float(default_logits[w]), key=f"logit_{w}", format="%f", step=0.01))

    T = st.number_input("Temperature T", min_value=0.01, max_value=10.0, value=2.0, step=0.01)
    r = st.number_input("Random number r", min_value=0.0, max_value=1.0, value=0.45, step=0.01, format="%f")

    e_vals, probs = softmax_probs(logits, T)
    sum_e = float(np.sum(e_vals))

    # Left table: V, Z, /T, e^, /Σ, Prob (with bar)
    left = {"V": [], "Z (logit)": [], "/T": [], "e^": [], "/Σ": [], "Prob": []}
    for i, w in enumerate(vocab):
        z = logits[i] / float(T)
        left["V"].append(w)
        left["Z (logit)"].append(f"{logits[i]:.2f}")
        left["/T"].append(f"{z:.2f}")
        left["e^"] .append(f"{e_vals[i]:.2f}")
        left["/Σ"].append(f"{e_vals[i]:.2f}/{sum_e:.2f}")
        bar = f"<div style='background:#e74c3c;width:{probs[i]*100:.1f}%;height:12px;border-radius:3px'></div>"
        left["Prob"].append(f"{probs[i]:.4f}<br>{bar}")

    # Right table sorted by prob desc with cumulative ranges
    order = np.argsort(-probs)
    sorted_vocab = [vocab[i] for i in order]
    sorted_probs = [float(probs[i]) for i in order]
    cum = 0.0
    right = {"Vocab": [], "Prob": [], "left": [], "< r <": []}
    for p, w in zip(sorted_probs, sorted_vocab):
        left_c = cum
        right_c = cum + p
        bar = f"<div style='background:#e74c3c;width:{p*100:.1f}%;height:12px;border-radius:3px'></div>"
        right["Vocab"].append(w)
        right["Prob"].append(f"{p:.4f}<br>{bar}")
        right["left"].append(f"{left_c:.2f}")
        right["< r <"].append(f"{right_c:.2f}")
        cum = right_c

    st.subheader("计算详情")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Input / 计算过程**")
        st.components.v1.html(render_compact_table(left), height=240, scrolling=False)
    with c2:
        st.markdown("**按概率排序（累计区间）**")
        st.components.v1.html(render_compact_table(right), height=240, scrolling=False)

    # determine sampled word
    sampled = None
    cum = 0.0
    for i in order:
        left_c = cum
        right_c = cum + probs[i]
        # include left bound, exclude right (except r==1.0)
        if (r >= left_c and r < right_c) or (r == 1.0 and right_c == 1.0):
            sampled = vocab[i]
            break
        cum = right_c

    st.markdown("**Sampled Word**")
    if sampled is None:
        st.warning("随机数未落入任何区间（浮点精度问题），默认选择概率最大项。")
        sampled = sorted_vocab[0]
    st.markdown(f"<div class='sampled'>{sampled}</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
