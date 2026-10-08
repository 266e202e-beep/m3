import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.family'] = 'IPAGothic'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
G3 = ['#9bbfe8', '#2a78d6', '#103c73']       # 低 / 中 / 高 理解度 (明→暗)
ORANGE = '#eb6834'
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
rng = np.random.default_rng(20260101)

FEAT = {1: '効果発現速度', 2: '適用症状の広さ', 3: '副作用の少なさ', 4: '他の薬との飲み合わせの良さ', 5: '24h持続性'}
FSHORT = {1: '効果発現\n速度', 2: '適用症状\nの広さ', 3: '副作用の\n少なさ', 4: '飲み合わせ\nの良さ', 5: '24h\n持続性'}
DEPTS = ['内科', '耳鼻咽喉科', '小児科', '眼科']
FOCUS = ['内科', '耳鼻咽喉科']
LV = ['低(1-2)', '中(3)', '高(4-5)']
SMALL = 30

d = pd.read_csv(SRC)                      # 読み込みのみ
assert len(d) == 5000
d['unp'] = d.Q3_1 - d.Q3_2
for k in FEAT:
    d[f'L{k}'] = pd.cut(d[f'Q1_{k}'], [0, 2, 3, 5], labels=LV)


def save(df, name, index=True):
    df.to_csv(os.path.join(OUT, name), encoding='utf-8-sig', index=index)


def style(ax):
    ax.set_facecolor(SURF)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)


def share(g): return g.Q3_2.sum() / g.Q3_1.sum() if len(g) and g.Q3_1.sum() > 0 else np.nan


# ============ 分析1: Importance × Knowledge Gap ============
rowsA, rowsA2, rowsA3 = [], [], []
for dep in DEPTS:
    g = d[d.診療科 == dep]; n = len(g)
    for k, nm in FEAT.items():
        top = g[g[f'Q2_{k}'] == 1]; imp = g[g[f'Q2_{k}'].notna()]
        gap = top[top[f'Q1_{k}'] <= 2]
        rowsA.append({'診療科': dep, '特性': nm, '診療科医師数': n,
                      '1位に選んだ医師数': len(top), '1位の割合': len(top) / n,
                      '重視(順位を付けた)医師数': len(imp), '重視の割合': len(imp) / n})
        for lab, grp in [('重視している医師(順位あり)', imp), ('1位に選んだ医師', top)]:
            vc = grp[f'L{k}'].value_counts().reindex(LV).fillna(0).astype(int)
            r = {'診療科': dep, '特性': nm, '対象': lab, '医師数': len(grp)}
            for l in LV: r[f'{l} 人数'] = vc[l]
            for l in LV: r[f'{l} 割合'] = vc[l] / len(grp) if len(grp) else np.nan
            rowsA2.append(r)
        rowsA3.append({'診療科': dep, '特性': nm, '1位に選んだ医師数': len(top),
                       '認知ギャップ群(1位かつ理解度1-2)': len(gap),
                       '1位選択者に占める割合': len(gap) / len(top) if len(top) else np.nan, '診療科の医師数に占める割合': len(gap) / n})
tA = pd.DataFrame(rowsA); tA2 = pd.DataFrame(rowsA2); tA3 = pd.DataFrame(rowsA3)
save(tA, 'A1_重視度_診療科×特性.csv', False); save(tA2, 'A2_理解度分布.csv', False); save(tA3, 'A3_認知ギャップ群.csv', False)
print('== A1\n', tA.round(3).to_string()); print('\n== A2\n', tA2.round(3).to_string()); print('\n== A3\n', tA3.round(3).to_string())

# ============ 分析2: Market Potential ============
rowsB = []
for dep in DEPTS:
    g = d[d.診療科 == dep]; dep_unp = g.unp.sum()
    rowsB.append({'診療科': dep, '特性': '(診療科全体)', '医師数': len(g), '花粉症患者数合計': g.Q3_1.sum(),
                  '薬剤A処方患者数合計': g.Q3_2.sum(), '潜在未処方患者数合計': dep_unp, '薬剤A処方割合': share(g),
                  '平均Q4': g.Q4.mean(), '診療科の潜在未処方に占める割合': 1.0})
    for k, nm in FEAT.items():
        gp = g[(g[f'Q2_{k}'] == 1) & (g[f'Q1_{k}'] <= 2)]
        rowsB.append({'診療科': dep, '特性': nm, '医師数': len(gp), '花粉症患者数合計': gp.Q3_1.sum(),
                      '薬剤A処方患者数合計': gp.Q3_2.sum(), '潜在未処方患者数合計': gp.unp.sum(), '薬剤A処方割合': share(gp),
                      '平均Q4': gp.Q4.mean(), '診療科の潜在未処方に占める割合': gp.unp.sum() / dep_unp})
tB = pd.DataFrame(rowsB); save(tB, 'B_認知ギャップ群の市場規模.csv', False)
print('\n== B\n', tB.round(3).to_string())
# 重複: 1位は1人1項目なので、ギャップ群は互いに排他。合計=ユニーク人数
for dep in DEPTS:
    s = tB[(tB.診療科 == dep) & (tB.特性 != '(診療科全体)')]
    g = d[d.診療科 == dep]
    uniq = sum(((g[f'Q2_{k}'] == 1) & (g[f'Q1_{k}'] <= 2)).astype(int) for k in FEAT)
    print(dep, '5群の医師数合計', int(s.医師数.sum()), 'ユニーク', int((uniq > 0).sum()), '重複(2群以上)', int((uniq > 1).sum()),
          ' 潜在未処方比', round(s.潜在未処方患者数合計.sum() / g.unp.sum(), 3))

# 感度確認: 「重視(順位あり)かつ理解度1-2」だと重複が出る
rowsS = []
for dep in DEPTS:
    g = d[d.診療科 == dep]
    flags = pd.DataFrame({k: g[f'Q2_{k}'].notna() & (g[f'Q1_{k}'] <= 2) for k in FEAT})
    for k, nm in FEAT.items():
        rowsS.append({'診療科': dep, '特性': nm, '重視かつ理解度1-2の医師数': int(flags[k].sum())})
    rowsS.append({'診療科': dep, '特性': '(5群の延べ人数)', '重視かつ理解度1-2の医師数': int(flags.values.sum())})
    rowsS.append({'診療科': dep, '特性': '(ユニーク人数)', '重視かつ理解度1-2の医師数': int(flags.any(axis=1).sum())})
tS = pd.DataFrame(rowsS); save(tS, 'B2_感度確認_重視全体での重複.csv', False)
print('\n== B2 (重視=順位あり) 重複\n', tS.to_string())

# ============ 分析3: Prescription Association ============
def boot(a, b, fn, n=2000):
    """低群-高群 の差の95%ブートストラップ区間(医師の再抽出)"""
    out = []
    for _ in range(n):
        ra = a.iloc[rng.integers(0, len(a), len(a))]; rb = b.iloc[rng.integers(0, len(b), len(b))]
        out.append(fn(ra) - fn(rb))
    return np.nanpercentile(out, [2.5, 97.5])


rowsC, rowsD = [], []
for dep in DEPTS:
    g = d[d.診療科 == dep]
    for k, nm in FEAT.items():
        top = g[g[f'Q2_{k}'] == 1]
        grp = {l: top[top[f'L{k}'] == l] for l in LV}
        for l in LV:
            x = grp[l]
            rowsC.append({'診療科': dep, '特性': nm, '理解度': l, '医師数': len(x), '平均患者数': x.Q3_1.mean(),
                          '処方割合': share(x), '平均Q4': x.Q4.mean(), '平均潜在未処方患者数': x.unp.mean(),
                          '潜在未処方患者数合計': x.unp.sum(), '少数注意(n<30)': len(x) < SMALL})
        lo, hi = grp[LV[0]], grp[LV[2]]
        r = {'診療科': dep, '特性': nm, '低群n': len(lo), '高群n': len(hi)}
        if len(lo) >= 5 and len(hi) >= 5:
            q = boot(lo, hi, lambda z: z.Q4.mean()); s = boot(lo, hi, share)
            r.update({'Q4差(低−高)': lo.Q4.mean() - hi.Q4.mean(), 'Q4差 95%区間': f'[{q[0]:.2f}, {q[1]:.2f}]',
                      '処方割合差(低−高)': share(lo) - share(hi), '処方割合差 95%区間': f'[{s[0]:.3f}, {s[1]:.3f}]'})
        r['少数注意'] = min(len(lo), len(hi)) < SMALL
        rowsD.append(r)
tC = pd.DataFrame(rowsC); tD = pd.DataFrame(rowsD)
save(tC, 'C_1位選択者の理解度別比較.csv', False); save(tD, 'C2_低理解度−高理解度の差.csv', False)
print('\n== C\n', tC.round(3).to_string()); print('\n== C2\n', tD.round(3).to_string())

# 特性固有の理解ギャップか? (Q1の5項目は互いに強く相関するため確認)
rowsE = []
for dep in DEPTS:
    g = d[d.診療科 == dep]
    for k, nm in FEAT.items():
        top = g[g[f'Q2_{k}'] == 1]; gap = top[top[f'Q1_{k}'] <= 2]
        oth = [f'Q1_{j}' for j in FEAT if j != k]
        om = gap[oth].mean(axis=1)
        rowsE.append({'診療科': dep, '特性': nm, 'ギャップ群n': len(gap),
                      '他4特性の平均理解度(ギャップ群)': om.mean() if len(gap) else np.nan,
                      '他4特性の平均も2以下の割合': (om <= 2).mean() if len(gap) else np.nan,
                      '5特性すべて理解度2以下の割合': (gap[[f'Q1_{j}' for j in FEAT]] <= 2).all(axis=1).mean() if len(gap) else np.nan,
                      '当該特性だけ低い(他4特性平均が3以上)割合': (om >= 3).mean() if len(gap) else np.nan})
tE = pd.DataFrame(rowsE); save(tE, 'E_ギャップが特性固有かの確認.csv', False)
print('\n== E\n', tE.round(3).to_string())

# ============ 分析4: 比較表 ============
rows = []
for dep in FOCUS:
    a = tA[tA.診療科 == dep].set_index('特性'); b = tB[tB.診療科 == dep].set_index('特性')
    a3 = tA3[tA3.診療科 == dep].set_index('特性'); dd = tD[tD.診療科 == dep].set_index('特性')
    for nm in FEAT.values():
        rows.append({'診療科': dep, '特性': nm,
                     '①Importance 1位率': a.loc[nm, '1位の割合'],
                     '②Gap 1位選択者の低理解度率': a3.loc[nm, '1位選択者に占める割合'],
                     '②Gap ギャップ群医師数': a3.loc[nm, '認知ギャップ群(1位かつ理解度1-2)'],
                     '③Potential ギャップ群の潜在未処方患者数': b.loc[nm, '潜在未処方患者数合計'],
                     '③Potential 診療科の潜在未処方に占める割合': b.loc[nm, '診療科の潜在未処方に占める割合'],
                     '④Association Q4差(低−高)': dd.loc[nm, 'Q4差(低−高)'],
                     '④Association 処方割合差(低−高)': dd.loc[nm, '処方割合差(低−高)'],
                     '低群n': dd.loc[nm, '低群n'], '高群n': dd.loc[nm, '高群n']})
tF = pd.DataFrame(rows)
for c in ['①Importance 1位率', '②Gap ギャップ群医師数', '③Potential ギャップ群の潜在未処方患者数']:
    tF[c + '_順位'] = tF.groupby('診療科')[c].rank(ascending=False, method='min').astype(int)
tF['④Association Q4差(低−高)_順位'] = tF.groupby('診療科')['④Association Q4差(低−高)'].rank(ascending=True, method='min')
tF['④Association 処方割合差(低−高)_順位'] = tF.groupby('診療科')['④Association 処方割合差(低−高)'].rank(ascending=True, method='min')
save(tF, 'F_診療科×特性の優先度比較.csv', False)
print('\n== F\n', tF.round(3).to_string())

# ---------------- 図 ----------------
def fig_base(w, h): return plt.figure(figsize=(w, h), facecolor=SURF)


# 図1 認知ギャップ群: 医師数 と 潜在未処方
f = fig_base(11.5, 4.8)
for j, (col, ttl, fmt) in enumerate([('医師数', '認知ギャップ群の医師数(1位かつ理解度1-2)', '{:.0f}'),
                                       ('潜在未処方患者数合計', '認知ギャップ群の潜在未処方患者数(直近1年間・合計)', '{:.0f}')]):
    ax = f.add_subplot(1, 2, j + 1); style(ax)
    x = np.arange(5); w = .38
    for i, dep in enumerate(FOCUS):
        v = [tB[(tB.診療科 == dep) & (tB.特性 == nm)][col].iloc[0] for nm in FEAT.values()]
        bars = ax.bar(x + (i - .5) * w, v, w * .92, color=['#2a78d6', ORANGE][i], label=dep)
        for xi, vi in zip(x + (i - .5) * w, v):
            ax.text(xi, vi + max(v) * .015, fmt.format(vi), ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([FSHORT[k] for k in FEAT], color=INK, fontsize=9)
    ax.set_title(ttl, loc='left', color=INK, fontsize=11); ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
f.tight_layout(); f.savefig(os.path.join(OUT, 'figA_認知ギャップ群の規模.png'), dpi=150, facecolor=SURF); plt.close(f)

# 図2 理解度別の処方意向・処方割合 (内科/耳鼻 × 2指標)
f = fig_base(13, 8)
for r, dep in enumerate(FOCUS):
    for c, (col, ttl, scale, ylim) in enumerate([('平均Q4', '平均処方意向 Q4(1〜5)', 1, 5.6), ('処方割合', '薬剤A処方割合(%)', 100, 85)]):
        ax = f.add_subplot(2, 2, r * 2 + c + 1); style(ax)
        x = np.arange(5); w = .26
        for i, l in enumerate(LV):
            v, ns = [], []
            for nm in FEAT.values():
                row = tC[(tC.診療科 == dep) & (tC.特性 == nm) & (tC.理解度 == l)].iloc[0]
                v.append(row[col] * scale if row['医師数'] > 0 else np.nan); ns.append(int(row['医師数']))
            for xi, vi, ni in zip(x + (i - 1) * w, v, ns):
                small = ni < SMALL
                ax.bar(xi, 0 if np.isnan(vi) else vi, w * .92, color=G3[i], hatch='///' if small else None,
                       edgecolor=SURF if not small else INK2, linewidth=0 if not small else .6,
                       label=None)
                if not np.isnan(vi):
                    ax.text(xi, vi + ylim * .012, f'{vi:.1f}' if col == '平均Q4' else f'{vi:.0f}', ha='center', fontsize=7.5, color=INK)
                ax.text(xi, ylim * -.07, f'n={ni}', ha='center', fontsize=6.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([FSHORT[k] for k in FEAT], color=INK, fontsize=9); ax.tick_params(axis='x', pad=14)
        ax.set_ylim(0, ylim); ax.set_title(f'{dep}:1位に選んだ医師の{ttl}', loc='left', color=INK, fontsize=10.5)
        if r == 0 and c == 0:
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=G3[i], label=f'理解度{l}') for i, l in enumerate(LV)], frameon=False, fontsize=8.5, labelcolor=INK2, ncol=3, loc='upper left')
f.suptitle('各特性を1位に選んだ医師の、その特性の理解度別比較(斜線=n<30、相関であり因果ではない)', x=.01, ha='left', color=INK, fontsize=11.5)
f.tight_layout(rect=[0, 0, 1, .96]); f.savefig(os.path.join(OUT, 'figB_理解度別の処方意向と処方割合.png'), dpi=150, facecolor=SURF); plt.close(f)

# 図3 比較表ヒートマップ(各指標を同診療科内でスケール)
f = fig_base(12.5, 5.2)
cols = [('①Importance 1位率', '①Importance\n1位率', '{:.0%}'), ('②Gap 1位選択者の低理解度率', '②Gap\n1位者の\n低理解度率', '{:.0%}'),
        ('③Potential ギャップ群の潜在未処方患者数', '③Potential\nギャップ群の\n潜在未処方', '{:.0f}'),
        ('④Association Q4差(低−高)', '④Association\nQ4差\n(低−高)', '{:.2f}'), ('④Association 処方割合差(低−高)', '④Association\n処方割合差\n(低−高)', '{:.0%}')]
from matplotlib.colors import LinearSegmentedColormap
SEQ = LinearSegmentedColormap.from_list('s', ['#eef4fc', '#2a78d6', '#103c73'])
for r, dep in enumerate(FOCUS):
    ax = f.add_subplot(1, 2, r + 1); ax.set_facecolor(SURF)
    t = tF[tF.診療科 == dep].reset_index(drop=True)
    M = np.zeros((5, len(cols)))
    for j, (c, _, _) in enumerate(cols):
        v = t[c].astype(float).values
        M[:, j] = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
        if '④' in c: M[:, j] = (np.abs(v) - np.nanmin(np.abs(v))) / (np.nanmax(np.abs(v)) - np.nanmin(np.abs(v)) + 1e-9)
    ax.imshow(np.nan_to_num(M), cmap=SEQ, vmin=0, vmax=1, aspect='auto')
    for i in range(5):
        for j, (c, _, fm) in enumerate(cols):
            ax.text(j, i, '—' if pd.isna(t.loc[i, c]) else fm.format(t.loc[i, c]), ha='center', va='center', color='white' if M[i, j] > .55 else INK, fontsize=10)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], color=INK, fontsize=8)
    ax.set_yticks(range(5)); ax.set_yticklabels([FEAT[k] for k in FEAT], color=INK, fontsize=9.5)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0); ax.xaxis.tick_top()
    ax.set_title(dep, loc='left', color=INK, fontsize=12, pad=52)
f.suptitle('色は列ごとの相対値(④は差の絶対値が大きいほど濃い)。適用症状の広さは1位選択者が0人のため算出不可', x=.01, ha='left', color=INK2, fontsize=9.5)
f.tight_layout(rect=[0, 0, 1, .96]); f.savefig(os.path.join(OUT, 'figC_診療科×特性の比較.png'), dpi=150, facecolor=SURF); plt.close(f)
print('done')
