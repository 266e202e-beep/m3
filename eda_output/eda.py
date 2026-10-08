import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm

SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.family'] = 'IPAGothic'
plt.rcParams['axes.unicode_minus'] = False

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
SEQ = LinearSegmentedColormap.from_list('seq', ['#eef4fc', '#2a78d6', '#103c73'])
DEPTS = ['内科', '小児科', '耳鼻咽喉科', '眼科']
pd.set_option('display.width', 250)

d = pd.read_csv(SRC)          # 元データは読み込みのみ。書き戻さない
assert len(d) == 5000 and d['医師ID'].is_unique
d['潜在未処方'] = d.Q3_1 - d.Q3_2
d['share'] = d.Q3_2 / d.Q3_1.where(d.Q3_1 > 0)   # 個人の処方割合(Q3_1=0は算出不可)
Q1 = [f'Q1_{i}' for i in range(1, 6)]
Q2 = [f'Q2_{i}' for i in range(1, 6)]
d['Q1平均'] = d[Q1].mean(axis=1)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), encoding='utf-8-sig')


def style(ax):
    ax.set_facecolor(SURF)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, lw=.6)
    ax.set_axisbelow(True)


def fig(w, h):
    f = plt.figure(figsize=(w, h), facecolor=SURF)
    return f


# ---------------- 1. 診療科別 ----------------
def dept_table(g):
    n = len(g)
    p, t = g.Q3_1.sum(), g.Q3_2.sum()
    return pd.Series({
        '医師数': n, '平均花粉症患者数': g.Q3_1.mean(), '花粉症患者数合計': p,
        '薬剤A処方患者数合計': t, '薬剤A処方割合': t / p,
        '潜在未処方患者数合計': p - t, '医師一人当たり潜在未処方患者数': (p - t) / n})


t1 = d.groupby('診療科').apply(dept_table).loc[DEPTS]
t1.loc['全体'] = dept_table(d)
t1['潜在未処方の構成比'] = t1['潜在未処方患者数合計'] / t1.loc['全体', '潜在未処方患者数合計']
t1.loc['全体', '潜在未処方の構成比'] = 1.0
save(t1, 'table1_診療科別基礎集計.csv')
print('== 1 診療科別\n', t1.round(3).to_string())

# 図1
f = fig(11, 4.6)
ax = f.add_subplot(1, 2, 1); style(ax)
x = np.arange(len(DEPTS))
pres = t1.loc[DEPTS, '薬剤A処方患者数合計'] / t1.loc[DEPTS, '医師数']
unp = t1.loc[DEPTS, '医師一人当たり潜在未処方患者数']
ax.bar(x, pres, .55, color=BLUE, label='薬剤A処方患者')
ax.bar(x, unp, .55, bottom=pres + .0, color=ORANGE, label='潜在未処方患者(Q3_1−Q3_2)',
       linewidth=0, )
for i in range(len(DEPTS)):
    ax.text(i, pres.iloc[i] / 2, f'{pres.iloc[i]:.1f}', ha='center', va='center', color='white', fontsize=10)
    ax.text(i, pres.iloc[i] + unp.iloc[i] / 2, f'{unp.iloc[i]:.1f}', ha='center', va='center', color=INK, fontsize=10)
    ax.text(i, pres.iloc[i] + unp.iloc[i] + .15, f'計 {pres.iloc[i] + unp.iloc[i]:.1f}', ha='center', color=INK2, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(DEPTS, color=INK)
ax.set_ylabel('医師一人当たり花粉症患者数(期間不明)', color=INK2)
ax.set_ylim(0, 14)
ax.set_title('診療科別 医師一人当たり患者数の内訳', loc='left', color=INK, fontsize=12)
ax.legend(frameon=False, loc='upper right', fontsize=9, labelcolor=INK2)

ax = f.add_subplot(1, 2, 2); style(ax)
sh = t1.loc[DEPTS, '薬剤A処方割合'] * 100
ax.bar(x, sh, .55, color=BLUE)
for i, v in enumerate(sh):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', color=INK, fontsize=10)
ov = t1.loc['全体', '薬剤A処方割合'] * 100
ax.axhline(ov, color=INK2, lw=1, ls='--')
ax.text(len(DEPTS) - .45, ov + 1, f'全体 {ov:.1f}%', ha='right', color=INK2, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(DEPTS, color=INK)
ax.set_ylabel('薬剤A処方割合(処方患者計/患者計)', color=INK2)
ax.set_ylim(0, 60)
ax.set_title('診療科別 薬剤A処方割合', loc='left', color=INK, fontsize=12)
f.tight_layout(); f.savefig(os.path.join(OUT, 'fig1_診療科別市場規模と処方割合.png'), dpi=150, facecolor=SURF); plt.close(f)

# ---------------- 2. Q3 × Q4 ----------------
def q4_table(g):
    n = len(g); p, t = g.Q3_1.sum(), g.Q3_2.sum()
    return pd.Series({
        '医師数': n, '医師割合': n / len(d), '平均花粉症患者数': g.Q3_1.mean(),
        '処方割合(患者数加重)': t / p, '処方割合(医師平均)': g.share.mean(),
        '平均処方患者数': g.Q3_2.mean(), '平均潜在未処方患者数': g.潜在未処方.mean(),
        '潜在未処方患者数合計': g.潜在未処方.sum()})


t2 = d.groupby('Q4').apply(q4_table)
t2.loc['全体'] = q4_table(d)
t2['潜在未処方の構成比'] = t2['潜在未処方患者数合計'] / d.潜在未処方.sum()
save(t2, 'table2a_Q4別処方状況.csv')
print('\n== 2a Q4別\n', t2.round(3).to_string())

ov_share = d.Q3_2.sum() / d.Q3_1.sum()
valid = d.dropna(subset=['share']).copy()
bands = [-.001, 0, .1, .2, ov_share, .5, 1.0]
labels = ['0%', '0超〜10%', '10〜20%', f'20%〜全体({ov_share*100:.0f}%)未満', f'全体({ov_share*100:.0f}%)〜50%', '50%超']
bands = [-.001, 0, .1, .2, ov_share - 1e-9, .5, 1.0]
valid['処方割合帯'] = pd.cut(valid.share, bands, labels=labels)
t2b = pd.crosstab(valid['処方割合帯'], valid.Q4)
t2b.columns = [f'Q4={c}' for c in t2b.columns]
t2b_pct = (t2b / t2b.values.sum() * 100).round(1)
save(t2b, 'table2b_処方割合帯×Q4_医師数.csv')
print('\n== 2b 処方割合帯×Q4(医師数)\n', t2b.to_string())

# 低処方割合×高意向 (定義を複数で感度確認)
rows = []
for name, thr in [('10%未満', .10), ('20%未満', .20), (f'全体平均({ov_share*100:.0f}%)未満', ov_share)]:
    low = valid.share < thr
    for hname, hi in [('Q4=5', valid.Q4 == 5), ('Q4≥4', valid.Q4 >= 4)]:
        g = valid[low & hi]
        rows.append({'低処方割合の定義': name, '高意向の定義': hname, '医師数': len(g),
                     '全回答者(5,000)に占める割合': len(g) / len(d),
                     '平均花粉症患者数': g.Q3_1.mean(), '平均潜在未処方患者数': g.潜在未処方.mean(),
                     '潜在未処方患者数合計': g.潜在未処方.sum(),
                     '潜在未処方の全体比': g.潜在未処方.sum() / d.潜在未処方.sum()})
t2c = pd.DataFrame(rows)
save(t2c.set_index('低処方割合の定義'), 'table2c_低処方割合×高意向の医師数.csv')
print('\n== 2c 低処方割合×高意向\n', t2c.round(3).to_string())

# 高意向×低処方 の診療科内訳 (Q4>=4 & share<全体平均)
hl = valid[(valid.Q4 >= 4) & (valid.share < ov_share)]
t2d = pd.DataFrame({'医師数': hl.groupby('診療科').size(), '診療科内の割合': hl.groupby('診療科').size() / valid.groupby('診療科').size(),
                    '潜在未処方合計': hl.groupby('診療科').潜在未処方.sum()}).loc[DEPTS]
save(t2d, 'table2d_高意向×低処方_診療科別.csv')
print('\n== 2d 診療科別(Q4≥4 & 処方割合<全体平均)\n', t2d.round(3).to_string())

# 図2
f = fig(11, 4.6)
ax = f.add_subplot(1, 2, 1); style(ax)
qs = [1, 2, 3, 4, 5]
v = t2.loc[qs, '処方割合(患者数加重)'] * 100
ax.bar(qs, v, .6, color=BLUE)
for q, y in zip(qs, v):
    ax.text(q, y + 1, f'{y:.1f}%', ha='center', color=INK, fontsize=10)
ax.set_xticks(qs); ax.set_xticklabels([f'{q}\n(n={int(t2.loc[q, "医師数"]):,})' for q in qs], color=INK)
ax.set_xlabel('処方意向 Q4(1〜5段階)', color=INK2); ax.set_ylabel('薬剤A処方割合', color=INK2)
ax.set_ylim(0, 75); ax.set_title('処方意向別 薬剤A処方割合', loc='left', color=INK, fontsize=12)
ax = f.add_subplot(1, 2, 2); style(ax)
pr = t2.loc[qs, '平均処方患者数']; un = t2.loc[qs, '平均潜在未処方患者数']
ax.bar(qs, pr, .6, color=BLUE, label='薬剤A処方患者')
ax.bar(qs, un, .6, bottom=pr, color=ORANGE, label='潜在未処方患者')
for q in qs:
    ax.text(q, pr[q] + un[q] + .15, f'未処方 {un[q]:.1f}', ha='center', color=INK, fontsize=9)
ax.set_xticks(qs); ax.set_xticklabels(qs, color=INK)
ax.set_xlabel('処方意向 Q4(1〜5段階)', color=INK2); ax.set_ylabel('医師一人当たり患者数(期間不明)', color=INK2)
ax.set_ylim(0, 11.5); ax.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=INK2)
ax.set_title('処方意向別 医師一人当たり患者数の内訳', loc='left', color=INK, fontsize=12)
f.tight_layout(); f.savefig(os.path.join(OUT, 'fig2_処方意向別の処方状況.png'), dpi=150, facecolor=SURF); plt.close(f)

# 図3: ヒートマップ 処方割合帯 × Q4 (医師数)
f = fig(8.6, 4.8); ax = f.add_subplot(1, 1, 1); ax.set_facecolor(SURF)
M = t2b.values
im = ax.imshow(M, cmap=SEQ, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        c = 'white' if M[i, j] > M.max() * .55 else INK
        ax.text(j, i, f'{M[i, j]:,}', ha='center', va='center', color=c, fontsize=11)
ax.set_xticks(range(5)); ax.set_xticklabels([1, 2, 3, 4, 5], color=INK)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color=INK)
ax.set_xlabel('処方意向 Q4', color=INK2); ax.set_ylabel('個人の処方割合(Q3_2/Q3_1)', color=INK2)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
# 高意向×全体平均未満 の枠
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((2.5, -.5), 2, 4, fill=False, ec=ORANGE, lw=2.2))
ax.set_title('処方割合と処方意向の分布(医師数、Q3_1=0の47名を除く)\n橙枠:Q4が4以上かつ処方割合が全体平均未満', loc='left', color=INK, fontsize=11)
f.tight_layout(); f.savefig(os.path.join(OUT, 'fig3_処方割合×処方意向の分布.png'), dpi=150, facecolor=SURF); plt.close(f)

# ---------------- 3. Q1 ----------------
t3a = pd.DataFrame({'平均': d[Q1].mean(), '標準偏差': d[Q1].std(), '4以上の割合': (d[Q1] >= 4).mean(), '1の割合': (d[Q1] == 1).mean()})
t3a.loc['5項目平均'] = [d.Q1平均.mean(), d.Q1平均.std(), (d.Q1平均 >= 4).mean(), np.nan]
save(t3a, 'table3a_Q1全体.csv')
t3b = d.groupby('診療科')[Q1 + ['Q1平均']].mean().loc[DEPTS]
t3b.loc['全体'] = d[Q1 + ['Q1平均']].mean()
save(t3b, 'table3b_Q1診療科別平均.csv')
print('\n== 3a Q1全体\n', t3a.round(3).to_string()); print('\n== 3b Q1診療科別\n', t3b.round(3).to_string())

d['Q1帯'] = pd.cut(d.Q1平均, [0, 1.5, 2.5, 3.5, 4.5, 5.01], labels=['1.0〜1.4', '1.5〜2.4', '2.5〜3.4', '3.5〜4.4', '4.5〜5.0'], right=False)


def band_t(g):
    p, t = g.Q3_1.sum(), g.Q3_2.sum()
    return pd.Series({'医師数': len(g), '処方割合(患者数加重)': t / p, '平均花粉症患者数': g.Q3_1.mean(),
                      '平均潜在未処方患者数': g.潜在未処方.mean(), '平均Q4': g.Q4.mean()})


t3c = d.groupby('Q1帯', observed=True).apply(band_t)
save(t3c, 'table3c_Q1平均帯別の処方割合.csv')
print('\n== 3c Q1帯別\n', t3c.round(3).to_string())
sp = pd.DataFrame({'Spearman相関': {c: d[c].rank().corr(d.share.rank()) for c in Q1 + ['Q1平均']}})
sp['(参考)Q4との相関'] = {c: d[c].rank().corr(d.Q4.rank()) for c in Q1 + ['Q1平均']}
save(sp, 'table3d_Q1と処方割合の相関.csv')
print('\n== 3d 相関\n', sp.round(3).to_string())
# 診療科内でも見る(診療科構成の影響確認)
t3e = d.groupby(['診療科', 'Q1帯'], observed=True).apply(band_t)[['医師数', '処方割合(患者数加重)']].unstack('診療科')
save(t3e, 'table3e_診療科×Q1帯の処方割合.csv')
print('\n== 3e 診療科×Q1帯\n', t3e.round(3).to_string())

f = fig(11, 4.8)
ax = f.add_subplot(1, 2, 1); ax.set_facecolor(SURF)
H = t3b.loc[DEPTS + ['全体'], Q1 + ['Q1平均']]
im = ax.imshow(H.values, cmap=SEQ, vmin=1.5, vmax=3.5, aspect='auto')
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f'{H.values[i, j]:.2f}', ha='center', va='center', color='white' if H.values[i, j] > 2.9 else INK, fontsize=10)
ax.set_xticks(range(6)); ax.set_xticklabels(Q1 + ['5項目\n平均'], color=INK, fontsize=9)
ax.set_yticks(range(5)); ax.set_yticklabels(DEPTS + ['全体'], color=INK)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
ax.set_title('診療科別 平均理解度(1〜5)', loc='left', color=INK, fontsize=12)
ax = f.add_subplot(1, 2, 2); style(ax)
xs = np.arange(len(t3c))
vals = t3c['処方割合(患者数加重)'] * 100
ax.bar(xs, vals, .6, color=BLUE)
for i, (v, n) in enumerate(zip(vals, t3c['医師数'])):
    ax.text(i, v + 1.2, f'{v:.0f}%', ha='center', color=INK, fontsize=10)
ax.set_xticks(xs); ax.set_xticklabels([f'{l}\n(n={int(n):,})' for l, n in zip(t3c.index, t3c['医師数'])], color=INK, fontsize=9)
ax.set_xlabel('Q1 5項目の平均理解度', color=INK2); ax.set_ylabel('薬剤A処方割合', color=INK2)
ax.set_ylim(0, 100)
ax.set_title('理解度別 薬剤A処方割合(相関であり因果ではない)', loc='left', color=INK, fontsize=11)
f.tight_layout(); f.savefig(os.path.join(OUT, 'fig4_薬剤A理解度.png'), dpi=150, facecolor=SURF); plt.close(f)

# ---------------- 4. Q2 ----------------
rows = []
for c in Q2:
    r = d[c]
    rows.append({'項目': c, '順位付けした割合': r.notna().mean(), '1位の割合': (r == 1).mean(),
                 '2位以内の割合': (r <= 2).mean(), '3位以内の割合': (r <= 3).mean(),
                 '平均順位(順位付けした医師のみ)': r.mean(), '順位付けした医師数': int(r.notna().sum())})
t4a = pd.DataFrame(rows).set_index('項目')
save(t4a, 'table4a_Q2全体.csv')
print('\n== 4a Q2全体\n', t4a.round(3).to_string())
chk = d[Q2].apply(lambda r: r.dropna().min() == 1 if r.notna().any() else False, axis=1)
print('1位を付けた医師:', chk.sum(), ' / 全員に1位あり:', bool(chk.all()), '  1位率合計:', round(t4a['1位の割合'].sum(), 3))


def q2_dept(metric):
    out = {}
    for dep, g in d.groupby('診療科'):
        out[dep] = {c: metric(g[c]) for c in Q2}
    out['全体'] = {c: metric(d[c]) for c in Q2}
    return pd.DataFrame(out).T.loc[DEPTS + ['全体']]


t4b = q2_dept(lambda r: (r == 1).mean()); t4c = q2_dept(lambda r: (r <= 2).mean()); t4d = q2_dept(lambda r: r.notna().mean())
t4e = q2_dept(lambda r: r.mean())
save(t4b, 'table4b_Q2診療科別_1位の割合.csv'); save(t4c, 'table4c_Q2診療科別_2位以内の割合.csv')
save(t4d, 'table4d_Q2診療科別_順位付けした割合.csv'); save(t4e, 'table4e_Q2診療科別_平均順位.csv')
for n, t in [('1位の割合', t4b), ('2位以内の割合', t4c), ('順位付けした割合', t4d), ('平均順位(順位付けした医師のみ)', t4e)]:
    print(f'\n== 4 診療科別 {n}\n', t.round(3).to_string())
print('\n診療科別 順位付け項目数:\n', d.groupby('診療科').apply(lambda g: g[Q2].notna().sum(axis=1).mean()).round(2).to_dict())

f = fig(11.5, 4.8)
ax = f.add_subplot(1, 2, 1); style(ax); ax.yaxis.grid(False); ax.xaxis.grid(True, color=GRID, lw=.6)
y = np.arange(5)[::-1]; h = .26
ax.barh(y + h, t4a['順位付けした割合'] * 100, h * .9, color='#9bbfe8', label='順位を付けた')
ax.barh(y, t4a['2位以内の割合'] * 100, h * .9, color=BLUE, label='2位以内')
ax.barh(y - h, t4a['1位の割合'] * 100, h * .9, color='#103c73', label='1位')
for yy, a, b, c in zip(y, t4a['順位付けした割合'], t4a['2位以内の割合'], t4a['1位の割合']):
    ax.text(a * 100 + 1, yy + h, f'{a*100:.0f}%', va='center', color=INK, fontsize=9)
    ax.text(b * 100 + 1, yy, f'{b*100:.0f}%', va='center', color=INK, fontsize=9)
    ax.text(c * 100 + 1, yy - h, f'{c*100:.0f}%', va='center', color=INK, fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(Q2, color=INK); ax.set_xlim(0, 80)
ax.set_xlabel('回答者5,000名に占める割合(%)', color=INK2)
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc='lower right')
ax.set_title('Q2 項目別の選択率と上位率', loc='left', color=INK, fontsize=12)
ax = f.add_subplot(1, 2, 2); ax.set_facecolor(SURF)
H = t4b.values * 100
ax.imshow(H, cmap=SEQ, vmin=0, vmax=60, aspect='auto')
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        ax.text(j, i, f'{H[i, j]:.0f}%', ha='center', va='center', color='white' if H[i, j] > 33 else INK, fontsize=10)
ax.set_xticks(range(5)); ax.set_xticklabels(Q2, color=INK, fontsize=9)
ax.set_yticks(range(5)); ax.set_yticklabels(DEPTS + ['全体'], color=INK)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
ax.set_title('診療科別 1位に選んだ割合', loc='left', color=INK, fontsize=12)
f.tight_layout(); f.savefig(os.path.join(OUT, 'fig5_治療上の重視項目.png'), dpi=150, facecolor=SURF); plt.close(f)

# 補足: 診療科別の属性構成比(参考)と Q3_1 分布
print('\n補足: Q3_1 患者数 分位', d.Q3_1.quantile([.1, .25, .5, .75, .9, .99]).to_dict(), ' Q3_1=0:', int((d.Q3_1 == 0).sum()), ' 処方0人の医師:', int((d.Q3_2 == 0).sum()))
print('全体: 処方割合(加重)', round(ov_share, 4), ' 個人平均', round(d.share.mean(), 4), ' 潜在未処方合計', int(d.潜在未処方.sum()))
