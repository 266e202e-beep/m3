import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-m3/f8de41a6-071b-5062-a3cf-a392fa1f2d68/scratchpad')
from base import *
OUT = '/home/user/m3/targeting_review'
save = lambda df, n, idx=False: df.to_csv(os.path.join(OUT, n), encoding='utf-8-sig', index=idx)
rng = np.random.default_rng(20261105)
d, s25, att, visits, tg, lg, ct = load()
b = d.merge(s25, on='医師ID', suffixes=('_26', '_25'), validate='1:1')
b = b[b.診療科_26.isin(['内科', '耳鼻咽喉科'])].copy(); assert len(b) == 2641 and (b.診療科_25 == b.診療科_26).all()
b['dep'] = b.診療科_25
b['q1_25'] = b[[c + '_25' for c in Q1]].mean(axis=1); b['q1_26'] = b[[c + '_26' for c in Q1]].mean(axis=1)
b['unp25'] = b.Q3_1_25 - b.Q3_2_25; b['unp26'] = b.Q3_1_26 - b.Q3_2_26
b['dq1'] = b.q1_26 - b.q1_25; b['dq4'] = b.Q4_26 - b.Q4_25; b['dq32'] = b.Q3_2_26 - b.Q3_2_25; b['dq31'] = b.Q3_1_26 - b.Q3_1_25
print('分母: 両年回答の内科・耳鼻咽喉科', len(b), ' 内科', int((b.dep == '内科').sum()), ' 耳鼻', int((b.dep == '耳鼻咽喉科').sum()), ' 2025年潜在未処方合計', int(b.unp25.sum()))
# ---- 2025年の情報だけで分類(2026年の値は使わない)
def q1b(x): return np.where(x <= 2, 'Q1低(≦2)', np.where(x < 3.5, 'Q1中(2超〜3.5未満)', 'Q1高(≧3.5)'))
def q4b(x): return np.where(x <= 2, 'Q4低(≦2)', np.where(x == 3, 'Q4中(=3)', 'Q4高(≧4)'))
b['q1c'] = q1b(b.q1_25); b['q4c'] = q4b(b.Q4_25)
def grp(df):
    return np.select([(df.q1c == 'Q1低(≦2)') & (df.q4c == 'Q4低(≦2)'), (df.q1c == 'Q1中(2超〜3.5未満)') & (df.q4c == 'Q4中(=3)'), (df.q1c == 'Q1高(≧3.5)') & (df.q4c == 'Q4高(≧4)')],
                     ['A 低理解×低意向', 'B 中理解×中意向', 'C 高理解×高意向'], default='その他(上記に該当せず)')
b['grp'] = grp(b)
GR = ['A 低理解×低意向', 'B 中理解×中意向', 'C 高理解×高意向', 'その他(上記に該当せず)']
b['hp'] = b.unp25 >= 8
H = b[b.hp].copy()
print('高ポテンシャル(2025年の潜在未処方8人以上)', len(H), ' 潜在未処方', int(H.unp25.sum()), ' 両年回答内科・耳鼻の潜在未処方に占める割合', round(H.unp25.sum() / b.unp25.sum(), 3))
def sh(g, y): return g[f'Q3_2_{y}'].sum() / g[f'Q3_1_{y}'].sum() if g[f'Q3_1_{y}'].sum() > 0 else np.nan
def base_row(g, tot):
    return {'医師数': len(g), '内科': int((g.dep == '内科').sum()), '耳鼻咽喉科': int((g.dep == '耳鼻咽喉科').sum()), '2025年 潜在未処方患者数': int(g.unp25.sum()), '高ポテンシャル内の構成比(潜在未処方)': g.unp25.sum() / tot,
            '平均潜在未処方/医師': g.unp25.mean(), '2025年 平均患者数Q3_1': g.Q3_1_25.mean(), '2025年 平均Q1': g.q1_25.mean(), '2025年 平均Q4': g.Q4_25.mean(), '2025年 処方割合': sh(g, 25), '2025年 平均処方患者数Q3_2': g.Q3_2_25.mean()}
# ===== Step1 分群表 =====
rows = []
for thr in [8]:
    for k in GR: rows.append({'グループ': k, **base_row(H[H.grp == k], H.unp25.sum())})
    rows.append({'グループ': '高ポテンシャル全体', **base_row(H, H.unp25.sum())})
t1 = pd.DataFrame(rows); save(t1, 'R1_2025年基準の医師分群表.csv'); print(t1.round(3).to_string())
grid_n = pd.crosstab(H.q1c, H.q4c).reindex(index=['Q1低(≦2)', 'Q1中(2超〜3.5未満)', 'Q1高(≧3.5)'], columns=['Q4低(≦2)', 'Q4中(=3)', 'Q4高(≧4)']).fillna(0).astype(int)
grid_u = H.pivot_table(index='q1c', columns='q4c', values='unp25', aggfunc='sum').reindex(index=grid_n.index, columns=grid_n.columns).fillna(0).astype(int)
save(grid_n, 'R1b_Q1×Q4グリッド_医師数.csv', True); save(grid_u, 'R1b_Q1×Q4グリッド_潜在未処方.csv', True); print(grid_n, grid_u)
rows = []
for thr in [7, 8, 10]:
    gg = b[b.unp25 >= thr]
    for k in GR:
        x = gg[gg.grp == k]; rows.append({'基準(2025年潜在未処方≧)': thr, 'グループ': k, '医師数': len(x), '潜在未処方患者数': int(x.unp25.sum())})
save(pd.DataFrame(rows), 'R1c_基準感度_群の規模.csv'); print(pd.DataFrame(rows).pivot(index='グループ', columns='基準(2025年潜在未処方≧)').to_string())
# ===== Step2 変化の比較 =====
def ci_mean(x):
    x = x.dropna(); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan; return m, f'[{m-1.96*se:.2f}, {m+1.96*se:.2f}]'
def boot_sh(g, n=800):
    a = g[['Q3_1_25', 'Q3_2_25', 'Q3_1_26', 'Q3_2_26']].values; out = []
    for _ in range(n):
        r = a[rng.integers(0, len(a), len(a))]; out.append(r[:, 3].sum() / r[:, 2].sum() - r[:, 1].sum() / r[:, 0].sum())
    return np.percentile(out, [2.5, 97.5]) * 100
def change_row(g, label, small=30):
    r = {'区分': label, '医師数': len(g)}
    if len(g) < small: r['備考'] = f'n<{small}のため変化は省略'; return r
    for nm, col in [('Q1平均', 'dq1'), ('Q4', 'dq4'), ('Q3_2(処方患者数)', 'dq32'), ('Q3_1(患者数)', 'dq31')]:
        m, c = ci_mean(g[col]); r[f'{nm}の変化'] = m; r[f'{nm}の変化 95%区間'] = c
    r['処方割合の変化(pt)'] = (sh(g, 26) - sh(g, 25)) * 100; lo, hi = boot_sh(g); r['処方割合の変化 95%区間(pt)'] = f'[{lo:.1f}, {hi:.1f}]'
    r['2025年→2026年 処方割合'] = f'{sh(g,25):.3f}→{sh(g,26):.3f}'
    r['Q3_2の純増÷2025年潜在未処方'] = g.dq32.sum() / g.unp25.sum()
    r['2026年の潜在未処方(平均)'] = g.unp26.mean(); r['2025年の潜在未処方(平均)'] = g.unp25.mean()
    return r
rows = [change_row(H[H.grp == k], k) for k in GR] + [change_row(H, '高ポテンシャル全体')]
t2 = pd.DataFrame(rows); save(t2, 'R2_群別の変化.csv'); print(t2.round(3).T.to_string())
# 平均への回帰・接触の影響を見るための層別: 接触なし(面談なし・配信対象外) / 面談なし・MR君あり / 面談あり
H['cs'] = np.where(H.面談あり, '面談あり', np.where(H.配信対象, '面談なし・MR君配信対象', '面談なし・MR君対象外(接触なし)'))
rows = []
for k in GR[:3]:
    for c in ['面談なし・MR君対象外(接触なし)', '面談なし・MR君配信対象', '面談あり']:
        rows.append({'グループ': k, **change_row(H[(H.grp == k) & (H.cs == c)], c)})
t2b = pd.DataFrame(rows); save(t2b, 'R2b_群×接触状況別の変化.csv'); print(t2b[['グループ', '区分', '医師数', 'Q1平均の変化', 'Q4の変化', 'Q3_2(処方患者数)の変化', 'Q3_1(患者数)の変化', '処方割合の変化(pt)', 'Q3_2の純増÷2025年潜在未処方', '2025年→2026年 処方割合']].round(3).to_string())
# 群間差: B-A, C-A (全体・接触なしのみ・面談ありのみ)
def diff(g1, g2, label):
    r = {'比較': label, 'n(前)': len(g1), 'n(後)': len(g2)}
    if min(len(g1), len(g2)) < 30: r['備考'] = 'n<30'; return r
    for nm, col in [('Q1平均', 'dq1'), ('Q4', 'dq4'), ('Q3_2', 'dq32'), ('Q3_1', 'dq31')]:
        a, c = g1[col], g2[col]; m = a.mean() - c.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c)); r[f'{nm}の変化の差'] = m; r[f'{nm} 95%区間'] = f'[{m-1.96*se:.2f}, {m+1.96*se:.2f}]'
    sa = (sh(g1, 26) - sh(g1, 25)) * 100; sc = (sh(g2, 26) - sh(g2, 25)) * 100; r['処方割合の変化の差(pt)'] = sa - sc
    out = []
    for _ in range(800):
        x = g1.iloc[rng.integers(0, len(g1), len(g1))]; y = g2.iloc[rng.integers(0, len(g2), len(g2))]; out.append(((sh(x, 26) - sh(x, 25)) - (sh(y, 26) - sh(y, 25))) * 100)
    lo, hi = np.percentile(out, [2.5, 97.5]); r['処方割合の差 95%区間(pt)'] = f'[{lo:.1f}, {hi:.1f}]'
    r['純増÷潜在未処方の差'] = g1.dq32.sum() / g1.unp25.sum() - g2.dq32.sum() / g2.unp25.sum(); return r
rows = []
for sc_name, mask in [('全体', H.cs.notna()), ('接触なし(面談なし・MR君対象外)のみ', H.cs == '面談なし・MR君対象外(接触なし)'), ('面談ありのみ', H.cs == '面談あり'), ('面談なし(MR君は問わない)', H.cs != '面談あり')]:
    X = H[mask]
    for a_, c_ in [('B 中理解×中意向', 'A 低理解×低意向'), ('C 高理解×高意向', 'A 低理解×低意向')]:
        rows.append({'範囲': sc_name, **diff(X[X.grp == a_], X[X.grp == c_], f'{a_[0]} − {c_[0]}')})
t2c = pd.DataFrame(rows); save(t2c, 'R2c_群間差_B−A_C−A.csv'); print(t2c.round(3).T.to_string())
# ===== Step3 接触状況 =====
rows = []
for k in GR + ['高ポテンシャル全体']:
    g = H if k == '高ポテンシャル全体' else H[H.grp == k]
    rows.append({'グループ': k, '医師数': len(g), 'MR面談あり率': g.面談あり.mean(), '平均面談回数': g.面談回数.mean(), '面談ありの医師の平均回数': g[g.面談あり].面談回数.mean(), 'MR君配信対象率': g.配信対象.mean(),
                 '既読率(全医師比)': g.既読あり.mean(), '既読率(配信対象内)': g[g.配信対象].既読あり.mean() if g.配信対象.any() else np.nan, '平均既読本数': g.既読本数.mean(),
                 '接触なし(面談なし・対象外)の割合': (g.cs == '面談なし・MR君対象外(接触なし)').mean()})
t3 = pd.DataFrame(rows); save(t3, 'R3_群別の接触状況.csv'); print(t3.round(3).to_string())
H['c4'] = np.where(H.面談あり, '面談あり', '面談なし') + '・' + np.where(H.既読あり, 'MR君既読あり', 'MR君既読なし')
rows = []
for k in GR[:3] + ['高ポテンシャル全体']:
    g = H if k == '高ポテンシャル全体' else H[H.grp == k]
    for c in ['面談なし・MR君既読なし', '面談なし・MR君既読あり', '面談あり・MR君既読なし', '面談あり・MR君既読あり']:
        rows.append({'グループ': k, **change_row(g[g.c4 == c], c)})
t3b = pd.DataFrame(rows); save(t3b, 'R3b_群×面談×既読の4状態別の変化.csv'); print(t3b[['グループ', '区分', '医師数', 'Q1平均の変化', 'Q4の変化', 'Q3_2(処方患者数)の変化', '処方割合の変化(pt)', 'Q3_2の純増÷2025年潜在未処方']].round(3).to_string())
# ===== 参考: 2026年定義の316名との関係(既存結果の再確認のみ) =====
g8 = pd.read_csv('/tmp/claude-0/-home-user-m3/f8de41a6-071b-5062-a3cf-a392fa1f2d68/scratchpad/_work_g8.csv'); A26 = set(g8[g8.grp == 'A 低理解×低意向'].医師ID)
bb = b[b.医師ID.isin(A26)]
print('2026年定義316名のうち両年回答', len(bb), ' うち2025年分類: ', bb.grp.value_counts().to_dict(), ' 2025年高ポテンシャル', int(bb.hp.sum()))
print('2025年A群(高ポテンシャル×低理解×低意向)のうち、2026年の316名に残った', int(H[H.grp == GR[0]].医師ID.isin(A26).sum()), '/', int((H.grp == GR[0]).sum()))
H.to_csv('/tmp/claude-0/-home-user-m3/f8de41a6-071b-5062-a3cf-a392fa1f2d68/scratchpad/_H25.csv', index=False)
