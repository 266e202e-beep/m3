import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-m3/f8de41a6-071b-5062-a3cf-a392fa1f2d68/scratchpad')
from base import *
OUT = '/home/user/m3/visit_dose'
save = lambda df, n, idx=False: df.to_csv(os.path.join(OUT, n), encoding='utf-8-sig', index=idx)
rng = np.random.default_rng(20261201)
d, s25, att, visits, tg, lg, ct = load()
b = d.merge(s25, on='医師ID', suffixes=('_26', '_25'), validate='1:1')
b = b[b.診療科_26.isin(['内科', '耳鼻咽喉科'])].copy(); assert len(b) == 2641
b['dep'] = b.診療科_26
b['q1_25'] = b[[c + '_25' for c in Q1]].mean(axis=1); b['q1_26'] = b[[c + '_26' for c in Q1]].mean(axis=1)
b['unp25'] = b.Q3_1_25 - b.Q3_2_25
b['dq1'] = b.q1_26 - b.q1_25; b['dq4'] = b.Q4_26 - b.Q4_25; b['dq32'] = b.Q3_2_26 - b.Q3_2_25; b['dq31'] = b.Q3_1_26 - b.Q3_1_25
S1 = b[(b.q1_25 <= 2) & (b.Q4_25 <= 2)].copy(); A = S1[S1.unp25 >= 8].copy()
print('S1', len(S1), ' A', len(A))
print('面談回数の分布(A):', A.面談回数.value_counts().sort_index().to_dict()); print('面談回数の分布(S1):', S1.面談回数.value_counts().sort_index().to_dict())
def sh(g, y): return g[f'Q3_2_{y}'].sum() / g[f'Q3_1_{y}'].sum()
def vlab(v): return str(int(v)) if v <= 4 else '5以上'
def fine(v): return str(int(v)) if v <= 8 else '9以上'
def cim(x):
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan; return m, m - 1.96 * se, m + 1.96 * se
def boot_sh(g, n=500):
    a = g[['Q3_1_25', 'Q3_2_25', 'Q3_1_26', 'Q3_2_26']].values; out = []
    for _ in range(n):
        r = a[rng.integers(0, len(a), len(a))]; out.append(r[:, 3].sum() / r[:, 2].sum() - r[:, 1].sum() / r[:, 0].sum())
    return np.percentile(out, [2.5, 97.5]) * 100
def table(df, key, order, nm, minn=20):
    r1, r2 = [], []
    for k in order:
        g = df[df[key] == k]
        if len(g) == 0: continue
        r1.append({'群': nm, '面談回数': k, '医師数': len(g), '内科': int((g.dep == '内科').sum()), '耳鼻咽喉科': int((g.dep == '耳鼻咽喉科').sum()), '2025年 平均患者数Q3_1': g.Q3_1_25.mean(), '2025年 平均処方患者数Q3_2': g.Q3_2_25.mean(),
                   '2025年 処方割合': sh(g, 25), '2025年 平均Q1': g.q1_25.mean(), '2025年 平均Q4': g.Q4_25.mean(), '2025年 平均潜在未処方': g.unp25.mean(), 'MR君配信対象率': g.配信対象.mean(), 'MR君既読率(全医師)': g.既読あり.mean(), '少数(n<20)': len(g) < minn})
        r = {'群': nm, '面談回数': k, '医師数': len(g)}
        if len(g) >= 10:
            for lab, col in [('Q3_2(処方患者数)', 'dq32'), ('Q3_1(患者数)', 'dq31'), ('Q1', 'dq1'), ('Q4', 'dq4')]:
                m, lo, hi = cim(g[col]); r[f'{lab}の変化'] = m; r[f'{lab} 95%区間'] = f'[{lo:.2f}, {hi:.2f}]'
            r['処方割合の変化(pt)'] = (sh(g, 26) - sh(g, 25)) * 100; lo, hi = boot_sh(g); r['処方割合 95%区間(pt)'] = f'[{lo:.1f}, {hi:.1f}]'
            r['処方割合 25→26'] = f'{sh(g,25):.3f}→{sh(g,26):.3f}'
        else: r['備考'] = 'n<10のため省略'
        r['少数(n<20)'] = len(g) < minn
        r2.append(r)
    return pd.DataFrame(r1), pd.DataFrame(r2)
A['v6'] = A.面談回数.map(vlab); S1['v6'] = S1.面談回数.map(vlab); A['vf'] = A.面談回数.map(fine); S1['vf'] = S1.面談回数.map(fine)
O6 = ['0', '1', '2', '3', '4', '5以上']; OF = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9以上']
for nm, df in [('A群346名(主)', A), ('S1群986名(参考)', S1)]:
    t1, t2 = table(df, 'v6', O6, nm); tf1, tf2 = table(df, 'vf', OF, nm)
    tag = 'A' if nm.startswith('A') else 'S1'
    save(t1, f'V1_{tag}_面談回数別の医師数と2025年の特徴.csv'); save(t2, f'V2_{tag}_面談回数別の2025→2026年の変化.csv'); save(tf1, f'V1b_{tag}_5回以上を細分.csv'); save(tf2, f'V2b_{tag}_5回以上を細分_変化.csv')
    print('\n====', nm); print(t1.round(3).to_string()); print(t2.round(3).to_string()); print('細分:'); print(tf1[['面談回数', '医師数']].T.to_string()); print(tf2[['面談回数', '医師数', 'Q3_2(処方患者数)の変化', '処方割合の変化(pt)']].round(2).to_string() if 'Q3_2(処方患者数)の変化' in tf2 else tf2)
# ---- Task3 隣り合う回数の比較(素の差 + 層別に揃えた差) ----
def strata(df, mode):
    st = df.dep.astype(str)
    st = st + '|' + pd.cut(df.Q3_1_25, [-1, 9, 12, 100], labels=list('xyz')).astype(str)
    if mode == 'full': st = st + '|' + pd.cut(df.Q3_2_25, [-1, 0, 2, 100], labels=list('abc')).astype(str) + '|' + df.Q4_25.clip(1, 3).astype(str)
    return st
def pair(df, lo_v, hi_v, mode, label, minn=20, B=300):
    g0 = df[df.vcat == lo_v]; g1 = df[df.vcat == hi_v]; r = {'比較': label, 'n(少ない回数)': len(g0), 'n(多い回数)': len(g1)}
    if min(len(g0), len(g1)) < minn: r['備考'] = f'n<{minn}のため比較しない'; return r
    r['2025年Q3_2(少/多)'] = f'{g0.Q3_2_25.mean():.2f}/{g1.Q3_2_25.mean():.2f}'; r['2025年Q3_1(少/多)'] = f'{g0.Q3_1_25.mean():.1f}/{g1.Q3_1_25.mean():.1f}'
    r['2025年Q4(少/多)'] = f'{g0.Q4_25.mean():.2f}/{g1.Q4_25.mean():.2f}'; r['2025年Q1(少/多)'] = f'{g0.q1_25.mean():.2f}/{g1.q1_25.mean():.2f}'
    def est(x0, x1, strat=False):
        if not strat:
            return {'dq32': x1.dq32.mean() - x0.dq32.mean(), 'dq31': x1.dq31.mean() - x0.dq31.mean(), 'dsh': ((sh(x1, 26) - sh(x1, 25)) - (sh(x0, 26) - sh(x0, 25))) * 100, 'cov': 1.0}
        z = pd.concat([x0.assign(_t=0), x1.assign(_t=1)]); z['_s'] = strata(z, mode); w, o = [], {'dq32': [], 'dq31': [], 'dsh': []}; ntot = (z._t == 1).sum(); nsup = 0
        for s, gg in z.groupby('_s'):
            a, c = gg[gg._t == 0], gg[gg._t == 1]
            if len(a) < 3 or len(c) < 3: continue
            nsup += len(c); w.append(len(c)); o['dq32'].append(c.dq32.mean() - a.dq32.mean()); o['dq31'].append(c.dq31.mean() - a.dq31.mean())
            o['dsh'].append(((sh(c, 26) - sh(c, 25)) - (sh(a, 26) - sh(a, 25))) * 100)
        if nsup / ntot < 0.7: return None
        return {k: float(np.average(v, weights=w)) for k, v in o.items()} | {'cov': nsup / ntot}
    raw = est(g0, g1); r['素の差 Q3_2'] = raw['dq32']; r['素の差 Q3_1'] = raw['dq31']; r['素の差 処方割合(pt)'] = raw['dsh']
    ci = []
    for _ in range(B):
        ci.append([est(g0.iloc[rng.integers(0, len(g0), len(g0))], g1.iloc[rng.integers(0, len(g1), len(g1))])[k] for k in ['dq32', 'dq31', 'dsh']])
    ci = np.percentile(np.array(ci), [2.5, 97.5], axis=0); r['素の差 Q3_2 95%区間'] = f'[{ci[0,0]:.2f}, {ci[1,0]:.2f}]'; r['素の差 処方割合 95%区間(pt)'] = f'[{ci[0,2]:.1f}, {ci[1,2]:.1f}]'
    adj = est(g0, g1, True)
    if adj is None: r['層別差'] = '共通サポート不足(群の70%未満しか比較可能な層がない)のため算出しない'
    else:
        r['層別差 Q3_2'] = adj['dq32']; r['層別差 Q3_1'] = adj['dq31']; r['層別差 処方割合(pt)'] = adj['dsh']; r['比較可能な割合'] = adj['cov']
        cc = []
        for _ in range(B):
            e = est(g0.iloc[rng.integers(0, len(g0), len(g0))], g1.iloc[rng.integers(0, len(g1), len(g1))], True); cc.append([e[k] for k in ['dq32', 'dsh']] if e else [np.nan, np.nan])
        cc = np.nanpercentile(np.array(cc), [2.5, 97.5], axis=0); r['層別差 Q3_2 95%区間'] = f'[{cc[0,0]:.2f}, {cc[1,0]:.2f}]'; r['層別差 処方割合 95%区間(pt)'] = f'[{cc[0,1]:.1f}, {cc[1,1]:.1f}]'
    return r
for nm, df, mode in [('A群346名(主)', A, 'coarse'), ('S1群986名(参考)', S1, 'full')]:
    df = df.copy(); df['vcat'] = df.面談回数.clip(upper=5).astype(int)
    rows = [pair(df, a, bb, mode, f'{bb}回{"以上" if bb==5 else ""} vs {a}回') for a, bb in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]]
    # 参考: 区間をまとめた比較
    df['vgrp'] = pd.cut(df.面談回数, [-1, 0, 2, 3, 100], labels=[0, 1, 2, 3]).astype(int)
    t = pd.DataFrame(rows); tag = 'A' if nm.startswith('A') else 'S1'; save(t, f'V3_{tag}_隣り合う回数の比較.csv'); print('\n====', nm, '隣り合う回数の比較'); print(t.round(3).T.to_string())
    # まとめ比較: 1-2回, 3回, 4+回 vs 0回
    df['vc'] = pd.cut(df.面談回数, [-1, 0, 2, 3, 100], labels=['0回', '1〜2回', '3回', '4回以上']).astype(str)
    rows = []
    for k in ['1〜2回', '3回', '4回以上']:
        g0 = df[df.vc == '0回']; g1 = df[df.vc == k]; 
        r = {'比較': f'{k} vs 0回', 'n(0回)': len(g0), f'n': len(g1), 'Q3_2の変化(0回/該当)': f'{g0.dq32.mean():.2f}/{g1.dq32.mean():.2f}', '処方割合の変化pt(0回/該当)': f'{(sh(g0,26)-sh(g0,25))*100:.1f}/{(sh(g1,26)-sh(g1,25))*100:.1f}'}
        rows.append(r)
    t = pd.DataFrame(rows); save(t, f'V3b_{tag}_0回との比較(まとめ).csv'); print(t.to_string())
# 診療科別(粗い区分)
rows = []
for dep in ['内科', '耳鼻咽喉科']:
    g = A[A.dep == dep].copy(); g['vc'] = pd.cut(g.面談回数, [-1, 0, 2, 3, 100], labels=['0回', '1〜2回', '3回', '4回以上']).astype(str)
    for k in ['0回', '1〜2回', '3回', '4回以上']:
        x = g[g.vc == k]; r = {'診療科': dep, '面談回数': k, '医師数': len(x)}
        if len(x) >= 10: r.update({'Q3_2の変化': x.dq32.mean(), 'Q3_1の変化': x.dq31.mean(), '処方割合の変化(pt)': (sh(x, 26) - sh(x, 25)) * 100, '2025年Q3_2': x.Q3_2_25.mean(), '2025年Q3_1': x.Q3_1_25.mean()})
        else: r['備考'] = 'n<10のため省略'
        rows.append(r)
t = pd.DataFrame(rows); save(t, 'V4_A群の診療科別.csv'); print(t.round(2).to_string())
# 上限: 処方患者数の変化は2025年の潜在未処方を超えられない(天井)
for nm, df in [('A', A), ('S1', S1)]:
    print(nm, '2025年潜在未処方の平均', round(df.unp25.mean(), 2), ' 面談4回以上のQ3_2変化 平均', round(df[df.面談回数 >= 4].dq32.mean(), 2), ' 同 2025年潜在未処方', round(df[df.面談回数 >= 4].unp25.mean(), 2))
