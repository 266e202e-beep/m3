# -*- coding: utf-8 -*-
"""最終発表v5の分析: 現状の配信の評価 → 処方拡大の機会(WHO) → 重点医師の特徴(BENCHMARK)と診療科別の重視項目 → 配信計画の入力

使い方: DATA_DIR に元データ(CSV 7本)のフォルダを指定して実行する。
    DATA_DIR=/path/to/data python3 v5_analysis.py   (その後 python3 v5_plan.py)
出力: ./output/ に集計CSV(医師単位のデータは出力しない)と params_v5.json(Excel・PPT用の入力値)。
注意: 観察データの比較であり、差は因果効果を示さない。完読・コンテンツ別の効果・長さの分析は使わない。
"""
import os, glob, json
import numpy as np, pandas as pd

DATA = os.environ.get('DATA_DIR', '/root/.claude/uploads/f8de41a6-071b-5062-a3cf-a392fa1f2d68')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'); os.makedirs(OUT, exist_ok=True)
save = lambda df, n, idx=False: df.to_csv(os.path.join(OUT, n), encoding='utf-8-sig', index=idx)
rd = lambda n: pd.read_csv(glob.glob(os.path.join(DATA, f'*{n}.csv'))[0])
rng = np.random.default_rng(2026)
F = {1: '効果発現速度', 2: '適用症状の広さ', 3: '副作用の少なさ', 4: '飲み合わせ', 5: '24h持続性'}
Q1, Q2 = [f'Q1_{i}' for i in range(1, 6)], [f'Q2_{i}' for i in range(1, 6)]
FOCUS = ['内科', '耳鼻咽喉科']

# ==================== 0. 読み込み・結合・変数作成 ====================
s26, s25, att, vis, tg, lg, ct = (rd(n) for n in ['survey_2026', 'survey_2025', 'doctor_attributes', 'mr_visits', 'mrkun_targets', 'mrkun_logs', 'mrkun_contents_summary'])
for t in (s26, s25, att, vis, tg): assert t.医師ID.is_unique
assert len(lg) == len(lg.drop_duplicates(['医師ID', 'コンテンツID']))           # 医師×コンテンツの重複なし
d = s26.merge(vis, on='医師ID', how='left', validate='1:1')
assert d.面談回数.notna().all() and len(d) == 5000
chk = d.merge(att, on='医師ID', suffixes=('', '_属性'))
assert all((chk[c] == chk[c + '_属性']).all() for c in ['性別', '年代', '診療科', '地域', '施設区分'])   # アンケートと属性ファイルの属性は一致
d['潜在未処方'] = d.Q3_1 - d.Q3_2                                              # 花粉症患者数 − 薬剤A処方患者数(直近1年・自己申告)
d['処方率'] = d.Q3_2 / d.Q3_1.replace(0, np.nan)                              # 患者0人の医師は欠損
d['理解度'] = d[Q1].mean(axis=1)                                               # 5特性の平均(1〜5)
d['面談あり'] = d.面談回数 > 0
d['配信対象'] = d.医師ID.isin(tg.医師ID)
d['既読件数'] = d.医師ID.map(lg.groupby('医師ID').size()).fillna(0).astype(int)
d['既読あり'] = d.既読件数 > 0
d['最重視'] = d[Q2].idxmin(axis=1).where(d[Q2].notna().any(axis=1)).map(lambda q: F[int(q[-1])] if isinstance(q, str) else '未回答')
d['意向高'] = d.Q4 >= 4
d['潜在帯'] = pd.cut(d.潜在未処方, [-1, 2, 7, 100], labels=['0〜2人', '3〜7人', '8人以上']).astype(str)
d['接触'] = np.where(d.面談あり, 'MR面談あり', np.where(d.配信対象, 'MR君のみ', '未接触'))
P = {}   # Excel・PPTに渡す値

# ==================== 1. WHO ====================
w1 = d.groupby('診療科').agg(医師数=('医師ID', 'size'), 花粉症患者=('Q3_1', 'sum'), 処方患者=('Q3_2', 'sum'), 潜在未処方=('潜在未処方', 'sum'))
w1['潜在未処方の構成比'] = w1.潜在未処方 / w1.潜在未処方.sum(); w1['処方率'] = w1.処方患者 / w1.花粉症患者
save(w1.reset_index(), 'W1_診療科別_潜在未処方.csv')
nj = d[d.診療科.isin(FOCUS)].copy()
w2 = nj.潜在未処方.value_counts().sort_index().rename('医師数').reset_index().rename(columns={'index': '潜在未処方'})
save(w2, 'W2_内科耳鼻_潜在未処方の分布.csv')
w3 = nj.groupby('潜在帯').agg(医師数=('医師ID', 'size'), 潜在未処方=('潜在未処方', 'sum'), 平均処方率=('処方率', 'mean'), MR君配信率=('配信対象', 'mean'), MR面談率=('面談あり', 'mean'),
                            未接触=('接触', lambda x: (x == '未接触').sum()), 理解度=('理解度', 'mean'), 処方意向=('Q4', 'mean'))
w3['医師の割合'] = w3.医師数 / len(nj); w3['潜在の割合'] = w3.潜在未処方 / nj.潜在未処方.sum(); save(w3.reset_index(), 'W3_内科耳鼻_潜在帯別.csv')
un = nj[nj.接触 == '未接触'].copy(); un['帯'] = pd.cut(un.潜在未処方, [-1, 2, 4, 7, 100], labels=['0〜2人', '3〜4人', '5〜7人', '8人以上']).astype(str)
w4 = un.groupby('帯').agg(医師数=('医師ID', 'size'), 潜在未処方=('潜在未処方', 'sum'), 理解度=('理解度', 'mean'), 処方意向=('Q4', 'mean')); save(w4.reset_index(), 'W4_未接触医師_潜在帯別.csv')
P.update(潜在_全体=int(d.潜在未処方.sum()), 潜在_内科耳鼻=int(nj.潜在未処方.sum()), 医師_内科耳鼻=len(nj), 重点医師=int((nj.潜在未処方 >= 8).sum()),
         重点_潜在=int(nj[nj.潜在未処方 >= 8].潜在未処方.sum()), 未接触_内科耳鼻=len(un), 未接触_潜在=int(un.潜在未処方.sum()),
         未接触_8人以上=int((un.潜在未処方 >= 8).sum()), 未接触_5to7=int(un.潜在未処方.between(5, 7).sum()), 未接触_5to7_潜在=int(un[un.潜在未処方.between(5, 7)].潜在未処方.sum()),
         配信率_0to2=float(d[d.潜在帯 == '0〜2人'].配信対象.mean()), 配信率_8plus=float(d[d.潜在帯 == '8人以上'].配信対象.mean()))
print('WHO', {k: P[k] for k in list(P)[:10]})

# ==================== 2. WHY(重点医師 vs 内科・耳鼻のその他の医師) ====================
nj['重点'] = nj.潜在未処方 >= 8
T = nj[nj.重点].copy()
def prof(g):
    r = {'医師数': len(g), '花粉症患者(平均)': g.Q3_1.mean(), '処方率(平均)': g.処方率.mean(), '潜在未処方(平均)': g.潜在未処方.mean(), '理解度(5特性平均)': g.理解度.mean()}
    r.update({f'理解度:{F[i]}': g[f'Q1_{i}'].mean() for i in range(1, 6)})
    r.update({'処方意向(平均)': g.Q4.mean(), '処方意向4以上の割合': g.意向高.mean(), 'MR面談ありの割合': g.面談あり.mean(), '面談回数(平均)': g.面談回数.mean(),
              'MR君配信率': g.配信対象.mean(), 'MR君既読(配信対象1人あたり/年)': g[g.配信対象].既読件数.mean(), 'MR君既読医師率(配信対象)': g[g.配信対象].既読あり.mean()})
    r.update({f'最重視:{f}': (g.最重視 == f).mean() for f in F.values()})
    r.update({f'情報源:{q}': (g.Q5 == q).mean() for q in ['MRからの対面での説明', 'MR君によるWEBでの説明', '学会・論文', '同僚', 'その他']})
    for a in ['施設区分', '年代']: r.update({f'{a}:{k}': (g[a] == k).mean() for k in sorted(d[a].unique())})
    return pd.Series(r)
y1 = pd.concat({'重点医師(潜在8人以上)': prof(T), '内科耳鼻のその他(潜在7人以下)': prof(nj[~nj.重点])}, axis=1); y1['差'] = y1.iloc[:, 0] - y1.iloc[:, 1]
save(y1.reset_index().rename(columns={'index': '項目'}), 'Y1_重点医師とその他の比較.csv')
# 仮説①〜⑤の判定表(重点医師 vs 内科・耳鼻のその他の医師)
o = nj[~nj.重点]
y2 = pd.DataFrame([
    ('① 理解度が低い', '確認', f"5特性平均 {T.理解度.mean():.2f} vs {o.理解度.mean():.2f}(5特性すべてで低い)"),
    ('② 処方意向が低い', '確認', f"平均 {T.Q4.mean():.2f} vs {o.Q4.mean():.2f}"),
    ('③ 意向はあるが処方が少ない', '一部', f"重点医師の{T.意向高.mean():.1%}({int(T.意向高.sum())}名)が処方意向4以上"),
    ('④ MRとの接触が少ない', '確認(MR面談)', f"MR面談あり {T.面談あり.mean():.1%} vs {o.面談あり.mean():.1%}。MR君の配信率は {T.配信対象.mean():.1%} vs {o.配信対象.mean():.1%}で差は小さい"),
    ('⑤ 求める情報が診療科で異なる', '確認', f"最も重視する特性: 内科の重点医師は飲み合わせ{(T[T.診療科 == '内科'].最重視 == '飲み合わせ').mean():.1%}、耳鼻咽喉科は24h持続性{(T[T.診療科 == '耳鼻咽喉科'].最重視 == '24h持続性').mean():.1%}・副作用{(T[T.診療科 == '耳鼻咽喉科'].最重視 == '副作用の少なさ').mean():.1%}")],
    columns=['仮説', '判定', '根拠(重点医師 vs その他)'])
save(y2, 'Y2_仮説の検証.csv')
# 処方率の関連要因(内科・耳鼻、花粉症患者8人以上、重回帰。因果ではない)
r8 = nj[nj.Q3_1 >= 8].copy()
def ols(df, y, xs):
    X = np.column_stack([np.ones(len(df))] + [df[x].astype(float) for x in xs]); yy = df[y].values.astype(float)
    b, *_ = np.linalg.lstsq(X, yy, rcond=None); e = yy - X @ b; XtXi = np.linalg.inv(X.T @ X); V = XtXi @ (X.T * e ** 2) @ X @ XtXi
    return pd.DataFrame({'係数': b, '標準誤差': np.sqrt(np.diag(V))}, index=['切片'] + xs)
dm = pd.get_dummies(r8[['施設区分', '地域', '年代']], drop_first=True).astype(float); r8 = pd.concat([r8, dm], axis=1)
r8['内科'] = (r8.診療科 == '内科').astype(float); r8['面談あり_'] = r8.面談あり.astype(float); r8['配信対象_'] = r8.配信対象.astype(float)
xs = ['理解度', 'Q4', '面談あり_', '配信対象_', '内科', 'Q3_1'] + list(dm.columns)
y4 = ols(r8.assign(処方率pt=r8.処方率 * 100), '処方率pt', xs); y4['下限'] = y4.係数 - 1.96 * y4.標準誤差; y4['上限'] = y4.係数 + 1.96 * y4.標準誤差
save(y4.reset_index().rename(columns={'index': '変数'}), 'Y4_処方率の関連要因_重回帰.csv')

# ==================== 3. BENCHMARK(診療科×患者規模×施設区分をそろえた比較) ====================
nj['患者帯'] = pd.cut(nj.Q3_1, [-1, 7, 11, 15, 20, 40], labels=['〜7', '8〜11', '12〜15', '16〜20', '21+']).astype(str)
KEY = ['診療科', '患者帯', '施設区分']
Tm = nj[nj.重点].copy(); Hm = nj[(nj.Q3_1 >= 8) & (nj.処方率 >= 0.5) & ~nj.重点].copy()   # 類似高処方医師: 患者8人以上・処方率50%以上
ov = Tm.groupby(KEY).size().rename('重点医師').to_frame().join(Hm.groupby(KEY).size().rename('類似高処方医師'), how='left').fillna(0)
save(ov.reset_index(), 'B1_比較条件の重なり.csv')
ok = ov[ov.類似高処方医師 > 0].index
Tm = Tm[Tm.set_index(KEY).index.isin(ok)]
def att_w(t, h):  # 重点医師の層構成に合わせた重み
    w = (t.groupby(KEY).size() / h.groupby(KEY).size()).rename('w').reset_index(); return h.drop(columns='w', errors='ignore').merge(w, on=KEY, how='left').w.values
Hm['w'] = att_w(Tm, Hm)
MET = {'理解度(5特性平均)': '理解度', **{f'理解度:{F[i]}': f'Q1_{i}' for i in range(1, 6)}, '処方意向(平均)': 'Q4', '処方意向4以上の割合': '意向高', 'MR面談ありの割合': '面談あり', '面談回数(平均)': '面談回数',
       'MR君配信率': '配信対象', 'MR君既読(1人あたり/年)': '既読件数', '情報源:MR対面': ('Q5', 'MRからの対面での説明'), '情報源:MR君': ('Q5', 'MR君によるWEBでの説明'), '情報源:学会・論文': ('Q5', '学会・論文'),
       '情報源:同僚': ('Q5', '同僚'), '情報源:その他': ('Q5', 'その他'), **{f'最重視:{f}': ('最重視', f) for f in F.values()}, '処方率': '処方率', '花粉症患者(平均)': 'Q3_1'}
def val(g, m, w=None):
    x = (g[m[0]] == m[1]).astype(float) if isinstance(m, tuple) else g[m].astype(float)
    return np.average(x, weights=w) if w is not None else x.mean()
def bench(t, h, label=''):
    rows = []
    boots = {k: [] for k in MET}
    for _ in range(500):
        tb = t.iloc[rng.integers(0, len(t), len(t))]; hb = h.iloc[rng.integers(0, len(h), len(h))].copy()
        hb = hb[hb.set_index(KEY).index.isin(tb.set_index(KEY).index.unique())]; hb['w'] = att_w(tb, hb); tb = tb[tb.set_index(KEY).index.isin(hb.set_index(KEY).index.unique())]
        for k, m in MET.items(): boots[k].append(val(tb, m) - val(hb, m, hb.w))
    for k, m in MET.items():
        a, b = val(t, m), val(h, m, h.w); lo, hi = np.percentile(boots[k], [2.5, 97.5])
        rows.append({'範囲': label, '項目': k, '重点医師': a, '類似高処方医師': b, '差': a - b, '差_下限': lo, '差_上限': hi, '重点医師数': len(t), '類似高処方医師数': len(h)})
    return pd.DataFrame(rows)
b3 = bench(Tm, Hm, '内科・耳鼻')
def bench_tgt(t, h):  # 配信対象の医師だけで比べた既読(1人あたり/年)と既読医師率
    t, h = t[t.配信対象], h[h.配信対象].copy(); h['w'] = att_w(t, h); h = h[h.w.notna()]
    out = []
    for k, m in [('MR君既読(配信対象1人あたり/年)', '既読件数'), ('MR君既読医師率(配信対象)', '既読あり')]:
        bs = []
        for _ in range(500):
            tb = t.iloc[rng.integers(0, len(t), len(t))]; hb = h.iloc[rng.integers(0, len(h), len(h))].copy()
            hb = hb[hb.set_index(KEY).index.isin(tb.set_index(KEY).index.unique())]; hb['w'] = att_w(tb, hb); tb = tb[tb.set_index(KEY).index.isin(hb.set_index(KEY).index.unique())]
            bs.append(val(tb, m) - val(hb, m, hb.w))
        a_, b_ = val(t, m), val(h, m, h.w); lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append({'範囲': '内科・耳鼻(配信対象のみ)', '項目': k, '重点医師': a_, '類似高処方医師': b_, '差': a_ - b_, '差_下限': lo, '差_上限': hi, '重点医師数': len(t), '類似高処方医師数': len(h)})
    return pd.DataFrame(out)
b3 = pd.concat([b3, bench_tgt(Tm, Hm)], ignore_index=True); save(b3, 'B3_重点医師と類似高処方医師の比較.csv')
bal = []
for a in ['地域', '年代', '性別']:
    for k in sorted(d[a].unique()): bal.append({'変数': f'{a}:{k}', '重点医師': (Tm[a] == k).mean(), '類似高処方医師': np.average((Hm[a] == k).astype(float), weights=Hm.w)})
save(pd.DataFrame(bal).assign(差=lambda x: x.重点医師 - x.類似高処方医師), 'B2_比較条件のバランス.csv')
b4 = pd.concat([bench(Tm[Tm.診療科 == dp], Hm[Hm.診療科 == dp], dp) for dp in FOCUS]); save(b4, 'B4_診療科別の比較.csv')
# 回帰でそろえた差(地域・年代・性別も調整)
rg = pd.concat([Tm.assign(重点_=1.0), Hm.assign(重点_=0.0)])
dm = pd.get_dummies(rg[['診療科', '患者帯', '施設区分', '地域', '年代', '性別']], drop_first=True).astype(float); rg = pd.concat([rg, dm], axis=1)
b5 = []
for k, m in MET.items():
    if isinstance(m, tuple): rg['_y'] = (rg[m[0]] == m[1]).astype(float)
    else: rg['_y'] = rg[m].astype(float)
    z = ols(rg.dropna(subset=['_y']), '_y', ['重点_'] + list(dm.columns)).loc['重点_']
    b5.append({'項目': k, '調整した差': z.係数, '下限': z.係数 - 1.96 * z.標準誤差, '上限': z.係数 + 1.96 * z.標準誤差})
save(pd.DataFrame(b5), 'B5_回帰で調整した差.csv')
P.update(bench_n_T=len(Tm), bench_n_H=len(Hm), bench_excluded=int(ov[ov.類似高処方医師 == 0].重点医師.sum()))
print('BENCH', b3[['項目', '重点医師', '類似高処方医師', '差', '差_下限', '差_上限']].round(3).to_string())

# ==================== 4. セグメント(重点医師701名、医師ごとに主要施策を1つ) ====================
# 判定順: ① MR面談あり → ② 未接触 → ③ 処方意向4以上 → ④ その他(理解度・意向とも低い)
T['セグメント'] = np.select([T.面談あり, ~T.配信対象, T.意向高], ['MR面談フォロー型', '接触不足型', '意向ギャップ型'], '理解不足型')
# 当初案の「情報ニーズ特化型」(最重視する特性だけ理解が低い)の確認
fm = {F[i]: f'Q1_{i}' for i in range(1, 6)}
T['最重視の理解度'] = [r[fm[r.最重視]] if r.最重視 in fm else np.nan for _, r in T.iterrows()]
spec_gap = ((T.理解度 - T.最重視の理解度) >= 1).sum()
P.update(ニーズ特化型_該当=int(spec_gap), 最重視理解度_差=float((T.最重視の理解度 - T.理解度).mean()))
seg_order = ['接触不足型', '理解不足型', '意向ギャップ型', 'MR面談フォロー型']
def sp(g):
    t_ = g[g.配信対象]
    r = {'医師数': len(g), '内科': int((g.診療科 == '内科').sum()), '耳鼻咽喉科': int((g.診療科 == '耳鼻咽喉科').sum()), '潜在未処方': int(g.潜在未処方.sum()), '1人あたり潜在未処方': g.潜在未処方.mean(),
         '花粉症患者': int(g.Q3_1.sum()), '処方率': g.Q3_2.sum() / g.Q3_1.sum(), '理解度': g.理解度.mean(), '処方意向': g.Q4.mean(), '処方意向4以上': g.意向高.mean(),
         'MR君配信率': g.配信対象.mean(), 'MR面談率': g.面談あり.mean(), '既読(配信対象1人/年)': t_.既読件数.mean() if len(t_) else np.nan, '既読医師率(配信対象)': t_.既読あり.mean() if len(t_) else np.nan,
         '既読件数': int(t_.既読件数.sum())}
    for dp in FOCUS:
        gg = g[g.診療科 == dp]; top = gg.最重視.value_counts(normalize=True)
        r[f'{dp}:最重視1位'] = f'{top.index[0]} {top.iloc[0]:.0%}' if len(top) else ''
    q5 = g.Q5.value_counts(normalize=True); r['情報源1位'] = f'{q5.index[0]} {q5.iloc[0]:.0%}'; r['情報源2位'] = f'{q5.index[1]} {q5.iloc[1]:.0%}'
    return pd.Series(r)
s1 = pd.concat({s: sp(T[T.セグメント == s]) for s in seg_order}, axis=1).T
save(s1.reset_index().rename(columns={'index': 'セグメント'}), 'S1_セグメント別プロファイル.csv')
bm = b3.set_index('項目').類似高処方医師
print(s1[['医師数', '内科', '耳鼻咽喉科', '潜在未処方', '処方率', '理解度', '処方意向', '既読(配信対象1人/年)', '既読医師率(配信対象)', '情報源1位']].to_string())

# ==================== 5. HOW の入力(現状の配信・既読) ====================
# 現状: 配信対象1名に10本(課題資料「10コンテンツを対象医師に配信」)。MR君・MR面談は2025→2026の1年間の活動(データ仕様書)
N_DELIV = 10
cur = d[d.配信対象]
h1 = d[d.配信対象].groupby('潜在帯').agg(配信対象=('医師ID', 'size'), 既読件数=('既読件数', 'sum'), 既読医師率=('既読あり', 'mean'))
h1['既読率(1配信あたり)'] = h1.既読件数 / (h1.配信対象 * N_DELIV); save(h1.reset_index(), 'H1_潜在帯別の既読.csv')
low = d[d.配信対象 & (d.潜在未処方 <= 2)]                                           # A案で回数を減らす: 配信中で潜在未処方0〜2人(全診療科)
mid_ref = nj[nj.配信対象 & ~nj.面談あり & nj.潜在未処方.between(5, 7)]                 # 未接触・潜在5〜7人の既読率の参照(同じ条件で配信中の医師)
def grp(g, name, ref=None):
    ref = g if ref is None else ref; rt = ref[ref.配信対象]
    return {f'{name}_n': len(g), f'{name}_潜在': int(g.潜在未処方.sum()), f'{name}_既読率': float(rt.既読件数.sum() / (len(rt) * N_DELIV)), f'{name}_既読医師率10': float(rt.既読あり.mean())}
P.update(現状_配信対象=len(cur), 現状_既読=int(cur.既読件数.sum()), 現状_潜在カバー=int(cur.潜在未処方.sum()), 配信本数_現状=N_DELIV, 既読単価=400, 売上=50e8,
         回答者_処方患者=int(d.Q3_2.sum()), 全医師=10000, 回答医師=5000)
for s in ['理解不足型', '意向ギャップ型', 'MR面談フォロー型']: P.update(grp(T[T.セグメント == s], s))
P.update(grp(T[T.セグメント == '接触不足型'], '接触不足型', ref=T[T.セグメント == '理解不足型']))   # 新規配信の既読率は、似た状態で配信中の理解不足型の実測値
P.update(grp(un[un.潜在未処方.between(5, 7)], '未接触5to7', ref=mid_ref)); P.update(grp(low, '低優先'))
fx = T[(T.セグメント == 'MR面談フォロー型') & T.配信対象]
P.update({'MR面談フォロー型_配信中': len(fx), 'MR面談フォロー型_配信中_潜在': int(fx.潜在未処方.sum()), 'MR面談フォロー型_配信中_既読': int(fx.既読件数.sum())})
# 参考: 配信対象と対象外の処方率変化の差(MR面談なしの両年回答、診療科・2025年の状態で調整)。シナリオの目安にのみ使う
b = d.merge(s25[['医師ID', 'Q3_1', 'Q3_2', 'Q4'] + Q1], on='医師ID', suffixes=('', '_25'))
b = b[(b.Q3_1 > 0) & (b.Q3_1_25 > 0) & ~b.面談あり].copy()
b['変化'] = (b.Q3_2 / b.Q3_1 - b.Q3_2_25 / b.Q3_1_25) * 100; b['処方率25'] = b.Q3_2_25 / b.Q3_1_25 * 100; b['理解度25'] = b[[c + '_25' for c in Q1]].mean(axis=1)
for k, v in {'内科_': '内科', '耳鼻_': '耳鼻咽喉科', '小児_': '小児科'}.items(): b[k] = (b.診療科 == v).astype(float)
b['tgt'] = b.配信対象.astype(float)
z = ols(b, '変化', ['tgt', '内科_', '耳鼻_', '小児_', '処方率25', '理解度25', 'Q4_25']).loc['tgt']
reach = b[b.配信対象].既読あり.mean(); unpf = 1 - b[b.配信対象].Q3_2_25.sum() / b[b.配信対象].Q3_1_25.sum()
P.update(参考_差=float(z.係数), 参考_差_下限=float(z.係数 - 1.96 * z.標準誤差), 参考_差_上限=float(z.係数 + 1.96 * z.標準誤差), 参考_n=len(b), 参考_既読医師率=float(reach), 参考_未処方割合=float(unpf),
         参考_転換率=float(z.係数 / reach / unpf / 100), 参考_転換率_下限=float((z.係数 - 1.96 * z.標準誤差) / reach / unpf / 100))
bal25 = {v: float((b[b.配信対象][v].mean() - b[~b.配信対象][v].mean()) / np.sqrt((b[b.配信対象][v].var() + b[~b.配信対象][v].var()) / 2)) for v in ['処方率25', '理解度25', 'Q4_25']}
P['参考_2025バランス_最大SMD'] = max(abs(x) for x in bal25.values())
print('参考', {k: round(v, 4) for k, v in P.items() if k.startswith('参考')})

# ==================== 6. 配信計画の群(施策①: 潜在未処方と接触状況で配信対象・回数を決める) ====================
# 重点医師 = 内科・耳鼻で潜在8人以上。MR面談なしで配信中(392名)は増回、未接触(191名)は新規、MR面談あり(118名)は現状維持
on = T[T.配信対象 & ~T.面談あり]
P.update(grp(on, '配信中重点'))                                                    # 配信中・MR面談なしの重点医師
P.update(grp(T[~T.配信対象 & ~T.面談あり], '未接触重点', ref=on))                  # 新規配信先の既読率は、配信中・MR面談なしの重点医師の実測値
P.update(grp(T[T.面談あり], '面談あり重点'))

# ==================== 7. 現状の配信の評価(良い点と改善の余地) ====================
A = att.merge(vis, on='医師ID', how='left', validate='1:1'); A['配信'] = A.医師ID.isin(tg.医師ID); A['面談'] = A.面談回数 > 0
P.update(全医師_配信対象=int(A.配信.sum()), 全医師_配信率=float(A.配信.mean()), 面談なし_全医師=int((~A.面談).sum()), 面談なし_配信率=float(A[~A.面談].配信.mean()),
         面談なし_配信=int((A.配信 & ~A.面談).sum()), 全医師_既読件数=len(lg), 全医師_既読医師=int(lg.医師ID.nunique()), 全医師_既読医師率=float(lg.医師ID.nunique() / A.配信.sum()))
q5 = d[~d.面談あり]; P.update(情報源_面談なし_MR君=float((q5.Q5 == 'MR君によるWEBでの説明').mean()), 情報源_面談なし配信中_MR君=float((q5[q5.配信対象].Q5 == 'MR君によるWEBでの説明').mean()),
                         情報源_面談なし_n=len(q5), 情報源_面談なし配信中_n=int(q5.配信対象.sum()))
c0 = pd.concat({'面談なしの回答医師': q5.Q5.value_counts(normalize=True), '面談なし・配信中の回答医師': q5[q5.配信対象].Q5.value_counts(normalize=True)}, axis=1)
save(c0.reset_index().rename(columns={'index': '最も参考になった情報源(Q5)', 'Q5': '最も参考になった情報源(Q5)'}), 'C0_面談なしの医師の情報源.csv')
d['潜在帯4'] = pd.cut(d.潜在未処方, [-1, 2, 4, 7, 100], labels=['0〜2人', '3〜4人', '5〜7人', '8人以上']).astype(str)
c1 = pd.concat([d.groupby('潜在帯').agg(医師数=('医師ID', 'size'), 配信対象=('配信対象', 'sum'), 配信率=('配信対象', 'mean'), MR面談率=('面談あり', 'mean')).assign(区分='3区分'),
                d.groupby('潜在帯4').agg(医師数=('医師ID', 'size'), 配信対象=('配信対象', 'sum'), 配信率=('配信対象', 'mean'), MR面談率=('面談あり', 'mean')).assign(区分='4区分')])
save(c1.reset_index().rename(columns={'index': '潜在帯'}), 'C1_潜在帯別の配信率.csv')
P.update(配信率_3to7=float(d[d.潜在帯 == '3〜7人'].配信対象.mean()), 重点_未接触=int((~T.配信対象 & ~T.面談あり).sum()), 重点_MR君のみ=int((T.配信対象 & ~T.面談あり).sum()), 重点_面談あり=int(T.面談あり.sum()))

# ==================== 8. 診療科別の重視項目(施策②: 診療科ごとに重視項目を割り付ける) ====================
def needs(g):
    return pd.Series({'医師数': len(g), **{f'最重視:{f}': (g.最重視 == f).mean() for f in F.values()}, **{f'2位以内:{F[i]}': (g[f'Q2_{i}'] <= 2).mean() for i in range(1, 6)}})
n1 = pd.concat({dp: needs(T[T.診療科 == dp]) for dp in FOCUS}, axis=1).T; save(n1.reset_index().rename(columns={'index': '診療科'}), 'N1_重点医師の重視項目_診療科別.csv')
n2 = pd.concat({dp: needs(d[d.診療科 == dp]) for dp in sorted(d.診療科.unique())}, axis=1).T; save(n2.reset_index().rename(columns={'index': '診療科'}), 'N2_全回答医師の重視項目_診療科別.csv')
for dp in FOCUS:
    for f in F.values(): P[f'重視_{dp}_{f}'] = float(n1.loc[dp, f'最重視:{f}']); P[f'重視2位以内_{dp}_{f}'] = float(n1.loc[dp, f'2位以内:{f}'])
    P[f'重点_{dp}_n'] = int(n1.loc[dp, '医師数'])

# ==================== 9. 読まれやすさ: 本人の最重視テーマのコンテンツ vs その他(配信対象の全回答医師、1配信あたりの既読率) ====================
TH = dict(zip(ct.コンテンツID, ct.主要テーマ.replace({'他の薬との飲み合わせの良さ': '飲み合わせ'})))
RD = lg.groupby(['医師ID', 'コンテンツID']).size().unstack(fill_value=0)
def match_rates(g):
    g = g[g.配信対象]; m, o = [], []
    for _, r in g.iterrows():
        mine = [c for c, th in TH.items() if th == r.最重視]
        if not mine: m.append(np.nan); o.append(np.nan); continue
        row = RD.loc[r.医師ID] if r.医師ID in RD.index else pd.Series(0, index=RD.columns)
        m.append(row.reindex(mine, fill_value=0).sum() / len(mine)); o.append((row.sum() - row.reindex(mine, fill_value=0).sum()) / (10 - len(mine)))
    return g.assign(既読率_重視=m, 既読率_その他=o)
allt = match_rates(d[d.最重視.isin(list(F.values()))])
rows = []
for dp in ['全体'] + sorted(d.診療科.unique()):
    g = allt if dp == '全体' else allt[allt.診療科 == dp]; g = g.dropna(subset=['既読率_重視']); dif = (g.既読率_重視 - g.既読率_その他).values
    bs = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(2000)]
    rows.append({'診療科': dp, '配信対象の医師数': len(g), '重視テーマの既読率': g.既読率_重視.mean(), 'その他の既読率': g.既読率_その他.mean(), '差_下限': np.percentile(bs, 2.5), '差_上限': np.percentile(bs, 97.5)})
r2 = pd.DataFrame(rows); r2['倍率'] = r2.重視テーマの既読率 / r2.その他の既読率; save(r2, 'R1_重視テーマの既読率_配信対象の全医師.csv'); print(r2.round(4).to_string())
P.update(全_既読率_重視=float(r2.iloc[0].重視テーマの既読率), 全_既読率_その他=float(r2.iloc[0].その他の既読率), 全_既読率差_下限=float(r2.iloc[0].差_下限), 全_既読率差_上限=float(r2.iloc[0].差_上限), 全_既読率_n=int(r2.iloc[0].配信対象の医師数))

# ==================== 10. 平均への回帰(翌年の潜在未処方 ÷ 今年。MR面談・MR君の接点がない両年回答医師) ====================
# 患者数は自己申告で年ごとに変わるため、今年の潜在が多い医師は翌年に減り、少ない医師は増える。効果は翌年の潜在で計算する
bb2 = d.merge(s25[['医師ID', 'Q3_1', 'Q3_2']], on='医師ID', suffixes=('', '_25'))
bb2 = bb2[~bb2.面談あり & ~bb2.配信対象].copy(); bb2['u25'] = bb2.Q3_1_25 - bb2.Q3_2_25
bb2['帯'] = pd.cut(bb2.u25, [-1, 2, 4, 7, 100], labels=['0〜2人', '3〜4人', '5〜7人', '8人以上']).astype(str)
rows = []
for scope, g0 in [('内科・耳鼻', bb2[bb2.診療科.isin(FOCUS)]), ('全診療科', bb2)]:
    for k, g in g0.groupby('帯'):
        bsr = [(lambda y: y.潜在未処方.sum() / y.u25.sum())(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(1000)]
        rows.append({'範囲': scope, '2025年の潜在帯': k, '医師数': len(g), '2025年の潜在(平均)': g.u25.mean(), '2026年の潜在(平均)': g.潜在未処方.mean(), '翌年比': g.潜在未処方.sum() / g.u25.sum(), '下限': np.percentile(bsr, 2.5), '上限': np.percentile(bsr, 97.5)})
m1 = pd.DataFrame(rows); save(m1, 'M1_平均への回帰_翌年比.csv'); print(m1.round(3).to_string())
RT = {(r.範囲, r['2025年の潜在帯']): r.翌年比 for _, r in m1.iterrows()}
P.update(回帰_8人以上=float(RT[('内科・耳鼻', '8人以上')]), 回帰_5to7=float(RT[('内科・耳鼻', '5〜7人')]), 回帰_0to2=float(RT[('全診療科', '0〜2人')]))

json.dump(P, open(os.path.join(OUT, 'params_v5.json'), 'w'), ensure_ascii=False, indent=1, default=float)
print('saved', OUT)
