import sys, os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SRCDIR, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams['font.family'] = 'IPAGothic'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
rng = np.random.default_rng(20260102)


def rd(name): return pd.read_csv(glob.glob(os.path.join(SRCDIR, f'*{name}.csv'))[0])


s26, s25, att = rd('survey_2026'), rd('survey_2025'), rd('doctor_attributes')
visits, tg, lg, ct = rd('mr_visits'), rd('mrkun_targets'), rd('mrkun_logs'), rd('mrkun_contents_summary')
save = lambda df, n, idx=False: df.to_csv(os.path.join(OUT, n), encoding='utf-8-sig', index=idx)
FEAT = {1: '効果発現速度', 2: '適用症状の広さ', 3: '副作用の少なさ', 4: '他の薬との飲み合わせの良さ', 5: '24h持続性'}
Q1 = [f'Q1_{i}' for i in range(1, 6)]
DEPTS = ['内科', '耳鼻咽喉科', '小児科', '眼科']

# ================= 0. データ品質チェック =================
print('== 0 品質チェック')
for n, t in [('survey_2026', s26), ('survey_2025', s25), ('attributes', att), ('visits', visits), ('targets', tg)]:
    print(f'{n}: 行={len(t)} 医師ID重複={t.医師ID.duplicated().sum()}')
print('logs: 行', len(lg), '(医師,コンテンツ)重複', lg.duplicated(['医師ID', 'コンテンツID']).sum())
for n, t in [('2026', s26), ('2025', s25)]:
    print(n, 'Q3_2>Q3_1:', int((t.Q3_2 > t.Q3_1).sum()), ' Q3_1==0:', int((t.Q3_1 == 0).sum()), ' Q1範囲', int(t[Q1].min().min()), int(t[Q1].max().max()),
          ' Q4範囲', int(t.Q4.min()), int(t.Q4.max()))
print('logs∖targets:', len(set(lg.医師ID) - set(tg.医師ID)), ' 視聴時間0秒の既読:', int((lg.視聴時間_秒 == 0).sum()))

# ================= 基礎テーブル =================
d = s26.copy()
d['unp'] = d.Q3_1 - d.Q3_2
d = d.merge(att, on='医師ID', how='left', suffixes=('', '_att'), validate='1:1')
assert len(d) == 5000
mismatch = {c: int((d[c] != d[c + '_att']).sum()) for c in ['性別', '年代', '診療科', '地域', '施設区分']}
print('属性の不一致(2026 vs 属性ファイル):', mismatch)
d = d.merge(visits, on='医師ID', how='left', validate='1:1'); assert d.面談回数.notna().all()
d['配信対象'] = d.医師ID.isin(tg.医師ID)
lgm = lg.merge(ct[['コンテンツID', 'コンテンツ長_秒']], on='コンテンツID', how='left')
lgm['完了率'] = lgm.視聴時間_秒 / lgm.コンテンツ長_秒
agg = lgm.groupby('医師ID').agg(既読本数=('コンテンツID', 'nunique'), 総視聴秒=('視聴時間_秒', 'sum'), 平均完了率=('完了率', 'mean')).reset_index()
d = d.merge(agg, on='医師ID', how='left', validate='1:1')
d['既読本数'] = d.既読本数.fillna(0).astype(int); d['総視聴秒'] = d.総視聴秒.fillna(0)
d['既読あり'] = d.既読本数 > 0
d['面談あり'] = d.面談回数 > 0
assert len(d) == 5000 and d.医師ID.is_unique
# 認知ギャップ群
for k in FEAT: d[f'gap{k}'] = (d[f'Q2_{k}'] == 1) & (d[f'Q1_{k}'] <= 2)
d['gap_any'] = d[[f'gap{k}' for k in FEAT]].any(axis=1)
d['gA'] = (d.診療科 == '内科') & d.gap4; d['gB'] = (d.診療科 == '耳鼻咽喉科') & d.gap5; d['gC'] = (d.診療科 == '内科') & d.gap3
d['gABC'] = d.gA | d.gB | d.gC


def sh(g): return g.Q3_2.sum() / g.Q3_1.sum() if g.Q3_1.sum() > 0 else np.nan


# ================= 4. Double Check =================
rows = []
def chk(item, existing, new, tol=0.0005):
    ok = abs(existing - new) <= tol if isinstance(existing, float) else existing == new
    rows.append({'項目': item, '既存の数値': existing, '再計算': round(new, 4) if isinstance(new, float) else new, '一致': '○' if ok else '×'})


ex = {'内科': (2072, 13380, 3842, 9538, .287), '小児科': (1232, 5165, 1177, 3988, .228), '耳鼻咽喉科': (1214, 14148, 6020, 8128, .426), '眼科': (482, 1521, 241, 1280, .158)}
for dep, (n, p, t, u, r) in ex.items():
    g = d[d.診療科 == dep]
    chk(f'{dep} 医師数', n, len(g)); chk(f'{dep} 花粉症患者数', p, int(g.Q3_1.sum())); chk(f'{dep} 処方患者数', t, int(g.Q3_2.sum()))
    chk(f'{dep} 潜在未処方', u, int(g.unp.sum())); chk(f'{dep} 処方割合', r, round(g.Q3_2.sum() / g.Q3_1.sum(), 3))
tot = int(d.unp.sum()); chk('潜在未処方 全体', 22934, tot)
chk('内科+耳鼻咽喉科の割合(約77%)', .77, round(d[d.診療科.isin(['内科', '耳鼻咽喉科'])].unp.sum() / tot, 3), .005)
for q, u in zip([1, 2, 3, 4, 5], [4991, 5065, 5508, 3972, 3398]): chk(f'Q4={q} 潜在未処方', u, int(d[d.Q4 == q].unp.sum()))
chk('Q4=1〜3の潜在未処方の割合(約68%)', .68, round(d[d.Q4 <= 3].unp.sum() / tot, 3), .005)
for nm, col, n_, u_ in [('A 内科×飲み合わせ', 'gA', 600, 3416), ('B 耳鼻咽喉科×24h持続性', 'gB', 190, 1657), ('C 内科×副作用', 'gC', 283, 1542)]:
    chk(f'{nm} 医師数', n_, int(d[col].sum())); chk(f'{nm} 潜在未処方', u_, int(d[d[col]].unp.sum()))
# 相関
v = d.dropna(subset=['Q3_1']).copy(); v = v[v.Q3_1 > 0]; v['share'] = v.Q3_2 / v.Q3_1; v['q1m'] = v[Q1].mean(axis=1)
rho = v.q1m.rank().corr(v.share.rank()); chk('Q1平均×処方割合 順位相関(0.77)', .77, round(rho, 2), .005)
t_dc = pd.DataFrame(rows); save(t_dc, 'D1_既存分析との照合表.csv')
print(t_dc.to_string()); print('不一致', int((t_dc.一致 == '×').sum()), '/', len(t_dc))
# 以前の記述の再確認
hi5 = v[v.Q4 == 5]; ov = d.Q3_2.sum() / d.Q3_1.sum()
print(f'Q4=5で処方割合が全体平均({ov:.3f})以上: {(hi5.share >= ov).sum()}/{len(hi5)} = {(hi5.share >= ov).mean():.3f}  (旧記述「約95%」)')
chk_dept = {c: int((d.診療科 != d.診療科_att).sum()) for c in ['診療科']}; print('診療科の不一致', chk_dept)

# --- 潜在未処方が多い医師 vs 認知ギャップ群
sub = d[d.診療科.isin(['内科', '耳鼻咽喉科'])].copy()
rows = []
sub['top20'] = sub.unp >= sub.unp.quantile(.8)   # 同数タイ含むため20%を少し超える場合あり
thr = sub.unp.quantile(.8)
print(f'\n上位20%の閾値 unp>={thr}, 該当 {int(sub.top20.sum())}名 / {len(sub)}')
for nm, col in [('認知ギャップ群 A+B+C', 'gABC'), ('認知ギャップ群(内科・耳鼻の4特性すべて)', 'gap_any')]:
    gg = sub[sub[col]]
    rows.append({'群': nm, '医師数': len(gg), '潜在未処方合計': int(gg.unp.sum()), '一人当たり潜在未処方': gg.unp.mean(),
                 '内科+耳鼻の潜在未処方に占める割合': gg.unp.sum() / sub.unp.sum(), '群内の上位20%該当率': gg.top20.mean(),
                 '上位20%医師のうち当群の割合': (sub.top20 & sub[col]).sum() / sub.top20.sum(), '平均Q4': gg.Q4.mean(), '処方割合': sh(gg)})
non = sub[~sub.gap_any]
rows.append({'群': '(参考)ギャップ群以外', '医師数': len(non), '潜在未処方合計': int(non.unp.sum()), '一人当たり潜在未処方': non.unp.mean(),
             '内科+耳鼻の潜在未処方に占める割合': non.unp.sum() / sub.unp.sum(), '群内の上位20%該当率': non.top20.mean(),
             '上位20%医師のうち当群の割合': (sub.top20 & ~sub.gap_any).sum() / sub.top20.sum(), '平均Q4': non.Q4.mean(), '処方割合': sh(non)})
N = int(sub.gABC.sum()); topN = sub.sort_values(['unp', 'Q4'], ascending=False).head(N)
rows.append({'群': f'(参考)潜在未処方の多い順の上位{N}名(A+B+Cと同数)', '医師数': N, '潜在未処方合計': int(topN.unp.sum()), '一人当たり潜在未処方': topN.unp.mean(),
             '内科+耳鼻の潜在未処方に占める割合': topN.unp.sum() / sub.unp.sum(), '群内の上位20%該当率': topN.top20.mean(),
             '上位20%医師のうち当群の割合': topN.top20.sum() / sub.top20.sum(), '平均Q4': topN.Q4.mean(), '処方割合': sh(topN)})
t_ov = pd.DataFrame(rows); save(t_ov, 'D2_潜在未処方上位医師とギャップ群の関係.csv'); print(t_ov.round(3).to_string())
# 集中度
srt = sub.sort_values('unp', ascending=False); cum = srt.unp.cumsum() / srt.unp.sum()
print('潜在未処方の集中(内科+耳鼻): 上位10%医師', round(cum.iloc[int(len(srt) * .1) - 1], 3), ' 上位20%', round(cum.iloc[int(len(srt) * .2) - 1], 3), ' 上位50%', round(cum.iloc[int(len(srt) * .5) - 1], 3))
# 潜在未処方上位の医師のQ1・Q4・接触
print('上位20%(unp>=thr)医師: 平均Q4', round(sub[sub.top20].Q4.mean(), 2), ' 平均Q1', round(sub[sub.top20][Q1].mean().mean(), 2), ' 処方割合', round(sh(sub[sub.top20]), 3))
print('  ギャップ群A+B+Cの Q1', round(sub[sub.gABC][Q1].mean().mean(), 2))
print('  上位20%医師の Q1<=2 割合', round((sub[sub.top20][Q1].mean(axis=1) <= 2).mean(), 3))

# ================= 5. Reach =================
def cov(r):
    return {(False, False): 'A 面談なし・MR君対象外', (True, False): 'B 面談あり・MR君対象外',
            (False, True): 'C 面談なし・MR君対象', (True, True): 'D 面談あり・MR君対象'}[(bool(r.面談あり), bool(r.配信対象))]


d['coverage'] = d.apply(cov, axis=1)
COV = ['A 面談なし・MR君対象外', 'B 面談あり・MR君対象外', 'C 面談なし・MR君対象', 'D 面談あり・MR君対象']


def summarize(g, base_unp=None):
    return {'医師数': len(g), '潜在未処方患者数': int(g.unp.sum()), '平均患者数': g.Q3_1.mean(), '処方割合': sh(g), '平均Q4': g.Q4.mean(),
            '平均潜在未処方(一人当たり)': g.unp.mean(), **({'潜在未処方の構成比': g.unp.sum() / base_unp} if base_unp else {})}


scopes = [('全診療科(5,000)', d), ('内科', d[d.診療科 == '内科']), ('耳鼻咽喉科', d[d.診療科 == '耳鼻咽喉科']), ('小児科', d[d.診療科 == '小児科']), ('眼科', d[d.診療科 == '眼科']),
          ('ギャップ群A 内科×飲み合わせ', d[d.gA]), ('ギャップ群B 耳鼻×24h持続性', d[d.gB]), ('ギャップ群C 内科×副作用', d[d.gC]), ('ギャップ群A+B+C', d[d.gABC])]
rows = []
for nm, g in scopes:
    for c in COV:
        x = g[g.coverage == c]
        rows.append({'対象': nm, 'カバー状況': c, **summarize(x, g.unp.sum()), '医師数の構成比': len(x) / len(g)})
t_r = pd.DataFrame(rows); save(t_r, 'R1_カバー状況別比較.csv'); print('\n== R1\n', t_r.round(3).to_string())
# 面談/配信の単独割合
for nm, g in scopes[:3] + scopes[-4:]:
    print(nm, '面談あり率', round(g.面談あり.mean(), 3), ' 配信対象率', round(g.配信対象.mean(), 3), ' 既読あり率', round(g.既読あり.mean(), 3), ' 配信対象内の既読率', round(g[g.配信対象].既読あり.mean(), 3))

# ================= 6. Engagement =================
def mrk(r): return '① MR君配信対象外' if not r.配信対象 else ('③ 配信対象かつ既読あり' if r.既読あり else '② 配信対象だが既読なし')


d['mrk'] = d.apply(mrk, axis=1)
d['meet'] = pd.cut(d.面談回数, [-1, 0, 1, 3, 100], labels=['0回', '1回', '2〜3回', '4回以上'])
d['comb'] = np.select([(~d.面談あり) & (~d.既読あり), d.面談あり & (~d.既読あり), (~d.面談あり) & d.既読あり, d.面談あり & d.既読あり],
                      ['α 面談なし・既読なし', 'β 面談あり・既読なし', 'γ 面談なし・既読あり', 'δ 面談あり・既読あり'], default='')
scopes = [('全診療科(5,000)', d), ('内科', d[d.診療科 == '内科']), ('耳鼻咽喉科', d[d.診療科 == '耳鼻咽喉科']), ('小児科', d[d.診療科 == '小児科']), ('眼科', d[d.診療科 == '眼科']),
          ('ギャップ群A 内科×飲み合わせ', d[d.gA]), ('ギャップ群B 耳鼻×24h持続性', d[d.gB]), ('ギャップ群C 内科×副作用', d[d.gC]), ('ギャップ群A+B+C', d[d.gABC])]
COMB = ['α 面談なし・既読なし', 'β 面談あり・既読なし', 'γ 面談なし・既読あり', 'δ 面談あり・既読あり']
rows = []
for nm, g in scopes:
    for col, order in [('mrk', ['① MR君配信対象外', '② 配信対象だが既読なし', '③ 配信対象かつ既読あり']), ('meet', ['0回', '1回', '2〜3回', '4回以上']), ('comb', COMB)]:
        for c in order:
            x = g[g[col] == c]
            rows.append({'対象': nm, '区分': {'mrk': 'MR君', 'meet': 'MR面談回数', 'comb': '面談×既読'}[col], '群': c, **summarize(x, g.unp.sum()),
                         '平均面談回数': x.面談回数.mean(), '平均既読本数': x.既読本数.mean()})
t_e = pd.DataFrame(rows); save(t_e, 'E1_接触状況別比較.csv'); print('\n== E1\n', t_e.round(3).to_string())
x = d[d.既読あり]
print('既読医師: 平均既読本数', round(x.既読本数.mean(), 2), ' 平均完了率', round(x.平均完了率.mean(), 3), ' 視聴0秒の既読ログ', int((lg.視聴時間_秒 == 0).sum()), '/', len(lg))
print('配信対象のうち既読あり', round(d[d.配信対象].既読あり.mean(), 3))
# 推定費用
d['費用'] = d.面談回数 * 10000 + d.既読本数 * 400
print('回答者5,000名の推定費用: 面談', int((d.面談回数 * 10000).sum()), '円 / MR君', int((d.既読本数 * 400).sum()), '円')

# ================= 7. Outcome =================
b = d.merge(s25, on='医師ID', how='inner', suffixes=('_26', '_25'), validate='1:1')
print('\n== 7 Outcome: 両年回答', len(b))
print('診療科/年代/地域などの2年間の不一致:', {c: int((b[c + '_26'] != b[c + '_25']).sum()) for c in ['性別', '年代', '診療科', '地域', '施設区分']})
b['q1_25'] = b[[c + '_25' for c in Q1]].mean(axis=1); b['q1_26'] = b[[c + '_26' for c in Q1]].mean(axis=1)
b['dq1'] = b.q1_26 - b.q1_25; b['dq4'] = b.Q4_26 - b.Q4_25; b['dq32'] = b.Q3_2_26 - b.Q3_2_25; b['dq31'] = b.Q3_1_26 - b.Q3_1_25
b['sh25i'] = b.Q3_2_25 / b.Q3_1_25.where(b.Q3_1_25 > 0); b['sh26i'] = b.Q3_2_26 / b.Q3_1_26.where(b.Q3_1_26 > 0)


def mci(x):
    x = x.dropna(); m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - 1.96 * se, m + 1.96 * se


def shdiff_boot(g, n=1000):
    if len(g) < 5: return (np.nan, np.nan, np.nan)
    pt = g.Q3_2_26.sum() / g.Q3_1_26.sum() - g.Q3_2_25.sum() / g.Q3_1_25.sum(); out = []
    a = g[['Q3_1_25', 'Q3_2_25', 'Q3_1_26', 'Q3_2_26']].values
    for _ in range(n):
        r = a[rng.integers(0, len(a), len(a))]
        out.append(r[:, 3].sum() / r[:, 2].sum() - r[:, 1].sum() / r[:, 0].sum())
    return (pt, *np.nanpercentile(out, [2.5, 97.5]))


# (a) 比較可能性: 2025年の初期状態・診療科構成・n
bs = [('全診療科', b), ('内科', b[b.診療科_26 == '内科']), ('耳鼻咽喉科', b[b.診療科_26 == '耳鼻咽喉科'])]
rows = []
for nm, g in bs:
    for c in COMB:
        x = g[g.comb == c]
        rows.append({'対象': nm, '接触群': c, '医師数(両年回答)': len(x), '2025年 Q1平均': x.q1_25.mean(), '2025年 Q4': x.Q4_25.mean(),
                     '2025年 患者数Q3_1': x.Q3_1_25.mean(), '2025年 処方患者数Q3_2': x.Q3_2_25.mean(),
                     '2025年 処方割合': x.Q3_2_25.sum() / x.Q3_1_25.sum() if len(x) else np.nan,
                     '内科の割合': (x.診療科_26 == '内科').mean(), '耳鼻咽喉科の割合': (x.診療科_26 == '耳鼻咽喉科').mean(),
                     '小児科の割合': (x.診療科_26 == '小児科').mean(), '眼科の割合': (x.診療科_26 == '眼科').mean(),
                     '開業医の割合': (x.施設区分_26 == '開業医').mean()})
t_o1 = pd.DataFrame(rows); save(t_o1, 'O1_前年比較の可否_初期状態と構成.csv'); print('\n== O1\n', t_o1.round(3).to_string())
# (b) 変化量
rows = []
for nm, g in bs + [('全診療科・2025年Q1平均2以下', b[b.q1_25 <= 2]), ('内科・2025年Q1平均2以下', b[(b.診療科_26 == '内科') & (b.q1_25 <= 2)]),
                   ('耳鼻咽喉科・2025年Q1平均2以下', b[(b.診療科_26 == '耳鼻咽喉科') & (b.q1_25 <= 2)])]:
    for c in COMB:
        x = g[g.comb == c]
        if len(x) == 0: continue
        r = {'対象': nm, '接触群': c, '医師数': len(x)}
        for lab, col in [('Q1平均の変化', 'dq1'), ('Q4の変化', 'dq4'), ('Q3_2(処方患者数)の変化', 'dq32'), ('Q3_1(患者数)の変化', 'dq31')]:
            m, lo, hi = mci(x[col]); r[lab] = m; r[lab + ' 95%区間'] = f'[{lo:.2f}, {hi:.2f}]'
        pt, lo, hi = shdiff_boot(x); r['処方割合の変化(pt)'] = pt * 100; r['処方割合の変化 95%区間'] = f'[{lo*100:.1f}, {hi*100:.1f}]'
        r['2025年処方割合'] = x.Q3_2_25.sum() / x.Q3_1_25.sum(); r['2026年処方割合'] = x.Q3_2_26.sum() / x.Q3_1_26.sum()
        r['少数注意(n<30)'] = len(x) < 30
        rows.append(r)
t_o2 = pd.DataFrame(rows); save(t_o2, 'O2_接触群別の前年からの変化.csv'); print('\n== O2\n', t_o2.round(3).to_string())
# 配信対象外の群との差(接触なし=α基準でなく MR君別): 補助
rows = []
for nm, g in bs:
    for c in ['① MR君配信対象外', '② 配信対象だが既読なし', '③ 配信対象かつ既読あり']:
        x = g[g.mrk == c]
        rows.append({'対象': nm, 'MR君': c, '医師数': len(x), '2025年Q1': x.q1_25.mean(), 'Q1変化': x.dq1.mean(), '2025年Q4': x.Q4_25.mean(), 'Q4変化': x.dq4.mean(),
                     'Q3_2変化': x.dq32.mean(), '処方割合変化(pt)': shdiff_boot(x, 300)[0] * 100})
t_o3 = pd.DataFrame(rows); save(t_o3, 'O3_MR君3区分別の変化.csv'); print('\n== O3\n', t_o3.round(3).to_string())
# 全体の変化(接触に関係なく)
print('両年回答全体: Q1', round(b.dq1.mean(), 3), ' Q4', round(b.dq4.mean(), 3), ' Q3_2', round(b.dq32.mean(), 3), ' Q3_1', round(b.dq31.mean(), 3))
print('2025Q1の水準別: ', b.groupby(pd.cut(b.q1_25, [0, 1.5, 2.5, 3.5, 5.1], right=False)).dq1.agg(['count', 'mean']).round(3).to_dict())
print('接触群と2025Q1の相関(順位):', round(b.q1_25.rank().corr((b.面談回数 + b.既読本数).rank()), 3), ' 接触と2025 Q4', round(b.Q4_25.rank().corr((b.面談回数 + b.既読本数).rank()), 3))
print('面談あり×診療科:', pd.crosstab(b.診療科_26, b.面談あり, normalize='index').round(3).to_dict())

# ================= 図 =================
def style(ax):
    ax.set_facecolor(SURF)
    for s_ in ['top', 'right']: ax.spines[s_].set_visible(False)
    for s_ in ['left', 'bottom']: ax.spines[s_].set_color(GRID)
    ax.tick_params(colors=INK2, length=0); ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)


# 図R: カバー状況別 潜在未処方(構成比)
f = plt.figure(figsize=(12.5, 4.8), facecolor=SURF)
sc = ['内科', '耳鼻咽喉科', 'ギャップ群A 内科×飲み合わせ', 'ギャップ群B 耳鼻×24h持続性', 'ギャップ群C 内科×副作用']
lab = ['内科', '耳鼻咽喉科', 'A 内科×\n飲み合わせ', 'B 耳鼻×\n24h持続性', 'C 内科×\n副作用']
ax = f.add_subplot(1, 1, 1); style(ax)
left = np.zeros(len(sc)); x = np.arange(len(sc)); gap = .004
for i, c in enumerate(COV):
    v = np.array([t_r[(t_r.対象 == s_) & (t_r.カバー状況 == c)]['潜在未処方の構成比'].iloc[0] * 100 for s_ in sc])
    ax.bar(x, v, .6, bottom=left, color=CAT[i], edgecolor=SURF, linewidth=1.5, label=c)
    for xi, vi, li in zip(x, v, left):
        if vi >= 6: ax.text(xi, li + vi / 2, f'{vi:.0f}%', ha='center', va='center', color='white' if i in (0, 1) else INK, fontsize=10)
    left += v
ax.set_xticks(x); ax.set_xticklabels(lab, color=INK); ax.set_ylim(0, 100); ax.set_ylabel('潜在未処方患者数の構成比(%)', color=INK2)
ax.legend(handles=[Patch(color=CAT[i], label=COV[i]) for i in range(4)], frameon=False, fontsize=9, labelcolor=INK2, ncol=4, loc='upper center', bbox_to_anchor=(.5, -.16))
ax.set_title('プロモーションのカバー状況別 潜在未処方患者数の構成比(アンケート回答者)', loc='left', color=INK, fontsize=12)
f.tight_layout(); f.savefig(os.path.join(OUT, 'figR_カバー状況別の潜在未処方.png'), dpi=150, facecolor=SURF); plt.close(f)

# 図E: 接触状況(面談×既読)別 医師数/潜在未処方一人当たり/Q4 ・ 処方割合
f = plt.figure(figsize=(13, 4.6), facecolor=SURF)
for j, (col, ttl, fmt, sc_) in enumerate([('医師数', '医師数', '{:.0f}', 1), ('平均Q4', '平均処方意向 Q4', '{:.2f}', 1), ('処方割合', '薬剤A処方割合(%)', '{:.0f}', 100)]):
    ax = f.add_subplot(1, 3, j + 1); style(ax); x = np.arange(4); w = .38
    for i, dep in enumerate(['内科', '耳鼻咽喉科']):
        v = [t_e[(t_e.対象 == dep) & (t_e.区分 == '面談×既読') & (t_e.群 == c)][col].iloc[0] * sc_ for c in COMB]
        ax.bar(x + (i - .5) * w, v, w * .92, color=['#2a78d6', '#eb6834'][i], label=dep)
        for xi, vi in zip(x + (i - .5) * w, v): ax.text(xi, vi * 1.01, fmt.format(vi), ha='center', fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(['面談なし\n既読なし', '面談あり\n既読なし', '面談なし\n既読あり', '面談あり\n既読あり'], color=INK, fontsize=8.5)
    ax.set_title(ttl, loc='left', color=INK, fontsize=11)
    if j == 0: ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
f.suptitle('接触状況別の比較(MR面談の有無×MR君既読の有無・アンケート回答者。相関であり因果ではない)', x=.01, ha='left', color=INK, fontsize=11.5)
f.tight_layout(rect=[0, 0, 1, .94]); f.savefig(os.path.join(OUT, 'figE_接触状況別の比較.png'), dpi=150, facecolor=SURF); plt.close(f)

# 図O: 2025年初期状態と変化
f = plt.figure(figsize=(13, 8.2), facecolor=SURF)
mets = [('2025年 Q1平均', 'q1_25', 'Q1平均の変化', 1), ('2025年 Q4', 'Q4_25', 'Q4の変化', 1), ('2025年 処方割合', None, '処方割合の変化(pt)', 1)]
for r_, dep in enumerate(['全診療科', '内科']):
    for j, (blab, bcol, dcol, _) in enumerate(mets):
        ax = f.add_subplot(2, 3, r_ * 3 + j + 1); style(ax); x = np.arange(4)
        sub1 = t_o1[(t_o1.対象 == dep)]; sub2 = t_o2[(t_o2.対象 == dep)]
        v1 = [sub1[sub1.接触群 == c][blab].iloc[0] for c in COMB]; v2 = [sub2[sub2.接触群 == c][dcol].iloc[0] for c in COMB]
        w = .38; sc1 = 100 if '処方割合' in blab else 1
        ax.bar(x - w / 2, np.array(v1) * sc1, w * .92, color='#9bbfe8', label='2025年の水準(左軸)')
        for xi, vi in zip(x - w / 2, np.array(v1) * sc1): ax.text(xi, vi * 1.01, f'{vi:.1f}' if sc1 == 1 else f'{vi:.0f}%', ha='center', fontsize=8, color=INK)
        ax2 = ax.twinx() if False else None
        ax.bar(x + w / 2, v2, w * .92, color='#103c73', label='2025→2026の変化(同じ軸)')
        for xi, vi in zip(x + w / 2, v2): ax.text(xi, max(vi, 0) + (.03 if sc1 == 1 else .6), f'{vi:+.2f}' if sc1 == 1 else f'{vi:+.1f}pt', ha='center', fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(['面談なし\n既読なし', '面談あり\n既読なし', '面談なし\n既読あり', '面談あり\n既読あり'], color=INK, fontsize=8)
        ax.set_title(f'{dep}:{blab.replace("2025年 ", "")}', loc='left', color=INK, fontsize=10.5)
f.suptitle('両年回答4,000名:2025年の初期水準と2025→2026年の変化(接触群別・記述統計。プロモーションの因果効果ではない)', x=.01, ha='left', color=INK, fontsize=11.5)
f.legend(handles=[Patch(color='#9bbfe8', label='2025年の水準'), Patch(color='#103c73', label='2025→2026年の変化(同じ目盛り)')], loc='lower center', ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
f.tight_layout(rect=[0, .03, 1, .95]); f.savefig(os.path.join(OUT, 'figO_前年比較.png'), dpi=150, facecolor=SURF); plt.close(f)
print('done')
