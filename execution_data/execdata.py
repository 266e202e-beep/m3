import sys; sys.path.insert(0, '/tmp/claude-0/-home-user-m3/f8de41a6-071b-5062-a3cf-a392fa1f2d68/scratchpad')
from base import *
OUT = '/home/user/m3/execution_data'
save = lambda df, n, idx=False: df.to_csv(os.path.join(OUT, n), encoding='utf-8-sig', index=idx)
d, s25, att, visits, tg, lg, ct = load()
d = d.merge(att[['医師ID', '地域', '施設区分', '年代', '性別']].rename(columns=lambda c: c if c == '医師ID' else c + '_att'), on='医師ID', validate='1:1')
assert (d.年代 == d.年代_att).all() and (d.施設区分 == d.施設区分_att).all()
# ===== 0. 照合 =====
sub = d[d.診療科.isin(['内科', '耳鼻咽喉科'])].copy(); H = sub[sub.unp >= 8].copy()
A = H[(H.q1m <= 2) & (H.Q4 <= 2)].copy(); B = H[(H.q1m > 2) & (H.q1m < 3.5) & (H.Q4 == 3)].copy()
chk = [('回答医師数', 5000, len(d)), ('潜在未処方(全体)', 22934, int(d.unp.sum())), ('内科+耳鼻咽喉科の潜在未処方', 17666, int(sub.unp.sum())), ('同割合(%)', 77.0, round(sub.unp.sum() / d.unp.sum() * 100, 1)),
       ('両診療科の回答医師数', 3286, len(sub)), ('潜在未処方8人以上の医師数', 701, len(H)), ('同 潜在未処方', 7187, int(H.unp.sum())), ('同 両診療科比(%)', 40.7, round(H.unp.sum() / sub.unp.sum() * 100, 1)),
       ('A層 医師数', 316, len(A)), ('A層 潜在未処方', 3355, int(A.unp.sum())), ('A層 配信対象外', 120, int((A.mrk == '①対象外').sum())), ('A層 対象だが既読なし', 111, int((A.mrk == '②対象だが未読').sum())),
       ('A層 既読あり', 85, int((A.mrk == '③既読あり').sum())), ('B層 医師数', 117, len(B)), ('B層 潜在未処方', 1196, int(B.unp.sum()))]
t0 = pd.DataFrame([{'項目': a, '既存値': b, '再計算': c, '一致': '○' if b == c else '×'} for a, b, c in chk]); save(t0, 'E0_既存数値との照合.csv'); print(t0.to_string())
# ===== 集計関数 =====
def prof(g, nm):
    r = {'群': nm, '医師数': len(g), '内科': int((g.診療科 == '内科').sum()), '耳鼻咽喉科': int((g.診療科 == '耳鼻咽喉科').sum())}
    r.update({'潜在未処方 合計': int(g.unp.sum()), '潜在未処方 平均': g.unp.mean(), '潜在未処方 中央値': g.unp.median(), '潜在未処方 Q1': g.unp.quantile(.25), '潜在未処方 Q3': g.unp.quantile(.75),
              '花粉症患者数Q3_1 平均': g.Q3_1.mean(), '薬剤A処方患者数Q3_2 平均': g.Q3_2.mean(), '処方割合(合計比)': sh(g)})
    for k, nmf in FEAT.items(): r[f'理解度 {nmf}'] = g[f'Q1_{k}'].mean()
    r.update({'理解度 5特性平均': g.q1m.mean(), '処方意向Q4 平均': g.Q4.mean(), '面談あり': int(g.面談あり.sum()), '面談なし': int((~g.面談あり).sum()), '面談あり率': g.面談あり.mean(), '面談回数 合計': int(g.面談回数.sum()),
              '面談回数 平均(全医師)': g.面談回数.mean(), '面談回数 平均(面談あり医師)': g[g.面談あり].面談回数.mean() if g.面談あり.any() else np.nan,
              '女性の割合': (g.性別 == '女性').mean(), '開業医の割合': (g.施設区分 == '開業医').mean(), '50代以上の割合': g.年代.isin(['50代', '60代', '70代以上']).mean(),
              '既読件数 合計': int(g.既読本数.sum()), '既読医師1人当たり平均既読件数': g[g.既読あり].既読本数.mean() if g.既読あり.any() else np.nan})
    return r
def smd(x, y):
    sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2); return (x.mean() - y.mean()) / sp if sp > 0 else np.nan
ST = ['①対象外', '②対象だが未読', '③既読あり']; LB = {'①対象外': 'a MR君配信対象外', '②対象だが未読': 'b 配信対象だが既読記録なし', '③既読あり': 'c 既読記録あり'}
for nm, G, tag in [('A層(2026年基準)', A, 'A'), ('B層(2026年基準)', B, 'B')]:
    rows = [prof(G, f'{nm} 全体')] + [prof(G[G.mrk == s], f'{nm} {LB[s]}') for s in ST]
    t = pd.DataFrame(rows); save(t, f'E1_{tag}層_配信状況別の特徴.csv'); print('\n=====', nm); print(t.set_index('群').T.round(3).to_string())
    n = len(G); rows = [{'区分': LB[s], '医師数': int((G.mrk == s).sum()), '割合(分母=層全体)': (G.mrk == s).sum() / n} for s in ST]
    rows.append({'区分': '配信対象(b+c)', '医師数': int(G.配信対象.sum()), '割合(分母=層全体)': G.配信対象.mean()}); rows.append({'区分': 'c÷配信対象', '医師数': int(G.既読あり.sum()), '割合(分母=層全体)': G.既読あり.sum() / G.配信対象.sum()})
    save(pd.DataFrame(rows), f'E1_{tag}層_配信状況の人数.csv'); print(pd.DataFrame(rows).round(3).to_string())
# a vs その他のSMD (A層)
a = A[A.mrk == '①対象外']; o = A[A.mrk != '①対象外']
rows = []
for lab, col in [('潜在未処方', 'unp'), ('花粉症患者数', 'Q3_1'), ('処方患者数', 'Q3_2'), ('理解度(5特性平均)', 'q1m'), ('処方意向Q4', 'Q4'), ('面談回数', '面談回数')]:
    rows.append({'指標': lab, 'a 配信対象外(120)': a[col].mean(), 'b+c 配信対象(196)': o[col].mean(), 'b 未読(111)': A[A.mrk == '②対象だが未読'][col].mean(), 'c 既読あり(85)': A[A.mrk == '③既読あり'][col].mean(), 'SMD(a vs b+c)': smd(a[col], o[col])})
for lab, mask in [('内科の割合', lambda g: g.診療科 == '内科'), ('面談ありの割合', lambda g: g.面談あり), ('開業医の割合', lambda g: g.施設区分 == '開業医')]:
    rows.append({'指標': lab, 'a 配信対象外(120)': mask(a).mean(), 'b+c 配信対象(196)': mask(o).mean(), 'b 未読(111)': mask(A[A.mrk == '②対象だが未読']).mean(), 'c 既読あり(85)': mask(A[A.mrk == '③既読あり']).mean(), 'SMD(a vs b+c)': np.nan})
t = pd.DataFrame(rows); save(t, 'E1b_A層_配信対象外と配信対象の比較.csv'); print(t.round(3).to_string())
print('配信対象外の診療科別: ', a.診療科.value_counts().to_dict(), ' 地域', a.地域_att.value_counts().to_dict())
# ===== 2. コンテンツ整理 =====
cnt = ct.copy()
kind = {'C01': '作用機序の解説', 'C02': '症例集', 'C03': '安全性データの解説', 'C04': '実践ガイド(併用時の注意)', 'C05': '臨床エビデンス', 'C06': '疾患トレンドの総括(一般情報)', 'C07': '複合訴求(効果発現×持続性)',
        'C08': '患者アウトカム(QOL)', 'C09': '実践アプローチ(多剤併用患者の処方設計)', 'C10': '長期データ(安全性・有効性)'}
cnt['内容の種類(タイトルからの推定)'] = cnt.コンテンツID.map(kind)
cnt['タイトルに含まれる他の特性(推定)'] = cnt.コンテンツID.map({'C07': '24h持続性', 'C10': '副作用の少なさ(安全性)', 'C06': '(特性横断的なトレンド)'}).fillna('')
cl = lg.groupby('コンテンツID').size().rename('既読件数(全ログ)'); ca = lg[lg.医師ID.isin(A.医師ID)].groupby('コンテンツID').size().rename('既読件数(A層316名)'); cb = lg[lg.医師ID.isin(B.医師ID)].groupby('コンテンツID').size().rename('既読件数(B層117名)')
cnt = cnt.set_index('コンテンツID').join([cl, ca, cb]).fillna(0).astype({'既読件数(A層316名)': int, '既読件数(B層117名)': int}); save(cnt, 'E2_既存10コンテンツの整理.csv', True); print(cnt.to_string())
print('ログ列:', list(lg.columns), ' 視聴0秒の行', int((lg.視聴時間_秒 == 0).sum()))
# ===== 4. 試験対象規模 =====
rows = []
for dep in ['内科', '耳鼻咽喉科']:
    x = A[(A.診療科 == dep)]; n120 = x[x.mrk == '①対象外']
    rows.append({'段階': '第1段階候補(A層・配信対象外)', '診療科': dep, '人数': len(n120), '配信資格を判定できる変数': 'なし', '配信資格が不明な人数': len(n120), '2群へ分割(概算)': f'{len(n120)//2}/{len(n120)-len(n120)//2}', '面談あり': int(n120.面談あり.sum())})
n120 = A[A.mrk == '①対象外']; rows.append({'段階': '第1段階候補(A層・配信対象外)', '診療科': '合計', '人数': len(n120), '配信資格を判定できる変数': 'なし', '配信資格が不明な人数': len(n120), '2群へ分割(概算)': f'{len(n120)//2}/{len(n120)-len(n120)//2}', '面談あり': int(n120.面談あり.sum())})
t4a = pd.DataFrame(rows); save(t4a, 'E4a_第1段階の候補人数.csv'); print(t4a.to_string())
rows = []
ex = pd.concat([A.assign(層='A層'), B.assign(層='B層')]); ex = ex[ex.配信対象]
for (lay, dep), g in ex.groupby(['層', '診療科']): rows.append({'層': lay, '診療科': dep, '既存配信対象の人数': len(g), '2群へ分割(概算)': f'{len(g)//2}/{len(g)-len(g)//2}', '既読あり': int(g.既読あり.sum()), '既読なし': int((~g.既読あり).sum())})
for lay, g in ex.groupby('層'): rows.append({'層': lay, '診療科': '合計', '既存配信対象の人数': len(g), '2群へ分割(概算)': f'{len(g)//2}/{len(g)-len(g)//2}', '既読あり': int(g.既読あり.sum()), '既読なし': int((~g.既読あり).sum())})
rows.append({'層': 'A+B', '診療科': '合計', '既存配信対象の人数': len(ex), '2群へ分割(概算)': f'{len(ex)//2}/{len(ex)-len(ex)//2}', '既読あり': int(ex.既読あり.sum()), '既読なし': int((~ex.既読あり).sum())})
t4b = pd.DataFrame(rows); save(t4b, 'E4b_第2段階の候補人数.csv'); print(t4b.to_string())
# 検出力の目安: 2025年→2026年の個人変化のばらつき(観測SD)
b = d.merge(s25, on='医師ID', suffixes=('_26', '_25'), validate='1:1'); b = b[b.診療科_26.isin(['内科', '耳鼻咽喉科'])].copy()
b['q1_25'] = b[[c + '_25' for c in Q1]].mean(axis=1); b['unp25'] = b.Q3_1_25 - b.Q3_2_25; b['dq32'] = b.Q3_2_26 - b.Q3_2_25
b['ds'] = (b.Q3_2_26 / b.Q3_1_26.where(b.Q3_1_26 > 0)) - (b.Q3_2_25 / b.Q3_1_25.where(b.Q3_1_25 > 0))
sets = {'2025年A層(低理解×低意向×高ポテンシャル)・面談なし': b[(b.q1_25 <= 2) & (b.Q4_25 <= 2) & (b.unp25 >= 8) & (~b.面談あり)],
        '同・面談なし・MR君配信対象外': b[(b.q1_25 <= 2) & (b.Q4_25 <= 2) & (b.unp25 >= 8) & (~b.面談あり) & (~b.配信対象)],
        '2025年B層(中×中×高ポテンシャル)・面談なし': b[(b.q1_25 > 2) & (b.q1_25 < 3.5) & (b.Q4_25 == 3) & (b.unp25 >= 8) & (~b.面談あり)]}
rows = []
for nm, g in sets.items():
    sd32, sds = g.dq32.std(), g.ds.std()
    for n_ in [60, 48, 40, 30, 98, 49, 24]:
        rows.append({'SDの取得元(2025→2026の個人変化)': nm, 'SD取得元のn': len(g), 'Q3_2変化のSD': sd32, '処方割合変化のSD': sds, '1群あたり医師数': n_, '検出できる最小差 Q3_2(人)': 2.8 * sd32 * np.sqrt(2 / n_), '検出できる最小差 処方割合(pt)': 2.8 * sds * np.sqrt(2 / n_) * 100})
t4c = pd.DataFrame(rows); save(t4c, 'E4c_検出力の目安_観測SDからの試算.csv'); print(t4c[t4c['1群あたり医師数'].isin([60, 48, 30, 98, 24])].round(3).to_string())
# ===== 5. 費用計算に使える実測 =====
rows = []
for nm, G in [('A層', A), ('B層', B)]:
    ex_ = G[G.配信対象]; rd_ = G[G.既読あり]
    rows.append({'層': nm, '層全体の医師数': len(G), '配信対象': len(ex_), '既読あり医師数': len(rd_), '既読件数 合計': int(G.既読本数.sum()), '既読医師1人当たり平均既読件数': rd_.既読本数.mean(), '配信対象内の既読率': len(rd_) / len(ex_) if len(ex_) else np.nan,
                 '既読件数の費用(400円×件数) 現在の接触分': int(G.既読本数.sum()) * 400, '0秒既読の件数': int(lg[lg.医師ID.isin(G.医師ID) & (lg.視聴時間_秒 == 0)].shape[0]),
                 '既読本数の分布(0/1/2/3/4+)': '/'.join(str(int((G.既読本数 == k).sum()) if k < 4 else int((G.既読本数 >= 4).sum())) for k in range(5))})
t5 = pd.DataFrame(rows); save(t5, 'E5_MR君費用計算に使える実測情報.csv'); print(t5.round(3).to_string())
for nm, G in [('A層', A), ('B層', B)]:
    for dep in ['内科', '耳鼻咽喉科']:
        g = G[G.診療科 == dep]; r_ = g[g.既読あり]; print(nm, dep, '配信対象', int(g.配信対象.sum()), '既読医師', len(r_), '既読件数', int(g.既読本数.sum()), '平均', round(r_.既読本数.mean(), 2) if len(r_) else None)
