# -*- coding: utf-8 -*-
"""HOW と BUSINESS IMPACT の計算(効果試算_v4.xlsx と同じ式)。v4_analysis.py の後に実行する。

既読医師率のモデル: 年間 f 回配信したときの既読医師率 = 1 − (1 − 10回配信時の既読医師率)^(f/10)
  (配信ごとの既読のしやすさが一定と仮定。10回配信時の値は実測)
追加処方患者 = 既読医師の増加数 × 1人あたり潜在未処方 × 増分転換率(シナリオの仮定)
"""
import os, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'output')
P = json.load(open(os.path.join(OUT, 'params_v4.json')))
SALES_PER_PT = P['売上'] * 1.0 / (P['回答者_処方患者'] * P['全医師'] / P['回答医師'])   # 50億円 ÷ 全医師の処方患者推計
CONV = {'保守': 0.01, '基準': 0.03, '楽観': 0.06}
UNIT = P['既読単価']; PROD_COST = 0; OPS_COST = 0                                          # 制作費・運用費は未確認(既存10本の活用を前提に0)

# 群: (名前, 医師数, 潜在未処方, 1配信あたり既読率, 10回配信時の既読医師率, 現状回数, A案回数, B案回数)
g4 = {'n': P['MR面談フォロー型_配信中'], 'unp': P['MR面談フォロー型_配信中_潜在'], 'reads': P['MR面談フォロー型_配信中_既読']}
G = [('接触不足型', P['接触不足型_n'], P['接触不足型_潜在'], P['接触不足型_既読率'], P['接触不足型_既読医師率10'], 0, 12, 12),
     ('理解不足型', P['理解不足型_n'], P['理解不足型_潜在'], P['理解不足型_既読率'], P['理解不足型_既読医師率10'], 10, 12, 12),
     ('意向ギャップ型', P['意向ギャップ型_n'], P['意向ギャップ型_潜在'], P['意向ギャップ型_既読率'], P['意向ギャップ型_既読医師率10'], 10, 24, 24),
     ('MR面談フォロー型(配信中)', g4['n'], g4['unp'], g4['reads'] / (g4['n'] * 10), P['MR面談フォロー型_既読医師率10'], 10, 10, 10),
     ('未接触×潜在5〜7人', P['未接触5to7_n'], P['未接触5to7_潜在'], P['未接触5to7_既読率'], P['未接触5to7_既読医師率10'], 0, 0, 12),
     ('低優先(配信中×潜在0〜2人)', P['低優先_n'], P['低優先_潜在'], P['低優先_既読率'], P['低優先_既読医師率10'], 10, 6, 6)]
used_n = sum(g[1] for g in G if g[5] > 0); used_reads = round(sum(g[1] * g[5] * g[3] for g in G if g[5] > 0))
other_n = P['現状_配信対象'] - used_n; other_reads = P['現状_既読'] - used_reads
other_unp = P['現状_潜在カバー'] - sum(g[2] for g in G if g[5] > 0)
G.append(('その他の配信中(変更なし)', other_n, other_unp, other_reads / (other_n * 10), None, 10, 10, 10))
reach = lambda r10, f: 0.0 if f == 0 else 1 - (1 - r10) ** (f / 10)

rows = []
for name, n, unp, rr, r10, f0, fa, fb in G:
    row = {'群': name, '医師数': n, '潜在未処方': unp, '1配信あたり既読率': rr, '10回時の既読医師率': r10, '現状回数': f0, 'A案回数': fa, 'B案回数': fb}
    for sc, f in [('現状', f0), ('A案', fa), ('B案', fb)]:
        row[f'{sc}_配信医師数'] = n if f > 0 else 0; row[f'{sc}_配信件数'] = n * f; row[f'{sc}_既読件数'] = n * f * rr; row[f'{sc}_費用'] = n * f * rr * UNIT
        row[f'{sc}_潜在カバー'] = unp if f > 0 else 0
        row[f'{sc}_既読医師率'] = reach(r10, f) if r10 is not None else None
    for sc in ['A案', 'B案']:
        d_reached = 0.0 if r10 is None else n * (row[f'{sc}_既読医師率'] - row['現状_既読医師率'])
        row[f'{sc}_既読医師の増加'] = d_reached
        for c, v in CONV.items(): row[f'{sc}_追加処方患者_{c}'] = d_reached * (unp / n) * v
    rows.append(row)
H = pd.DataFrame(rows)
H.to_csv(os.path.join(OUT, 'H2_配信計画_群別.csv'), encoding='utf-8-sig', index=False)
tot = {}
for sc in ['現状', 'A案', 'B案']:
    t = {k: H[f'{sc}_{k}'].sum() for k in ['配信医師数', '配信件数', '既読件数', '費用', '潜在カバー']}
    t['費用'] += (PROD_COST + OPS_COST) if sc != '現状' else 0
    for c in CONV:
        t[f'追加処方患者_{c}'] = H[f'{sc}_追加処方患者_{c}'].sum() if sc != '現状' else 0.0
        t[f'追加売上_{c}'] = t[f'追加処方患者_{c}'] * SALES_PER_PT
    tot[sc] = t
for sc in ['A案', 'B案']: tot[sc]['追加費用'] = tot[sc]['費用'] - tot['現状']['費用']
tot['現状']['追加費用'] = 0
tot['B案−A案'] = {k: tot['B案'][k] - tot['A案'][k] for k in tot['B案']}
S = pd.DataFrame(tot); S.index.name = '項目'
S.to_csv(os.path.join(OUT, 'H3_予算シナリオ比較.csv'), encoding='utf-8-sig')
pd.set_option('display.width', 220); pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
print(H[['群', '医師数', '潜在未処方', '1配信あたり既読率', '10回時の既読医師率', '現状回数', 'A案回数', 'B案回数', 'A案_既読件数', 'A案_既読医師の増加', 'A案_追加処方患者_基準', 'B案_追加処方患者_基準']].to_string())
print(S.to_string()); print('1処方患者あたり年間売上', round(SALES_PER_PT, 1))
json.dump({'sales_per_pt': SALES_PER_PT, 'conv': CONV, 'groups': G, 'totals': tot}, open(os.path.join(OUT, 'plan_v4.json'), 'w'), ensure_ascii=False, indent=1, default=float)
