# -*- coding: utf-8 -*-
"""v11: 配信計画の費用・効果を「科別」に計算し、内科・耳鼻咽喉科だけを全国へ外挿する。

前提(v5_plan.py と同じ群・既読率・翌年比を使う):
  既読件数 = 医師数 × 年間回数 × 1配信あたり既読率(群別の実測)
  費用 = 既読件数 × 400円
  既読医師率(f回) = 1 − (1 − 10回時の既読医師率)^(f/10)            [仮定: 配信ごとの読まれやすさは一定]
  追加処方患者 = Δ既読医師率 × 潜在未処方 × 翌年比 × 転換率          [転換率は仮定]
全国(内科・耳鼻のみ):
  全国の現状費用 = サンプルの現状既読件数(科別) × 科別倍率 × 400円
  全国の変化 = サンプルの変化(科別) × 科別倍率 × 実施可能割合(仮定90%)
  追加売上 = 全国の追加処方患者 × 処方患者1人あたり年間売上(参考値、未検証)
"""
import os, glob, json
import pandas as pd

DATA = os.environ.get('DATA_DIR', '/root/.claude/uploads/f8de41a6-071b-5062-a3cf-a392fa1f2d68')
HERE = os.path.dirname(os.path.abspath(__file__)); V5 = os.path.join(HERE, '..', '..', 'v5', 'analysis', 'output')
PL = json.load(open(os.path.join(V5, 'plan_v5.json'))); P = json.load(open(os.path.join(V5, 'params_v5.json')))
rd = lambda n: pd.read_csv(glob.glob(os.path.join(DATA, f'*{n}.csv'))[0])
s26, vis, tg, lg = rd('survey_2026'), rd('mr_visits'), rd('mrkun_targets'), rd('mrkun_logs')
d = s26.merge(vis, on='医師ID', validate='1:1'); d['配信'] = d.医師ID.isin(tg.医師ID); d['面談'] = d.面談回数 > 0; d['unp'] = d.Q3_1 - d.Q3_2
d['既読'] = d.医師ID.map(lg.groupby('医師ID').size()).fillna(0)

UNIT = 400; FEAS = 0.9; CONV = {'保守': 0.01, '基準': 0.03, '楽観': 0.06}
NAT = {'内科': 62161, '耳鼻咽喉科': 9330}                     # 厚労省 令和6年 医療施設従事医師・主たる診療科(内科は確認、耳鼻は暫定)
DEPTS = ['内科', '耳鼻咽喉科', '小児科', '眼科']; FOC = ['内科', '耳鼻咽喉科']
SAMP = {k: int((d.診療科 == k).sum()) for k in DEPTS}
FAC = {k: NAT[k] / SAMP[k] for k in FOC}
G = {g[0][0]: g for g in PL['groups']}                       # (名前, n, 潜在, 既読率, 10回既読医師率, 翌年比, 現状, A, B)
masks = {'①': lambda x: (x.診療科.isin(FOC)) & (x.unp >= 8) & ~x.配信 & ~x.面談,
         '②': lambda x: (x.診療科.isin(FOC)) & (x.unp >= 8) & x.配信 & ~x.面談,
         '③': lambda x: (x.unp <= 2) & x.配信,
         '④': lambda x: (x.診療科.isin(FOC)) & (x.unp >= 8) & x.配信 & x.面談,
         '⑤': lambda x: (x.診療科.isin(FOC)) & x.unp.between(5, 7) & ~x.配信 & ~x.面談}
reach = lambda r10, f: 0.0 if f == 0 else 1 - (1 - r10) ** (f / 10)
rows = []
for k, m in masks.items():
    _, n_all, unp_all, rr, r10, kk, f0, fa, fb = G[k]
    sub = d[m(d)]; assert len(sub) == n_all and sub.unp.sum() == unp_all, k
    for dp in DEPTS:
        x = sub[sub.診療科 == dp]
        if len(x) == 0: continue
        r = {'群': k, '診療科': dp, '医師数': len(x), '潜在未処方': int(x.unp.sum()), '既読率': rr, '10回既読医師率': r10, '翌年比': kk, '現状回数': f0, 'A案回数': fa, 'B案回数': fb}
        for sc, f in [('現状', f0), ('A案', fa), ('B案', fb)]: r[f'{sc}_既読件数'] = len(x) * f * rr; r[f'{sc}_費用'] = len(x) * f * rr * UNIT
        for sc, f in [('A案', fa), ('B案', fb)]:
            dre = reach(r10, f) - reach(r10, f0)
            for c, v in CONV.items(): r[f'{sc}_追加処方患者_{c}'] = dre * x.unp.sum() * kk * v
        rows.append(r)
R = pd.DataFrame(rows)
# サンプル全体(4科)の費用: 変更なしの医師は現状のまま
cur_total = PL['totals']['現状']['費用']
samp = {'現状': cur_total}
for sc in ['A案', 'B案']: samp[sc] = cur_total + (R[f'{sc}_費用'] - R['現状_費用']).sum()
# 科別のサンプル変化
byd = R.groupby('診療科')[[c for c in R.columns if c.endswith('_費用') or '追加処方患者' in c]].sum()
for sc in ['A案', 'B案']: byd[f'{sc}_費用変化'] = byd[f'{sc}_費用'] - byd['現状_費用']
# 全国(内科・耳鼻)
cur_reads = {dp: float(d[(d.診療科 == dp) & d.配信].既読.sum()) for dp in FOC}
nat = {'現状費用': sum(cur_reads[dp] * FAC[dp] * UNIT for dp in FOC)}
for sc in ['A案', 'B案']:
    nat[f'{sc}_費用変化'] = sum(byd.loc[dp, f'{sc}_費用変化'] * FAC[dp] * FEAS for dp in FOC)
    nat[f'{sc}_費用'] = nat['現状費用'] + nat[f'{sc}_費用変化']
    for c in CONV: nat[f'{sc}_追加処方患者_{c}'] = sum(byd.loc[dp, f'{sc}_追加処方患者_{c}'] * FAC[dp] * FEAS for dp in FOC)
cnt = lambda mask: {dp: int((mask & (d.診療科 == dp)).sum()) for dp in FOC}
hp = cnt((d.unp >= 8)); nat['高ポテンシャル医師'] = sum(hp[dp] * FAC[dp] for dp in FOC)
for k in ['①', '②', '③', '⑤']:
    c_ = R[(R.群 == k) & R.診療科.isin(FOC)].set_index('診療科').医師数
    nat[f'{k}_医師数'] = sum(c_.get(dp, 0) * FAC[dp] for dp in FOC)
# 処方患者1人あたり年間売上(参考値、未検証): 50億円 × 内科・耳鼻の処方患者の割合 ÷ 全国の内科・耳鼻の処方患者(推計)
rx = {dp: float(d[d.診療科 == dp].Q3_2.sum()) for dp in DEPTS}; rx_all = sum(rx.values())
share = (rx['内科'] + rx['耳鼻咽喉科']) / rx_all
nat_rx = sum(rx[dp] * FAC[dp] for dp in FOC)
price = P['売上'] * share / nat_rx
for sc in ['A案', 'B案']:
    for c in CONV: nat[f'{sc}_追加売上_{c}'] = nat[f'{sc}_追加処方患者_{c}'] * price
out = {'SAMP': SAMP, 'NAT': NAT, 'FAC': FAC, 'FEAS': FEAS, 'CONV': CONV, 'rx': rx, 'share': share, 'nat_rx': nat_rx, 'price': price, 'samp': samp, 'nat': nat,
       'hp_sample': hp, 'cur_reads': cur_reads, 'byd': byd.reset_index().to_dict(orient='records'), 'rows': R.to_dict(orient='records')}
os.makedirs(os.path.join(HERE, 'output'), exist_ok=True)
R.to_csv(os.path.join(HERE, 'output', 'N1_群×科別の費用と追加処方患者_サンプル.csv'), encoding='utf-8-sig', index=False)
json.dump(out, open(os.path.join(HERE, 'output', 'national_v11.json'), 'w'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 220)
print(R[['群', '診療科', '医師数', '潜在未処方', '現状_費用', 'A案_費用', 'B案_費用', 'A案_追加処方患者_基準', 'B案_追加処方患者_基準']].round(1).to_string())
print(byd[['A案_費用変化', 'B案_費用変化', 'A案_追加処方患者_基準', 'B案_追加処方患者_基準']].round(1))
print('samp', {k: round(v) for k, v in samp.items()}); print('FAC', FAC, 'share', share, 'nat_rx', nat_rx, 'price', price)
print({k: round(v, 1) for k, v in nat.items()})
