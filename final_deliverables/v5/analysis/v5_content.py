# -*- coding: utf-8 -*-
"""施策②: 診療科ごとの配信順。v5_analysis.py の後に実行する。

規則: 5つの特性の代表コンテンツ1本ずつを、その特性を最も重視する医師が多い順に並べ(同率は2位以内の割合)、各特性の2本目以降のコンテンツを後半に置く。
重視の割合は、内科・耳鼻咽喉科は重点医師、小児科・眼科は全回答医師(潜在0〜2人の医師への隔月配信で使う)。
コンテンツはコンテンツ名と主要テーマ(mrkun_contents_summary)だけで分類しており、本文は確認していない。
"""
import json, pandas as pd
import os
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output') + os.sep
FEATS = ['飲み合わせ', '副作用の少なさ', '24h持続性', '効果発現速度', '適用症状の広さ']
FIRST = {'飲み合わせ': ['C04'], '副作用の少なさ': ['C03'], '24h持続性': ['C05'], '効果発現速度': ['C01'], '適用症状の広さ': ['C02']}
SECOND = {'飲み合わせ': ['C09'], '副作用の少なさ': [], '24h持続性': ['C10'], '効果発現速度': ['C07', 'C06'], '適用症状の広さ': ['C08']}
TITLE = {'C01': 'つらい症状にすぐ届く！薬剤Aの即効メカニズム', 'C02': '鼻炎だけじゃない！薬剤Aの幅広い適応症例集', 'C03': '安心の処方を支える薬剤Aの副作用プロファイル',
         'C04': '他剤との飲み合わせガイド〜併用時の注意点と実践〜', 'C05': '朝から夜まで効果持続！24時間カバーの臨床エビデンス', 'C06': '花粉症治療2026年最新トレンドまとめ',
         'C07': '速く効いて長く続く〜薬剤Aの即効性×持続性〜', 'C08': '患者さんの声から見る薬剤AのQOL改善効果', 'C09': '多剤併用患者への処方設計〜実践的アプローチ〜', 'C10': '薬剤Aの長期投与における安全性と有効性'}
SHORT = {'C01': '即効メカニズム', 'C02': '幅広い適応症例集', 'C03': '副作用プロファイル', 'C04': '飲み合わせガイド', 'C05': '24時間カバーの臨床エビデンス', 'C06': '2026年最新トレンド',
         'C07': '即効性×持続性', 'C08': 'QOL改善効果', 'C09': '多剤併用患者への処方設計', 'C10': '長期投与の安全性と有効性'}
THEME = {'C01': '効果発現速度', 'C02': '適用症状の広さ', 'C03': '副作用の少なさ', 'C04': '飲み合わせ', 'C05': '24h持続性', 'C06': '効果発現速度', 'C07': '効果発現速度', 'C08': '適用症状の広さ', 'C09': '飲み合わせ', 'C10': '24h持続性'}
def order(shares, shares2):
    fo = sorted(FEATS, key=lambda f: (-shares[f], -shares2[f]))
    return [c for f in fo for c in FIRST[f]] + [c for f in fo for c in SECOND[f]], fo
n1 = pd.read_csv(O + 'N1_重点医師の重視項目_診療科別.csv').set_index('診療科'); n2 = pd.read_csv(O + 'N2_全回答医師の重視項目_診療科別.csv').set_index('診療科')
OUTD = {}
for dp in ['内科', '耳鼻咽喉科', '小児科', '眼科']:
    src, base = (n1, '重点医師') if dp in n1.index else (n2, '全回答医師')
    sh = {f: float(src.loc[dp, f'最重視:{f}']) for f in FEATS}; sh2 = {f: float(src.loc[dp, f'2位以内:{f}']) for f in FEATS}
    od, fo = order(sh, sh2)
    assert sorted(od) == sorted(TITLE), (dp, od)
    cum3 = sum(sh[THEME[c]] for c in od[:3])
    OUTD[dp] = {'基準': base, '医師数': int(src.loc[dp, '医師数']), '配信順': od, '特性の順': fo, '最重視の割合': sh, '2位以内の割合': sh2, '3本目までに最重視テーマを受け取る割合': cum3}
    print(dp, base, od, round(cum3, 3))
json.dump({'order': OUTD, 'title': TITLE, 'short': SHORT, 'theme': THEME}, open(O + 'content_order_v5.json', 'w'), ensure_ascii=False, indent=1)
