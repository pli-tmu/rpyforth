# ベンチマークセット計画（ECOOP / stack fragment 論文向け）

rpyforth の stack fragment JIT 論文（投稿目標: ECOOP）で使うベンチマークの選定根拠、分類、現状、計測方法をまとめた文書。

関連: [`STACK_FRAGMENT_JIT.md`](STACK_FRAGMENT_JIT.md)（研究プラン全体）、[`SHOOTOUT_FULL_SUPPORT_PLAN.md`](SHOOTOUT_FULL_SUPPORT_PLAN.md)（shootout port 手順）、[`SHOOTOUT_COVERAGE.md`](../SHOOTOUT_COVERAGE.md)（自動生成ステータス）

---

## 1. 設計方針

### 1.1 何を証明するか

| 主張 | 必要なベンチ |
|------|-------------|
| 再帰・深い call で SP 正規化が効く | カテゴリ **R** |
| flat / memory 系で退化しない | カテゴリ **F**, **M** |
| fragment 確保コストが問題になる条件 | カテゴリ **C** |
| 単一マイクロのチューニングではない | **composite** + 12+ 本 |
| trace optimizer が stack access を消す | **jitlog** 対象 4 本 |
| Forth 文脈での位置づけ | **Gforth 比較** 8 本 + 可能なら **appbench** |

### 1.2 Forth に「大規模スイート」はない

Java の DaCapo や SPEC のような学会標準の巨大スイートは Forth には存在しない。ECOOP で「大規模」と言える根拠は次の 3 層の組み合わせとする。

| 層 | スイート | 論文での役割 |
|----|---------|-------------|
| **大規模** | [Forth appbench](https://www.complang.tuwien.ac.at/forth/appbench-1.4.zip) | Ertl ECOOP 2024 が「substantial / idiomatic Forth」と位置づけた実アプリ寄りベンチ |
| **中規模** | Classic Shootout 拡張（lists, hash, moments 等） | アルゴリズム幅 + Gforth 比較しやすさ |
| **合成** | `composite.fs` | 2SOM 型の擬似アプリ（単一ベンチ最適化批判への防御） |

マイクロベンチ（ack, fibo 等）だけでは機構証明には足りるが、ECOOP では **分類された 10+ 本 + 大規模または合成 1 本** が実質下限。

---

## 2. カテゴリ分類（R / F / M / C）

論文の表・図にそのまま使う分類。1 ベンチが複数特性を持つ場合は主目的で分類する。

```
R — Recursive / deep call     fragment 改善の主対象
F — Flat stack depth          SP 正規化の恩恵が少ない（回帰確認）
M — Memory (@/! heap)         本最適化と直交（回帰確認）
C — Call-heavy shallow        fragment alloc コストが見えやすい
```

| カテゴリ | fragment への期待 | 論文での扱い |
|---------|------------------|-------------|
| R | 大きく改善（speedup + jitlog 削減） | 主結果 |
| F | 中立〜微悪化 | 退化なしを示す |
| M | 中立 | 既存 heap 最適化と分離 |
| C | 要測定（alloc vs trace 簡素化） | 負の結果・設計空間 |

---

## 3. ティア定義（論文投稿ライン）

| ティア | 本数 | 用途 | 会場 |
|--------|------|------|------|
| **Tier A** | 6 + jitlog 2–3 | 機構証明の最低ライン | ICOOOLPS / 内部マイルストーン |
| **Tier B** | 8–12 + jitlog 3–4 | ECOOP 投稿の実質下限 | ECOOP（条件付き可） |
| **Tier C** | 12–17 + composite + appbench 1 | ECOOP 本命 | ECOOP |

### Tier A（6 本）

| カテゴリ | 本数 | ベンチ |
|---------|------|--------|
| R | 3 | ack, fibo, recurse（要追加） |
| F | 2 | nestedloop, sieve |
| M | 1 | ary または heap |

+ jitlog Before/After: ack, fibo（必須に近い）

### Tier B（8–12 本）

Tier A + 次を追加:

| 追加 | ベンチ |
|------|--------|
| call-heavy | cd16sim または空定義連打ベンチ |
| memory 幅 | heap と ary 両方 |
| shootout 拡張 | lists, matrix, hash, moments のうち port 済み分 |

+ カテゴリ別 geomean / median  
+ jitlog: ack, fibo, nestedloop, ary（3–4 本）

### Tier C（ECOOP 本命）

- shootout 実行可能 12–17 本（`check_coverage.py` 参照）
- **composite.fs** 1 本
- **appbench 大規模 1 本**（推奨: cd16sim）
- Gforth 比較 8–10 本
- baseline 3–4 構成（fixed / virtualizable / no-virtualize / fragment）

---

## 4. 論文用セット（確定案）

### 4.1 Mandatory 8 本

| # | ベンチ | カテゴリ | ソース | rpyforth | 目的 |
|---|--------|---------|--------|----------|------|
| 1 | ack | R | shootout | **対応済** | 深い非末尾再帰。SP 正規化の主戦場 |
| 2 | fibo | R | shootout | **対応済** | 小さい word の再帰密度 |
| 3 | recurse | R | SOM/AWFY 由来 | **未** | ack/fibo だけという批判を避ける |
| 4 | nestedloop | F | shootout | **対応済** | call なし・ループ中心の回帰 |
| 5 | sieve | F/M | shootout | **対応済** | byte array + loop |
| 6 | ary | M | shootout | **対応済** | unboxed `@`/`!` |
| 7 | heap | M | shootout | **対応済** | float heap / sort |
| 8 | call-heavy | C | 新規 or cd16sim | **未** | 浅い call 大量・fragment コスト |

### 4.2 ECOOP stretch +4 本

| # | ベンチ | カテゴリ | shootout phase | 主な blocker |
|---|--------|---------|----------------|--------------|
| 9 | lists | M/C | 4 | STRUCT, %ALLOT |
| 10 | matrix | M/F | 5 | V*, POSTPONE |
| 11 | hash | M/C | 3 | WORDLIST, SEARCH-WORDLIST |
| 12 | moments | M | 4 | READ-LINE, F,, F$ |

### 4.3 大規模ベンチ（appbench）

Ertl & Paysan (ECOOP 2024) が採用。[appbench-1.4](https://www.complang.tuwien.ac.at/forth/appbench-1.4.zip)

| プログラム | 説明 | 行数 | 特性 | 本論文での優先度 |
|-----------|------|------|------|-----------------|
| **cd16sim** | 16-bit CPU エミュレータ | ~937 | calls, app | **最優先**（カテゴリ C + 大規模） |
| lexex | スキャナジェネレータ | ~3655 | calls, app | 大規模本命候補 |
| brainless | チェス | ~3648 | calls, app | lexex の代替 |
| fcp | チェス（別実装） | ~2046 | calls, app | brainless の代替 |
| benchgc | GC ベンチ | ~1155 | calls | GC 評価向け。stack fragment とは直交 |

**推奨**: 大規模は **cd16sim 1 本** を論文に入れる。port 可能なら lexex を追加。

### 4.4 合成ベンチ composite.fs

2SOM (ECOOP 2025) と同様、複数ベンチを 1 プロセスで固定順に連続実行する。

```
ack → fibo → sieve → nestedloop → ary → heap
```

| 項目 | 仕様 |
|------|------|
| 目的 | 「単一マイクロのチューニング」批判への防御 |
| 実行時間 | 各サブの iteration を調整し、全体 3–10 秒 |
| 主張の位置づけ | 機構証明はカテゴリ別ベンチ + jitlog。composite は補助 |
| ファイル | `shootout/composite.fs`（未作成） |

### 4.5 jitlog 対象（4 本）

| ベンチ | カテゴリ | 載せる理由 |
|--------|---------|-----------|
| ack | R | `ds_ptr_*` 削減の Before/After が最も明確 |
| fibo | R | 再帰 trace の代表 |
| nestedloop | F | 退化しないことの trace 証拠 |
| ary | M | stack 最適化と heap 最適化の分離 |

論文本文: 上記 4 本の optimized trace op 分解を必須とする。

### 4.6 Gforth 比較セット（8 本）

Mandatory 8 本と重複させる。目的は Gforth に勝つことではなく、Ertl 2024 文脈での meta-tracing 設計分析。

```
ack, fibo, nestedloop, sieve, ary, heap, recurse, call-heavy（または cd16sim）
```

`benchmark/run_shootout.py --compare gforth` で計測。

---

## 5. ベンチマークカタログ

### 5.1 Classic Shootout（23 プログラム）

Gforth 初期比較で使われた [Doug Bagley / Win32 Shootout](http://www.bagley.org/~doug/shootout/) Forth 版。

**現状: 6/23 対応済**（`./rpyforth-c` で実行可能）

| ID | タイトル | ファイル | Phase | 状態 | 論文カテゴリ | 論文採用 |
|----|---------|---------|-------|------|-------------|---------|
| ack | Ackermann | `shootout/ack.fs` | 0 | supported | R | **必須** |
| ary | Array access | `shootout/ary.fs` | 0 | supported | M | **必須** |
| fibo | Fibonacci | `shootout/fibo.fs` | 0 | supported | R | **必須** |
| heap | Heapsort | `shootout/heap.fs` | 0 | supported | M | **必須** |
| nestedloop | Nested loops | `shootout/nestedloop.fs` | 0 | supported | F | **必須** |
| sieve | Sieve | `shootout/sieve.fs` | 0 | supported | F/M | **必須** |
| hello | Hello world | `shootout/hello.fs` | 1 | missing | — | 対象外 |
| sumcol | Sum column | `shootout/sumcol.fs` | 1 | missing | I/O | 任意 |
| strcat | String concat | `shootout/strcat.fs` | 1 | missing | M | 任意 |
| reversefile | Reverse lines | `shootout/reversefile.fs` | 1 | missing | I/O | 任意 |
| wc | Word count | `shootout/wc.fs` | 1 | missing | I/O | 任意 |
| except | Exceptions | `shootout/except.fs` | 2 | missing | — | 任意 |
| random | Random numbers | `shootout/random.fs` | 2 | missing | F | 任意 |
| hash | Hash table | `shootout/hash.fs` | 3 | missing | M/C | **stretch** |
| hash2 | Two-level hash | `shootout/hash2.fs` | 3 | missing | M/C | 任意 |
| spellcheck | Spell checker | `shootout/spellcheck.fs` | 3 | missing | I/O+C | 任意 |
| wordfreq | Word frequency | `shootout/wordfreq.fs` | 3 | missing | M/C | 任意 |
| lists | Linked lists | `shootout/lists.fs` | 4 | missing | M/C | **stretch** |
| moments | Statistical moments | `shootout/moments.fs` | 4 | missing | M | **stretch** |
| matrix | Matrix multiply | `shootout/matrix.fs` | 5 | missing | M/F | **stretch** |
| methcall | Method calls | — | 6 | out_of_scope | — | 対象外 |
| objinst | Object instantiation | — | 6 | out_of_scope | — | 対象外 |
| prodcons | Producer/consumer | — | 6 | out_of_scope | — | 対象外 |
| regexmatch | Regex scan | — | 6 | out_of_scope | — | 対象外 |

Phase 6（Gforth `objects.fs` / `tasker.fs` / `gray.fs` 依存）は Forth 一般のベンチとして不適切。対象外。

**curve バリアント**: `shootout/curve/{ack,ary,fibo,heap,nestedloop,sieve}.fs` — 論文本文には載せず、開発・回帰・スケーリング曲線用。

### 5.2 論文固有ベンチ（shootout 外）

| ID | ソース | カテゴリ | 状態 | 備考 |
|----|--------|---------|------|------|
| recurse | SOM / AWFY 由来を Forth 化 | R | 未作成 | Tier A 必須 |
| call-heavy | 空定義連打 or cd16sim 簡易版 | C | 未作成 | Tier B 必須 |
| composite | 複数 shootout 連結 | 合成 | 未作成 | Tier C 必須 |

### 5.3 Gforth 小型ベンチ（Ertl 2024 付録）

appbench に加え Ertl が併用。shootout と重複するものあり。

| プログラム | 説明 | 特性 | shootout との関係 |
|-----------|------|------|------------------|
| siev | 素数ふるい | counted loops | = sieve |
| bubble | バブルソート | loops, cond. branch | shootout 外 |
| matrix | 整数行列積 | counted loops | = shootout matrix |
| fib | 再帰 | calls | ≈ fibo |
| fft-bench | FFT | calls in loop | 任意 |
| pentomino | パズル | cond. branches | 任意 |
| sha512 | 暗号 | huge loop body | 任意 |

論文では shootout 版を優先し、Gforth 小型ベンチは Gforth 比較時の重複確認用とする。

### 5.4 類似論文のベンチ規模

| 論文 | 規模 | 教訓 |
|------|------|------|
| Bolz & Tratt (SCP 2013) | 3 + 付録 8 = 11 | 合成・分類で十分な先例 |
| 2SOM (ECOOP 2025) | SOM 17 + 合成 20 | マイクロ + 合成 workload |
| Gaißert et al. (OOPSLA 2025) | ~15 + AWFY ~9 | 機構別分類 |
| Ertl (ECOOP 2024) | appbench 5 + Gforth 小型 5+ | **Forth の大規模の先例** |
| ICOOOLPS | 1 現実負荷 or SOM 一部 | 途中成果向け |

→ VM 実装論の下限: **分類された 10 前後 + jitlog +（大規模 1 または合成 1）**

---

## 6. baseline 構成とメトリクス

### 6.1 baseline 4 構成

| ID | 構成 | 環境変数 / コマンド |
|----|------|---------------------|
| B0 | fixed stack, JIT on | `./rpyforth-c`（現状） |
| B1 | fixed + virtualizable | `./rpyforth-c`（virtualizable 有効） |
| B2 | fixed, no virtualize | `./rpyforth-c-novirt` または `RPYFORTH_NO_VIRTUALIZE` |
| B3 | stack fragment | 実装後の fragment 版 |

比較の主軸: **B1 vs B3**（問題の診断と解決）。B2 は virtualizable の効果分離用。

### 6.2 必須メトリクス

| メトリクス | 用途 | 集計方法 |
|------------|------|----------|
| elapsed time (median) | 全体性能 | `benchmark/run_shootout.py --iterations N` |
| geomean / カテゴリ別 geomean | 論文表 | R/F/M/C ごとに別集計 |
| warmup-inclusive time | tracing JIT の現実コスト | 初回実行含む |
| peak time | 最適化後の純効果 | warmup 後 |
| `ds_ptr_*` getfield/setfield | SP が trace に残る証拠 | `benchmark/jitlog_analysis.py` |
| `getarrayitem_gc` / `setarrayitem_gc` | stack array access | jitlog |
| guard / bridge 数 | trace fragmentation 副作用 | jitlog |
| allocation / GC time | fragment 確保コスト | call-heavy, cd16sim |

### 6.3 Research Questions とベンチの対応

| RQ | 内容 | 使うベンチ |
|----|------|-----------|
| RQ1 | 固定スタックで SP/array ops が trace に残るか | ack, fibo（baseline jitlog） |
| RQ2 | fragment で R 系は速くなるか | ack, fibo, recurse |
| RQ3 | `ds_ptr_*` は消えるか | ack, fibo, nestedloop, ary |
| RQ4 | fragment コストはどこで出るか | call-heavy, cd16sim |
| RQ5 | Gforth 系との性能特性差 | Gforth 比較 8 本 |

---

## 7. 査読批判と対策

| 想定批判 | 対策 | 該当ベンチ |
|---------|------|-----------|
| toy benchmark だけ | 8+ 本 + カテゴリ表 | Tier B 以上 |
| 最適化と無関係なベンチがない | F/M で回帰なし | nestedloop, sieve, ary, heap |
| 実アプリではどうか | composite + appbench | composite.fs, cd16sim |
| wall-clock だけ | jitlog 4 本 | §4.5 |
| Gforth と比較していない | 8 本 Gforth 比較 | §4.6 |
| 単一カテゴリのゲーミング | カテゴリ別 geomean | 全カテゴリ |

---

## 8. 実装優先順位

```
1. Phase 0: ack/fibo baseline jitlog（計測のみ）
2. Tier A 6 本が動く状態の確認（recurse 追加）
3. fragment prototype → A/B/C 比較
4. call-heavy または cd16sim 簡易版
5. composite.fs
6. shootout stretch（lists, hash, moments, matrix）— port phase 順
7. cd16sim full port（appbench）
```

### shootout port phase と blocker 概要

| Phase | 解禁されるベンチ | 主な追加 word |
|-------|-----------------|--------------|
| 1 | hello, sumcol, strcat, wc | File I/O, RESIZE, CASE |
| 2 | except, random | CATCH, VALUE/TO |
| 3 | hash, hash2, spellcheck, wordfreq | WORDLIST, SEARCH-WORDLIST |
| 4 | lists, moments | STRUCT, %ALLOT, READ-LINE |
| 5 | matrix | V*, POSTPONE |

詳細: [`SHOOTOUT_FULL_SUPPORT_PLAN.md`](SHOOTOUT_FULL_SUPPORT_PLAN.md)

---

## 9. ツールとファイル

| ファイル | 役割 |
|---------|------|
| `benchmark/run_shootout.py` | ベンチ一括実行、`--compare gforth/jit/virt`、jitlog 集計 |
| `check_coverage.py` | Core / shootout / **appbench** 対応状況 → `FORTH2012_COVERAGE.md`, `SHOOTOUT_COVERAGE.md`, `APPBENCH_COVERAGE.md` |
| `benchmark/jitlog_analysis.py` | trace op 分解・可視化 |
| `shootout/*.fs` | ベンチ本体 |
| `shootout/curve/*.fs` | スケーリング用（論文外） |

### 実行例

```bash
# 全 shootout 実行
./benchmark/run_shootout.py

# Gforth 比較（8 本 manual でも可）
./benchmark/run_shootout.py --compare gforth --iterations 5

# カバレッジレポート更新
python check_coverage.py
```

---

## 10. 論文用表のテンプレート

### Table: Benchmark suite

| Benchmark | Cat. | Source | Lines | Fragment expectation |
|-----------|------|--------|-------|---------------------|
| ack | R | shootout | ~30 | large speedup |
| fibo | R | shootout | ~20 | large speedup |
| recurse | R | SOM-derived | TBD | large speedup |
| nestedloop | F | shootout | ~15 | neutral |
| sieve | F | shootout | ~40 | neutral |
| ary | M | shootout | ~50 | neutral |
| heap | M | shootout | ~80 | neutral |
| call-heavy | C | new | TBD | measure cost |
| cd16sim | C | appbench | ~937 | measure cost |
| composite | — | synthetic | — | mixed |

### Table: Speedup by category (geomean)

| Category | Benchmarks | B1 (fixed+virt) | B3 (fragment) |
|----------|------------|-----------------|---------------|
| R | ack, fibo, recurse | 1.00 | TBD |
| F | nestedloop, sieve | 1.00 | TBD |
| M | ary, heap | 1.00 | TBD |
| C | call-heavy | 1.00 | TBD |

（計測後に値を埋める）

---

## 11. チェックリスト（ECOOP 投稿前）

- [ ] Tier B 8 本計測完了
- [ ] Tier C stretch 4 本（可能な範囲）
- [ ] composite.fs 作成・計測
- [ ] cd16sim または同等 call-heavy 大規模 1 本
- [ ] baseline B0–B3 すべて計測
- [ ] jitlog 4 本（ack, fibo, nestedloop, ary）
- [ ] カテゴリ別 geomean 表
- [ ] F/M で退化なし確認
- [ ] C でコスト分析済み
- [ ] Gforth 8 本比較
- [ ] `benchmark/run_shootout.py` + 再現手順（artifact）

---

*作成: 2025-06 — stack fragment / ECOOP 論文向けベンチマーク計画*  
*データソース: `check_coverage.py`, `SHOOTOUT_FULL_SUPPORT_PLAN.md`, Ertl ECOOP 2024, 会話での大規模ベンチ分析*
