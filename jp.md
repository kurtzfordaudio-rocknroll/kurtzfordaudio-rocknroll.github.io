---
layout: default
title: 音楽のステム分離
lang: ja
---

[English]({{ "/" | relative_url }}) | [한국어]({{ "/kr.html" | relative_url }}) | **日本語**

# 音楽ファイルから楽器ごとの音とボーカルを取り出そう

**HTDemucs 6-StemのチェックポイントをCore ML FP16モデルに変換し、自分の音楽ファイルで試してみよう**

**所要時間の目安：** 約30分  
**完成するもの：** macOS 13以降での利用を想定した音源分離用Core MLパッケージ `HTDemucs_CoreML_FP16.mlpackage`

ボーカル、ドラム、ベース、ギター、ピアノなど、さまざまな音がミックスされた楽曲から、それぞれの楽器の音を取り出すことはできるでしょうか。

AI研究者の**Alexandre Défossez氏と共同研究者たち**は、この課題に取り組むため、音源分離モデル**Demucs**を開発しました。研究チームは**MUSDB HQなどの音楽データ**を用いてモデルを学習させ、後続の研究に役立てられるようにコードの一部と学習済みモデルのチェックポイントを公開しました。その後登場した**Hybrid Transformer Demucs（HTDemucs）**は、波形とスペクトログラムの両方を処理し、ドラム、ベース、ボーカル、ギター、ピアノ、その他の音という6つのステムに分離します。

このチュートリアルでは、チェックポイントを自分でダウンロードし、MacでCore ML形式に変換した後、テスト音源を6つのステムに分離して実際に聴いてみます。

<figure style="margin: 24px 0; text-align: center;">
  <img
    src="./images/manlistening.jpg"
    alt="音楽を集中して聴いている人"
    width="500"
    height="500"
    style="width: 500px; height: 500px; object-fit: cover; object-position: center bottom; border-radius: 8px;"
    loading="lazy"
  />
  <figcaption style="margin-top: 8px; font-size: 14px; color: #666;">
    写真提供：
    <a href="https://www.pexels.com/@wahlex-1869682520/" target="_blank" rel="noopener noreferrer">Wahlex</a>,
    pexels.com
  </figcaption>
</figure>

## 目次

- [何を作るのか？](#何を作るのか)
- [準備するもの](#準備するもの)
- [ステップ1. ターミナルを開いて作業フォルダを作る](#ステップ1-ターミナルを開いて作業フォルダを作る)
- [ステップ2. チェックポイントをダウンロードする](#ステップ2-チェックポイントをダウンロードする)
- [ステップ3. Appleのコマンドラインツールをインストールする](#ステップ3-appleのコマンドラインツールをインストールする)
- [ステップ4. Core ML変換ツールをダウンロードする](#ステップ4-core-ml変換ツールをダウンロードする)
- [ステップ5. Pythonをインストールして専用環境を作る](#ステップ5-pythonをインストールして専用環境を作る)
- [ステップ6. チェックポイントをPyTorchのキャッシュに置く](#ステップ6-チェックポイントをpytorchのキャッシュに置く)
- [ステップ7. 変換コードを少し修正する](#ステップ7-変換コードを少し修正する)
- [ステップ8. FP16のCore MLモデルを作成する](#ステップ8-fp16のcore-mlモデルを作成する)
- [ステップ9. モデルの入力と出力を確認する](#ステップ9-モデルの入力と出力を確認する)
- [ステップ10. ステムを分離して聴いてみる](#ステップ10-ステムを分離して聴いてみる)
- [トラブルシューティングFAQ](#トラブルシューティングfaq)
- [おわりに](#おわりに)
- [参考資料](#参考資料)

## 何を作るのか？

**HTDemucs_CoreML_FP16.mlpackage**というCore MLのAIモデルパッケージを作ります。このモデルは44.1 kHzのステレオ音声を10秒ずつ受け取り、ドラム、ベース、ボーカル、ギター、ピアノ、その他の音に分離します。最後の実習では、提供されるtest_10s.wavを入力して6つのWAVファイルを生成し、それぞれを聴いてみます。適切な処理を追加すれば、10秒より長い音楽ファイルも複数の区間に分けて処理し、結果をつなぎ合わせて連続したステムファイルを作ることができます。

## 準備するもの

- **macOS 13以降を搭載したMac**（Apple SiliconとIntelの両方に対応）
- **インターネット接続**
- **ターミナル（Terminal）アプリ**（macOSに標準搭載）
- **Webブラウザ**（チェックポイントなどのダウンロードに使用）
- **実習用の音声 `test_10s.wav`**（このチュートリアルで提供 — [`test_10s.wav`](resources/test_10s.wav)）

---

## ステップ1. ターミナルを開いて作業フォルダを作る

### 1-1. ターミナルとは？

**ターミナル（Terminal）**は、文字でコマンドを入力してMacに指示を出すアプリです。マウスでフォルダを作ったり移動したりする代わりに、短いコマンドを使って操作できます。

<img src="https://help.apple.com/assets/67DB80CC8ED20D93270BDBE5/67DB80CD888B30021203ACDC/en_US/4e8328e6927032645dfb715a76f5a527.png" width="100" alt="Appleのターミナルアプリのアイコン — 黒い背景にコマンドプロンプトの記号">

ターミナルの開き方：

1. Finderを開きます。
2. **アプリケーション（Applications）** → **ユーティリティ（Utilities）**フォルダに移動します。
3. 上のアイコンのような**ターミナル（Terminal）**をダブルクリックします。

または、`⌘ Command + Space`を押して`Terminal`を検索し、Returnキーを押します。

ターミナルを開くと、次のような**プロンプト（prompt）**が表示されます。`~`はMacユーザーのホームフォルダを表します。つまり、プロンプトは**現在の作業場所がホームフォルダである**ことを示しています。

```text
ユーザー名@Mac名 ~ %
```

プロンプトの末尾にコマンドを入力できます。

### 1-2. 作業フォルダを作る

次のコマンドをターミナルにコピーして貼り付け、Return（Enter）キーを押してください。自分で入力しても構いません。

```
mkdir ~/htdemucs
```

`mkdir`は**新しいフォルダを作る**という意味です。ホームフォルダ内に`htdemucs`というフォルダを作成します。正常に完了した場合、通常は何も表示されません。同じフォルダがすでに存在すると`File exists`と表示されることがありますが、その場合は既存のフォルダを使えます。

作成したフォルダに移動します。

```
cd ~/htdemucs
```

`cd`は**フォルダを移動する**コマンドです。次に現在の場所を確認します。

```
pwd
```

表示されたパスが`/htdemucs`で終わっていれば成功です。最後に、フォルダ内にどのようなファイルがあるか確認します。

```
ls
```

`ls`は**現在のフォルダ内のファイル一覧**を表示します。新しいフォルダなら何も表示されないことがあります。

---

## ステップ2. チェックポイントをダウンロードする

**チェックポイント（checkpoint）**とは、AIモデルが学習によって獲得した重みを保存したファイルです。学習済みモデルの中核となる重要なファイルで、これを利用すれば大量の学習データを集めてモデルを一から学習させる必要がありません。

ダウンロードするファイル名は`5c90dfd2-34c22ccb.th`です。次のいずれかのリンクを開いてダウンロードしてください。

- **Metaのサーバー：** [5c90dfd2-34c22ccb.thをダウンロード](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th)
- **Hugging Face — lainlives/audio-separator-models：** [ファイルページ](https://huggingface.co/lainlives/audio-separator-models/blob/main/5c90dfd2-34c22ccb.th)
- **Hugging Face — lainlives/audio-separator-models-old：** [ファイルページ](https://huggingface.co/lainlives/audio-separator-models-old/blob/a78be39e563e8337bc7f2e5241d8063256d86167/5c90dfd2-34c22ccb.th)

Finderで`Downloads`フォルダを開き、`5c90dfd2-34c22ccb.th`があることを確認します。以降は`~/Downloads/5c90dfd2-34c22ccb.th`に保存されているものとして説明します。

---

## ステップ3. Appleのコマンドラインツールをインストールする

**Apple Command Line Tools**は、Macで開発関連のコマンドを利用するための基本ツールです。

まず、すでにインストールされているか確認します。ターミナルで次のコマンドを実行します。

```
xcode-select -p
```

`/Library/Developer/CommandLineTools`または別のインストール先のパスが表示されたら、すでにインストールされています。

エラーが表示された場合は、次のコマンドを実行します。

```
xcode-select --install
```

インストール画面が表示されたら、**インストール（Install）**をクリックして案内に従ってください。完了後、`xcode-select -p`を再実行してパスを確認します。

Apple Command Line Toolsに含まれる**Git**も使えるか確認します。

```
git --version
```

Gitのバージョン番号が表示されれば準備完了です。

---

## ステップ4. Core ML変換ツールをダウンロードする

HTDemucsをCore MLに変換するため、[dexxdean/htdemucs-coreml](https://github.com/dexxdean/htdemucs-coreml)プロジェクトを使用します。この変換ツールは、もともと4ステム用に設定されています。6ステムに対応できるよう少し修正します。

次のコマンドを順番に実行します。4行まとめてターミナルに貼り付けても構いません。`~/htdemucs`への移動、Gitリポジトリの取得、プロジェクトフォルダへの移動、必要な2つのファイルの確認を行います。

```
cd ~/htdemucs
git clone https://github.com/dexxdean/htdemucs-coreml.git
cd htdemucs-coreml
ls convert.py requirements.txt
```

成功すると、次のような出力が表示されます。

```
Cloning into 'htdemucs-coreml'...
remote: Enumerating objects: 12, done.
remote: Counting objects: 100% (12/12), done.
remote: Compressing objects: 100% (9/9), done.
remote: Total 12 (delta 1), reused 12 (delta 1), pack-reused 0 (from 0)
Receiving objects: 100% (12/12), 18.83 KiB | 9.42 MiB/s, done.
Resolving deltas: 100% (1/1), done.
convert.py       requirements.txt
```

---

## ステップ5. Pythonをインストールして専用環境を作る

**Python（パイソン）**は、今回の変換プログラムを実行するためのプログラミング言語です。ここではPython 3.9を使用します。

### 5-1. Pythonのインストールを確認する

ターミナルで次のコマンドを実行します。

```
python3.9 --version
```

`Python 3.9.x`と表示されればステップ5-2に進みます。`command not found`などのメッセージが表示された場合は、[Python公式サイト](https://www.python.org/downloads/macos/)からmacOS用Python 3.9のインストーラをダウンロードしてインストールしてください。Intel MacまたはApple Silicon Macに対応するインストーラを選びます。

インストール後、同じコマンドでもう一度確認してください。

### 5-2. 作業環境を作る

このチュートリアルでは、複数のPythonパッケージをインストールします。ほかのプロジェクトのパッケージと混ざらないように、独立した**仮想環境（virtual environment）**を作ります。

`~/htdemucs/htdemucs-coreml`に移動し、仮想環境を作成して有効にします。

```
cd ~/htdemucs/htdemucs-coreml
python3.9 -m venv .venv
source .venv/bin/activate
```

ターミナルのプロンプトの先頭に`(.venv)`が表示されれば、**このチュートリアル用の仮想環境が有効になっています**。

```text
(.venv) ユーザー名@Mac名 ~/htdemucs/htdemucs-coreml %
```

`(.venv)`の隣に`(base)`が表示されても問題ありません。

### 5-3. 必要なPythonパッケージをインストールする

変換ツールに必要なパッケージをインストールします。次のブロックをまとめてターミナルに貼り付け、実行してください。

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "numpy==2.0.2" scipy
python -m pip check
```

最初のコマンドはPythonのパッケージ管理ツール`pip`を更新します。2番目は変換ツールの依存パッケージをインストールし、3番目は`NumPy 2.0.2`と、最後の音源分離実習に必要な`scipy`をインストールします。最後のコマンドはパッケージ間の依存関係に矛盾がないか確認します。インストールには時間がかかる場合があります。

`requirements.txt`にも`NumPy`が含まれていますが、このチュートリアルではバージョンを2.0.2に固定するために別途インストールします。

完了したら、次のコマンドで必要なパッケージを読み込めるか確認します。

```
python -c 'import torch, demucs, coremltools, numpy, einops; print("Python packages: OK")'
```

`Python packages: OK`と表示されれば読み込みに成功しています。エラーが出た場合は、後述の**トラブルシューティングFAQ**を確認してください。

**作業中にターミナルを閉じ、後で新しいターミナルを開いた場合は？** 最初からインストールし直す必要はありません。再開前に次の2つのコマンドを実行します。

```
cd ~/htdemucs/htdemucs-coreml
source .venv/bin/activate
```

---

## ステップ6. チェックポイントをPyTorchのキャッシュに置く

次のブロック全体をターミナルに貼り付け、Returnキーを押してください。ステップ2でダウンロードしたチェックポイントファイル（`5c90dfd2-34c22ccb.th`）をPyTorchのキャッシュフォルダに自動でコピーします。

```
python - <<'PY'
from pathlib import Path
import shutil
import torch

source = Path.home() / 'Downloads' / '5c90dfd2-34c22ccb.th'
target_dir = Path(torch.hub.get_dir()) / 'checkpoints'
if not source.is_file():
    raise SystemExit(f'ERROR: checkpoint not found: {source}')
target_dir.mkdir(parents=True, exist_ok=True)
target = target_dir / source.name
shutil.copy2(source, target)
print('Copied checkpoint to:', target)
PY
```

最後に`Copied checkpoint to:`と保存先のパスが表示されれば成功です。

---

## ステップ7. 変換コードを少し修正する

ここがこのガイドで最も重要な部分です。ダウンロードした変換ツールには4ステム用の設定が入っています。以下のように変更します。また、比較的古いmacOSでも変換したモデルを使えるように、Core MLの最小デプロイ対象をmacOS 13に設定します。

| 項目 | 元の値 | 変更後 |
| --- | --- | --- |
| モデル名 | `htdemucs` | `htdemucs_6s` |
| ステム数 | `4` | `6` |
| 出力の並べ替え | `[3, 0, 1, 2]` | `[0, 1, 3, 4, 5, 2]` |
| 出力名 | vocals, drums, bass, other | drums, bass, vocals, guitar, piano, other |
| 出力形状の説明 | `(1, 4, 2, …)` | `(1, 6, 2, …)` |
| 最小macOSバージョン | `macOS14` | `macOS13` |

元のDemucs 6-sourceモデルの出力順序は**drums, bass, other, vocals, guitar, piano**です。このガイドでは**drums, bass, vocals, guitar, piano, other**の順序に変更します。

### 7-1. 補助スクリプトをダウンロードする

このチュートリアルに付属する[`prepare_six_stem.py`](resources/prepare_six_stem.py)をダウンロードしてください。`~/Downloads`に保存されたものとして説明します。

次のコマンドを1行ずつ実行します。

```
cd ~/htdemucs/htdemucs-coreml
cp ~/Downloads/prepare_six_stem.py .
ls convert.py prepare_six_stem.py
```

最初のコマンドは、別のフォルダに移動していた場合に備えたものです。すでに正しいフォルダにいる場合は省略できます。
`cp`はスクリプトを現在のフォルダにコピーし、`ls`は両方のファイルがあることを確認します。確認できたら補助スクリプトを実行します。

```
python prepare_six_stem.py
```

このスクリプトは`convert.py`を`convert.py.original-4stem`としてバックアップし、`convert.py`の数か所を修正します。エラーが起きた場合には詳しい内容が表示される設計です。

> 元の変換コードに想定した箇所が見つからない場合、不完全な修正を避けるために処理を中止します。その場合は下のFAQを確認してください。

**想定される出力：**

```text
SUCCESS: convert.py adapted for six stems.
Backup: convert.py.original-4stem
Model: htdemucs_6s
Output: drums, bass, vocals, guitar, piano, other
```

### 7-2. 変更内容を確認する

次のブロックをまとめてターミナルに貼り付け、Returnキーを押します。

```
python -m py_compile convert.py
python - <<'PY'
from pathlib import Path
s = Path('convert.py').read_text()
for line in s.splitlines():
    if line.startswith(('MODEL_NAME =', 'NUM_SOURCES =', 'SOURCE_REORDER =', 'SOURCE_NAMES =')):
        print(line)
PY
```

次のように表示されれば正常です。

```python
MODEL_NAME = "htdemucs_6s"
NUM_SOURCES = 6
SOURCE_REORDER = [0, 1, 3, 4, 5, 2]
SOURCE_NAMES = ["drums", "bass", "vocals", "guitar", "piano", "other"]
```

### なぜ変換ツールを修正するのか？

元のHTDemucsには、`coremltools`でそのまま変換することが難しい演算があります。公開されている変換ツールは複素数演算を実数ベースの表現に置き換え、attentionやoverlap-addの処理をCore MLに変換できる形で実装しています。**モデルの重みを再学習するわけではなく**、学習済みネットワークをCore ML形式に移すことが目的です。

---

## ステップ8. FP16のCore MLモデルを作成する

いよいよ**Core MLパッケージ**を作る最終段階です。

作業場所を`htdemucs-coreml`フォルダに移し、変換プログラム`convert.py`を実行します。ターミナルのプロンプトの先頭に`(.venv)`があることを確認してください。

```
cd ~/htdemucs/htdemucs-coreml
python convert.py --fp16 --compute-units cpu_and_gpu
```

変換中は、次のような処理段階が表示されます。

```text
[1/5] Loading Demucs 'htdemucs_6s' ...
[2/5] Building real-valued wrapper ...
[3/5] PyTorch forward pass ...
[4/5] torch.jit.trace ...
[5/5] Core ML conversion ...
```

複雑なニューラルネットワークの変換には多くのメモリが必要になる場合があります。変換と検証が**エラーなく終了したこと**を確認してください。次のような検証結果が表示されれば正常です。

```
[Val] Validating Core ML vs. PyTorch reference ...

      Max diff:  0.005583
      Mean diff: 0.000232
      Validation OK.
```

出力値の最大差（`Max diff`）と平均差（`Mean diff`）は、小さいほど良い値です。`Validation OK.`は変換ツールの検証に合格したことを示します。

最後に表示される`Next step:`はXcode開発者向けの案内なので、無視して構いません。次のステップでは、作成したCore MLモデルを使って実際の音声を分離します。

モデルパッケージ`HTDemucs_CoreML_FP16.mlpackage`が作成されたことを確認します。

```
ls -ld HTDemucs_CoreML_FP16.mlpackage
```

---

## ステップ9. モデルの入力と出力を確認する

`coremltools`を使って、作成したモデルの入力・出力仕様を確認します。

```
python - <<'PY'
import coremltools as ct

path = 'HTDemucs_CoreML_FP16.mlpackage'
model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
print('Inputs:')
for item in model.get_spec().description.input:
    print(' ', item.name, list(item.type.multiArrayType.shape))
print('Outputs:')
for item in model.get_spec().description.output:
    print(' ', item.name, list(item.type.multiArrayType.shape))
PY
```

確認には少し時間がかかる場合があります。次のような値が表示されます。

```text
Inputs:
  audio [1, 2, 441000]
Outputs:
  sources [1, 6, 2, 441000]
```

`[1, 6, 2, 441000]`は、1回の音声処理で6つのステムが生成され、各ステムが左右2チャンネルと441,000サンプル（44.1 kHzで10秒分）で構成されることを意味します。

ステップ10では、分離の品質と出力順序を必ず実際に聴いて確認してください。例えば`drums.wav`にはドラムの音が、`guitar.wav`には別の楽器ではなく主にギターの音が入っていることを確認します。

---

## トラブルシューティングFAQ

**Q. `python3.9`が見つかりません。**  
A. [Python公式サイト](https://www.python.org/downloads/macos/)からPython 3.9.xをダウンロードしてインストールしてください。その後、ターミナルで`python3.9 --version`を実行します。

**Q. `No module named demucs`などのPythonエラーが表示されます。**  
A. `source .venv/bin/activate`を再度実行し、`python -m pip install -r requirements.txt`が正常に完了しているか確認してください。

**Q. 4つのステムしか生成されません。**  
A. `MODEL_NAME`、`NUM_SOURCES`、`SOURCE_REORDER`、`SOURCE_NAMES`がステップ7の値になっているか確認し、以前に作成した`.mlpackage`を間違って使っていないか確認してください。

**Q. Core MLパッケージは作成できましたが、読み込みや予測に失敗します。**  
A. macOSのバージョン、`coremltools`のバージョン、空きディスク容量、メモリ、`CPU_AND_GPU`の設定を確認してください。検証せずに計算デバイスの設定を変更することは推奨しません。

**Q. 仮想環境を終了するには？**  
A. ターミナルで`deactivate`を実行します。再開するときは作業フォルダで`source .venv/bin/activate`を実行してください。

---

## ステップ10. ステムを分離して聴いてみる

作成したCore MLモデルが、実際の音楽を楽器ごとにどの程度分離できるか確認してみましょう。

10秒間の[`test_10s.wav`](resources/test_10s.wav)をダウンロードしてください。

1. このGitHubリポジトリで**`test_10s.wav`**をクリックし、**Download raw file**アイコンを押してダウンロードします。
2. ファイルがMacの`Downloads`フォルダに保存されていることを確認します。
3. 次のコマンドを1行ずつ実行して変換ツールの作業フォルダに移動し、テスト音源をコピーします。

```
cd ~/htdemucs/htdemucs-coreml
source .venv/bin/activate
cp ~/Downloads/test_10s.wav .
ls -lh test_10s.wav HTDemucs_CoreML_FP16.mlpackage
```

次のコードブロック全体をターミナルにコピーして貼り付け、Returnキーを押してください。音源が6つのWAVファイルに分離されます。

```
python - <<'PY'
from pathlib import Path
import numpy as np
import coremltools as ct
from scipy.io import wavfile

rate, data = wavfile.read('test_10s.wav')
if rate != 44100 or data.ndim != 2 or data.shape[1] != 2:
    raise ValueError('44.1 kHzのステレオWAVファイルが必要です。')

data = data.astype(np.float32)
segment_samples = 441000
if len(data) < segment_samples:
    data = np.pad(data, ((0, segment_samples - len(data)), (0, 0)))
else:
    data = data[:segment_samples]

audio = np.transpose(data, (1, 0))[None, :, :]
model = ct.models.MLModel(
    'HTDemucs_CoreML_FP16.mlpackage',
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
)
sources = np.asarray(model.predict({'audio': audio})['sources'])[0]

names = ['drums', 'bass', 'vocals', 'guitar', 'piano', 'other']
if sources.shape != (6, 2, segment_samples):
    raise ValueError(f'想定外の出力形状: {sources.shape}')

output_dir = Path('separated_stems')
output_dir.mkdir(exist_ok=True)
for name, stem in zip(names, sources):
    file_path = output_dir / f'{name}.wav'
    wavfile.write(file_path, rate, stem.T.astype(np.float32))
    print('作成:', file_path)
PY
```

処理が完了したら、次のコマンドで結果フォルダを開きます。

```
open separated_stems
```

`drums.wav`、`bass.wav`、`vocals.wav`、`guitar.wav`、`piano.wav`、`other.wav`が作成されています。それぞれのファイルをダブルクリックして聴き、元の`test_10s.wav`と比べてみてください。

---

## おわりに

HTDemucsのチェックポイントを使って自分のCore MLモデルを作成し、実際の音楽から6つのステムを分離できました。AIモデルを一から学習させる代わりに、学習済みモデルをCore MLパッケージに変換して新しい音源に適用しました。

6つのWAVファイルを聴くと、楽器によって分離品質が異なり、ほかの楽器の音が少し残っている場合もあります。それでも、1つのミックス音源から個々の楽器の音を取り出せるのは興味深い技術です。

**作成した`HTDemucs_CoreML_FP16.mlpackage`を大切に保管してください！** 次回同じモデルを使う際に、複雑な変換作業を繰り返さずに済みます。

---

## 参考資料

- [Meta Demucsの元のリポジトリ](https://github.com/facebookresearch/demucs)
- [HTDemucs Core ML変換ツール — dexxdean/htdemucs-coreml](https://github.com/dexxdean/htdemucs-coreml)
- [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553)
- [Python公式ダウンロード](https://www.python.org/downloads/macos/)
