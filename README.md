# Tracknet-ForPytorch

## 前準備

csvのデータを作ります

```bash
python makefile.py
```

## 使い方

### インストール

```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge opencv
conda install -c anaconda numpy
conda install -c conda-forge tqdm
pip install torchsummary
```

### 実行方法　例

学習
```bash
python train.py -trainname "csv/tracknet.csv" -path "tracknet" -batch 2 -e 500 -tqdm True
```

予測
```bash
python predict.py -weights "weights/tracknet.pth" -output "Tracknet" -tqdm True
```
