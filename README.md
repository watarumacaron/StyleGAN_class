# StyleGAN
## このpythonファイルは何なのか☕️
このpythonファイルは、StyleGANをクラス化したものである。個人的に画像の表示や画像の保存、潜在変数の保存をよく行うため、これらをクラスのメソッドとしてわかりやすい名前で作成した。面白い関数や必要な関数があれば、ぜひ追加してほしい。

## 使い方
[NVlabs/stylegan][stylegan]をgit cloneし、その中にこのpythonファイルを保存する。
Googlecolabで動かす場合は、以下の文を実行する。以下の文を実行することで、簡単にtensorflowの1系を使用することができる。
```
%tensorflow_version 1.x
```

```
import os
os.chdir('styleganを指すパス')

### もしくは

cd 'styleganを指すパス'
```
```
stylegan = StyleGAN('重みのパス')
```
