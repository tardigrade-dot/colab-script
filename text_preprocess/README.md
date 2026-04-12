
``` bash
pandoc '自由的窄廊國家與社會如何決定自由的命運 = The Narrow Corridor States, Societies, and the Fate of Liberty (戴倫 · 艾塞默魯, Daron Acemoglu, 詹姆斯 · 羅賓森 etc.) (z-library.sk, 1lib.sk, z-lib.sk)_简体.epub' -o ziyoudezhailang.md \
   --toc --toc-depth=3 \
   --strip-comments \
   --markdown-headings=atx \
   --wrap=none \
   --lua-filter=clean.lua   

pandoc -f epub '/Volumes/sw/books/财产、法律与政府 ( etc.) (Z-Library).epub' -t plain --wrap=none -o caichanfalvyuzhengfu.txt

```

biliup下载:

``` bash

https://github.com/biliup/biliup/releases

```

biliup使用:

``` bash

biliup --user-cookie /content/drive/MyDrive/biliup_cookies.json upload -c /content/drive/MyDrive/bili_config.yml

```