
``` bash
pandoc '/Volumes/sw/books/自由与权力 ([英]阿克顿 [未知]) (Z-Library).epub' -o ziyouyuquanli.md \
   --toc --toc-depth=3 \
   --strip-comments \
   --markdown-headings=atx \
   --wrap=none \
   --lua-filter=clean.lua   
```