mkdir dataset/tashkeela/train -p
mkdir dataset/tashkeela/val -p
mkdir dataset/tashkeela/test -p
mkdir dataset/tashkeela/preds -p

wget -P dataset/tashkeela/train https://github.com/AliOsm/arabic-text-diacritization/raw/master/dataset/train.txt
wget -P dataset/tashkeela/val https://github.com/AliOsm/arabic-text-diacritization/raw/master/dataset/val.txt
wget -P dataset/tashkeela/test https://github.com/AliOsm/arabic-text-diacritization/raw/master/dataset/test.txt