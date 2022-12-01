unzip Shell2.zip
cd Shell2/Documents/
rm urls.txt
sort < words.txt > sortedwords.txt 
wc -l < words.txt >> sortedwords.txt
cd ../Scripts
./script1 &
./script2 &
./script3 &
jobs > log.txt
cd ..
cp ../urls.txt ./Documents/
cd Documents/
wget -i urls.txt
mv 0906-13-00903-jpg 0906-13-00903.jpg
mv 0906-13-00903.jpg ../Photos/
mv hubble-image-jpg hubble_image.jpg
mv hubble_image.jpg ../Photos/
cd ../Documents/
awk ' BEGIN{ FS = "\t" }; {print $7,$9 | "sort -ur" };  ' < files.txt > date_modified.txt
cd ..
cd ..
rm -v Shell2.tar.gz
tar -zcpf Shell2.tar.gz Shell2/*