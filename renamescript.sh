i=0
for file in *.jpg
do
  mv "$file" "img_${i}.jpg"
  (( i=$i+1 ))
done
