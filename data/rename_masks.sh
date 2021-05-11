# this script makes sure that the masks have the same names as the images
# replace "road" in each filename with "sat"
# save png files as jpg such
echo "Start renaming files"
find . -type f -name "extra*"| while read FILE ; do
    newfile="$(echo ${FILE} |sed -e 's/road/sat/'|sed -e 's/png/jpg/')" ;
    mv "${FILE}" "${newfile}" ;
done
echo "Renaming files done"