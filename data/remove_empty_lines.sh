#remove empty lines \r\n from all *.out files

for file in *.out; do
    echo $file
    cat $file | tr -s '\r\n' > $file.tmp
    mv $file.tmp $file
done