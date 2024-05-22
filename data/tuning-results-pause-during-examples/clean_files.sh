#go trough all .out files in the folder and remove empty lines using sed

for file in *.out; do
    echo "Cleaning $file"
    cat $file | tr -s '\r\n' > $file.clean
    mv $file.clean $file
done