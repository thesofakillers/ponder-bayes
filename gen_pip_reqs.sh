poetry export --without-hashes -o requirements.txt
echo "# local package" >> requirements.txt
echo "-e ." >> requirements.txt
