import zipfile

with zipfile.ZipFile('images.zip', 'r') as z:
    z.extractall('images')
print("Done.")