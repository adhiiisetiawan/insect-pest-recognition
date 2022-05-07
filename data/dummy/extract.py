import os
import zipfile

zip_ref = zipfile.ZipFile('rps.zip')
zip_ref.extractall()

zip_ref = zipfile.ZipFile('rps-test-set.zip')
zip_ref.extractall()
zip_ref.close()