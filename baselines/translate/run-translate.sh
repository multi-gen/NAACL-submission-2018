#!/usr/bin/env bash

npm init --yes
npm install --save google-translate-api
npm install --save fs
npm install --save csv-parse
npm install --save csv-write-stream

echo 'node modules installed'

node translate_ar_test.js > tmp_ar_test.txt
echo 'Arabic Test translated'
node translate_ar_validate.js > tmp_ar_validate.txt
echo 'Arabic Validate translated'
node translate_eo_test.js > tmp_eo_test.txt
echo 'Esperanto Test translated'
node translate_eo_validate.js > tmp_eo_validate.txt
echo 'Esperanto Validate translated'