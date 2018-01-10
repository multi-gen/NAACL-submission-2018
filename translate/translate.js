const translate = require('google-translate-api');
//const fs = require('fast-csv')
const fs = require('fs')
var parse = require('csv-parse')
var csvWriter = require('csv-write-stream')
var writer = csvWriter({ headers: ['Main-Item','Number of Triples','Summary 1','Target']})

var inputData = require('./translate_input_ar.json')
//var inputPath = require('./translate_input_eo.csv')

writer.pipe(fs.createWriteStream('out-ar.csv'))
for(var entityId in inputData){
	arr = inputData[entityId]
	nr_triples = arr[0]
	sentence = arr[1]
	target = arr[2]
	translation(entityId, nr_triples, target)
}

function translation(entityId, nr_triples, target){
	translate(sentence, {from: 'en', to: 'ar'}).then(res => {
	    writer.write([entityId, nr_triples, res.text, target])
	    console.log(target)
	    return res.text
	}).catch(err => {
	    console.error(err);
	});
}