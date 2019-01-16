const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const _ = require('lodash')


const INPUT_DIR = path.resolve("./input");
const OUTPUT_IMAGES = path.resolve("./output_images");
const OUTPUT_MASKS = path.resolve("./output_masks");
if (!fs.existsSync(OUTPUT_IMAGES)) {
  fs.mkdirSync(OUTPUT_IMAGES);
} else {
  for (file of fs.readdirSync(OUTPUT_IMAGES)) {
    fs.unlinkSync(path.join(OUTPUT_IMAGES, file));
  }
}

if (!fs.existsSync(OUTPUT_MASKS)) {
  fs.mkdirSync(OUTPUT_MASKS);
} else {
  for (file of fs.readdirSync(OUTPUT_MASKS)) {
    fs.unlinkSync(path.join(OUTPUT_MASKS, file));
  }
}

function augmentImage(input, outputFile, rotation = 0, flipX = false) {
  let image = sharp(input);
  image.rotate(rotation);
  image.flop(flipX);

  image.resize(226,226,{
    kernel: "nearest"
  })
  image.toFile(outputFile);
}

const inputFiles = fs.readdirSync(INPUT_DIR);

let outputIndex = -1;

const inputFilesNoMasks = inputFiles.filter(filename => {
  const regex = /^(\d)*\.jpg$/;
  return regex.test(filename);
});


let randomNumbers = new Array(400)
  for(let j = 0; j < randomNumbers.length; j++){
    randomNumbers[j] = j
  }
  randomNumbers = _.shuffle(_.shuffle(randomNumbers))

for (let i = 0; i < inputFilesNoMasks.length; i++) {
  let filename = inputFilesNoMasks[i];
  let file = path.join(INPUT_DIR, filename);

  const fileNumber = /(\d*).jpg/.exec(filename);
  const maskRegex = new RegExp("^" + fileNumber[1] + "_mask.png", "i");
  


  let maskFile = path.join(INPUT_DIR, inputFiles.find(x => maskRegex.test(x)));



  augmentImage(file, path.join(OUTPUT_IMAGES, randomNumbers[++outputIndex] + ".jpg"), 0);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, randomNumbers[outputIndex] + ".png"), 0);
  
  augmentImage(file, path.join(OUTPUT_IMAGES, randomNumbers[++outputIndex] + ".jpg"), 90);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, randomNumbers[outputIndex] + ".png"), 90);
  
  augmentImage(file, path.join(OUTPUT_IMAGES, randomNumbers[++outputIndex] + ".jpg"), 180);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, randomNumbers[outputIndex] + ".png"), 180);
  
  augmentImage(file, path.join(OUTPUT_IMAGES, randomNumbers[++outputIndex] + ".jpg"), 270);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, randomNumbers[outputIndex] + ".png"), 270);

  augmentImage(file, path.join(OUTPUT_IMAGES, randomNumbers[++outputIndex] + ".jpg"), 0, true);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, randomNumbers[outputIndex] + ".png"), 0, true);
  
  augmentImage(file, path.join(OUTPUT_IMAGES, randomNumbers[++outputIndex] + ".jpg"), 90, true);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, randomNumbers[outputIndex] + ".png"), 90, true);
  
  augmentImage(file, path.join(OUTPUT_IMAGES, randomNumbers[++outputIndex] + ".jpg"), 180, true);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, randomNumbers[outputIndex] + ".png"), 180, true);
  
  augmentImage(file, path.join(OUTPUT_IMAGES, randomNumbers[++outputIndex] + ".jpg"), 270, true);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, randomNumbers[outputIndex] + ".png"), 270, true);
  

}
