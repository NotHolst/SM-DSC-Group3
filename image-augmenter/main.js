const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

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

  image.toFile(outputFile);
}

const inputFiles = fs.readdirSync(INPUT_DIR);
let outputIndex = 0;

const inputFilesNoMasks = inputFiles.filter(filename => {
  const regex = /^(\d)*\.jpg$/;
  return regex.test(filename);
});

for (let i = 0; i < inputFilesNoMasks.length; i++) {
  let filename = inputFilesNoMasks[i];
  let file = path.join(INPUT_DIR, filename);

  const fileNumber = /(\d*).jpg/.exec(filename);
  const maskRegex = new RegExp(fileNumber[1] + "_mask.jpg");
  let maskFile = path.join(INPUT_DIR, inputFiles.find(x => maskRegex.test(x)));

  augmentImage(file, path.join(OUTPUT_IMAGES, ++outputIndex + ".jpg"), 0);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, outputIndex + ".png"), 0);
  augmentImage(file, path.join(OUTPUT_IMAGES, ++outputIndex + ".jpg"), 90);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, outputIndex + ".png"), 90);
  augmentImage(file, path.join(OUTPUT_IMAGES, ++outputIndex + ".jpg"), 180);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, outputIndex + ".png"), 180);
  augmentImage(file, path.join(OUTPUT_IMAGES, ++outputIndex + ".jpg"), 270);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, outputIndex + ".png"), 270);

  augmentImage(file, path.join(OUTPUT_IMAGES, ++outputIndex + ".jpg"), 0, true);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, ++outputIndex + ".png"), 0, true);
  augmentImage(file, path.join(OUTPUT_IMAGES, ++outputIndex + ".jpg"), 90, true);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, ++outputIndex + ".png"), 90, true);
  augmentImage(file, path.join(OUTPUT_IMAGES, ++outputIndex + ".jpg"), 180, true);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, ++outputIndex + ".png"), 180, true);
  augmentImage(file, path.join(OUTPUT_IMAGES, ++outputIndex + ".jpg"), 270, true);
  augmentImage(maskFile, path.join(OUTPUT_MASKS, ++outputIndex + ".png"), 270, true);
  

}