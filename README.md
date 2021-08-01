# Technical part
## OCR for three file types
This is a small project written as a part of interview session in a few evenings.

The task was to implement an OCR with tesseract to be able to read .png, .jpeg and .pdf files and  extract readable text.
Preprocessing, tesseract extraction and postprocessing parts must be included.

### Prerequisites:

bash install_tesseract_nltk.sh

pip install -r requirements.txt 

wget http://norvig.com/ngrams/count_1w.txt -O data/count_1w.txt


### Useful articles which helped me to understand the task and write some code blocks:

https://tesseract-ocr.github.io/tessdoc/#usage

https://tesseract-ocr.github.io/tessdoc/ImproveQuality

https://www.pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/

https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033

http://www.sk-spell.sk.cx/tesseract-ocr-parameters-in-302-version

https://medium.com/states-title/using-nlp-bert-to-improve-ocr-accuracy-385c98ae174c

https://towardsdatascience.com/extracting-text-from-scanned-pdf-using-pytesseract-open-cv-cd670ee38052

https://qna.habr.com/q/841085

### Launch with:

python3 Tesseract.py --input test.png --output Result.txt --verbose type

as "type" you can set 0, 1, 2

2 - save images on some stages and full logging;

1 - stands for logging all processing stages;

0 - hidden logs.

All logs are saved to tesseract.log

All images are saved to transformed_images/

Tried to implement BERT masked model for text correction, but had some bugs and the part is putted into comments.

# Challendge
This part was made by adopting code from https://github.com/stackchain/ocr-cnh-2-json.

The test image is in the folder "data".

### Launch with 
python3 OcrIDs.py --input "data/image.png"

In the output you will get an image of the face find_face.png and json with all info final.json.

Due to time difficulties was not able to make it work well.

### Prerequisites
Download:
haarcascade_frontalface_default.xml for face detection.

### Datasets

In my opinion for making more precise model one need to search for existing images of IDs and try to take into account all the difficult/unusual cases, 
then proper dataset should be made, maybe by taking photos of IDs of all the visitors who came into the company during month, then balance datasets (in the sense of different types of IDs and country dependence - with some kind of augmentation) 
and train the NN model (Bert can be OK here as it contains a lot of information about language).

### Improvements
I am quite new comer in the theme of not toy image processing and this was a hard task to do for several evenings, but I new a lot and there are several possible ways of improvements.
Start with trying different kernels to improve image quality and end with correct model training for NER task or some kind of classification. 