## OCR for three file types
This is a small project written as a part of interview session in a few evenings.

The task was to implement an OCR with tesseract to be able to read .png, .jpeg and .pdf files and  extract readable text.
Preprocessing, tesseract extraction and postprocessing parts must be included.

#### Prerequisites (assume numpy, datetime, re and os are already installed):

pip install opencv-python;

pip install pytesseract;

pip install click;

pip install pdf2image;

pip install python-poppler;

pip install imutils;

pip install pytorch-pretrained-bert;

pip install pyenchant;

pip install nltk;

import nltk
nltk.download('averaged_perceptron_tagger'); 
nltk.download('maxent_ne_chunker'); 
nltk.download('words'); 

#### Useful articles which helped me to understand the task and write some code blocks:

https://tesseract-ocr.github.io/tessdoc/#usage

https://tesseract-ocr.github.io/tessdoc/ImproveQuality

https://www.pyimagesearch.com/2017/07/17/credit-card-ocr-with-opencv-and-python/

https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

https://medium.com/cashify-engineering/improve-accuracy-of-ocr-using-image-preprocessing-8df29ec3a033

http://www.sk-spell.sk.cx/tesseract-ocr-parameters-in-302-version

https://medium.com/states-title/using-nlp-bert-to-improve-ocr-accuracy-385c98ae174c

https://towardsdatascience.com/extracting-text-from-scanned-pdf-using-pytesseract-open-cv-cd670ee38052

https://qna.habr.com/q/841085

Launch with:

python3 Tesseract.py --input test.png --output Result.txt --verbose type

as "type" you can set 0, 1, 2

2 - save images on some stages and full logging;

1 - stands for logging all processing stages;

0 - hidden logs.

All logs are saved to tesseract.log

All images are saved to transformed_images/

Tried to implement BERT masked model for text correction, but had some bugs and the part is putted into comments.
