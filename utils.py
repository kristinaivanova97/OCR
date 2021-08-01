import pytesseract
from pytesseract import Output
import os
import cv2
from pdf2image import convert_from_path
from matplotlib import pyplot as plt
from datetime import date, datetime
import functools
import math
import numpy as np
import gzip

# constants
PDF_DPI = 350
IM_DIR = 'transformed_images'


def check_and_read_input(input_file, output_file):
    """
    Check correctness of input and output file formats
    input: input_file - file to be OCRed in english, output_file - '.txt' file with obtained text
    output: read image object (list of several objects in case of long pdf file)
    """

    assert input_file.endswith(('.png', '.jpg', '.pdf')), "Input file should be in .png, .jpeg or .pdf formats."
    assert output_file.endswith('.txt'), "Output file should be in .txt format."
    imgs = []
    if input_file.endswith('.pdf'):
        pages = convert_from_path(input_file, PDF_DPI)
        for i, page in enumerate(pages):
            image_name = "Page_" + str(i) + ".jpg"
            page.save(image_name, "JPEG")
            imgs.append(cv2.imread(image_name))
    else:
        imgs = [cv2.imread(input_file)]
    return imgs


def get_text_from_image(log, image, output_file, mode='w', verbose: int = 2):
    """Perform OCR-ing with tesseract"""
    custom_oem_psm_config = r'--oem 3 --psm 6'
    with open(output_file, mode=mode) as out_f:
        # object = pytesseract.image_to_string(image, lang='eng', config=custom_oem_psm_config)
        out_f.write(pytesseract.image_to_string(image, lang='eng', config=custom_oem_psm_config))
    details = pytesseract.image_to_data(image, lang='eng', output_type=Output.DICT, config=custom_oem_psm_config)
    total_boxes = len(details['text'])
    for sequence_number in range(total_boxes):
        if int(details['conf'][sequence_number]) > 20:

            (x, y, w, h) = (
            details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],
            details['height'][sequence_number])

            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if verbose == 2:
        plot_im(image, 'bounding_box', 0)


def plot_im(img, tras_name: str, num_of_img: int):
    """Save images of a transformation to a folder IM_DIR"""
    plt.figure(figsize=(20, 20))
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.savefig(os.path.join(IM_DIR, "image_{}_{}_transform_{}".format(num_of_img, tras_name, date.today().strftime("%Y%m%d_"))))


def split_pairs(word):
    return [(word[:i + 1], word[i + 1:]) for i in range(len(word))]


def segment(word, word_seq_fitness=None):
    if not word or not word_seq_fitness:
        return []
    all_segmentations = [[first] + segment(rest, word_seq_fitness=word_seq_fitness)
                         for (first, rest) in split_pairs(word)]
    return max(all_segmentations, key=word_seq_fitness)


class OneGramDist(dict):
    """
    1-gram probability distribution for corpora.
    Source: http://norvig.com/ngrams/count_1w.txt
    """
    def __init__(self, filename='data/count_1w_cleaned.txt'):
        self.total = 0
        _open = open
        if filename.endswith('gz'):
            _open = gzip.open
        with _open(filename) as handle:
            for line in handle:
                word, count = line.strip().split('\t')
                self[word] = int(count)
                self.total += int(count)

    def __call__(self, word):
        try:
            result = float(self[word]) / self.total
        except KeyError:
            return 1.0 / (self.total * 10**(len(word) - 2))
        return result


def onegram_log(onegrams, words):
    """
    Use the log trick to avoid tiny quantities.
    http://machineintelligence.tumblr.com/post/4998477107/the-log-sum-exp-trick
    """

    result = functools.reduce(lambda x, y: x + y, (math.log10(onegrams(w)) for w in words))
    return result


def rect2Box(rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def validDateString(dateText):
  try:
    return datetime.strptime(dateText, '%d/%m/%Y')
  except ValueError:
    print("Unable to parse this string to date: ", dateText)
