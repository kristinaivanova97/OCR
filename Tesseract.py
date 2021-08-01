# OCR for three document types. Launch possibilities and prerequisites are written in README.md
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import click
from logging import getLogger, basicConfig, INFO
from imutils import *
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import re
import nltk
from enchant.checker import SpellChecker
from difflib import SequenceMatcher
from transformers.tokenization_bert import BasicTokenizer
from utils import *


class PreprocessImage:
	def __init__(self, log, verbose):
		self.logger = log
		self.verbose = verbose

	def bilateral_filter(self, image: np.ndarray):
		"""Implement a Bilateral Filtering for an object as is effective at noise removal while preserving edges"""
		blurred = cv2.bilateralFilter(image, 9, 75, 75)
		# blurred = cv2.GaussianBlur(image, (5, 5), 0)
		changed_pixels = np.sum(np.where(image.flatten() != blurred.flatten(), 1, 0))
		if self.verbose:
			self.logger.info("%i pixels were changed during Bilateral Filtering.", changed_pixels)

		return blurred

	def rescale_image(self, image):
		"""
		Rescale image to make one of dimensions equal to 1024
		input: image object (np.ndarray)
		output: resized image object
		"""
		scale_factor = max(1.0, float(2048.0 / image.shape[1]))
		width = int(image.shape[1] * scale_factor)
		height = int(image.shape[0] * scale_factor)
		dim = (width, height)
		resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
		if self.verbose:
			self.logger.info("Image was rescaled from %i, %i size to %i, %i", image.shape[1], image.shape[0], width, height)
		return resized

	def image_threshold(self, image):
		"""Get black white tresholding mask and apply to an image"""
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		(thresh, black) = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		if self.verbose:
			self.logger.info("Image was transferred to black/white colors")
		return black

	def find_orientation_and_rotate(self, image):
		"""Rotate an image"""
		rot = re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0)
		angle = float(rot)
		if angle > 0:
			angle = 360 - angle
		(h, w) = image.shape[:2]
		center = (w // 2, h // 2)
		rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
		rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
		if self.verbose:
			self.logger.info("Angle of an image is {}".format(angle))
			self.logger.info("Info about image rotation after the rotation: {}".format(pytesseract.image_to_osd(rotated)))
		return rotated

	def smooth(self, image):
		"""Implement a smoothing technique with opening and closing kernels"""
		kernel = np.ones((1, 1), np.uint8)
		opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
		closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
		ret1, th1 = cv2.threshold(image, cv2.THRESH_BINARY, 255, cv2.THRESH_BINARY)
		ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		blur = cv2.GaussianBlur(th2, (1, 1), 0)
		ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		or_image = cv2.bitwise_or(th3, closing)
		if self.verbose:
			self.logger.info("Opening and closing kernels were applied to an image")
			self.logger.info("Several thresholding, Gaussian blurring, Otsu blurring")
		return or_image

	def apply_transformations(self, images: list, verbose: int = 2):
		"""Transform an image to be better for inserting in OCR"""
		updated = []
		for num_img, image in enumerate(images):
			image_history = []
			image = self.rescale_image(image)
			image_history.append((image, 'rescale'))
			image = self.bilateral_filter(image)
			image_history.append((image, 'bilateral_filtering'))
			image = self.image_threshold(image)
			image_history.append((image, 'thersholding'))
			image = self.find_orientation_and_rotate(image)
			image_history.append((image, 'rotate'))
			image = self.smooth(image)
			image_history.append((image, 'smoothing'))
			updated.append(image)
			if verbose == 2:
				for transform in image_history:
					plot_im(transform[0], transform[1], num_img)
		return updated


class PostprocessImage:
	def __init__(self, text_file, verbose, log):
		assert text_file.endswith('.txt')
		self.text_file = text_file
		self.spellcheker = SpellChecker("en_US")
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
		self.logger = log
		self.verbose = verbose

	def get_personslist(self, text):
		"""
		Spellcheker won't work well with person names, so they should be saved
		input: text
		output: list with all words with 'PERSON' pos tag
		"""
		personslist = []
		for sent in nltk.sent_tokenize(text):
			for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
				if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
					personslist.insert(0, (chunk.leaves()[0][0]))
		if self.verbose:
			self.logger.info("Got %i person names from text", len(personslist))
		return list(set(personslist))

	def find_bert_predictions(self, text):
		"""Predict words for masked elements with Bert"""
		tokenized_text = self.tokenizer.tokenize(text)
		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
		MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']
		# Create the segments tensors
		segs = [i for i, e in enumerate(tokenized_text) if e == "."]
		segments_ids = []
		prev = -1
		for k, s in enumerate(segs):
			segments_ids = segments_ids + [k] * (s - prev)
			prev = s
		segments_ids = segments_ids + [len(segs)] * (len(tokenized_text) - len(segments_ids))
		segments_tensors = torch.tensor([segments_ids])
		# prepare Torch inputs
		tokens_tensor = torch.tensor([indexed_tokens])
		# Predict all tokens
		with torch.no_grad():
			predictions = self.model(tokens_tensor, segments_tensors)
		return predictions, MASKIDS

	def predict_word(self, suggestedwords, MASKIDS, text_original, predictions):
		pred_words = []
		for i in range(len(MASKIDS)):
			preds = torch.topk(predictions[0, MASKIDS[i]], k=50)
			indices = preds.indices.tolist()
			list1 = self.tokenizer.convert_ids_to_tokens(indices)
			list2 = suggestedwords[i]
			simmax = 0
			predicted_token = ''
			for word1 in list1:
				for word2 in list2:
					s = SequenceMatcher(None, word1, word2).ratio()
					if s is not None and s > simmax:
						simmax = s
						predicted_token = word1
			text_original = text_original.replace('[MASK]', predicted_token, 1)
		return text_original

	def clean_unacceptable_chars(self, text):
		"""Remove long sequences of the same letter"""
		text_len = len(text)
		unacceptable = re.compile(r'([a-zA-Z])\1{2,10}')
		text = unacceptable.sub('', text)
		if self.verbose:
			self.logger.info("After removing long sequences of the same letter from text its lenghs dropped on %i chars", len(text) - text_len)
		return text

	def get_parts(self, text):

		words = text.split(" ")
		text_part = ''
		text_parts = []
		for word in words:
			text_part += word + ' '
			if len(text_part) >= 256:
				text_parts.append(text_part)
				text_part = ''
		if text_part:
			text_parts.append(text_part)
		return text_parts

	def transform(self):
		"""Transform text to a more grammatical version"""

		with open(self.text_file, 'r') as f:
			lines = f.readlines()
			lines = "\n".join([line.strip() for line in lines])
		lines = self.clean_unacceptable_chars(lines)
		# cleanup text
		rep = {'\\': ' ', '\"': '"', '-': ' ', '"': ' " ', ',': ' , ', '.': ' . ', '!': ' ! ',
			   '?': ' ? ', "n't": " not", "'ll": " will", '*': ' * ', '(': ' ( ', ')': ' ) ', "s'": "s '"}
		rep = dict((re.escape(k), v) for k, v in rep.items())
		pattern = re.compile("|".join(rep.keys()))
		text = pattern.sub(lambda m: rep[re.escape(m.group(0))], lines)
		personslist = self.get_personslist(text)
		ignorewords = personslist + ["!", ",", ".", '\"', "?", '(', ')', '*', "'"]  # using enchant.checker.SpellChecker, identify incorrect words
		datetype_pattern = re.compile('(\d*:\d*)|(\d*/\d*)|(\d*)')
		words = text.split()
		incorrectwords = [w for w in words if not self.spellcheker.check(w) and w not in ignorewords and
						  datetype_pattern.search(w) is None]
		# using enchant.checker.SpellChecker, get suggested replacements
		suggestedwords = [self.spellcheker.suggest(w) for w in incorrectwords]
		# replace incorrect words with [MASK]
		assert len(incorrectwords) == len(suggestedwords)
		for word_i, w in enumerate(incorrectwords):
			if suggestedwords[word_i]:
				text = text.replace(w, suggestedwords[word_i][0])
				lines = lines.replace(w, suggestedwords[word_i][0])
		# *** not workoing for now part ***
		# for w in incorrectwords:
		# 	text = text.replace(w, '[MASK]')
		# 	lines = lines.replace(w, '[MASK]')
		# many_masks = re.compile("(\[MASK\])*")
		# text = many_masks.sub('', text)
		# lines = many_masks.sub('', lines)
		# text_parts = self.get_parts(text)
		# text_parts_original = self.get_parts(lines)
		# for i, text_i in enumerate(text_parts):
		# 	predictions, MASKIDS = self.find_bert_predictions(text_i)
		# 	lines_i = self.predict_word(suggestedwords, MASKIDS, text_parts_original[i], predictions)
		# 	mode = 'w' if i == 0 else 'a'
		# 	with open(self.text_file.replace('_temporary.txt', '.txt'), mode=mode) as f:
		# 		f.write(lines_i)
		# *** end ***
		onegrams = OneGramDist(filename='data/count_1w.txt')
		onegram_fitness = functools.partial(onegram_log, onegrams)
		tok_text = text.split('\n')
		final_text = ''
		for text_i in tok_text:
			btokenizer = BasicTokenizer(do_lower_case=False)
			tokens = btokenizer.tokenize(text_i)
			for tok in tokens:
				findings = segment(tok.lower(), word_seq_fitness=onegram_fitness)
				if tok.istitle():
					findings[0] = findings[0].title()
				elif tok.isupper():
					findings[0] = findings[0].upper()
					if len(findings) > 1:
						findings[1] = findings[1].upper()
				final_text += " ".join(findings) + " "
			final_text += '\n'
		with open(self.text_file.replace('_temporary.txt', '.txt'), mode='w') as f:
			f.write(final_text)
		os.remove(self.text_file)


@click.command()
@click.option('--input', type=str, prompt='input file', help='The image or document for OCR processing. Possible formats: png, jpeg, pdf.')
@click.option('--output', type=str, prompt='output file', help='The .txt file, where extracted text information will be placed.')
@click.option('--verbose', type=int, prompt='verbose mode', help='Output detailed logs. "2" save images on some stages and full logging; '
																 '"1" stands for logging all processing stages; "0" - hidden logs')
def main(input, output, verbose):
	logger = getLogger(__name__)
	basicConfig(filename='tesseract.log', level=INFO)
	logger.info("Start preprocessing ...")
	imgs = check_and_read_input(input, output.replace('.txt', '_temporary.txt'))
	if verbose == 2:
		if not os.path.exists(IM_DIR):
			os.mkdir(IM_DIR)
	preprocessor = PreprocessImage(log=logger, verbose=verbose)
	preprocessed_imgs = preprocessor.apply_transformations(images=imgs, verbose=verbose)
	for j, preprocessed_img in enumerate(preprocessed_imgs):
		mode = 'w' if j == 0 else 'a'
		get_text_from_image(logger, preprocessed_img, output.replace('.txt', '_temporary.txt'), mode, verbose)
	postprocessor = PostprocessImage(output.replace('.txt', '_temporary.txt'), log=logger, verbose=verbose)
	postprocessor.transform()


if __name__ == '__main__':
	main()
