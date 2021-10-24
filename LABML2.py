import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils
from sklearn.neighbors import KNeighborsClassifier
import pandas
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
import re
import string
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import random
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

class LOG_REG():
	def __init__(self, dim):
		self.teta = np.zeros((dim + 1, ))
	def calculate(self, x):
		if len(x) + 1 != len(self.teta):
			raise ValueError
		sum = self.teta[0]
		for i in range(len(x)):
			sum += x[i]*self.teta[i + 1]
		if sum > 15.0:
			return 1.0
		elif sum < -15.0:
			return 0.0
		return 1.0/(1.0 + math.exp(-sum))
	def __x_to_t(self, x, t):
		t[0] = 1.0
		for i in range(len(x)):
			t[i + 1] = x[i]
	def fit(self, X, Y, bs = 0, a = 0.1):
		if len(X) != len(Y):
			raise ValueError
		if bs == 0:
			bs = len(X)
		sum = np.zeros((len(self.teta), ))
		x = np.zeros((len(self.teta), ))
		for i in range(len(X)):
			self.__x_to_t(X[i], x)
			sum += (float(Y[i]) - self.calculate(X[i]))*x
			if ((i + 1) % bs == 0) or (i + 1 == len(X)):
				self.teta += a*sum
	def test(self, X, Y):
		if len(X) != len(Y):
			raise ValueError
		count = 0
		for i in range(len(X)):
			ans = self.calculate(X[i])
			if ans >= 0.5:
				ans = 1
			else:
				ans = 0
			if (Y[i] == ans):
				count += 1
		return count/len(X)
	def answer(self, X):
		out = [0]*len(X)
		for i in range(len(X)):
			ans = self.calculate(X[i])
			if ans >= 0.5:
				out[i] = 1
		return out

class SVM():
	def __init__(self, dim):
		self.w = np.zeros((dim + 1, ))
	def calculate(self, x):
		if len(x) + 1 != len(self.w):
			raise ValueError
		sum = self.w[0]
		for i in range(len(x)):
			sum += x[i]*self.w[i + 1]
		if sum >= 0:
			return 1
		return -1
	def __x_to_t(self, x, t):
		t[0] = 1.0
		for i in range(len(x)):
			t[i + 1] = x[i]
	def fit(self, X, Y, a = 0.1, n = 0.1):
		if len(X) != len(Y):
			raise ValueError
		x = np.zeros((len(self.w), ))
		for i in range(len(X)):
			self.__x_to_t(X[i], x)
			if (Y[i]*np.dot(self.w, x) < 1):
				self.w -= n*(a*self.w - Y[i]*x)
				continue
			self.w -= n*a*self.w
	def test(self, X, Y):
		if len(X) != len(Y):
			raise ValueError
		count = 0
		for i in range(len(X)):
			if Y[i] == self.calculate(X[i]):
				count += 1
		return count/len(X)
	def answer(self, X):
		out = [0]*len(X)
		for i in range(len(X)):
			out[i] = self.calculate(X[i])
		return out

def entropy(X):
	freqs = {}
	freqs[0] = 0
	freqs[1] = 0
	for el in X:
		freqs[el[-1]] += 1
	tmp0 = float(freqs[0])/len(X)
	tmp1 = float(freqs[1])/len(X)
	if freqs[0] == 0:
		tmp0 = 0.0
	else:
		tmp0 = -tmp0*math.log(tmp0, 2)
	if freqs[1] == 0:
		tmp1 = 0.0
	else:
		-tmp1*math.log(tmp1, 2)
	return tmp0 + tmp1
def split(X, i, element):
	X_tmp = []
	for el in X:
		if el[i] == element:
			tmp = el[:i]
			tmp.extend(el[i + 1:])
			X_tmp.append(tmp)
	return X_tmp
def get_feat_to_del(X):
	start_ent = entropy(X)
	max_gain = 0.0
	max_feat = -1
	for i in range(len(X[0]) - 1):
		feats = [el[i] for el in X]
		set_feats = set(feats)
		ent = 0.0
		for el in set_feats:
			X_tmp = split(X, i, el)
			ent += len(X_tmp) * entropy(X_tmp) / len(X)
		gain = start_ent - ent
		if (gain > max_gain):
			max_gain = gain
			max_feat = i
	return max_feat
def top_class(Y):
	freqs = {}
	freqs[0] = 0
	freqs[1] = 0
	for ans in Y:
		freqs[ans] += 1
	if freqs[0] >= freqs[1]:
		return 0
	return 1
def create_tree(X, labels = [-1]):
	if len(labels) == 1 and labels[0] == -1:
		labels = [i for i in range(len(X[0]) - 1)]
	now_Y = [el[-1] for el in X]
	if now_Y.count(now_Y[0]) == len(now_Y):
		return now_Y[0]
	if len(now_Y) == 1:
		return top_class(now_Y)
	index = get_feat_to_del(X)
	top_ans = labels[index]
	tree = [top_ans, {}]
	del(labels[index])
	feats = [el[index] for el in X]
	set_feats = set(feats)
	for feat in set_feats:
		new_l = labels[:]
		tree[1][feat] = create_tree(split(X, index, feat), new_l)
	return tree
def predict(tree, x):
	if type(tree) != int:
		if x[tree[0]] not in tree[1]:
			keys = list(tree[1].keys())
			return predict(tree[1][keys[random.randint(0, len(keys) - 1)]], x)
		return predict(tree[1][x[tree[0]]], x)
	return tree
def test(tree, X):
	count = 0
	Y = []
	for x in X:
		Y.append(predict(tree, x))
		if Y[-1] == x[-1]:
			count += 1
	return count/len(X), Y

max_review = 1000

nltk.download('punkt')
nltk.download('wordnet')

max_words = 30

stopwords=['this','that','and','a','we','it','to','is','of','up', 'are', 'as','the']

def ampliment(texts, start, end):
	freq = {}
	ps = PorterStemmer()
	lem = WordNetLemmatizer()
	for i in range(start, end):
		step = 0
		texts[i] = texts[i].strip()
		texts[i] = texts[i].lower()
		tmp = texts[i].split(' ')
		for word in tmp:
			start_word = word
			word = ps.stem(word)
			word = re.sub(r'\d+', '', word)
			if word in stopwords:
				word = 'W'
			if word == start_word:
				continue
			if step == 0 or step + 1 == len(tmp):
				texts[i] = texts[i].replace(start_word + ' ', word + ' ')
			else:
				texts[i] = texts[i].replace(' ' + start_word + ' ', ' ' + word + ' ')
			step += 1
	an_text = ""
	for text in texts:
		text = text.replace('/', "")
		text = text.replace('<', "")
		text = text.replace('>', "")
		text = text.replace('-', " ")
		text = text.replace(',', "")
		text = text.replace('.', "")
		text = text.replace('?', "")
		text = text.replace('!', "")
		for word in text.split(' '):
			freq[word] = freq.get(word, 0) + 1
		an_text += text + " "
	b_tok = nltk.word_tokenize(an_text)
	print("count of bigramms: ", len(list(nltk.bigrams(b_tok))))
	sorted_val = sorted(freq.values(), reverse=True)
	sort_freq = {}
	for i in sorted_val:
		for k in freq.keys():
			if freq[k] == i:
				sort_freq[k] = freq[k]
				break
	#print("freq (W = count of stop word):")
	#print(sort_freq)
	tok = Tokenizer(num_words = 5000)
	tok.fit_on_texts(texts)
	#print(tok.word_index)
	seq = tok.texts_to_sequences(texts)
	return pad_sequences(seq, maxlen = max_words)

data = pandas.read_csv('base.csv')
x_train = data['review'][:max_review]

X = ampliment(x_train, 0, len(x_train))
print(X)

y_train = data['sentiment'][:max_review]
Y = [0]*max_review
for i in range(max_review):
	if y_train[i] == 'positive':
		Y[i] = 1
	else:
		Y[i] = 0

#generic synthetic data: (with line y = k*x + b)
"""
data_size = 1000
k = 2.0
b = 1.0
X = np.array([random.random() for i in range(data_size)])
Y = np.array([k*random.random() + b for i in range(data_size)])

y_train = []
x_train = [0]*data_size
for i in range(data_size):
	x_train[i] = [X[i], Y[i]]
	if Y[i] >= X[i]*k + b:
		y_train.append(1)
	else:
		y_train.append(0)
"""

x_train = X
y_train = Y

print("my_LR:")
lg = LOG_REG(len(x_train[0]))
lg.fit(x_train, y_train, 2.0)
print("accuracy: ", lg.test(x_train, y_train))
print("confusion_matrix:")
print(confusion_matrix(y_train, lg.answer(x_train)))

print("SK_LR:")
lg2 = LogisticRegression(solver='liblinear', multi_class = 'auto').fit(x_train, y_train)
print("accuracy: ", lg2.score(x_train, y_train))
print("confusion_matrix:")
print(confusion_matrix(y_train, lg2.predict(x_train)))

for i in range(len(y_train)):
	if (y_train[i] == 0):
		y_train[i] = -1

print("my_SVM:")
lg = SVM(len(x_train[0]))
lg.fit(x_train, y_train, 0.05, 0.01)
print("accuracy: ", lg.test(x_train, y_train))
print("confusion_matrix:")
print(confusion_matrix(y_train, lg.answer(x_train)))

for i in range(len(y_train)):
	if (y_train[i] == -1):
		y_train[i] = 0

print("SK_SVM:")
lg2 = LinearSVC(C = 0.00001, dual=True, max_iter=100000)
lg2.fit(x_train, y_train)
print("accuracy: ", lg2.score(x_train, y_train))
print("confusion_matrix:")
print(confusion_matrix(y_train, lg2.predict(x_train)))

"""
for i in range(len(x_train)):
	x_train[i][0] = int(x_train[i][0]*55)
	x_train[i][1] = int(x_train[i][1]*55)
"""

print("SK SOL_TREE:")
tree_sk = DecisionTreeClassifier()
tree_sk.fit(x_train[:800], y_train[:800])
print("accuracy: ", tree_sk.score(x_train[800:max_review], y_train[800:max_review]))
print("confusion_matrix:")
print(confusion_matrix(y_train[800:max_review], tree_sk.predict(x_train[800:max_review])))

x_train = [0]*len(X)

for i in range(len(x_train)):
	x_train[i] = X[i].tolist()
	x_train[i].append(y_train[i])

print("SOL_TREE:")
tree = create_tree(x_train[:800])
ac, Ys = test(tree, x_train[800:max_review])
print("accuracy: ", ac)
print("confusion_matrix:")
print(confusion_matrix(y_train[800:max_review], Ys))
