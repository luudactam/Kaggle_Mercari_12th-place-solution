import os ; os.environ['OMP_NUM_THREADS'] = '4'

import re
from datetime import datetime 
start_real = datetime.now()
#from time import time
from collections import Counter

import tensorflow as tf
tf.set_random_seed(103)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6)
import pandas as pd
import numpy as np
import time
start_time = time.time()

from nltk.stem.porter import PorterStemmer
from fastcache import clru_cache as lru_cache

from scipy.sparse import csr_matrix, hstack

from sklearn.preprocessing import LabelBinarizer

import sys

#Add https://www.kaggle.com/anttip/wordbatch to your kernel Data Sources, 
#until Kaggle admins fix the wordbatch pip package installation
#sys.path.insert(0, '../input/workbatch/wordbatch/')
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL


import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk.corpus import stopwords
import re
import math

def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))
# Tokenization and pipeline
np.random.seed(1525)
t_start = time.time()

stemmer = PorterStemmer()

@lru_cache(32768, typed=False)
def stem(s):
    return stemmer.stem(s)

whitespace = re.compile(r'\s+')
non_letter = re.compile(r'\W+')

@lru_cache(32768, typed=False)
def tokenize(text):
    text = text.lower()
    text = non_letter.sub(' ', text)

    tokens = []

    for t in text.split():
        t = stem(t)
        tokens.append(t)

    return tokens

class Tokenizer:
    def __init__(self, min_df=10, tokenizer=str.split):
        self.min_df = min_df
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = None

    def fit_transform(self, texts):
        tokenized = []
        doc_freq = Counter()
        n = len(texts)

        for text in texts:
            sentence = self.tokenizer(text)
            tokenized.append(sentence)
            doc_freq.update(set(sentence))

        vocab = sorted([t for (t, c) in doc_freq.items() if c >= self.min_df])
        vocab_idx = {t: (i + 1) for (i, t) in enumerate(vocab)}
        doc_freq = [doc_freq[t] for t in vocab]

        self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        max_len = 0
        result_list = []
        for text in tokenized:
            text = self.text_to_idx(text)
            max_len = max(max_len, len(text))
            result_list.append(text)

        self.max_len = max_len
        result = np.zeros(shape=(n, max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text

        return result

    def text_to_idx(self, tokenized):
        return [self.vocab_idx[t] for t in tokenized if t in self.vocab_idx]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = self.tokenizer(texts[i])
            text = self.text_to_idx(text)[:self.max_len]
            result[i, :len(text)] = text

        return result

    def vocabulary_size(self):
        return len(self.vocab) + 1

def size64_to_size32(df):
    for c in df.columns:
        if df[c].dtypes=='int64':
            df[c]=df[c].astype(np.int32)
        if df[c].dtypes=='float64':
            df[c]=df[c].astype(np.float32)
            

# reading train data
print('reading train data...')
df_train = pd.read_csv('../input/train.tsv', sep='\t')
#df_train = df_train[df_train.price != 0].reset_index(drop=True)
df_train = df_train.drop(df_train[(df_train.price < 3.0)].index)
print(df_train.shape)
price = df_train.pop('price')
y = np.log1p(price.values)
mean = y.mean()
std = y.std()
y = (y - mean) / std
y = y.reshape(-1, 1)

# process the NaN in name, category_name, brand_name
# and item_description
df_train.name.fillna('unkname', inplace=True)
df_train.category_name.fillna('unk_cat', inplace=True)
df_train.brand_name.fillna('unk_brand', inplace=True)
df_train.item_description.fillna('nodesc', inplace=True)


# Process category, title, description, brand and other features

print('processing category...')


def paths(tokens):
    all_paths = ['/'.join(tokens[0:(i+1)]) for i in range(len(tokens))]
    return ' '.join(all_paths)

@lru_cache(32768, typed=False)
def cat_process(cat):
    cat = cat.lower()
    cat = whitespace.sub('', cat)
    split = cat.split('/')
    return paths(split)

df_train.category_name = df_train.category_name.apply(cat_process)

print("Cat token")
cat_tok = Tokenizer(min_df=55)
X_cat = cat_tok.fit_transform(df_train.category_name)
cat_voc_size = cat_tok.vocabulary_size()

print('processing title...')

name_tok = Tokenizer(min_df=10, tokenizer=tokenize)
X_name = name_tok.fit_transform(df_train.name)
name_voc_size = name_tok.vocabulary_size()

print('processing description...')

desc_num_col = 54 #v0 40
desc_tok = Tokenizer(min_df=50, tokenizer=tokenize)
X_desc = desc_tok.fit_transform(df_train.item_description)
X_desc = X_desc[:, :desc_num_col]
desc_voc_size = desc_tok.vocabulary_size()

print('processing brand...')

df_train.brand_name = df_train.brand_name.str.lower()
df_train.brand_name = df_train.brand_name.str.replace(' ', '_')

brand_cnt = Counter(df_train.brand_name[df_train.brand_name != 'unk_brand'])
brands = sorted(b for (b, c) in brand_cnt.items() if c >= 50)
brands_idx = {b: (i + 1) for (i, b) in enumerate(brands)}

X_brand = df_train.brand_name.apply(lambda b: brands_idx.get(b, 0))
X_brand = X_brand.values.reshape(-1, 1)
brand_voc_size = len(brands) + 1

print('processing other features...')

X_item_cond = (df_train.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
X_shipping = df_train.shipping.astype('float32').values.reshape(-1, 1)

# Define the model
print('defining the model...')


def prepare_batches(seq, step):
    n = len(seq)
    res = []
    for i in range(0, n, step):
        res.append(seq[i:i + step])
    return res

@lru_cache(32768, typed=False)
def conv1d(inputs, num_filters, filter_size, padding='same'):
    he_std = np.sqrt(2 / (filter_size * num_filters))
    # filters = output size, kernel_size = number of filters
    out = tf.layers.conv1d(
        inputs=inputs, filters=num_filters, padding=padding,
        kernel_size=filter_size,
        activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(stddev=he_std))
    return out

@lru_cache(32768, typed=False)
def dense(X, size, reg=0.0, activation=None):
    he_std = np.sqrt(2 / int(X.shape[1]))
    # dimensionality of the output space.
    out = tf.layers.dense(X, units=size, activation=activation,
                          kernel_initializer=tf.random_normal_initializer(stddev=he_std),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(reg))
    return out

@lru_cache(32768, typed=False)
def embed(inputs, size, dim):
    std = np.sqrt(2 / dim)
    emb = tf.Variable(tf.random_uniform([size, dim], -std, std))
    lookup = tf.nn.embedding_lookup(emb, inputs)
    return lookup


name_embeddings_dim = 32
name_seq_len = X_name.shape[1]
desc_embeddings_dim = 32
desc_seq_len = X_desc.shape[1]

brand_embeddings_dim = 4

cat_embeddings_dim = 14
cat_seq_len = X_cat.shape[1]

graph = tf.Graph()
graph.seed = 1


with graph.as_default():
    place_name = tf.placeholder(tf.int32, shape=(None, name_seq_len))
    place_desc = tf.placeholder(tf.int32, shape=(None, desc_seq_len))
    place_brand = tf.placeholder(tf.int32, shape=(None, 1))
    place_cat = tf.placeholder(tf.int32, shape=(None, cat_seq_len))
    place_ship = tf.placeholder(tf.float32, shape=(None, 1))
    place_cond = tf.placeholder(tf.uint8, shape=(None, 1))

    place_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    place_lr = tf.placeholder(tf.float32, shape=(), )

    name = embed(place_name, name_voc_size, name_embeddings_dim)  # [batch_size x name_seq_len x name_embeddings_dim]
    desc = embed(place_desc, desc_voc_size, desc_embeddings_dim)
    brand = embed(place_brand, brand_voc_size, brand_embeddings_dim)
    cat = embed(place_cat, cat_voc_size, cat_embeddings_dim)

    name = conv1d(name, num_filters=16, filter_size=4)
    name = tf.layers.average_pooling1d(name, pool_size=name_seq_len, strides=1, padding='same')
    name = tf.layers.dropout(name, rate=0.5)
    name = tf.contrib.layers.flatten(name)
    print(name.shape)

    desc = conv1d(desc, num_filters=10, filter_size=3)
    desc = tf.layers.average_pooling1d(desc, pool_size=25, strides=1, padding='same')
    desc = tf.layers.dropout(desc, rate=0.5)
    desc = tf.contrib.layers.flatten(desc)
    print(desc.shape)

    brand = tf.contrib.layers.flatten(brand)
    print(brand.shape)

    cat = tf.layers.average_pooling1d(cat, pool_size=10, strides=1, padding='same')
    cat = tf.layers.dropout(cat, rate=0.5)
    cat = tf.contrib.layers.flatten(cat)
    print(cat.shape)

    ship = place_ship
    print(ship.shape)

    cond = tf.one_hot(place_cond, 5)
    cond = tf.contrib.layers.flatten(cond)
    print(cond.shape)

    out = tf.concat([name, desc, brand, cat, ship, cond], axis=1)
    print('concatenated dim:', out.shape)

    #out = dense(out, 256, activation=tf.nn.relu)
    #out = tf.layers.dropout(out, rate=0.5)
    out = dense(out, 128, activation=tf.nn.relu)
    out = tf.layers.dropout(out, rate=0.5)
    out = dense(out, 64, activation=tf.nn.relu)
    out = tf.layers.dropout(out, rate=0.5)
    #out = dense(out, 32, activation=tf.nn.relu)
    #out = tf.layers.dropout(out, rate=0.5)
    out = dense(out, 1)

    loss = tf.losses.mean_squared_error(place_y, out)
    rmse = tf.sqrt(loss)
    opt = tf.train.AdamOptimizer(learning_rate=place_lr)
    train_step = opt.minimize(loss)

    init = tf.global_variables_initializer()

session = tf.Session(config=session_conf, graph=graph)
session.run(init)


# Split training examples into train/dev examples.
print(X_name.shape)
X_name, X_name_dev = train_test_split(X_name, random_state=123, train_size=0.99)
print(X_name.shape, X_name_dev.shape)

# Split training examples into train/dev examples.
print(X_desc.shape)
X_desc, X_desc_dev = train_test_split(X_desc, random_state=123, train_size=0.99)
print(X_desc.shape, X_desc_dev.shape)

# Split training examples into train/dev examples.
print(X_brand.shape)
X_brand, X_brand_dev = train_test_split(X_brand, random_state=123, train_size=0.99)
print(X_brand.shape, X_brand_dev.shape)

# Split training examples into train/dev examples.

print(X_cat.shape)
X_cat, X_cat_dev = train_test_split(X_cat, random_state=123, train_size=0.99)
print(X_cat.shape, X_cat_dev.shape)

# Split training examples into train/dev examples.
print(X_item_cond.shape)
X_item_cond, X_item_cond_dev = train_test_split(X_item_cond, random_state=123, train_size=0.99)
print(X_item_cond.shape, X_item_cond_dev.shape)

# Split training examples into train/dev examples.
print(X_shipping.shape)
X_shipping, X_shipping_dev = train_test_split(X_shipping, random_state=123, train_size=0.99)
print(X_shipping.shape, X_shipping_dev.shape)

# Split training examples into train/dev examples.
print(y.shape)
y, y_dev = train_test_split(y, random_state=123, train_size=0.99)
print(y.shape, y_dev.shape)


print('training the model...')

for i in range(4):
    t0 = time.time()
    np.random.seed(i)
    train_idx_shuffle = np.arange(X_name.shape[0])
    np.random.shuffle(train_idx_shuffle)
    batches = prepare_batches(train_idx_shuffle, 500)

    if i <= 2:
        lr = 0.0025
    else:
        lr = 0.0008

    for idx in batches:
        feed_dict = {
            place_name: X_name[idx],
            place_desc: X_desc[idx],
            place_brand: X_brand[idx],
            place_cat: X_cat[idx],
            place_cond: X_item_cond[idx],
            place_ship: X_shipping[idx],
            place_y: y[idx],
            place_lr: lr,
        }
        session.run(train_step, feed_dict=feed_dict)

    took = time.time() - t0
    print('epoch %d took %.3fs' % (i, took))

# Training Step
del X_name, df_train,  X_desc,  X_brand, X_cat, X_item_cond, X_shipping; gc.collect()

n_dev = len(y_dev)
y_dev_pred = np.zeros(n_dev)

dev_idx = np.arange(n_dev)
batches = prepare_batches(dev_idx, 5000)

for idx in batches:

    feed_dict = {
        place_name: X_name_dev[idx],
        place_desc: X_desc_dev[idx],
        place_brand: X_brand_dev[idx],
        place_cat: X_cat_dev[idx],
        place_cond: X_item_cond_dev[idx],
        place_ship: X_shipping_dev[idx],
    }
    batch_dev_pred = session.run(out, feed_dict=feed_dict)
    y_dev_pred[idx] = batch_dev_pred[:, 0]

del X_name_dev,  X_desc_dev,  X_brand_dev, X_cat_dev, X_item_cond_dev, X_shipping_dev; gc.collect()
#y_pred = y_pred * std + mean
#tf_pred = np.expm1(y_pred)

y_dev_pred = y_dev_pred * std + mean
y_dev = y_dev* std + mean

y = y.reshape(-1, 1)
y_dev = y_dev.reshape(-1, 1)
y_dev_pred = y_dev_pred.reshape(-1, 1)

print("RMSL error on dev set:", rmsle(y_dev, y_dev_pred))


print('reading the test data...')

df_test = pd.read_csv('../input/test.tsv', sep='\t')
testid = df_test.test_id

df_test.name.fillna('unkname', inplace=True)
df_test.category_name.fillna('unk_cat', inplace=True)
df_test.brand_name.fillna('unk_brand', inplace=True)
df_test.item_description.fillna('nodesc', inplace=True)

df_test.category_name = df_test.category_name.apply(cat_process)
df_test.brand_name = df_test.brand_name.str.lower()
df_test.brand_name = df_test.brand_name.str.replace(' ', '_')

X_cat_test = cat_tok.transform(df_test.category_name)
X_name_test = name_tok.transform(df_test.name)

X_desc_test = desc_tok.transform(df_test.item_description)
X_desc_test = X_desc_test[:, :desc_num_col]

X_item_cond_test = (df_test.item_condition_id - 1).astype('uint8').values.reshape(-1, 1)
X_shipping_test = df_test.shipping.astype('float32').values.reshape(-1, 1)

X_brand_test = df_test.brand_name.apply(lambda b: brands_idx.get(b, 0))
X_brand_test = X_brand_test.values.reshape(-1, 1)

# Predict on the test set
print('applying the model to test...')

n_test = len(df_test)
y_pred = np.zeros(n_test)

test_idx = np.arange(n_test)
batches = prepare_batches(test_idx, 5000)

for idx in batches:

    feed_dict = {
        place_name: X_name_test[idx],
        place_desc: X_desc_test[idx],
        place_brand: X_brand_test[idx],
        place_cat: X_cat_test[idx],
        place_cond: X_item_cond_test[idx],
        place_ship: X_shipping_test[idx],
    }
    batch_pred = session.run(out, feed_dict=feed_dict)
    y_pred[idx] = batch_pred[:, 0]

y_pred = y_pred * std + mean
tf_pred = np.expm1(y_pred)
del X_name_test, df_test,  X_desc_test,  X_brand_test, X_cat_test, X_item_cond_test, X_shipping_test; gc.collect()
print('writing the results for tf')

tf_out = pd.DataFrame()
tf_out['test_id'] = testid
tf_out['price'] = tf_pred

#tf_out.to_csv('submission_tf.csv', index=False)


train_df = pd.read_table('../input/train.tsv')
test_df = pd.read_table('../input/test.tsv')
print(train_df.shape, test_df.shape)

# remove low prices
train_df = train_df.drop(train_df[(train_df.price < 1.0)].index)

# split category name into 3 parts
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
train_df['subcat_0'], train_df['subcat_1'], train_df['subcat_2'] = \
zip(*train_df['category_name'].apply(lambda x: split_cat(x)))
test_df['subcat_0'], test_df['subcat_1'], test_df['subcat_2'] = \
zip(*test_df['category_name'].apply(lambda x: split_cat(x)))

train_df.brand_name.fillna(value="missing", inplace=True)
test_df.brand_name.fillna(value="missing", inplace=True)

# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=123, train_size=0.99)

# Calculate number of train/dev/test examples.
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test_df.shape[0]
print("Training on", n_trains, "examples")
print("Validating on", n_devs, "examples")
print("Testing on", n_tests, "examples")

# Concatenate train - dev - test data for easy to handle
full_df = pd.concat([train_df, dev_df, test_df])

print("Handling missing values...")
full_df['category_name'] = full_df['category_name'].fillna('missing').astype(str)
full_df['subcat_0'] = full_df['subcat_0'].astype(str)
full_df['subcat_1'] = full_df['subcat_1'].astype(str)
full_df['subcat_2'] = full_df['subcat_2'].astype(str)
full_df['brand_name'] = full_df['brand_name'].fillna('missing').astype(str)
full_df['shipping'] = full_df['shipping'].astype(str)
full_df['item_condition_id'] = full_df['item_condition_id'].astype(str)
full_df['item_description'] = full_df['item_description'].fillna('No description yet').astype(str)

NUM_BRANDS = 4500
NUM_CATEGORIES = 1250
def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['subcat_0'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['subcat_0'].isin(pop_category1), 'subcat_0'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'
cutting(full_df)

stopwords = {x: 1 for x in stopwords.words('english')}
non_alphanums = re.compile(u'[^A-Za-z0-9]+')

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if x not in stopwords])
         

wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag,\
                                {"hash_ngrams": 3, "hash_ngrams_weights": [1.6, 0.8, 0.4],
                                "hash_size": 2 ** 28, "norm": "l2", "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze= True
X_name = wb.fit_transform(full_df['name'])
del(wb)
#X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
X_name = X_name[:, np.where(X_name.getnnz(axis=0) > 2)[0]]
#print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

wb = CountVectorizer()
X_category1 = wb.fit_transform(full_df['subcat_0'])
X_category2 = wb.fit_transform(full_df['subcat_1'])
X_category3 = wb.fit_transform(full_df['subcat_2'])
#print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

# wb= wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 3, "hash_ngrams_weights": [1.0, 1.0, 0.5],
wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag,\
                        {"hash_ngrams": 2, "hash_ngrams_weights": [2.5, 1.0],
                         "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                             "idf": None}) , procs=8)
wb.dictionary_freeze= True
X_description = wb.fit_transform(full_df['item_description'])
del(wb)
#X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
X_description = X_description[:, np.where(X_description.getnnz(axis=0) > 2)[0]]
#print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(full_df['brand_name'])
#print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

X_dummies = csr_matrix(pd.get_dummies(full_df[['item_condition_id', 'shipping']],
                                          sparse=True).values)
#print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape)
sparse_merge = hstack((X_dummies, X_description, X_brand,  X_category1, X_category2, X_category3, X_name)).tocsr()
#print('[{}] Create sparse merge completed'.format(time.time() - start_time))
del X_dummies,  X_description, X_brand,  X_category1, X_category2, X_category3, X_name; gc.collect()


print(sparse_merge.shape)
#mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
#sparse_merge = sparse_merge[:, np.where(sparse_merge.getnnz(axis=0) > 1)[0]]
X_train = sparse_merge[:n_trains]
Y_train = train_df.target.values

X_dev = sparse_merge[n_trains:n_trains+n_devs]
Y_dev = dev_df.target.values

X_test = sparse_merge[n_trains+n_devs:]

print(X_dev.shape, X_test.shape, Y_train.shape)
print(sparse_merge.shape)

mpr = sparse_merge.shape[1]

Y_train = Y_train.ravel()
Y_dev = Y_dev.ravel()

print("Fitting FM_FTRL model on training examples...")
#FM_FTRL_model = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=mpr, alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
#                    D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=4)

FM_FTRL_model = FM_FTRL(alpha=0.07, beta=0.05, L1=0.0001, L2=0.001,\
                D=mpr, alpha_fm=0.1, L2_fm=0.000, init_fm=0.08,
D_fm=100, e_noise=0.0001, iters=9, inv_link="identity", threads=4)

FM_FTRL_model.fit(X_train, Y_train)

Y_train = Y_train.reshape(-1, 1)
Y_dev = Y_dev.reshape(-1, 1)


Y_dev_preds_FM_FTRL = FM_FTRL_model.predict(X_dev)
Y_dev_preds_FM_FTRL = Y_dev_preds_FM_FTRL.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_FM_FTRL))

FM_FTRL_preds = FM_FTRL_model.predict(X_test)
FM_FTRL_preds = np.expm1(FM_FTRL_preds)

del FM_FTRL_model; gc.collect()
sparse_merge = sparse_merge[:, np.where(sparse_merge.getnnz(axis=0) > 100)[0]]
X_train = sparse_merge[:n_trains]
Y_train = train_df.target.values

X_dev = sparse_merge[n_trains:n_trains+n_devs]
Y_dev = dev_df.target.values

X_test = sparse_merge[n_trains+n_devs:]

print(X_dev.shape, X_test.shape, Y_train.shape)
print(sparse_merge.shape)
del sparse_merge; gc.collect()


print("Fitting Ridge model on training examples...")
ridge_model = Ridge(
    solver='auto', fit_intercept=True, alpha=20.0,
    max_iter=200, normalize=False, tol=0.01, random_state = 1,
)

ridge_model.fit(X_train, Y_train)

Y_train = Y_train.reshape(-1, 1)
Y_dev = Y_dev.reshape(-1, 1)

Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))

ridge_preds = ridge_model.predict(X_test)
ridge_preds = np.expm1(ridge_preds)

Y_train = Y_train.ravel()
Y_dev = Y_dev.ravel()


import lightgbm as lgb
params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'boosting' : 'dart',
        'max_drop' : 5,
        'drop_rate' : 0.01,
        'max_depth': 6,
        'num_leaves': 29,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 1,
        'bagging_freq': 5,
        'feature_fraction': 1,
        'feature_fraction_seed' : 2,
        'bagging_seed': 3,
        'seed' : 4, 
        'nthread': 4,
        'min_data_in_leaf': 15,
        #'max_bin': 40
    }

train_X, valid_X, train_y, valid_y = train_test_split(X_train, Y_train, test_size=0.08, random_state=100)
d_train = lgb.Dataset(train_X, label=train_y)
d_valid = lgb.Dataset(valid_X, label=valid_y)
watchlist = [d_train, d_valid]
lgb_model = lgb.train(params, train_set=d_train, num_boost_round=2300, valid_sets=watchlist, \
                      early_stopping_rounds=1000, verbose_eval=100)

del d_train; gc.collect()

Y_train = Y_train.reshape(-1, 1)
Y_dev = Y_dev.reshape(-1, 1)

Y_dev_preds_lgb = lgb_model.predict(X_dev)
Y_dev_preds_lgb = Y_dev_preds_lgb.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_lgb))

lgb_preds = lgb_model.predict(X_test)
lgb_preds = np.expm1(lgb_preds)


full_df["len_desc"] = full_df.item_description.apply\
            (lambda x: (len(x.split("."))-1) if x[-1]  == "." else (len(x.split("."))) )

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
print("Handling categorical variables...")
le = LabelEncoder()

#le.fit(np.hstack([train.category_name, test.category_name]))
full_df['category'] = le.fit_transform(full_df.category_name)

full_df['brand'] = le.fit_transform(full_df.brand_name)

full_df['sub_0'] = le.fit_transform(full_df.subcat_0)

full_df['sub_1'] = le.fit_transform(full_df.subcat_1)

full_df['sub_2'] = le.fit_transform(full_df.subcat_2)

#full_df = full_df_o
print("begin processing")

full_df.name = full_df.name.str.lower()
#test.name = test.name.str.lower()
full_df.item_description = full_df.item_description.str.lower()
#test.item_description = test.item_description.str.lower()


s = " ".join(full_df[:n_trains].name) # + " ".join(train.item_description)

import re
s =  re.findall(r"[\w']+", s)

s = pd.Series([x for x in s])

care = set(s.value_counts().loc[lambda x: x >=4].index)
func = lambda x: " ".join([i for i in re.findall(r"[\w']+", x) if i in care])


full_df.name = full_df.name.apply(func)
full_df.item_description = full_df.item_description.apply(func)
#test.name = test.name.apply(func)
#test.item_description = test.item_description.apply(func)
print("finish processing")

print("Text to seq process...")
print("   Fitting tokenizer...")
from keras.preprocessing.text import Tokenizer

tok_raw = Tokenizer()
tok_raw.fit_on_texts(full_df.name)
print("   Transforming text to seq...")
#train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
#test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
full_df["seq_item_description"] = tok_raw.texts_to_sequences(full_df.item_description)

full_df["seq_name"] = tok_raw.texts_to_sequences(full_df.name)

MAX_NAME_SEQ = 20 #17
MAX_ITEM_DESC_SEQ = 30 #60 #269
MAX_CATEGORY_NAME_SEQ = 20 #8
MAX_TEXT = np.max([np.max(full_df.seq_name.max())
                   , np.max(full_df.seq_item_description.max())])+10
MAX_TEXT_NAME = np.max(full_df.seq_name.max()) + 3
MAX_TEXT_DESC =  np.max(full_df.seq_item_description.max()) +3
MAX_CATEGORY = np.max(full_df.category.max()) +1
MAX_BRAND = np.max(full_df.brand.max()) +1
MAX_LEN_DESC  = np.max(full_df.len_desc.max()) + 1
MAX_SUB_1 = np.max(full_df.sub_1.max()) + 1
MAX_SUB_2 = np.max(full_df.sub_2.max()) + 1
MAX_CONDITION = 5
print('[{}] Finished EMBEDDINGS MAX VALUE...'.format(time.time() - start_time))
print('MAX_TEXT: ' + str(MAX_TEXT))
print('MAX_CATEGORY: ' + str(MAX_CATEGORY))

print('MAX_BRAND: ' + str(MAX_BRAND))
print('MAX_LEN_DESC: ' + str(MAX_LEN_DESC))
print('MAX_SUB_1: ' + str(MAX_SUB_1))
print('MAX_SUB_2: ' + str(MAX_SUB_2))

from keras.preprocessing.sequence import pad_sequences

def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description
                                    , maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand': np.array(dataset.brand)
        ,'category': np.array(dataset.category)
        #,'category_name': pad_sequences(dataset.seq_category_name
         #                               , maxlen=MAX_CATEGORY_NAME_SEQ)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'len_desc': np.array(dataset.len_desc)
        , 'general': np.array(dataset.sub_0)
        , 'sub_1': np.array(dataset.sub_1)
        , 'sub_2': np.array(dataset.sub_2)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X

X_train = get_keras_data(full_df[:n_trains])
X_valid = get_keras_data(full_df[n_trains: n_trains+n_devs])
X_test = get_keras_data(full_df[n_trains+n_devs:])

tf.set_random_seed(103)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
from keras.layers import Input, Dropout, Dense, BatchNormalization, \
    Activation, concatenate, GRU, Embedding, Flatten, AveragePooling1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
from keras import optimizers
from keras import initializers
import keras

class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        if (epoch == 0):
            K.set_value(model_RNN.optimizer.lr, 0.005)
        if (epoch == 1):
            K.set_value(model_RNN.optimizer.lr, 0.004)
        if (epoch == 2):
            K.set_value(model_RNN.optimizer.lr, 0.003)
        else:
            K.set_value(model_RNN.optimizer.lr, 0.001)
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

		
@lru_cache(32768, typed=False)
def get_model():
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    #category_name = Input(shape=[X_train["category_name"].shape[1]], 
    #                      name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    len_desc = Input(shape=[1], name="len_desc")
    general = Input(shape=[1], name="general")
    sub_1 = Input(shape=[1], name="sub_1")
    sub_2 = Input(shape=[1], name="sub_2")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    emb_size = 60
    
    emb_name = Embedding(MAX_TEXT_NAME, 20)(name)
    emb_name = AveragePooling1D(pool_size=15, strides=1, padding='valid') (emb_name)
    emb_item_desc = Embedding(MAX_TEXT, 30)(item_desc)
    emb_item_desc = AveragePooling1D(pool_size=25, strides=1, padding='valid') (emb_item_desc)
    #emb_category_name = Embedding(MAX_TEXT, emb_size//3)(category_name)
    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(7, 5)(item_condition)
    #emb_general = Embedding(15,10) (general)
    #emb_sub_1 = Embedding(MAX_SUB_1, 10) (sub_1)
    #emb_sub_2 = Embedding(MAX_SUB_2, 10) (sub_2)
    emb_len_desc = Embedding(MAX_LEN_DESC, 10) (len_desc)
    
    rnn_layer1 = GRU(13) (emb_item_desc)
    #rnn_layer2 = GRU(11) (emb_category_name)
    rnn_layer3 = GRU(12) (emb_name)
    
    
    main_l = concatenate([
        Flatten() (emb_brand)
        , Flatten() (emb_category)
        , Flatten() (emb_len_desc)
        #, Flatten() (emb_general)
        #, Flatten() (emb_sub_1)
        #, Flatten() (emb_sub_2)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        #, rnn_layer2
        , rnn_layer3
        , num_vars
    ])
    #main_l = Dropout(0.1)(BatchNormalization()(Dense(512,kernel_initializer='normal',activation='relu') (main_l)))
    #main_l = Dropout(0.1)(Dense(256,kernel_initializer='normal',activation='relu') (main_l))
    main_l = BatchNormalization()(Dense(128,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(64,kernel_initializer='normal',activation='relu') (main_l))
    #main_l = Dropout(0.1)(Dense(2,activation='relu') (main_l))
    
    #output
    output = Dense(1,activation="linear") (main_l)
    
    #model
    model = Model([name, item_desc, brand
                   , category
                   , item_condition, len_desc, general, sub_1, sub_2,  num_vars], output)
    #optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam()
    model.compile(loss="mse", 
                  optimizer=optimizer)
    return model

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
gc.collect()
np.random.seed(1337)
#FITTING THE MODEL
epochs = 3
BATCH_SIZE = 512 * 2
steps = int(len(X_train['name'])/BATCH_SIZE) * epochs
lr_init, lr_fin = 0.009, 0.0011
lr_decay = exp_decay(lr_init, lr_fin, steps)

target1 = full_df[:n_trains+n_devs].target.values
mean = target1.mean()
std = target1.std()
target1 = (target1 - mean) / std


with tf.Session() as sess:
    model_RNN = get_model()
    K.set_value(model_RNN.optimizer.lr, lr_init)
    K.set_value(model_RNN.optimizer.decay, lr_decay)

    history = model_RNN.fit(X_train, target1[:n_trains]
                        , epochs=epochs
                        , batch_size=BATCH_SIZE
                       # , validation_split=0.01
                        #, callbacks=[Histories()]
                        , verbose=2
                        )
    Y_dev_preds_rnn = model_RNN.predict(X_valid)
    Y_dev_preds_rnn = Y_dev_preds_rnn.reshape(-1, 1)
    Y_dev_preds_rnn = Y_dev_preds_rnn*std+mean
    Y_dev = Y_dev.reshape(-1, 1)
    
    rnn_preds = model_RNN.predict(X_test, batch_size = 5000)*std+mean
    rnn_preds = np.expm1(rnn_preds)
    print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_rnn))
    
Y_train = Y_train.ravel()
Y_dev = Y_dev.ravel()

blend_valid = []
blend_test = []

blend_valid.append(y_dev_pred)
#blend_valid.append(Y_dev_preds_FTRL)
blend_valid.append(Y_dev_preds_FM_FTRL)
blend_valid.append(Y_dev_preds_rnn)
blend_valid.append(Y_dev_preds_ridge)
#blend_valid.append(Y_dev_preds_huber)
blend_valid.append(Y_dev_preds_lgb)
tf_pred = tf_pred.reshape(-1, 1)
#PAR_preds = PAR_preds.reshape(-1, 1)
#FTRL_preds = FTRL_preds.reshape(-1, 1)
ridge_preds = ridge_preds.reshape(-1, 1)
#huber_preds = huber_preds.reshape(-1, 1)
FM_FTRL_preds = FM_FTRL_preds.reshape(-1, 1)
lgb_preds = lgb_preds.reshape(-1, 1)
blend_test.append(tf_pred)

blend_test.append(FM_FTRL_preds)
#blend_test.append(PAR_preds)
blend_test.append(rnn_preds)
blend_test.append(ridge_preds)
#blend_test.append(huber_preds)
blend_test.append(lgb_preds)

def create_placeholders(n_x, m):
    """
    Creates placeholdes for P and Y.
    
    Arguments:
        n_x -- size of one sample element
        m -- number of models' predictions
    Returns:
        P -- placeholder for P
        Y -- placeholder for Y
        """
    P = tf.placeholder(tf.float32, name="Preds", shape=[n_x, m])
    Y = tf.placeholder(tf.float32, name="Price", shape=[n_x, 1])
    return P, Y

def compute_cost(P, A, Y, lmbda=0.8):
    """
    Computes cost between predicted prices and actual ones.
    
    Arguments:
        P -- matrix of stacked predictions
        A -- vector of parameters
        Y -- actual prices
        lmbda -- regularazation parameter
    Returns:
        loss -- mean squared error + L1-regularization
    """
    prediction = tf.matmul(P, A) / tf.reduce_sum(A) # this is formula for weighted_predictions
    
    # Loss function + L1-regularization (You may want to try L2-regularization)
    loss = tf.reduce_mean(tf.squared_difference(prediction, Y)) + lmbda*tf.reduce_mean(tf.square(A))
    return loss

def initialize_parameters(m):
    """
    Initializes parameters A with ones.
    
    Arguments:
        m -- number of models' predictions
    Returns:
        A -- vector of parameters
    """
    A = tf.get_variable("Params", dtype=tf.float32, 
                        initializer=tf.constant(np.ones((m,1)).astype(np.float32)))
    return A

def optimize_weights(preds, actual_price, num_iterations=100):
    """
    Implements gradient descent optimizations for weighted_predictions.
    
    Arguments:
        pred -- matrix of models' predictions P
        actual_price -- actual price Y
        num_iterations -- number of iterations
    Returns:
        parameters -- vector A for weighted_predictions
    """
    np.random.seed(21)
    tf.reset_default_graph()
    
    (n_x, m) = preds.shape
    costs = []
    # create placeholders for P and Y
    P,  Y = create_placeholders(n_x, m)
    # initialize A
    A = initialize_parameters(m)
    # define loss as a function of A
    loss = compute_cost(P, A, Y)
    # Implement Gradient Descent optimization to minimize loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        # initialize global variables
        sess.run(init)
        for i in range(num_iterations):
            _ , current_cost = sess.run([optimizer, loss], feed_dict={P: preds,
                                                                      Y:actual_price})
            costs.append(current_cost)
        parameters = sess.run(A)
    return parameters
    
    
# transpose blend_train to get P
valid_preds = np.hstack(blend_valid)
actual_price = Y_dev.reshape(Y_dev.shape[0], -1)

# And finally let's find optimal weights
params = optimize_weights(valid_preds, actual_price, 613)
#params = optimize_weights(valid_preds, actual_price, 513)


Y_train = Y_train.reshape(-1, 1)
Y_dev = Y_dev.reshape(-1, 1)

Weighted_valid_pred = np.squeeze(np.dot(valid_preds, params) / np.sum(params))
Weighted_valid_pred = Weighted_valid_pred.reshape(-1, 1)
print("(Best) RMSL error for RNN + Ridge + RidgeCV on dev set:", rmsle(Y_dev, Weighted_valid_pred))

test_preds = np.hstack(blend_test)

Weighted_test_pred = np.squeeze(np.dot(test_preds, params) / np.sum(params))

sub = pd.DataFrame({'test_id': test_df.test_id,
                    'price': Weighted_test_pred})
sub.loc[sub['price'] < 0.0, 'price'] = 0.0
sub.to_csv('weighted_submission.csv', index=False)

t_min, t_sec = divmod((datetime.now() - start_real).total_seconds(), 60)
print(' It took {:.0f} minutes and {:.0f} seconds to run this notebook on kaggle'.format
              (t_min, t_sec))