import pandas as pd
import numpy as np
import spacy
from cleantext import clean
from datasets import load_dataset

# dataset = load_dataset("imdb")
dataset = load_dataset("sst")
np.random.seed(42)


# def import_train_imdb(rows_to_pick):
#    return dataset["train"][rows_to_pick]


def import_train_sst(split):
    return dataset[split]


def import_train_yelp(nrows, begin):
    to_skip = np.arange(begin)
    df = pd.read_csv(r'D:\Falko\Documents\Uni\11. WiSe 21-22\Masterarbeit\Programmierung\yelp_polarity\train.csv',
                     nrows=nrows, header=None, delimiter=",", skiprows=to_skip)

    return df


def make_split():
    df = pd.read_csv(r'D:\Falko\Documents\Uni\11. WiSe 21-22\Masterarbeit\Programmierung\easy_dataset\easy_text.txt',
                     header=None, delimiter="\t")
    reviews = df.iloc[:, 0].tolist()
    lengths = [len(review.split()) for review in reviews]
    text_labels = df.iloc[:, 1]
    labels = np.zeros(len(text_labels))
    for i in range(len(text_labels)):
        label = text_labels.iloc[i]
        if label == "positive":
            labels[i] = 1
        else:
            labels[i] = 0
    indices = np.arange(len(labels), dtype=np.int32)
    np.random.shuffle(indices)
    train = pd.DataFrame(list(zip(df.iloc[indices[:1000], 0].tolist(), labels[indices[:1000]].tolist())))
    train.to_csv(r'D:\Falko\Documents\Uni\11. WiSe 21-22\Masterarbeit\Programmierung\easy_dataset\train.txt', sep='\t',
                 header=False, index=False)
    dev = pd.DataFrame(list(zip(df.iloc[indices[1000:1250], 0].tolist(), labels[indices[1000:1250]].tolist())))
    dev.to_csv(r'D:\Falko\Documents\Uni\11. WiSe 21-22\Masterarbeit\Programmierung\easy_dataset\dev.txt', sep='\t',
               header=False, index=False)
    test = pd.DataFrame(list(zip(df.iloc[indices[1250:], 0].tolist(), labels[indices[1250:]].tolist())))
    test.to_csv(r'D:\Falko\Documents\Uni\11. WiSe 21-22\Masterarbeit\Programmierung\easy_dataset\test.txt', sep='\t',
                header=False, index=False)


def simple_w2v(split, eps, l_max, model, d_model, random_state):
    if split == 'train':
        df = pd.read_csv(r'D:\Falko\Documents\Uni\11. WiSe 21-22\Masterarbeit\Programmierung\easy_dataset\train.txt',
                         header=None, delimiter="\t")
    elif split == 'dev':
        df = pd.read_csv(r'D:\Falko\Documents\Uni\11. WiSe 21-22\Masterarbeit\Programmierung\easy_dataset\dev.txt',
                         header=None, delimiter="\t")
    else:
        df = pd.read_csv(r'D:\Falko\Documents\Uni\11. WiSe 21-22\Masterarbeit\Programmierung\easy_dataset\test.txt',
                         header=None, delimiter="\t")
    if random_state != None:
        df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    reviews = df.iloc[:, 0].tolist()
    nrows = len(reviews)
    labels = np.array(df.iloc[:, 1], dtype=np.int8)
    labels = labels * (1 - 2 * eps) + eps
    inverse_labels = 1 - labels
    two_class_labels = np.zeros((len(labels), 2))
    two_class_labels[:, 0] = labels
    two_class_labels[:, 1] = inverse_labels

    nlp = model
    tokenized = nlp.pipe(reviews)
    data = np.zeros((nrows, l_max, d_model))
    sentence_lengths = np.zeros(nrows, dtype=np.int32)
    k = 0
    for doc in tokenized:
        sentence_lengths[k] = np.minimum(len(doc), l_max)
        vectors = [token.vector for token in doc[:sentence_lengths[k]]]
        data[k, :sentence_lengths[k], :] = np.array(vectors)[:l_max, :]
        k = k + 1
    return [np.reshape(labels, (len(labels), 1)), data, sentence_lengths, nrows]


def word_to_vec(split, l_max, model, d_model):
    # for yelp
    # df = import_train_yelp(nrows, begin)
    # labels = (df.iloc[:, 0].to_numpy(dtype=np.int8) - 1) * (1 - 2 * eps) + eps
    # reviews = df.iloc[:, 1].tolist()

    # for imdb
    # dict = import_train_imdb(rows_to_pick)
    # labels = np.array(dict['label']) * (1 - 2 * eps) + eps
    # reviews = dict['text']

    # for sst
    dict = import_train_sst(split)
    labels = np.array(dict['label']).round(decimals=0)
    reviews = dict['sentence']

    nrows = len(labels)
    # print(nrows)

    # for two classes
    # inverse_labels = 1 - labels
    # two_class_labels = np.zeros((len(labels), 2))
    # two_class_labels[:, 0] = labels
    # two_class_labels[:, 1] = inverse_labels

    reviews = [clean(review, fix_unicode=True,  # fix various unicode errors
                     to_ascii=True,  # transliterate to closest ASCII representation
                     lower=False,  # lowercase text
                     no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                     no_urls=False,  # replace all URLs with a special token
                     no_emails=False,  # replace all email addresses with a special token
                     no_phone_numbers=False,  # replace all phone numbers with a special token
                     no_numbers=False,  # replace all numbers with a special token
                     no_digits=False,  # replace all digits with a special token
                     no_currency_symbols=False,  # replace all currency symbols with a special token
                     no_punct=False,  # remove punctuations
                     replace_with_punct="",  # instead of removing punctuations you may replace them
                     replace_with_url="<URL>",
                     replace_with_email="<EMAIL>",
                     replace_with_phone_number="<PHONE>",
                     replace_with_number="<NUMBER>",
                     replace_with_digit="0",
                     replace_with_currency_symbol="<CUR>",
                     lang="en")
               for review in reviews]

    nlp = model
    tokenized = nlp.pipe(reviews)
    data = np.zeros((nrows, l_max, d_model))
    sentence_lengths = np.zeros(nrows, dtype=np.int32)
    k = 0
    for doc in tokenized:
        # print(len(doc))
        sentence_lengths[k] = np.minimum(len(doc), l_max)
        vectors = [token.vector for token in doc[:sentence_lengths[k]]]
        data[k, :sentence_lengths[k], :] = np.array(vectors)[:l_max, :]
        k = k + 1

    # print(np.max(sentence_lengths))
    return [np.reshape(labels, (len(labels), 1)), data, sentence_lengths, nrows]


def one_hot_positions(word_matrix, h, I, sentence_lengths):
    positional = np.zeros((word_matrix.shape[0], word_matrix.shape[1], word_matrix.shape[1] + word_matrix.shape[2] + 4))
    positional[:, :, :word_matrix.shape[2]] = word_matrix
    for k in range(word_matrix.shape[0]):
        # positional[k, :, :word_matrix.shape[2]] = word_matrix
        positional[k, :sentence_lengths[k], word_matrix.shape[2]] = np.ones(sentence_lengths[k])
        positional[k, :sentence_lengths[k],
        word_matrix.shape[2] + 1: word_matrix.shape[2] + sentence_lengths[k] + 1] = np.eye(sentence_lengths[k])
        positional[k, :sentence_lengths[k], word_matrix.shape[2] + word_matrix.shape[1] + 2] = np.ones(
            sentence_lengths[k])
    repeated_positional = np.tile(positional, [1, 1, h * I])
    return repeated_positional


def make_positional_sine(nrows, d_model, l_max, sentence_lengths):
    positional_matrix = np.zeros((nrows, l_max, d_model))
    x = np.linspace(0, d_model - 1, d_model)
    for k in range(nrows):
        y = np.linspace(0, sentence_lengths[k] - 1, sentence_lengths[k])
        xx, yy = np.meshgrid(x, y)
        zz = np.sin((xx % 2) * np.pi / 2 + yy / (10000 ** ((xx - (xx % 2)) / d_model)))
        positional_matrix[k, :sentence_lengths[k], :] = zz
    # n x l_max x d_model
    return positional_matrix


def make_mask(nrows, l_max, h, sentence_lengths):
    np_mask = np.zeros((nrows, h, l_max, l_max))
    for k in range(nrows):
        bound = sentence_lengths[k]
        np_mask[k, :, :bound, :bound] = np.ones((1, h, bound, bound))
        np_mask[k, :, bound:, 0] = np.ones((1, h, l_max - bound))
    return np_mask


def fetch_data_vaswani(l_max, d_model, h, model, eps, easy, split, random_state):
    if easy:
        words = simple_w2v(split, eps, l_max, model, d_model, random_state)
    else:
        words = word_to_vec(split, l_max, model, d_model)
    positions = make_positional_sine(words[3], d_model, l_max, words[2])
    mask = make_mask(words[3], l_max, h, words[2])
    return words[0], words[1] + positions, mask


def fetch_data_kohler(l_max, d, h, I, model, eps, easy, split, random_state):
    if easy:
        words = simple_w2v(split, eps, l_max, model, d, random_state)
    else:
        words = word_to_vec(split, l_max, model, d)
    positions = one_hot_positions(words[1], h, I, words[2])
    mask = make_mask(words[3], l_max, h, words[2])
    return words[0], positions, mask


if __name__ == "__main__":
    nrows = 50
    begin = 0
    indices = np.arange(25000)
    # np.random.shuffle(indices)
    rows_to_pick = indices[begin: begin + nrows]
    nlp = spacy.load('en_core_web_lg')
    d_model = 300
    oov_vector = np.random.rand(d_model)
    whitespace_vector = np.random.rand(d_model)
    result = fetch_data_vaswani(l_max=100, d_model=d_model, h=6, model=nlp, eps=0,
                                easy=False, split='train', random_state=None)
    result2 = fetch_data_kohler(l_max=6, d=d_model, h=3, I=2, model=nlp, eps=0,
                                easy=True, split='train', random_state=None)
    print("asdf")
