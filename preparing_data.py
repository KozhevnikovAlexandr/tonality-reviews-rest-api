import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def load_model():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    import pathlib
    print(pathlib.Path(__file__).parent.absolute())
    with open(r'C:\Users\user\Desktop\Новая папка (6)\model.pickle', 'rb') as file:
        model = pickle.load(file)

    with open(r'C:\Users\user\Desktop\Новая папка (6)\vec.pickle', 'rb') as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


# убрать HTML разметку
def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


# убрать эмодзи
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


# напишем нашу функцию токенизации, с учетом всего рассмотренного


def mytokenize(text, stop_words, normalize=None):
    text = remove_html(text)
    text = remove_URL(text)
    text = remove_emoji(text)
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word.isalpha()]
    text = [word for word in text if not word in stop_words]
    if normalize == "s":
        porter = PorterStemmer()
        text = [porter.stem(word) for word in text]
    if normalize == "l":
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text]
    return text


def join_list(tab):
    return " ".join(tab)


def prepare_data(data):
    stop_words = set(stopwords.words("russian"))
    data['text_preprocessed'] = data['Text'].apply(mytokenize, stop_words=stop_words)
    data['text_preprocessed'] = data.text_preprocessed.apply(join_list)
    return data
