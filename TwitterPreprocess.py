'''
This file helps to clean all the tweets data:
1) regex to remove symbols, punctuation
2) text canonicalization
3) lemmatization
4) other stuffs -- adhoc

PS: Examples Included

@copyright Lun Li, Shengjie Liu
'''

# Import packages
import re
import enchant
import inflect
import wordsegment
import contractions
import pandas as pd
from wordsegment import segment
from nltk.corpus import stopwords
from itertools import groupby, product
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# GLOBAL VAR
path = '/Users/lunli/Desktop/'
norm_word_file = path + "emnlp_dict" + ".txt"
train_data = path + 'tweet-sentiment-extraction/train.csv' # Kaggle train
test_data = path + 'tweet-sentiment-extraction/test.csv'   # Kaggle test
covid_train = path + 'covid_train.csv' # COVID19 train
covid_test = path + 'covid_test.csv'   # COVID19 test
BAD_WORD = "BADWORD"

# initialize inflection engine
inf_eng = inflect.engine()
wordsegment.load()
words = enchant.Dict("en")
is_known_word = words.check
stemmer = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
norm_dict = pd.read_csv(norm_word_file, header = None, sep = "\t").set_index(0)[1].to_dict()

# lower case (adhoc `=>')
def lower_case(text):
    text = re.sub('`', "'", text)
    text = contractions.fix(text)
    return text.lower()

sample = "I LOVE you"
print(lower_case(sample))

# remove all mixed-up in url
def strip_https(text):
    """ strip any url """
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?]))''', "", text)

sample = "i love this website http://shrt.st/4ce"
print(strip_https(sample))

# remove all non-enlighs words
# in addition, deal with hashtag and @user
def process_punctuations_with_exceptions(text):
    # remove *
    new_text = []
    for each in text.split():
        if each[0] == "*":
            if len(each) >= 3:
                if each[1].isalpha():
                    new_text.append(each[1:-1])
                else:
                    new_text.append(BAD_WORD)
            else:
                new_text.append(BAD_WORD)
        elif each[0] == "#":
            new_text += segment(each[1:])
        else:
            new_text.append(each)
    text = ' '.join(new_text)
    # remove @ and hashtag
    return ' '.join(re.sub("(@\_[A-Za-z0-9]+)|(@ [A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^A-Za-z0-9 \t])", "", text).split())

sample = "@ peter! *sigh*! you are son of ***. #MacyGood  Thanks for the song! YouÃ¯Â¿Â½re awesome.  I can sing along all day! "
print(process_punctuations_with_exceptions(sample))

# remove duplicates in words 
# spelling correction
def text_canonicalization(text):
    
    # two utilities
    def remove_dups(s):
        return re.sub(r'(?i)(.)\1+', r'\1', s)

    def all_duplicates(word, max_repeat=float('inf')):
        chars = [[c*i for i in range(min(len(list(dups)), max_repeat), 0, -1)]
                 for c, dups in groupby(word)]
        return map(''.join, product(*chars))
    
    # revmove duplicates in words
    output = [next((e for e in all_duplicates(s) if e and is_known_word(e)), remove_dups(s)) for s in text.split()]
    
    # spelling correction
    """ spelling correction """
    output_n = []
    for each in output:
        if each.__contains__('hah'):
            output_n.append("ha")
        else:
            if each in norm_dict.keys(): 
                output_n.append(norm_dict[each])
            else: 
                output_n.append(each)
    return ' '.join(output_n)

sample = "omg a rat ewwwwwwwwww omg i cant wait for tha marra.. so ecxcited. annabel cant go tho"
print(text_canonicalization(sample))
    
# expand contractions e.g., he's => he is
# lemmatization
# optional: remove stop words
def normalize_text(text, remove_stop = False):
    lem = [lemmatizer.lemmatize(word, pos='v') for word in text.split()]
    stop_vec = lem
    if remove_stop:
        stop_vec = []
        for word in lem:
            if word not in stopwords.words('english'):
                stop_vec.append(word)
    return ' '.join(stop_vec)

sample = "I'd like biking on the road"
print(normalize_text(sample))

def clean_tweet(text):
    text = lower_case(text)
    text = strip_https(text)
    text = process_punctuations_with_exceptions(text)
    text = text_canonicalization(text)
    text = normalize_text(text)
    return text
    
samples = []
samples.append("i love this website http://shrt.st/4ce")
samples.append("@ peter! *sigh*! you are son of ***. #MacyGood  Thanks for the song! YouÃ¯Â¿Â½re awesome.  I can sing along all day! ")
samples.append("omg a rat ewwwwwwwwww omg i cant wait for tha marra.. so ecxcited. annabel cant go tho")
samples.append("I'd biking on the road")
for each in samples:
    print(clean_tweet(each))

# Clean kaggle training data
training = pd.read_csv(train_data)
training['text'] = training['text'].apply(lambda x: clean_tweet(str(x)))
training['selected_text'] = training['selected_text'].apply(lambda x: clean_tweet(str(x)))

# Drop row with only one letter
df_subset = pd.DataFrame(columns=['text'])
df_subset['text']= training['text'].apply(lambda x:' '.join([w for w in x.split() if len(w)>1 or w=='i']))
print(df_subset)