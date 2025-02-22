from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import json

# Load JSON file
def load_json(file, default={}):
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        data = default
    return data

# Recursively get all hyponyms of a synset
def get_all_hyponyms(synset):
    hyponyms = set()
    for hyponym in synset.hyponyms():
        hyponyms.add(hyponym)
        hyponyms.update(get_all_hyponyms(hyponym))  
    return hyponyms

def is_animal_related(word, synset_keyword):
    lemmatizer = WordNetLemmatizer()
    word = lemmatizer.lemmatize(word.lower()) 

    synsets = wn.synsets(synset_keyword)

    for synset in synsets:
        if word in [lemma.name().lower() for lemma in synset.lemmas()]:
            return True

        # Get all hyponyms of the synset recursively
        all_hyponyms = get_all_hyponyms(synset)
        for hyponym in all_hyponyms:
            if word in [lemma.name().lower() for lemma in hyponym.lemmas()]:
                return True

    return False

all_topics_file = "../../data_analysis/all_topics.json"
all_topics = load_json(all_topics_file)
synset_keyword = "animal"

cluster_keywords = list()
for topic in all_topics:
    if is_animal_related(topic, synset_keyword):
        cluster_keywords.append(topic)

print(cluster_keywords)
print(len(cluster_keywords))