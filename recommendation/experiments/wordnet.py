import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")
nltk.download("omw-1.4")

def get_hyponyms(synset):
    hyponyms = set()
    for syn in synset.closure(lambda s: s.hyponyms()):
        for lemma in syn.lemmas():
            word = lemma.name().replace("_", " ")
            if lemma.count() >= 2: 
                print(f"{word} ({lemma.count()})")
                hyponyms.add(word)
    return hyponyms

animals_subtopics = {
    "general": { 
        "savana": 0.3,
        "forest": 0.3,
        "desert": 0.3,
        "ocean": 0.3,
        "polar": 0.3,
        "grassland": 0.3,
        "wetlands": 0.3,
    },
    "classification": {},  
    "detailed": {}  
}

classification_synsets = {
    "mammals": wn.synset("mammal.n.01"),
    "reptiles": wn.synset("reptile.n.01"),
    "birds": wn.synset("bird.n.01"),
    "fish": wn.synset("fish.n.01"),
    "amphibians": wn.synset("amphibian.n.01"),
    "insects": wn.synset("insect.n.01"),
}

for category, synset in classification_synsets.items():
    animals_subtopics["classification"][category] = 0.5

for category, synset in classification_synsets.items():
    detailed_animals = get_hyponyms(synset)
    for animal in detailed_animals:
        animals_subtopics["detailed"][animal] = 0.8  # Higher weight for specificity

import json
with open("animals_subtopics.json", "w", encoding="utf-8") as f:
    json.dump(animals_subtopics, f, indent=2, ensure_ascii=False)
