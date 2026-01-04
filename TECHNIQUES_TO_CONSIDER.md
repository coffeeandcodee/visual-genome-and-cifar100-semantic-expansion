# NLP Techniques to Consider

## 🔤 Text Preprocessing

### Lemmatization
Currently "apple" and "apples", "running" and "run" are treated as different words. Lemmatization would normalize them:

```python
import spacy
nlp = spacy.load("en_core_web_sm")
def lemmatize(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc]
```

Could **significantly reduce vocabulary size** and give denser co-occurrences.

### Stop Word Removal
Words like "the", "a", "in", "of" dominate the network. Consider filtering them:

```python
from spacy.lang.en.stop_words import STOP_WORDS
```

---

## 📊 Better Co-occurrence Modeling

### Larger Context Windows
Currently only capturing **bigrams** (consecutive pairs). Skip-gram typically uses windows of 5-10:

```python
# Instead of just (tokens[i], tokens[i+1])
# Capture (tokens[i], tokens[i+k]) for k in range(-5, 6) if k != 0
```

### TF-IDF Weighting
Downweight common pairs:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
```

### PMI (Pointwise Mutual Information)
Captures which co-occurrences are **surprising** rather than just frequent:

```
PMI(a,b) = log( P(a,b) / (P(a) * P(b)) )
```

---

## 🌳 Semantic Knowledge

### WordNet Hierarchies
CIFAR-100 has a superclass structure. WordNet gives this for free:

```python
from nltk.corpus import wordnet as wn
apple = wn.synsets('apple')[0]
print(apple.hypernyms())  # ['fruit.n.01']
```

Could add edges based on semantic relationships, not just co-occurrence.

### ConceptNet
Knowledge graph with common-sense relationships:
- "apple" → RelatedTo → "fruit", "red", "tree", "pie"

---

## 📝 Data Augmentation

### Synonym Replacement
Augment descriptions by swapping words with synonyms:
```
"a red apple on the table" → "a crimson apple on the desk"
```

### Back-Translation
Translate to another language and back for paraphrasing.

---

## 🎯 Quick Wins Summary

| Technique | Difficulty | Impact |
|-----------|-----------|--------|
| **Lemmatization** | Easy | Medium |
| **Larger context window (5-10)** | Easy | Medium |
| **Stop word filtering** | Easy | Medium |
| **PMI instead of raw counts** | Medium | Medium |
| **WordNet hierarchy edges** | Medium | High |
| **Synonym replacement augmentation** | Easy | Medium |
