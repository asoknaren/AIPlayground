"""
Word2Vec Tokenization Demonstration

This program shows how word2vec tokenizes text and builds vocabularies.
We'll demonstrate:
1. Basic tokenization approaches
2. Text preprocessing
3. Word2Vec training and vocabulary
"""

import re
import string
from collections import Counter
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def basic_tokenization(text: str) -> list:
    """Simple tokenization by splitting on whitespace."""
    return text.split()


def advanced_tokenization(text: str) -> list:
    """Tokenization with preprocessing: lowercase, remove punctuation, split."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split on whitespace
    tokens = text.split()
    return tokens


def regex_tokenization(text: str) -> list:
    """Tokenization using regex patterns."""
    text = text.lower()
    # Match word characters and contractions
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
    return tokens


def nltk_tokenization(text: str) -> list:
    """Tokenization using NLTK's word_tokenize."""
    text = text.lower()
    tokens = word_tokenize(text)
    # Remove punctuation tokens
    tokens = [t for t in tokens if t.isalnum()]
    return tokens


def preprocess_sentences(sentences: list) -> list:
    """Preprocess sentences for word2vec training."""
    processed = []
    for sentence in sentences:
        # Lowercase
        sentence = sentence.lower()
        # Remove punctuation
        sentence = re.sub(r'[^\w\s]', '', sentence)
        # Tokenize
        tokens = sentence.split()
        # Filter out short tokens and stopwords (optional)
        tokens = [t for t in tokens if len(t) > 2]
        processed.append(tokens)
    return processed


def main():
    print("=" * 80)
    print("WORD2VEC TOKENIZATION DEMONSTRATION")
    print("=" * 80)

    # Sample text
    sample_text = (
        "Natural language processing is fascinating. "
        "Word2Vec models understand word relationships. "
        "Tokenization is the first step in NLP!"
    )

    print("\n1. ORIGINAL TEXT:")
    print(f"   {sample_text}\n")

    # ========== BASIC TOKENIZATION ==========
    print("-" * 80)
    print("2. BASIC TOKENIZATION (split on whitespace):")
    basic_tokens = basic_tokenization(sample_text)
    print(f"   Tokens: {basic_tokens}")
    print(f"   Token count: {len(basic_tokens)}\n")

    # ========== ADVANCED TOKENIZATION ==========
    print("-" * 80)
    print("3. ADVANCED TOKENIZATION (lowercase + remove punctuation):")
    advanced_tokens = advanced_tokenization(sample_text)
    print(f"   Tokens: {advanced_tokens}")
    print(f"   Token count: {len(advanced_tokens)}\n")

    # ========== REGEX TOKENIZATION ==========
    print("-" * 80)
    print("4. REGEX TOKENIZATION (word characters only):")
    regex_tokens = regex_tokenization(sample_text)
    print(f"   Tokens: {regex_tokens}")
    print(f"   Token count: {len(regex_tokens)}\n")

    # ========== NLTK TOKENIZATION ==========
    print("-" * 80)
    print("5. NLTK TOKENIZATION (NLTK word_tokenize):")
    nltk_tokens = nltk_tokenization(sample_text)
    print(f"   Tokens: {nltk_tokens}")
    print(f"   Token count: {len(nltk_tokens)}\n")

    # ========== VOCABULARY & FREQUENCY ==========
    print("-" * 80)
    print("6. VOCABULARY & FREQUENCY ANALYSIS:")
    token_freq = Counter(advanced_tokens)
    print("   Top 10 most frequent tokens:")
    for token, freq in token_freq.most_common(10):
        print(f"      '{token}': {freq}")
    print(f"   Total unique tokens: {len(token_freq)}\n")

    # ========== WORD2VEC TRAINING ==========
    print("-" * 80)
    print("7. WORD2VEC TOKENIZATION & TRAINING:")

    # Sample sentences for word2vec training
    sentences = [
        "Natural language processing is fascinating.",
        "Word2Vec models understand word relationships.",
        "Tokenization is the first step in NLP.",
        "Word embeddings capture semantic meaning.",
        "Machine learning uses word vectors.",
        "Deep learning processes text data.",
    ]

    print(f"   Training on {len(sentences)} sentences\n")

    # Preprocess sentences
    processed_sentences = preprocess_sentences(sentences)
    print("   Preprocessed sentences:")
    for i, sent in enumerate(processed_sentences, 1):
        print(f"      Sentence {i}: {sent}")

    # Train Word2Vec model
    print("\n   Training Word2Vec model...")
    model = Word2Vec(
        sentences=processed_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        seed=42,
    )

    print(f"   Model vocabulary size: {len(model.wv)}")
    print(f"   Vector dimensions: {model.wv.vector_size}\n")

    # ========== VOCABULARY & WORD VECTORS ==========
    print("-" * 80)
    print("8. WORD2VEC VOCABULARY:")
    print("   First 15 tokens in vocabulary:")
    for i, word in enumerate(list(model.wv.index_to_key)[:15], 1):
        print(f"      {i:2d}. '{word}'")

    print("\n   Word vector sample (first 10 dimensions of 'word'):")
    try:
        word_vector = model.wv['word']
        print(f"      {word_vector[:10]}\n")
    except KeyError:
        print("      'word' not in vocabulary\n")

    # ========== WORD SIMILARITY ==========
    print("-" * 80)
    print("9. WORD SIMILARITY (learned relationships):")
    try:
        similar_words = model.wv.most_similar('word', topn=5)
        print("   Words most similar to 'word':")
        for word, similarity in similar_words:
            print(f"      '{word}': {similarity:.4f}")
    except KeyError:
        print("   'word' not in vocabulary")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print("""
Different tokenization approaches:
1. Basic: Simple whitespace split (keeps punctuation)
2. Advanced: Lowercase + remove punctuation
3. Regex: Extract word patterns with regex
4. NLTK: Use Natural Language Toolkit

Word2Vec preprocessing typically involves:
- Converting to lowercase
- Removing punctuation
- Splitting into tokens
- Filtering short/rare words
- Building vocabulary
- Learning embeddings with context windows

The window parameter defines how many words before/after are used
for context. In this example, window=5 means 5 words left + 5 right.
""")


if __name__ == "__main__":
    main()
