"""A Subtitution Cypher Solver.

Playing around with neural network language models. Many ideas taken from
Quipster (2003):

    https://people.csail.mit.edu/hasinoff/pubs/hasinoff-quipster-2003.pdf

A collab version of this is at:

    https://colab.research.google.com/drive/1uDn1KVQkpxw4LLQ2UfzLDLBmXu2Ui4eh#scrollTo=xmHLz6760sm7
"""

import string
from functools import cache
from itertools import product
from random import choice
from typing import Callable

import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

WHITELIST = string.ascii_letters + "'"


def get_english_ngrams() -> str:
    """Get a dictionary of English 1,2,3-grams."""
    urls = [
        # Nietzsche
        "https://s3.amazonaws.com/text-datasets/nietzsche.txt",
        # The Great Gatsby
        "https://www.gutenberg.org/cache/epub/64317/pg64317.txt",
    ]
    s = []
    for url in urls:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        s.append(r.text)
    s = " ".join(s)

    cv = CountVectorizer(analyzer="char", ngram_range=(1, 3), strip_accents="unicode")
    X = cv.fit_transform([s])

    return dict(zip(cv.get_feature_names_out(), X.toarray().flatten()))


ENGLISH_NGRAMS = get_english_ngrams()


def markov_score(s: str) -> float:
    """Get the negative log likelihood of a string using a Markov model.

    Split into words and then use a 3-gram model on each word.
    """
    likelihood = 0
    for word in s.split(" "):
        affixed_word = "* " + word + " "
        for i in range(len(word) - 2):
            likelihood += get_neg_log_likelihood(affixed_word[i : i + 3])

    return likelihood


@cache
def get_neg_log_likelihood(s: str) -> float:
    """Get negative log likelihood of a 3-character string.

    Uses a simple 2,3-gram model based on English text.
    """
    if "*" not in s:
        numerator_count = ENGLISH_NGRAMS.get(s, 0)
        denominator_count = ENGLISH_NGRAMS.get(s[:2], 0)
    else:
        numerator_count = sum(ENGLISH_NGRAMS.get(prefix + s[1:], 0) for prefix in WHITELIST)
        denominator_count = sum(
            ENGLISH_NGRAMS.get(prefix + " " + suffix, 0) for prefix, suffix in product(WHITELIST, WHITELIST)
        )

    if numerator_count == 0 or denominator_count == 0:
        return 15.0
    else:
        return -np.log(numerator_count / denominator_count)


def get_solved_text(
    s: str,
    n_trials: int = 5,
    n_steps: int = 2000,
    score_func: Callable[[str], float] = markov_score,
) -> str:
    """Get the best solved text.

    Makes random swaps and keeps the ones that reduce the negative log
    likelihood.
    """
    chars = list(set(s) - {" "})
    best_scores = []
    for _ in range(n_trials):
        old_s, old_score = s, score_func(s)
        for _ in tqdm(range(n_steps)):
            x = y = None
            while x == y:
                x, y = choice(chars), choice(chars)
            new_s = old_s.translate({ord(x): y, ord(y): x})
            new_score = score_func(new_s)
            if new_score < old_score:
                old_s, old_score = new_s, new_score
        if new_s != s:
            best_scores += [(new_s, new_score)]

    return min(best_scores, key=lambda x: x[1])[0]


with open("puzzles.txt") as f:
    puzzles = f.readlines()


if __name__ == "__main__":
    with open("solutions.txt", "w") as g:
        for p in puzzles:
            s, _ = get_solved_text(p.strip())
            g.write(s + "\n")
