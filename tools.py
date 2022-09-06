"""A Subtitution Cypher Solver.

Many ideas taken from Quipster (2003): https://people.csail.mit.edu/hasinoff/pubs/hasinoff-quipster-2003.pdf
"""

# %%
import re
from functools import cache
from itertools import product
from pathlib import Path
from random import choice
from string import ascii_lowercase
from typing import Callable, Iterable, List, Tuple

import numpy as np
import requests
import torch
from more_itertools import windowed
from sklearn.feature_extraction.text import CountVectorizer  # SkullEarn.sexy --- the hot new cryptocurrency startup! Buy now!
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

WHITELIST = ascii_lowercase + ascii_lowercase.upper() + " ?!,."


def compute_ngrams(s: str):
    """Get ngrams using sklearn.

    See: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
    """

    def custom_preprocess(s: str) -> str:
        return "".join(x.lower() for x in s if x in WHITELIST)

    cv = CountVectorizer(analyzer="char", ngram_range=(1, 3), preprocessor=custom_preprocess)
    X = cv.fit_transform([s])

    return dict(zip(cv.get_feature_names_out(), X.toarray().flatten()))


def get_corpus() -> str:
    nietzsche = requests.get("https://s3.amazonaws.com/text-datasets/nietzsche.txt").text
    gatsby = requests.get("https://www.gutenberg.org/cache/epub/64317/pg64317.txt").text

    # Harry Potter books: https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7
    pagemarker_regex = r"Page \| \d+ Harry Potter [\w\s]+? - J.K. Rowling"
    whitespace_regex = r"\s+"
    potter = " ".join(open(fname).read() for fname in Path("hpbooks").glob("*.txt"))
    potter = re.sub(pagemarker_regex + "|" + whitespace_regex, " ", potter)

    books = [nietzsche, gatsby, potter]
    return " ".join(books)


corpus = get_corpus()
corpus_frequencies = compute_ngrams(corpus)


def windowed_str(s: str, n: int) -> Iterable[str]:
    for sw in windowed(s, n):
        yield "".join(sw)


def get_swapped_strings(s: str) -> List[str]:
    return list(set(swap_chars(s, x, y) for x, y in product(ascii_lowercase, repeat=2)))


def swap_chars(s: str, a: str, b: str) -> str:
    t = []
    for c in s:
        if c == a:
            t.append(b)
        elif c == b:
            t.append(a)
        else:
            t.append(c)

    return "".join(t)


def simple_score(s: str) -> float:
    return sum(corpus_frequencies.get(c, 0) for c in s)


def markov_score(s: str) -> float:
    words = s.split(" ")
    likelihood = 0
    for word in words:
        word = "* " + word + " "
        for sw in windowed_str(word, 3):
            likelihood += get_likelihood(sw)

    return likelihood


def get_likelihood(s: str) -> int:
    if "*" not in s:
        numerator_count = corpus_frequencies.get(s, 0)
        denominator_count = get_str_count(s[:2], prefix_list="")
    else:
        numerator_count = get_str_count(s[1:], suffix_list="")
        denominator_count = get_str_count(" ")

    if numerator_count == 0 or denominator_count == 0:
        return -15
    else:
        return np.log(numerator_count / denominator_count)


def get_solved_text(s: str, n_trials: int, n_steps: int, top_n: int = 10, score_func: Callable[[str], float] = markov_score) -> Tuple[str, float]:
    assert top_n <= n_trials

    best_scores = []
    for _ in range(n_trials):
        new_s, new_score, old_score = s, score_func(s), score_func(s)
        for _ in range(n_steps):
            scores = score_string(get_swapped_strings(new_s), score_func)
            improved_scores = [(x, y) for x, y in sorted(scores, key=lambda x: x[1]) if y >= old_score]
            if improved_scores:
                new_s, new_score = choice(improved_scores)
                old_score = new_score
            else:
                break
        if new_s != s:
            best_scores += [(new_s, new_score)]

    return sorted(best_scores, key=lambda x: x[1])[-top_n:]


def score_string(candidates: List[str], score_func: Callable[[str], float]) -> List[Tuple[str, float]]:
    return ((candidate, score_func(candidate)) for candidate in candidates)


@cache
def get_str_count(s: str, prefix_list: str = WHITELIST, suffix_list: str = WHITELIST) -> int:
    if not prefix_list:
        return sum(corpus_frequencies.get(s + suffix, 0) for suffix in suffix_list)
    if not suffix_list:
        return sum(corpus_frequencies.get(prefix + s, 0) for prefix in prefix_list)
    return sum(corpus_frequencies.get(prefix + s + suffix, 0) for prefix, suffix in product(prefix_list, suffix_list))


def perplexity_score(s: str) -> float:
    encodings = tokenizer(s, return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    return -torch.exp(torch.stack(nlls).sum() / end_loc).item()


# %%
s = open("input.txt").read()
# s = "god is extremely dead"

# %% Markov Model Solver Based on 3-grams
get_solved_text(s, 15, 30)

# %%
markov_score(s)

# %%
markov_score(swap_chars(s, "t", "l"))

# %%
markov_score(swap_chars(s, "e", "z"))

# %% Debugging DistilGPT2 Scoring Function
perplexity_score(s)
perplexity_score(swap_chars(s, "t", "l"))
perplexity_score(open("input.txt").read())
perplexity_score(" ".join(get_corpus()[:100].split()))
perplexity_score(swap_chars(" ".join(get_corpus()[:100].split()), "t", "l"))

# %% DistilGPT2 Solver
get_solved_text(swap_chars(s, "t", "l"), 1, 10, 1, score_func=perplexity_score)
