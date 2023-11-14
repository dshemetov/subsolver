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
from typing import Callable, Tuple

import numpy as np
import requests
import torch
from joblib import Memory
from more_itertools import windowed
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
)

device = "cpu"
# model_id = "gpt2"
model_id = "sshleifer/tiny-gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

WHITELIST = string.ascii_letters + "'"
disk_cache = Memory(".subsolver")


@disk_cache.cache
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
        if (r := requests.get(url)).status_code == 200:
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
        for sw in windowed("* " + word + " ", 3):
            likelihood += get_neg_log_likelihood("".join(sw))

    return likelihood


@cache
def get_neg_log_likelihood(s: str) -> int:
    """Get negative log likelihood of a 3-character string.

    Uses a simple 2,3-gram model based on English text.
    """
    if "*" not in s:
        numerator_count = ENGLISH_NGRAMS.get(s, 0)
        denominator_count = ENGLISH_NGRAMS.get(s[:2], 0)
    else:
        numerator_count = sum(
            ENGLISH_NGRAMS.get(prefix + s[1:], 0) for prefix in WHITELIST
        )
        denominator_count = sum(
            ENGLISH_NGRAMS.get(prefix + " " + suffix, 0)
            for prefix, suffix in product(WHITELIST, WHITELIST)
        )

    if numerator_count == 0 or denominator_count == 0:
        return 15
    else:
        return -np.log(numerator_count / denominator_count)


def perplexity_score(s: str) -> float:
    """Get the perplexity of a string using a neural network language model."""
    encodings = tokenizer(s, return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.stack(nlls).mean()


def perplexity_score2(s: str) -> float:
    """Much simpler version of perplexity_score."""
    tokens_tensor = tokenizer.encode(s, return_tensors="pt").to(device)
    loss = model(tokens_tensor, labels=tokens_tensor).loss
    return loss.cpu().detach().numpy()


def perplexity_score3(s: str) -> float:
    """Another attempt to get perplexity.

    Random idea from this repo:
      https://github.com/samer-noureddine/GPT-2-for-Psycholinguistic-Applications/blob/master/get_probabilities.py

    Something about separating the sentence by words to help with the tokenizer.
    """
    encoding = []
    for x in s.split(" "):
        encoding.extend(tokenizer.encode(x))

    tokens_tensor = torch.tensor([encoding]).to(device)
    return model(tokens_tensor, labels=tokens_tensor).loss.cpu().detach().numpy()


def get_solved_text(
    s: str,
    n_trials: int = 5,
    n_steps: int = 2000,
    score_func: Callable[[str], float] = markov_score,
) -> Tuple[str, float]:
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
                x = choice(chars)
                y = choice(chars)
            new_s = old_s.translate({ord(x): y, ord(y): x})
            new_score = score_func(new_s)
            if new_score < old_score:
                old_s, old_score = new_s, new_score
        if new_s != s:
            best_scores += [(new_s, new_score)]

    return sorted(best_scores, key=lambda x: x[1])


def get_puzzle_text(i: int):
    if i == 0:
        s = """
        the bekenarh aitenert onslrec ot the meetaiw pf uewnorrer toue dsnnspsnotec or at gor pf the rtotlette ar edhsec ai the rlprexleit dsnnerysiceide sb thsre ghs otteicec outhslwh rdoit meitasi sddlnr ai the bsnmou ylpuadotasir sb the rsdaetf doltasi ar the banrt done sb thsre oddlrtsmec ts bode sddorasiou dhonuotoinf oic amysrtlne uewnorre bsn rsme tame ueit the amowe ts ynsberrsn gepp plt ot the uottenr ceoth at gor netlniec ts ham oic nemoair ai har ysrrerrasi ghene a kaegec at ist usiw ows at ar tnluf o tennapue thaiw oic limartovopuf ovai ts the cneom rdluytlne sb fsliw gaudsq
        """.strip()
    else:
        raise NotImplementedError
    return s


print(get_puzzle_text(0))
print(get_solved_text(get_puzzle_text(0)))
