"""A Subtitution Cypher Solver.

Many ideas taken from Quipster (2003):
  https://people.csail.mit.edu/hasinoff/pubs/hasinoff-quipster-2003.pdf

Also playing around with neural network language models.
"""

import re
from functools import cache
from itertools import product
from pathlib import Path
from random import choice
from string import ascii_lowercase
from typing import Callable, Tuple

import numpy as np
import requests
import torch
from joblib import Memory
from more_itertools import windowed
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
)
from webdriver_manager.firefox import GeckoDriverManager

device = "cpu"
# model_id = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
# tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")


WHITELIST = ascii_lowercase + ascii_lowercase.upper() + " ?!,."
disk_cache = Memory(".subsolver")


@disk_cache.cache
def get_english_ngrams() -> str:
    nietzsche = requests.get(
        "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
    ).text
    gatsby = requests.get("https://www.gutenberg.org/cache/epub/64317/pg64317.txt").text

    # Harry Potter books: https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7
    # Replace whitespace and page markers
    pagemarker_regex = r"Page \| \d+ Harry Potter [\w\s]+? - J.K. Rowling"
    whitespace_regex = r"\s+"
    potter = " ".join(open(fname).read() for fname in Path("hpbooks").glob("*.txt"))
    potter = re.sub(pagemarker_regex + "|" + whitespace_regex, " ", potter)

    books = [nietzsche, gatsby, potter]
    s = " ".join(books)

    def custom_preprocess(s: str) -> str:
        return "".join(x.lower() for x in s if x in WHITELIST)

    cv = CountVectorizer(
        analyzer="char", ngram_range=(1, 3), preprocessor=custom_preprocess
    )
    X = cv.fit_transform([s])

    return dict(zip(cv.get_feature_names_out(), X.toarray().flatten()))


CORPUS_FREQS = get_english_ngrams()


def simple_score(s: str) -> float:
    return sum(CORPUS_FREQS.get(c, 0) for c in s)


def markov_score(s: str) -> float:
    likelihood = 0
    for word in s.split(" "):
        for sw in windowed("* " + word + " ", 3):
            likelihood += get_likelihood("".join(sw))

    return likelihood


@cache
def get_likelihood(s: str) -> int:
    if "*" not in s:
        numerator_count = CORPUS_FREQS.get(s, 0)
        denominator_count = sum(
            CORPUS_FREQS.get(s[:2] + suffix, 0) for suffix in WHITELIST
        )
    else:
        numerator_count = sum(
            CORPUS_FREQS.get(prefix + s[1:2], 0) for prefix in WHITELIST
        )
        denominator_count = sum(
            CORPUS_FREQS.get(prefix + " " + suffix, 0)
            for prefix, suffix in product(WHITELIST, WHITELIST)
        )

    if numerator_count == 0 or denominator_count == 0:
        return -15
    else:
        return np.log(numerator_count / denominator_count)


def perplexity_score(s: str, tile: bool = False) -> float:
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
        if tile:
            input_ids = torch.tile(input_ids, (2**4, 1))
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

    return -torch.exp(torch.stack(nlls).mean())


def get_solved_text(
    s: str,
    n_trials: int = 5,
    n_steps: int = 2000,
    score_func: Callable[[str], float] = markov_score,
) -> Tuple[str, float]:
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
            if new_score >= old_score:
                old_s, old_score = new_s, new_score
        if new_s != s:
            best_scores += [(new_s, new_score)]

    return sorted(best_scores, key=lambda x: x[1])


def get_puzzle_text(puzzle_num: int) -> str:
    """This function extracts the puzzle text from the Subsolver website.

    It's not a simple HTTP GET complicated because the puzzle text is rendered via
    Javascript and requires a keypress to reveal the text.
    """
    firefox_options = webdriver.FirefoxOptions()
    driver = webdriver.Firefox(
        executable_path=GeckoDriverManager().install(), options=firefox_options
    )
    puzzle_num = 1
    driver.get(f"https://www.subsolver.com/classic/{puzzle_num}")

    ActionChains(driver).key_down("f").key_down("j").perform()
    ActionChains(driver).key_up("f").key_up("j").perform()

    s = "".join(
        e.text if e.text != "" else " "
        for e in driver.find_elements(By.CLASS_NAME, "puzzle-letter")
    )

    driver.close()

    return s


s = """
fe ode usocceivriv tse mpra leboila frts tse leegeat deageut mnd obedruoi lebnudouw oil o cnye nm nhd unhitdw fe pecreye rt fnhcl pe ri tse peat ritedeat nm eyedwnie tn ateg pouk oil uniarled tse rbgcruotrnia fsrce fe pecreye tse mpra riteitrnia ode vnnl rt fnhcl pe fdniv mnd tse vnyedibeit tn mndue ha tn phrcl o pouklnnd ritn nhd gdnlhuta oil hctrbotecw fe meod tsot tsra leboil fnhcl hiledbrie tse yedw mdeelnba oil crpedtw nhd vnyedibeit ra beoit tn gdnteut
""".strip()
print(s)
print(get_solved_text(s, 5, 2500, score_func=markov_score))
# print(get_solved_text(s, 1, 600, score_func=perplexity_score))
