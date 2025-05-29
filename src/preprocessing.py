import logging
import re

import nltk
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer

logger = logging.getLogger(__name__)

# Downlod WordNet if needed
try:
    nltk.data.find("corpora/wordnet")
    logger.info("WordNet is already downloaded.")
except LookupError:
    logger.info("WordNet not found. Downloading...")
    nltk.download("wordnet")

REGEX_RULES = {
    "clean_whitespaces": (re.compile(r"([\s&&[^ ]]|\s{2,})"), " "),

    "internet/replace_urls": (re.compile(r"https?://\S+|www\.\S+"), "=url="),
    "internet/replace_emails": (re.compile(r"\w+@\w+\.\w+"), "=email="),
    "internet/repalce_usernames": (re.compile(r"@\w+"), "=username="),

    "punctuation/clean_!!": (re.compile(r"!{3,}"), "!!"),
    "punctuation/clean_??": (re.compile(r"\?{3,}"), "??"),
    "punctuation/clean_?!": (re.compile(r"\?+[\?!]+"), "?!"),
    "punctuation/clean_!?": (re.compile(r"!+[\?!]+"), "!?"),

    "contractions/replace_'m": (re.compile(r"(\w+)'m"), "\1 am"),
    "contractions/replace_'re": (re.compile(r"(\w+)'re"), "\1 are"),
    "contractions/replace_'s": (re.compile(r"(\w+)'s"), "\1 is"),
    "contractions/replace_'ve": (re.compile(r"(\w+)'ve"), "\1 have"),
    "contractions/replace_'ll": (re.compile(r"(\w+)'ll"), "\1 will"),
    "contractions/replace_'d": (re.compile(r"(\w+)'d"), "\1 would"),
    "contractions/replace_'t": (re.compile(r"(\w+)'t"), "\1 not"),

    "remove_repeated_chars": (re.compile(r"(.)\1{4,}"), "\1\1\1"),
    "remove_special_chars": (re.compile(r"[^a-zA-Z0-9\!\?= ]"), ""),

    # "@ to at": (re.compile(r" \@ "), " at "),
    # "2 to to": (re.compile(r" 2 "), " to "),

    #"long_numbers": (re.compile(r"[1-9]\d{2,}"), "99"),
    #"single chars": (re.compile(r" [^ai] "), " "),

    # "formatting *": (re.compile(r"\*+([^\* ]+)\*+"), "\1"),
    # "formatting _": (re.compile(r"_+([^\* ]+)_+"), "\1"),

    # "not": (re.compile(r"not (\w+)"), "not_\1"),
}  # fmt: off

ALL_RULES = [
    "clean_whitespaces",

    "internet",
    "internet/replace_urls",
    "internet/replace_emails",
    "internet/repalce_usernames",

    "punctuation",
    "punctuation/clean_!!",
    "punctuation/clean_??",
    "punctuation/clean_?!",
    "punctuation/clean_!?",

    "contractions",
    "contractions/replace_'m",
    "contractions/replace_'re",
    "contractions/replace_'s",
    "contractions/replace_'ve",
    "contractions/replace_'ll",
    "contractions/replace_'d",
    "contractions/replace_'t",

    "remove_repeated_chars",
    "remove_special_chars",
    "remove_stopwords",

    "stem",
    "lemmatize",

    "lowercase",
]  # fmt: off

# collection of common stopwords
STOPWORDS = {
    "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",

    "what", "which", "who", "whom", "when", "where", "why", "how",
    "this", "that", "these", "those",

    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",

    "a", "an", "the",
    "and", "or", "if", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "on", "over", "further",
    "then", "once", "here", "there", "both", "each", "more", "most", "some", "such", "own",
    "so", "can", "will", "just", "don", "should", "now",
}  # fmt: off


def preprocess_text(
    text: str,
    active_rules: set[str],
    stats: dict[str, int] = None,
):
    # apply regex rules
    new_stats = set()
    for name, rule in REGEX_RULES.items():
        # get rule prefixes
        rule_prefixes = name.split("/")
        for i in range(1, len(rule_prefixes)):
            rule_prefixes[i] = f"{rule_prefixes[i-1]}/{rule_prefixes[i]}"

        pattern, replacement = rule

        # save stats if requested
        if stats is not None and pattern.search(text):
            for prefix in rule_prefixes:
                new_stats.add(prefix)

        # check if rule is active
        if not any(prefix in active_rules for prefix in rule_prefixes):
            logger.info(f"Skipping inactive rule: {name}")
            continue

        # apply regex rule
        text = pattern.sub(replacement, text)

    if stats is not None:
        for prefix in new_stats:
            stats[prefix] = stats.get(prefix, 0) + 1

    # make lowercase
    if "lowercase" in active_rules:
        if stats is not None and text != text.lower():
            stats["lowercase"] = stats.get("lowercase", 0) + 1
        text = text.lower()

    # word stemming or lemmatization
    if "stem" in active_rules and "lemmatize" in active_rules:
        raise ValueError("Cannot use both stemming and lemmatization at the same time.")
    elif "stem" in active_rules: # stem
        stemmer = PorterStemmer()
        text_stemmed = " ".join(stemmer.stem(word) for word in text.split())
        if stats is not None and text != text_stemmed:
            stats["stem"] = stats.get("stem", 0) + 1
        text = text_stemmed
    elif "lemmatize" in active_rules: # lemmatize
        lemmatizer = WordNetLemmatizer()
        text_lemmatized = " ".join(lemmatizer.lemmatize(word) for word in text.split())
        if stats is not None and text != text_lemmatized:
            stats["lemmatize"] = stats.get("lemmatize", 0) + 1
        text = text_lemmatized

    # remove stopwords
    if "remove_stopwords" in active_rules:
        text_without_stopwords = " ".join(word for word in text.split() if word.lower() not in STOPWORDS)
        if stats is not None and text != text_without_stopwords:
            stats["remove_stopwords"] = stats.get("remove_stopwords", 0) + 1
        text = text_without_stopwords

    return text


def apply_preprocessing(
    sentences: pd.Series,
    active_rules: set[str],
    print_stats: bool = False,
):
    # check if active_rules only contains valid rules
    if not active_rules.issubset(ALL_RULES):
        raise ValueError(f"Invalid preprocessing rules: {active_rules - set(ALL_RULES)}")

    # copy the sentences to avoid modifying the original
    sentences = sentences.copy()

    stats = {}

    # apply preprocessing rules on each sentence
    sentences = sentences.apply(lambda x: preprocess_text(x, active_rules=active_rules, stats=stats))

    if print_stats:
        for name in sorted(stats.keys()):
            if stat:=stats[name]:
                logger.info(f"Preprocessing rule '{name}' applied {stat} ({100*stat/len(sentences):.02f}\%).")

    logger.info(f"Applied preprocessing rules: {active_rules}")
    return sentences
