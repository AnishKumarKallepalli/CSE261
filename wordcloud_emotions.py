"""
Generate one word cloud per dataset (Reddit and Twitter) using only
emotion-indicative words (lexicon filter). Uses cached data (run data_load.py
or run_all.py first). Saves outputs/figures/wordclouds/wordcloud_reddit.png and
wordcloud_twitter.png.
"""

import os
import re

import config
from data_load import get_cached

try:
    from wordcloud import WordCloud
except ImportError:
    print("Install wordcloud: pip install wordcloud")
    raise

# Emotion lexicon: words that signal one of our 6 emotions. Embedded so it works offline.
_LEX_BY_EMOTION = {
    "sadness": (
        "sad saddened sadness grieving grief sorrow sorrowful mourn miserable misery unhappy "
        "depressed depression despair hopeless helpless lonely loneliness tear tears crying cry weep "
        "heartbroken broken disappointed disappointment regret remorse guilt melancholy gloom dismal "
        "downcast dejected wretched"
    ).split(),
    "joy": (
        "happy happiness joy joyful joyous glad delight delighted cheerful cheer excitement excited "
        "pleasure pleased amazing wonderful great fantastic awesome fun funny laugh laughing "
        "smile smiling celebrate celebration grateful gratitude proud pride hopeful hope blessed thankful"
    ).split(),
    "love": (
        "love loving beloved adore adoring affection caring compassionate compassion kindness "
        "sweet heart romantic romance devoted devotion trust trusting faithful friendship friend "
        "cherish precious dear beautiful grateful"
    ).split(),
    "anger": (
        "angry anger mad furious rage enraged hate hatred hateful annoyed annoyance frustrated "
        "frustration irritated irritation outrage outraged resent bitter bitterness disgust "
        "disgusted hostile aggression aggressive"
    ).split(),
    "fear": (
        "fear fearful afraid scared scare scary terrified terror anxiety anxious worry worried "
        "panic panicked nervous dread horrified horror frightened frightening shock shocked "
        "alarm alarmed uneasy"
    ).split(),
    "surprise": (
        "surprise surprised surprising shock shocked amazing wow unexpected suddenly "
        "astonished astonishment stunned unbelievable incredible"
    ).split(),
}
EMOTION_WORDS = set()
for words in _LEX_BY_EMOTION.values():
    EMOTION_WORDS.update(w.lower() for w in words)
EMOTION_WORDS.update({
    "loved", "loves", "relieved", "relief", "hopeless", "hopelessness", "devastated",
    "miss", "missing", "laughed", "awful", "horrible", "happiness", "sadness",
})

# Simple English stopwords (subset; extend as needed)
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "don't", "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
    "ain", "aren", "aren't", "couldn", "couldn't", "didn", "didn't", "doesn",
    "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn",
    "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't",
    "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't", "rt", "amp", "gt", "lt",
}

MAX_WORDS = 200
FIGURE_DPI = 120


def _tokenize_for_cloud(text):
    """Lowercase, keep letters only, split; filter short and stopwords."""
    text = (text or "").lower()
    words = re.findall(r"[a-z']+", text)
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def _all_texts(data_dict):
    """Aggregate all splits into one list of texts (whole dataset)."""
    texts = []
    for v in data_dict.values():
        texts.extend(t for t in v["texts"] if isinstance(t, str) and t.strip())
    return texts


def _make_wordcloud(texts, title, out_path, max_words=MAX_WORDS, emotion_only=True):
    """Build one word cloud from a list of texts and save to out_path. If emotion_only, keep only lexicon words."""
    combined = " ".join(t for t in texts if isinstance(t, str) and t.strip())
    words = _tokenize_for_cloud(combined)
    if not words:
        print("  (no words for", title + ")")
        return
    word_freq = {}
    for w in words:
        if emotion_only and w not in EMOTION_WORDS:
            continue
        word_freq[w] = word_freq.get(w, 0) + 1
    if not word_freq:
        print("  (no emotion-lexicon words for", title + ")")
        return
    wc = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(word_freq)
    fig = wc.to_image()
    # Optional: add title via PIL if you want; here we just save the cloud
    fig.save(out_path, dpi=(FIGURE_DPI, FIGURE_DPI))
    print("  Saved:", out_path)


def main():
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    wordcloud_dir = os.path.join(config.FIGURES_DIR, "wordclouds")
    os.makedirs(wordcloud_dir, exist_ok=True)

    print("Loading cached data...")
    reddit, twitter = get_cached()

    print("Reddit: one word cloud for full dataset")
    reddit_texts = _all_texts(reddit)
    _make_wordcloud(reddit_texts, "Reddit", os.path.join(wordcloud_dir, "wordcloud_reddit.png"))

    print("Twitter: one word cloud for full dataset")
    twitter_texts = _all_texts(twitter)
    _make_wordcloud(twitter_texts, "Twitter", os.path.join(wordcloud_dir, "wordcloud_twitter.png"))

    print("Done. Word clouds in:", wordcloud_dir)


if __name__ == "__main__":
    main()
