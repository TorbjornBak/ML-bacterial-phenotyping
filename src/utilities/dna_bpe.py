from collections import Counter
from typing import Dict, List, Tuple, Iterable

def kmerize(seq: str, k: int = 6) -> List[str]:
    # non-overlapping k-mers (keeps last shorter chunk, same as your version)
    return [seq[i:i + k] for i in range(0, len(seq), k)]

def get_word_freqs(sequences: Iterable[str], k: int = 6) -> Dict[str, int]:
    freqs = Counter()
    for seq in sequences:
        for tok in kmerize(str(seq), k=k):
            freqs[tok] += 1
    return dict(freqs)

def get_alphabet(word_freqs: Dict[str, int]) -> List[str]:
    # unique characters from all tokens
    letters = {c for w in word_freqs for c in w}
    return sorted(letters)

def _init_splits(word_freqs: Dict[str, int]) -> Dict[str, List[str]]:
    # per-token list of chars
    return {w: list(w) for w in word_freqs}

def compute_pair_freqs(
    splits: Dict[str, List[str]],
    word_freqs: Dict[str, int],
) -> Counter:
    # count adjacent char-pair frequencies weighted by token frequency
    pair_freqs = Counter()
    for w, freq in word_freqs.items():
        s = splits[w]
        if len(s) < 2 or freq == 0:
            continue
        for a, b in zip(s, s[1:]):
            pair_freqs[(a, b)] += freq
    return pair_freqs

def merge_pair(
    a: str,
    b: str,
    splits: Dict[str, List[str]],
    word_freqs: Dict[str, int],
) -> None:
    # in-place merge of best pair across all splits
    merged = a + b
    for w in word_freqs:
        s = splits[w]
        if len(s) < 2:
            continue
        out = []
        i = 0
        while i < len(s):
            if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(s[i])
                i += 1
        splits[w] = out

def train_tokenizer(
    corpus: Iterable[str],
    vocab_size: int = 100,
    k: int = 6,
):
    # learn merges until reaching vocab_size or no pairs left
    word_freqs = get_word_freqs(corpus, k=k)
    vocab = get_alphabet(word_freqs)
    splits = _init_splits(word_freqs)
    merges: Dict[Tuple[str, str], str] = {}  # insertion order preserved

    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits, word_freqs)
        if not pair_freqs:
            break
        (best_a, best_b), _ = pair_freqs.most_common(1)[0]
        merged_token = best_a + best_b
        merges[(best_a, best_b)] = merged_token
        vocab.append(merged_token)
        merge_pair(best_a, best_b, splits, word_freqs)

    return merges, vocab  # maps (a,b) -> "ab"

def encode_tokens(
    text: str,
    merges: Dict[Tuple[str, str], str],
    k: int = 6,
) -> List[str]:
    """Return BPE tokens (strings), no id mapping."""
    tokens = kmerize(text, k=k)
    splits = [list(w) for w in tokens]
    for (a, b), merged in merges.items():
        for idx, s in enumerate(splits):
            if len(s) < 2:
                continue
            out = []
            i = 0
            while i < len(s):
                if i < len(s) - 1 and s[i] == a and s[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            splits[idx] = out
    out_tokens: List[str] = []
    for s in splits:
        out_tokens.extend(s)
    return out_tokens


def build_vocab_from_merges(
    corpus: Iterable[str],
    merges: Dict[Tuple[str, str], str],
    k: int = 6,
    add_specials: bool = True,
):
    """Create a vocab from what encode_tokens actually emits on the training corpus."""
    seen = set()
    for seq in corpus:
        seen.update(encode_tokens(seq, merges, k=k))
    vocab_list = sorted(seen)
    if add_specials:
        vocab_list = ["[PAD]", "[UNK]"] + vocab_list
    vocab_dict = {tok: i for i, tok in enumerate(vocab_list)}
    pad_id = vocab_dict.get("[PAD]", 0)
    unk_id = vocab_dict.get("[UNK]", 0)
    return vocab_list, vocab_dict, pad_id, unk_id

def build_vocab(vocab: List[str], add_specials: bool = True):
    """Fallback: build from training-time list; may miss inference tokens."""
    specials = ["[PAD]", "[UNK]"] if add_specials else []
    vocab_list = specials + list(vocab)
    vocab_dict = {tok: i for i, tok in enumerate(vocab_list)}
    pad_id = vocab_dict.get("[PAD]", 0)
    unk_id = vocab_dict.get("[UNK]", 0)
    return vocab_list, vocab_dict, pad_id, unk_id

def tokenize(
    id: str,
    text: str,
    merges: Dict[Tuple[str, str], str],
    vocab_dict: dict,
    k: int = 6,
) -> List[str]:
    # OOV-safe id mapping
    toks = encode_tokens(text, merges, k=k)
    unk_id = vocab_dict.get("[UNK]", 0)
    ids = [vocab_dict.get(t, unk_id) for t in toks]
    return {id: ids}