import numpy as np
import pandas as pd
from collections import defaultdict

"""===== Data Loading & Preprocessing ====="""

# Load data, skip lines with errors
ner = pd.read_csv("ner.csv", encoding="latin-1", on_bad_lines="skip")
ner_ds = pd.read_csv("ner_dataset.csv", encoding="latin-1", on_bad_lines="skip")

# print(ner.head())
# print()
# print(ner_ds.head())
# print()

# Associate rows with sentence numbers
ner_ds["Sentence #"] = ner_ds["Sentence #"].ffill()

# Drop mssing words or POS tags
ner_ds = ner_ds.dropna(subset=["Word", "POS"])

# group senentences by sentence # as list of tuples of obserervation and hidden state (word, POS tag)
sentences = (
    ner_ds.groupby("Sentence #", group_keys=False)
    .apply(
        lambda g: list(zip(g["Word"].str.strip(), g["POS"].str.strip()))
        )
    .tolist()
)

"""===== HMM Training ====="""

# Hidden state at start of seqeunce
init_counts  = defaultdict(int)
# Transition probability (hidden state -> hidden state)
trans_counts = defaultdict(lambda: defaultdict(int))
# Emission probability (hidden state -> example_word)
emit_counts  = defaultdict(lambda: defaultdict(int))
# Hidden state raw counts 
tag_counts   = defaultdict(int)

# Loop through each sentence
for sentence in sentences:
    # Ensure sentence not blank
    if not sentence:
        continue
    words, tags = zip(*sentence)

    # Count first state in sentence
    init_counts[tags[0]] += 1

    # Loop through each observation/state pair
    for i, (word, tag) in enumerate(zip(words, tags)):
        # count state occurence
        tag_counts[tag] += 1
        # count state emission of word
        emit_counts[tag][word.lower()] += 1

        # count transitions between states
        if i > 0:
            trans_counts[tags[i-1]][tag] += 1

# Get all tags
all_tags = sorted(tag_counts.keys())
# Index tags
tag_index = {t: i for i, t in enumerate(all_tags)}
n_tags = len(all_tags)

# Get all words
vocab = sorted({w.lower() for s in sentences for w, _ in s})
# Index words
word_index = {w: i for i, w in enumerate(vocab)}
n_words = len(vocab)

# Get total sentences
total_sentences = len(sentences)

# Initialize initial hidden state counts as zero probability (-inf)
log_pi = np.full(n_tags, -np.inf)

# Loop through each hidden state
for t, ti in tag_index.items():
    # Calculate hidden state probability using MLE with Laplace (add-one)
    log_pi[ti] = np.log((init_counts[t] + 1) / (total_sentences + n_tags))

# Initialize transition probabilities as 0 for each possible transition between states
log_A = np.zeros((n_tags, n_tags))
# Loop through all "from-state" tags
for ti_str, ti in tag_index.items():
    # Set denominator as tag count + total tag size (add-one smoothing)
    denom = tag_counts[ti_str] + n_tags
    # Loop through each "to-state" tags
    for tj_str, tj in tag_index.items():
        # Set numerator as count of from-state -> to-state with add-one smoothing
        count = trans_counts[ti_str][tj_str] + 1
        # Calculate transition probability
        log_A[ti, tj] = np.log(count / denom)

# Initialize emission probability dicts
log_B   = {}
log_unk = {}
# Loop through each hidden state
for t, ti in tag_index.items():
    # Denominator uses add-one smoothing
    denom = tag_counts[t] + n_words + 1
    arr = np.zeros(n_words)
    # Loop through each observation (word)
    for w, wi in word_index.items():
        # Calculate probability of current observation (Word) being emitted by the current hidden state (POS tag)
        arr[wi] = np.log((emit_counts[t][w] + 1) / denom)
    # Save all emission probabilities for current tag
    log_B[ti]   = arr
    # OOV fallback to smoothed value to avoid zero-probability
    log_unk[ti] = np.log(1 / denom)

"""===== Viterbi Inference ====="""
def viterbi(words):
    # Get length of sequence
    T  = len(words)

    # Handle empty sequence
    if T == 0:
        return []

    # Initialise probabilities as -inf and backpointers as 0 
    viterbi_mat = np.full((T, n_tags), -np.inf)
    backpointer = np.zeros((T, n_tags), dtype=int)

    # Calculate initial state probabilities
    w0 = words[0].lower()
    wi = word_index.get(w0)
    for ti in range(n_tags):
        # Emission probability of initial state
        emit = log_B[ti][wi] if wi is not None else log_unk[ti]
        # Probability score of  initial tag with emission of initial tag
        viterbi_mat[0, ti] = log_pi[ti] + emit

    # Recurse through remaining words
    for t in range(1, T):
        # Get current word
        w  = words[t].lower()
        wi = word_index.get(w)
        # Loop through each state for each word
        for tj in range(n_tags):
            # Get emission probability for word if in vocab, otherwise use OOV smoothed emission probability
            emit   = log_B[tj][wi] if wi is not None else log_unk[tj]
            # probability scores are combination of previous score and transition probability of current state
            scores = viterbi_mat[t-1, :] + log_A[:, tj]
            # Set best score for current word
            best   = int(np.argmax(scores))
            viterbi_mat[t, tj]  = scores[best] + emit
            # Set backpointer to best score for downstream inference
            backpointer[t, tj]  = best

    # Backtrack through the best scoring state for each word to build the seqeuence in reverse
    tags_out = [int(np.argmax(viterbi_mat[T-1, :]))]
    for t in range(T-1, 0, -1):
        tags_out.append(backpointer[t, tags_out[-1]])
    # Reverse the reversed hidden state (POS tag) seqeuence
    tags_out.reverse()

    return [all_tags[ti] for ti in tags_out]

"""===== Evaluate trained model ====="""
# Get subset of last 500 sentences for accuracy testing
test_sentences = sentences[-500:]
# Counts of correct and total tags
correct_tokens = 0
total_tokens = 0

# Loop through test sentences 
for s in test_sentences:
    # Get observations and hidden states in sentence
    words, true_tags = zip(*s)
    # Calculate best seqeunce probability using viterbi
    pred_tags = viterbi(list(words))
    
    # Loop through viterbi predictions to calculate accuracy
    for correct, predicted in zip(true_tags, pred_tags):
        # Count correct predictions
        correct_tokens += (correct == predicted)
        # Count total predictions
        total_tokens += 1

# Calculate accuraccy for viterbi predictions
accuracy = correct_tokens / total_tokens * 100
print(f"Accuracy on last 500 sentences: {accuracy:.2f}%")

# Unseen data test
test_sentences_new = [
    "John Freeman who was Gordon Freemans brother was one day in an office typing on a computer",
    "This Smith and Wesson got me moving like an invasive species",
    "Mahjong causes great damage to the human soul without a single benefit",
    "Congratulations, your item has sold on the Steam community market for 0.03 cents",
    "Snooping around as usual I see",
]

# Evaluate on un-seen data
# Loop through each unseen sentence
for sentence in test_sentences_new:
    # Split into words
    words     = sentence.split()
    # Predict sequence using viterbi
    pred_tags = viterbi(words)
    print(f"\nSentence : {sentence}")
    print(f"{'Word'} {'Predicted POS'}")
    print("-" * 35)
    for word, tag in zip(words, pred_tags):
        print(f"  {word} {tag}")