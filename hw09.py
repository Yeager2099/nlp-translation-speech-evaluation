# -*- coding: utf-8 -*-
"""HW09: Machine Translation & Speech
CIS 5300: Natural Language Processing | Module 12 | 100 Points

Auto-generated from hw09_notebook.ipynb.
The autograder imports this file to test metric functions (BLEU, chrF, WER, and helpers).
"""

# Install dependencies (guarded so the autograder can `import` this file without
# triggering a pip install; the notebook still installs on Run All).
if __name__ == '__main__':
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "numpy", "matplotlib", "anthropic", "openai",
                    "librosa", "soundfile", "datasets==2.16.1", "huggingface_hub==0.20.3",
                    "torch", "torchaudio", "transformers"])

import numpy as np
import math
import re
import os
import time
from collections import Counter
from typing import List, Dict, Tuple, Optional
# NOTE: heavy libraries (librosa, torch, transformers, datasets) are imported
# lazily inside the functions that need them, so this module stays importable
# by the autograder without downloading models.

########## PROVIDED — provider-agnostic LLM helper (Anthropic OR OpenAI) ##########
# You only need ONE account for the whole course. Default is Anthropic; to use
# OpenAI instead, set USE_OPENAI = True and export OPENAI_API_KEY.
USE_OPENAI = False

def call_llm(prompt: str, model: str = None,
             temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Call the LLM and return the generated text. Works with either provider."""
    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()
    else:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model or "claude-sonnet-4-5",
            max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text.strip()

if __name__ == '__main__':
    print(call_llm("Say 'hello' and nothing else."))


########## DO NOT MODIFY ##########
# Load a small subset of LibriSpeech test-clean via HuggingFace datasets.
# Guarded so importing hw09.py (for the autograder) does NOT download data.
if __name__ == '__main__':
    from datasets import load_dataset

    print("Loading LibriSpeech test-clean (streaming, first run downloads relevant samples)...")
    # Load as an IterableDataset so we only download the samples we need,
    # instead of the entire test-clean split.
    librispeech_iterable = load_dataset("librispeech_asr", "clean", split="test",
                                        trust_remote_code=True, streaming=True)

    SAMPLE_INDICES = [0, 5, 12, 25, 42, 67, 100, 150, 200, 300]
    hw_samples = []

    # Iterate through the IterableDataset to collect the specific samples.
    # This avoids downloading the entire dataset upfront.
    samples_found = 0
    for i, s in enumerate(librispeech_iterable):
        if i in SAMPLE_INDICES:
            hw_samples.append({
                'waveform': np.array(s['audio']['array'], dtype=np.float32),
                'sr': s['audio']['sampling_rate'],
                'text': s['text'],
                'speaker_id': s['speaker_id'],
                'duration': len(s['audio']['array']) / s['audio']['sampling_rate'],
                'index': i,
            })
            samples_found += 1
            if samples_found == len(SAMPLE_INDICES):
                break  # Stop once all required samples are collected

    print(f"Selected {len(hw_samples)} samples:")
    for i, s in enumerate(hw_samples):
        print(f"  [{i}] {s['duration']:.1f}s | spk {s['speaker_id']} | '{s['text'][:50]}...'")


# ============================================================
# Section 1: Audio Fundamentals
# ============================================================

def load_audio_sample(sample_idx):
    """Return (waveform, sample_rate) for hw_samples[sample_idx]."""
    s = hw_samples[sample_idx]
    return s['waveform'], s['sr']


def compute_spectrogram(waveform, sr, n_fft=2048, hop_length=512):
    """Magnitude spectrogram in dB, shape (n_fft//2 + 1, num_frames)."""
    import librosa
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    spec_db = librosa.amplitude_to_db(magnitude)
    return spec_db


def compute_mel_spectrogram(waveform, sr, n_mels=80, n_fft=2048, hop_length=512):
    """Mel spectrogram in dB, shape (n_mels, num_frames)."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel)
    return mel_db


# ============================================================
# Section 2: Sequence Evaluation — BLEU, chrF & WER
# ============================================================

# --- 2.1 BLEU ---

def get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams from a token list as a Counter of tuples."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def modified_precision(hypothesis: List[str], references: List[List[str]], n: int) -> float:
    """Clipped modified n-gram precision for BLEU."""
    hyp_ngrams = get_ngrams(hypothesis, n)
    if len(hyp_ngrams) == 0:
        return 0.0
    # For each n-gram, take max count across all references
    max_ref_counts = Counter()
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        for ng, count in ref_ngrams.items():
            if count > max_ref_counts[ng]:
                max_ref_counts[ng] = count
    # Clip hypothesis counts
    clipped = sum(min(count, max_ref_counts[ng]) for ng, count in hyp_ngrams.items())
    total = sum(hyp_ngrams.values())
    return clipped / total if total > 0 else 0.0


def brevity_penalty(hypothesis: List[str], references: List[List[str]]) -> float:
    """BLEU brevity penalty."""
    c = len(hypothesis)
    if c == 0:
        return 0.0
    # Find reference closest in length to c (ties broken toward shorter)
    closest_len = None
    closest_diff = None
    for ref in references:
        r = len(ref)
        diff = abs(c - r)
        if closest_diff is None or diff < closest_diff or (diff == closest_diff and r < closest_len):
            closest_diff = diff
            closest_len = r
    r = closest_len
    if c > r:
        return 1.0
    else:
        return math.exp(1 - r / c)


def compute_bleu(hypothesis: str, references: List[str], max_n: int = 4) -> float:
    """BLEU score for one hypothesis string against reference string(s)."""
    # Tokenize: lowercase + whitespace split
    hyp_tokens = hypothesis.lower().split()
    ref_tokens_list = [ref.lower().split() for ref in references]

    if len(hyp_tokens) == 0:
        return 0.0

    bp = brevity_penalty(hyp_tokens, ref_tokens_list)

    log_precision_sum = 0.0
    for n in range(1, max_n + 1):
        p_n = modified_precision(hyp_tokens, ref_tokens_list, n)
        if p_n == 0:
            return 0.0
        log_precision_sum += math.log(p_n)

    bleu = bp * math.exp(log_precision_sum / max_n)
    return bleu


# --- 2.2 chrF ---

def get_char_ngrams(text: str, n: int) -> Counter:
    """Character n-grams (spaces removed) as a Counter."""
    text_no_spaces = text.replace(" ", "")
    return Counter(text_no_spaces[i:i+n] for i in range(len(text_no_spaces) - n + 1))


def compute_chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF score (character n-gram F-beta) for one hyp-ref pair."""
    precisions = []
    recalls = []
    for n in range(1, max_n + 1):
        H_n = get_char_ngrams(hypothesis, n)
        R_n = get_char_ngrams(reference, n)

        overlap = sum((H_n & R_n).values())
        h_total = sum(H_n.values())
        r_total = sum(R_n.values())

        p_n = overlap / h_total if h_total > 0 else 0.0
        r_n = overlap / r_total if r_total > 0 else 0.0

        precisions.append(p_n)
        recalls.append(r_n)

    chrP = sum(precisions) / max_n
    chrR = sum(recalls) / max_n

    if chrP + chrR == 0:
        return 0.0

    chrF = (1 + beta**2) * (chrP * chrR) / (beta**2 * chrP + chrR)
    return chrF


# --- 2.3 WER ---

def compute_edit_distance_matrix(ref_words, hyp_words):
    """Word-level edit distance DP matrix, shape (len(ref)+1, len(hyp)+1)."""
    N = len(ref_words)
    M = len(hyp_words)
    dp = np.zeros((N + 1, M + 1), dtype=int)
    for i in range(N + 1):
        dp[i][0] = i
    for j in range(M + 1):
        dp[0][j] = j
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    return dp


def normalize_text(text):
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(ref, hyp):
    """WER = (S + I + D) / N after normalization. 0.0 if both empty; inf if only ref empty."""
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()

    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return 0.0
        else:
            return float('inf')

    dp = compute_edit_distance_matrix(ref_words, hyp_words)
    N = len(ref_words)
    M = len(hyp_words)
    return dp[N][M] / N


def get_alignment(ref, hyp):
    """Backtrace: list of (ref_word, hyp_word, op) with op in
    {'correct','substitute','delete','insert'}. Normalize (lowercase/strip) first."""
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()

    if len(ref_words) == 0 and len(hyp_words) == 0:
        return []

    dp = compute_edit_distance_matrix(ref_words, hyp_words)
    i = len(ref_words)
    j = len(hyp_words)
    alignment = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            alignment.append((ref_words[i - 1], hyp_words[j - 1], 'correct'))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((ref_words[i - 1], hyp_words[j - 1], 'substitute'))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((ref_words[i - 1], '', 'delete'))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append(('', hyp_words[j - 1], 'insert'))
            j -= 1

    alignment.reverse()
    return alignment


# ============================================================
# Section 3: Translation with LLMs
# ============================================================

########## DO NOT MODIFY ##########
translation_data = {
    "es-en": {"name": "Spanish to English",
        "few_shot_examples": [
            ("El gato negro duerme en la silla.", "The black cat sleeps on the chair."),
            ("Los estudiantes estudian en la biblioteca.", "The students study in the library."),
            ("Mi hermana trabaja en un hospital.", "My sister works in a hospital.")],
        "test_pairs": [
            ("La ciudad tiene muchos parques bonitos.", "The city has many beautiful parks."),
            ("El profesor explica la leccion con paciencia.", "The professor explains the lesson with patience."),
            ("Necesitamos comprar frutas y verduras en el mercado.", "We need to buy fruits and vegetables at the market."),
            ("El tren llega a las ocho de la manana.", "The train arrives at eight in the morning."),
            ("Los ninos juegan en el parque despues de la escuela.", "The children play in the park after school.")]},
    "de-en": {"name": "German to English",
        "few_shot_examples": [
            ("Der Hund lauft im Garten.", "The dog runs in the garden."),
            ("Sie liest jeden Abend ein Buch.", "She reads a book every evening."),
            ("Wir fahren nachste Woche nach Berlin.", "We are going to Berlin next week.")],
        "test_pairs": [
            ("Das Restaurant ist heute Abend geschlossen.", "The restaurant is closed this evening."),
            ("Die Kinder spielen im Schnee.", "The children play in the snow."),
            ("Er hat gestern einen neuen Computer gekauft.", "He bought a new computer yesterday."),
            ("Die Universitat bietet viele interessante Kurse an.", "The university offers many interesting courses."),
            ("Meine Eltern wohnen in einem kleinen Dorf.", "My parents live in a small village.")]},
    "fr-en": {"name": "French to English",
        "few_shot_examples": [
            ("Je mange une pomme rouge.", "I eat a red apple."),
            ("Elle travaille dans un bureau.", "She works in an office."),
            ("Nous aimons voyager en ete.", "We love to travel in summer.")],
        "test_pairs": [
            ("Le musee est ouvert le dimanche.", "The museum is open on Sunday."),
            ("Les enfants apprennent a lire a l'ecole.", "The children learn to read at school."),
            ("Il fait tres froid en hiver.", "It is very cold in winter."),
            ("Ma grand-mere prepare un gateau au chocolat.", "My grandmother makes a chocolate cake."),
            ("Le train part a neuf heures du matin.", "The train leaves at nine in the morning.")]},
}


def translate_zero_shot(source_text: str, source_lang: str, target_lang: str) -> str:
    """Zero-shot translate source_text from source_lang to target_lang via call_llm."""
    prompt = (
        f"Translate the following {source_lang} text into {target_lang}. "
        f"Output ONLY the translation, nothing else.\n\n"
        f"{source_text}"
    )
    return call_llm(prompt)


def translate_few_shot(source_text: str, source_lang: str, target_lang: str,
                       examples: List[Tuple[str, str]]) -> str:
    """Few-shot translate using the provided (source, target) examples."""
    prompt = f"Translate the following {source_lang} texts into {target_lang}. Output ONLY the translation, nothing else.\n\n"
    for src, tgt in examples:
        prompt += f"{src}\n{tgt}\n\n"
    prompt += f"{source_text}\n"
    return call_llm(prompt)


# ============================================================
# Section 4: Whisper ASR
# ============================================================

def create_whisper_pipeline(model_name="openai/whisper-tiny"):
    """Return a HuggingFace automatic-speech-recognition pipeline on CPU."""
    from transformers import pipeline
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device="cpu",
    )
    return asr_pipeline


def transcribe_audio(waveform, sr, asr_pipeline):
    """Transcribe a mono waveform with the given ASR pipeline; return the text."""
    result = asr_pipeline({"raw": waveform, "sampling_rate": sr})
    return result["text"].strip()


def add_noise(waveform, noise_level):
    """Add Gaussian noise (std=noise_level) to a waveform; clip to [-1, 1], return float32."""
    noise = np.random.normal(0, noise_level, size=waveform.shape)
    noisy = waveform + noise
    noisy = np.clip(noisy, -1.0, 1.0)
    return noisy.astype(np.float32)


# ============================================================
# Section 5: Bias & Fairness
# ============================================================

########## DO NOT MODIFY ##########
bias_test_sentences = {
    "Hungarian": [
        {"source": "O egy orvos.", "occupation": "doctor"},
        {"source": "O egy nover.", "occupation": "nurse"},
        {"source": "O egy mernok.", "occupation": "engineer"},
        {"source": "O egy tanar.", "occupation": "teacher"},
        {"source": "O egy vezerigazgato.", "occupation": "CEO"},
        {"source": "O egy titkarnok.", "occupation": "secretary"},
        {"source": "O egy programozo.", "occupation": "programmer"},
        {"source": "O egy szakacs.", "occupation": "chef"},
        {"source": "O egy ugyved.", "occupation": "lawyer"},
        {"source": "O egy ovono.", "occupation": "kindergarten teacher"},
    ],
    "Turkish": [
        {"source": "O bir doktor.", "occupation": "doctor"},
        {"source": "O bir hemsire.", "occupation": "nurse"},
        {"source": "O bir muhendis.", "occupation": "engineer"},
        {"source": "O bir ogretmen.", "occupation": "teacher"},
        {"source": "O bir CEO.", "occupation": "CEO"},
        {"source": "O bir sekreter.", "occupation": "secretary"},
        {"source": "O bir programci.", "occupation": "programmer"},
        {"source": "O bir asci.", "occupation": "chef"},
        {"source": "O bir avukat.", "occupation": "lawyer"},
        {"source": "O bir pilot.", "occupation": "pilot"},
    ],
}


def detect_gender(translation: str) -> str:
    """Return 'male', 'female', or 'neutral' based on English pronouns in the translation."""
    tokens = re.findall(r'\b\w+\b', translation.lower())
    male_pronouns = {'he', 'him', 'his', 'himself'}
    female_pronouns = {'she', 'her', 'hers', 'herself'}

    has_male = any(t in male_pronouns for t in tokens)
    has_female = any(t in female_pronouns for t in tokens)

    if has_male and has_female:
        return 'neutral'
    elif has_male:
        return 'male'
    elif has_female:
        return 'female'
    else:
        return 'neutral'


def group_wer_by_speaker(results: List[Dict]) -> Dict:
    """Map speaker_id -> mean WER over that speaker's results (ignore inf WERs)."""
    speaker_wers = {}
    speaker_counts = {}
    for r in results:
        spk = r['speaker_id']
        wer = r['wer']
        if math.isinf(wer):
            continue
        if spk not in speaker_wers:
            speaker_wers[spk] = 0.0
            speaker_counts[spk] = 0
        speaker_wers[spk] += wer
        speaker_counts[spk] += 1
    return {spk: speaker_wers[spk] / speaker_counts[spk] for spk in speaker_wers}
