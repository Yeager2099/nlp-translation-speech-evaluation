# Report transcription

Text extracted from the author-provided course report. Figures are not reproduced and tables may lose formatting. Student identifiers and institutional email addresses are removed.

## Page 1

CIS 5300: Natural Language Processing
Homework 9: Machine Translation & Speech
Report Template
Name: Yeager Ye
Penn ID: [STUDENT_ID_REDACTED]
Learning Objectives
1. Represent audio as spectrograms and mel spectrograms;
2. Implement the core evaluation metrics from scratch—BLEU and chrF(MT) and
WER(ASR);
3. Translate with an LLM(zero-shot and few-shot) and score the output with your
own metrics;
4. Transcribe speech with Whisper and measure its accuracy and noise robustness;
5. Probe bias and fairness across both modalities (gender bias in MT, speaker
disparity in ASR).
Points Breakdown
Section Points Description
Answer 3.3 5 pts Translation Analysis
Answer 4.3 5 pts ASR Error Analysis
Answer 5.3 3 pts Cross-Modal Bias
Answer 6.1 5 pts Evaluating a Speech-to-Speech Translation
System
Total 18 pts

## Page 2

CIS 5300: Natural Language Processing
Answer 3.3: Translation Analysis
1. Metric (dis)agreement: do BLEU and chrF agree on which translations are good
vs. bad? Find at least one translation where they disagree and explain why, in
terms of what each metric measures (word-level n-gram precision vs. character-
level F-score).
BLEU and chrF broadly agree on which translations are good vs. bad — when a
translation is near-perfect (e.g., es-en sentence 0: "The city has many beautiful
parks."), both metrics score 1.0000. However, they disagree sharply on translations
with small word-level differences. The most striking disagreement occurs in de-en
sentence 1 (few-shot): the hypothesis "The children are playing in the snow" vs.
reference "The children play in the snow." BLEU scores this 0.0000 because the 4-
gram "the children are playing" does not appear in the reference, causing p4 = 0,
which zeros out the entire geometric mean. chrF scores it 0.7921 because at the
character level, "playing" and "play" share substantial character n-gram overlap
("play" appears in both), and the rest of the sentence is identical. This illustrates the
fundamental difference: BLEU is a word-level n-gram precision metric that is brittle
to any 4-gram mismatch (a single different word can zero the score), while chrF is a
character-level F-score that rewards partial matches at the sub-word level, making
it more forgiving of morphological variations (e.g., tense changes, progressive
aspect) that are semantically minor but lexically disruptive.
2. Zero-shot vs. few-shot: did few-shot examples improve quality overall? Looking
at your own BLEU/chrF numbers broken down by language pair, name the
language pair(s) where few-shot helped the most (by your metrics)and
speculate why in-context examples might help more for that pair than the
others.
Language Pair Sentence ZS-BLEU ZS-chrF FS-BLEU FS-chrF

## Page 3

CIS 5300: Natural Language Processing
Spanish to Average of 5 0.8701 0.9282 0.8608 0.9216
English sentences
German to Average of 5 0.9095 0.9373 0.8000 0.9584
English sentences
French to Average of 5 0.7520 0.9513 0.7693 0.9048
English sentences
Few-shot examples did not consistently improve quality across all language pairs.
For Spanish-English (es-en), zero-shot slightly outperformed few-shot (BLEU
0.8701 vs. 0.8608), likely because Claude already translates simple Spanish
sentences near-perfectly zero-shot, and the few-shot examples occasionally primed
the model toward alternative word choices (e.g., "ninos" -> "kids" instead of
"children", "ocho" -> "eight o'clock" instead of "eight"). For German-English (de-en),
zero-shot was clearly better (BLEU 0.9095 vs. 0.8000) — the few-shot examples
led the model to use progressive tense ("are playing" vs. "play"), which zeroed the
BLEU score on that sentence due to 4-gram mismatch. For French-English (fr-en),
few-shot slightly helped (BLEU 0.7520 vs. 0.7693) — this is the pair where few-
shot was most beneficial, likely because French has more morphological variation
and nuance in phrasing (e.g., "prepare" vs. "makes" for "gateau"), and the in-context
examples helped the model calibrate to the expected English register. Overall, few-
shot helped most for fr-en, the pair where zero-shot performance was lowest,
suggesting that in-context examples are most valuable when the model has more
uncertainty about the target language's conventions — but can hurt when they
introduce stylistic variation that penalizes exact-match metrics like BLEU.
Answer 4.3: ASR Error Analysis
1. Error types: using `get_alignment` on 2–3 of your Whisper transcriptions,
categorize the errors (substitutions / insertions / deletions). What kinds of words
does Whisper get wrong (proper nouns, function words, homophones)? Quote
specific `(reference_word, hypothesis_word)` pairs.

## Page 4

CIS 5300: Natural Language Processing
Sample Error Category Example (ref → hyp)
(substitutions / insertions
/ deletions)
Sample 2 (spk 4078) 1 substitution(s) "off" -> "of"
(homophone/substitution)
Sample 3 (spk 1215) 1 substitution(s) "dazed" -> "dazzled"
(homophone/substitution)
Sample 4 (spk 2094) 1 substitution(s) "bother" -> "brother"
(homophone/substitution)
Using get_alignment on the Whisper transcriptions, the dominant error type is
substitutions (6 out of 10 samples had at least one substitution, with 0 insertions and
0 deletions across all samples). The specific substitution pairs reveal that Whisper-tiny
struggles primarily with: (1) Homophones and near-homophones: "manner" -> "manor"
(sample 7), "dazed" -> "dazzled" (sample 3), and "bother" -> "brother" (sample 4) —
these are phonetically similar words where the tiny model lacks the capacity to
disambiguate using broader context. (2) Function word substitutions: "off" -> "of"
(sample 2) and "a" -> "the" (sample 9) — these are short, high-frequency words where
the acoustic signal is brief and easily confused. (3) Archaic/formal vocabulary:
"whence" -> "when" (sample 8) — the model substitutes a rare word with a more
common one, reflecting its training distribution. No insertions or deletions were
observed, suggesting that whisper-tiny on clean LibriSpeech audio maintains correct
word count but occasionally substitutes acoustically similar words. The average WER
of 4.53% is consistent with reported whisper-tiny performance on LibriSpeech test-
clean.
2. Noise: describe how WER changed with noise level in 4.2 and explain why the
degradation curve looks the way it does (threshold effects, hallucinated words
at high noise, etc.).

## Page 5

CIS 5300: Natural Language Processing
WER increases monotonically with noise level, following a roughly exponential
degradation curve. At noise level 0.0 (clean audio), the average WER is 4.5%. At noise
level 0.005, WER rises slightly to 6.2%. At 0.01, WER jumps to 9.8%. At 0.02, WER
reaches 15.5%. At 0.05, WER climbs to 30.2%, and at 0.1, WER reaches 54.7%. The
degradation curve shows a threshold effect: below noise level ~0.01, Whisper-tiny is
largely robust (WER increase < 2x), but above 0.02, the model begins to hallucinate
words as the speech signal becomes buried in noise. At noise level 0.1, more than half
the words are wrong, indicating the model is generating plausible-sounding English
rather than actually transcribing the audio. This is consistent with known Whisper
behavior: at high noise levels, the model's language model component takes over from
its acoustic model, producing fluent but incorrect transcriptions. The per-sample
variation is also notable: samples with shorter, clearer utterances (e.g., sample 0)
degrade more gracefully than longer, more complex sentences. This suggests that
noise robustness depends not only on the noise level but also on the acoustic
complexity and duration of the utterance. (See the WER-vs-noise plot produced in
Section 4.2 of the notebook.)
Answer 5.3: Cross-Modal Bias
1. MT bias: from 5.1, which occupations did the model default to "he" vs. "she"?
What stereotype pattern (if any) do you see (e.g. does it track real-world
occupational demographics or common stereotypes), and where does it come
from (i.e. what in the model's training likely produced this pattern)?
The model defaults to "he" for doctor, engineer, CEO, programmer, chef, lawyer, and
pilot (7 of 10 occupations in Turkish, 6 of 10 in Hungarian), and "she" for nurse,
teacher, secretary, and kindergarten teacher. This pattern closely tracks common
occupational gender stereotypes rather than real-world demographics: while the
model assigns "he" to high-status, technical, and leadership roles (CEO, engineer,
programmer, lawyer), it assigns "she" to caregiving and administrative roles (nurse,
teacher, secretary, kindergarten teacher). Notably, the split is more pronounced in
Turkish (7 male / 3 female) than Hungarian (6 male / 4 female), possibly because
Turkish training data contains stronger gender-occupation correlations. The root cause
is the model's training data: web-crawled text, books, and articles that describe

## Page 6

CIS 5300: Natural Language Processing
professionals using gendered pronouns in stereotypical ways. When translating from a
gender-neutral-pronoun language (Hungarian "O", Turkish "O"), the model must
choose a gender, and it defaults to the statistically most common association in its
training corpus. This is a direct consequence of training on data that reflects historical
gender imbalances in the workforce and biased language patterns in written text.
2. ASR fairness: ASR WER on CORAAL (African American English) is ~16% vs.
~1.4% on LibriSpeech clean. Why do ASR systems underperform on non-
standard dialects, and what are the real-world consequences?
ASR systems underperform on non-standard dialects like African American English
(AAE, as in the CORAAL corpus) primarily because their training data is dominated
by Standard American English (SAE) — LibriSpeech, for example, consists of
audiobook recordings read by speakers trained in SAE pronunciation. AAE differs
from SAE in systematic ways: consonant cluster reduction (e.g., "last" -> "las"),
vowel shifts, grammatical constructions (habitual "be"), and intonation patterns.
When the acoustic model has rarely encountered these patterns during training, it
interprets them as noise or errors rather than legitimate phonetic variants, leading to
higher substitution and deletion rates. The real-world consequences are severe: (1)
Voice assistants (Siri, Alexa) fail more often for AAE speakers, creating unequal
access to technology. (2) Court transcription systems produce inaccurate records for
AAE-speaking defendants or witnesses, potentially affecting legal outcomes. (3)
Automated captioning on video platforms marginalizes AAE content creators by
producing garbled subtitles. (4) In education, ASR-based language learning tools
may penalize AAE-speaking students. The 16% vs. 1.4% WER gap is not just a
technical metric — it represents a systemic barrier to equal participation in voice-
mediated services.
3. Common cause: MT gender bias and ASR dialect disparity are both
representation problems — meaning: the training data over-represents some
groups/varieties and under-represents others, so the model performs worse
(or reproduces bias) for whatever was underrepresented. Explain the shared
root cause — why the data ends up skewed in the first place — in your own
words, and propose one mitigation that would help both.
The shared root cause of both MT gender bias and ASR dialect disparity is training
data skew: the data used to train these models over-represents certain demographic
groups and language varieties while under-representing others. For MT, the parallel

## Page 7

CIS 5300: Natural Language Processing
text corpora (e.g., Europarl, OPUS) and web-crawled data disproportionately contain
text written by and about men in professional contexts, encoding gender-occupation
stereotypes. For ASR, the audio corpora (LibriSpeech, CommonVoice) overwhelmingly
feature SAE speakers reading prepared text, leaving non-standard dialects
underrepresented. This skew arises because: (1) Data collection is often convenience-
based — audiobooks are easy to source but represent a narrow demographic; web text
is abundant but reflects existing power structures. (2) Annotation costs favor majority-
language varieties, as annotators are easier to recruit for SAE. (3) Open-source data
contributions (e.g., CommonVoice) reflect the demographics of contributors, who skew
toward tech-literate, majority-group speakers. One mitigation that would help both
problems is targeted data collection and curation: actively recruiting speakers of
underrepresented dialects for ASR training data, and curating parallel translation
corpora that explicitly counter stereotypical gender-occupation associations (e.g.,
including texts where women are engineers and men are nurses). Additionally, bias-
aware decoding strategies — such as gender-neutral translation options or dialect-
adaptive acoustic models — can reduce harm even without fully retraining. The key
insight is that both problems stem from the same underlying issue: models learn the
statistical regularities of their training data, and when that data is biased, the model
inherits and amplifies the bias.
Answer 6.1: Evaluating a Speech-to-Speech Translation
System
Imagine a product that listens to Spanish speech and outputs spoken
English (ASR → MT → text-to-speech). Design an evaluation protocol.
a) Automatic metrics: which automatic metrics would you compute at each
stage, and what are their limitations — citing what you learned implementing
BLEU/chrF/WER?
At the ASR (recognition) stage, I would compute WER (Word Error Rate) between
the Spanish speech transcription and a gold reference transcript, broken down into
S/I/D counts using alignment. WER directly measures recognition accuracy and is
the standard ASR metric. Its limitation, as I learned implementing it, is that it
penalizes all errors equally — a function word substitution ("the" -> "a") costs the

## Page 8

CIS 5300: Natural Language Processing
same as a content word substitution ("doctor" -> "doctor's"), even though the
latter is more semantically damaging. WER also cannot detect cases where the
transcription is semantically correct but lexically different (e.g., contractions:
"don't" vs. "do not"). At the MT (translation) stage, I would compute both BLEU
and chrF between the English translation and reference translations. BLEU
captures word-level n-gram precision and is the standard MT metric, but as I
observed in Section 3.3, it is brittle: a single 4-gram mismatch can zero the score
even when the translation is semantically perfect (e.g., "are playing" vs. "play").
chrF, being character-level, is more robust to morphological variation and should
be computed alongside BLEU to catch cases where BLEU underestimates quality.
Both metrics require reference translations, which limits their use in production
(where references are unavailable). Neither metric captures fluency, adequacy, or
pragmatic appropriateness — a translation can score high on BLEU but sound
unnatural to a human reader. At the TTS (speech synthesis) stage, I would
compute MOS (Mean Opinion Score) through human ratings, as there is no reliable
automatic metric for speech naturalness. Automatic metrics like Mel-cepstral
distortion exist but correlate poorly with human perception of quality.
b) Human evaluation: what human evaluation would you add to the pipeline
(who would judge what, at which stage(s)), and what problems would that
catch that BLEU/chrF/WER miss?
I would add human evaluation at three stages: (1) At the ASR stage, bilingual
Spanish-English speakers would verify that the transcription correctly captures
the speaker's intended meaning, catching cases where WER is low but the
transcription misses pragmatically important elements (e.g., politeness markers,
discourse particles, or proper nouns that WER treats as single-word
substitutions). (2) At the MT stage, native English speakers would rate the
translation on fluency (does it sound natural?) and adequacy (does it preserve the
original meaning?) using a 1-5 Likert scale, following the methodology from
Lecture M11.3. This would catch problems that BLEU/chrF miss: awkward
phrasing ("The museum is open on Sundays" vs. "The museum is open on Sunday"
— both score similarly on chrF but differ in naturalness), pragmatic errors (wrong
register, lost politeness), and cultural inappropriateness. (3) At the TTS stage,
listeners would rate the synthesized English speech for naturalness, intelligibility,
and pronunciation accuracy, catching issues like mispronounced words, unnatural
prosody, or robotic-sounding speech that no text-based metric can detect. I would
also conduct end-to-end evaluation: Spanish speakers would speak a prompt, and
English-speaking listeners would rate whether they understood the intended

## Page 9

CIS 5300: Natural Language Processing
message, testing the full pipeline rather than individual components. This catches
integration errors (e.g., ASR produces a slightly wrong transcription that MT then
translates faithfully but incorrectly, compounding the error).
c) Fairness checks: what fairness checks would you run before shipping, given
Section 5? A general protocol/principle is fine (you don't need a specific
numeric threshold).
Before shipping, I would run the following fairness checks based on the biases
observed in Section 5: (1) ASR fairness across dialects and accents: measure WER
broken out by speaker demographic groups — specifically, test with speakers of
Mexican Spanish, Caribbean Spanish, Castilian Spanish, and indigenous-language-
influenced Spanish (e.g., Quechua-Spanish bilinguals). If WER varies by more than
2x between groups, the ASR component has a fairness problem that needs
addressing through targeted data collection or model fine-tuning, analogous to the
LibriSpeech vs. CORAAL gap observed in Section 5.2. (2) MT gender bias: following
the methodology in Section 5.1, test with gender-neutral source sentences (e.g.,
from a language with gender-neutral pronouns, or constructed test sentences) and
verify that the translation does not systematically default to one gender for certain
occupations. If gender skew is detected, implement gender-neutral translation
options or bias-corrected decoding. (3) Intersectional fairness: test the full pipeline
(ASR + MT + TTS) with speakers who belong to multiple underrepresented groups
(e.g., female speakers of non-standard dialects), as biases can compound across
stages. (4) Representation audit: examine the training data for the ASR and MT
components to verify that diverse dialects, accents, genders, and socioeconomic
groups are adequately represented. The general principle is: measure performance
broken out by demographic group before launch, flag any group with
disproportionately high error rates, and either improve the model for those groups or
document the limitation transparently. This follows directly from the Section 5
finding that both MT gender bias and ASR dialect disparity stem from training data
underrepresentation — the fix is to measure, identify gaps, and close them through
targeted data collection and model improvement.
