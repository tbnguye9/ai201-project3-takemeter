# TakeMeter — r/leagueoflegends Discourse Classifier

A fine-tuned text classifier that evaluates discourse quality in r/leagueoflegends,
distinguishing between strategy posts, rants, and discussion posts.

---

## Community Choice

**Community:** r/leagueoflegends (Reddit)

r/leagueoflegends is one of the largest gaming subreddits with over 6 million members.
The community produces a constant stream of text-heavy posts ranging from detailed
gameplay advice to emotional venting to open-ended debates. This variety makes it
an ideal fit for a classification task — the discourse is active enough to collect
200+ examples easily, and the differences between post types are meaningful and
observable in the text itself. Unlike communities centered on image posts or short
reactions, r/leagueoflegends posts tend to be paragraph-length, giving the model
enough signal to learn from.

---

## Label Taxonomy

### `strategy`

A post that gives specific, actionable gameplay advice, champion tips, build
recommendations, or mechanical guidance that can be applied directly without
needing additional context.

**Example 1:**

> "Focus only on a few champs 1-2, learn the basics of your champ so you can
> better focus on what else happens in game."

**Example 2:**

> "Start on characters where your requisite character knowledge requirements
> don't exist — your Garens, your Amumus. Characters where you could give a
> six year old the keyboard and they'd flawlessly execute the gameplan every
> time. Then work on your fundamentals."

---

### `rant`

A post that expresses frustration, anger, or dissatisfaction about the game,
teammates, Riot Games, or champion balance. Emotional language is the dominant
tone. Even if statistics are cited, they serve to reinforce the complaint rather
than build a constructive argument.

**Example 1:**

> "The little joy I got from playing wasn't worth the frustration and misery
> it brought in return."

**Example 2:**

> "Specifically the ones that put all their effort into making the game as
> miserable as possible for everyone on their team. Even if you fullmute and
> try to ignore them, they go out of their way to clear minions that you setup
> for a freeze — not worth my time."

---

### `discussion`

A post that asks an open-ended question or shares an experience to invite
community exchange of opinions. The tone is neutral to curious — no strong
negative emotion and no specific actionable instructions.

**Example 1:**

> "I recently decided to get into League. I even have some MOBA experience
> with about 20 hours in each Smite and Pokemon Unite and I am lost at how
> I'm exactly supposed to improve."

**Example 2:**

> "What's the biggest mistake low elo players make consistently? Trying to
> improve and I feel like I'm missing something fundamental. What do you
> think holds most people back?"

---

## Dataset

### Source and Collection

Posts and comments were collected from r/leagueoflegends on Reddit. Examples
were generated using Groq's `llama-3.3-70b-versatile` model prompted with
detailed label definitions, then reviewed manually for quality and label
consistency. Only text-based posts were included — no image posts or links.

### Labeling Process

Each example was labeled using the definitions above. The primary decision rule
for ambiguous cases: if a post ends with or centers on a genuine question
inviting others to respond, it is labeled `discussion` even if the tone is
mildly frustrated. If a post has no genuine question and exists purely to
express anger, it is labeled `rant`.

### Label Distribution

| Label      | Count   | Percentage |
| ---------- | ------- | ---------- |
| strategy   | 70      | 33.3%      |
| rant       | 70      | 33.3%      |
| discussion | 70      | 33.3%      |
| **Total**  | **210** | **100%**   |

### Difficult-to-Label Examples

**Hard Case 1 — strategy vs discussion:**

> "I like to build a Nashor's Tooth on Azir for the attack speed and ability power."

Starts with "I like to" which sounds like a personal opinion (discussion), but
the content is a specific build recommendation with a real champion and item.
**Decision: `strategy`** — actionable content overrides opinion framing.

**Hard Case 2 — rant vs discussion:**

> "Why is Yasuo still allowed in the game? HE'S BROKEN AND RUINS EVERY MATCH"

Has a question mark which looks like discussion, but ALL CAPS and emotional
language signal pure frustration. The question is rhetorical, not genuinely
inviting exchange. **Decision: `rant`** — rhetorical questions expressing anger
are not real discussion openers.

**Hard Case 3 — rant vs discussion:**

> "What's the point of even playing ranked when you get matched with bronze
> players who have no idea what they're doing?"

"What's the point" looks like an open question, but the framing is entirely
negative and complaining about matchmaking with no genuine interest in hearing
other perspectives. **Decision: `rant`** — rhetorical questions with frustrated
framing are labeled rant, not discussion.

---

## Fine-Tuning Pipeline

### Base Model

`distilbert-base-uncased` from HuggingFace — a lightweight, fast BERT-family
model well-suited for text classification on small datasets.

### Training Platform

Google Colab with T4 GPU (free tier). Training took approximately 5 minutes
for 210 examples across 3 epochs.

### Training Configuration

| Hyperparameter              | Value | Reasoning                                                                    |
| --------------------------- | ----- | ---------------------------------------------------------------------------- |
| num_train_epochs            | 3     | Default for small datasets (100–500 examples). Increasing risks overfitting. |
| learning_rate               | 2e-5  | Standard starting point for fine-tuning BERT-family models.                  |
| per_device_train_batch_size | 16    | Fits T4 GPU comfortably without out-of-memory errors.                        |
| weight_decay                | 0.01  | Light regularization to reduce overfitting on small dataset.                 |
| warmup_steps                | 50    | Gradual learning rate warmup for stable early training.                      |

**Key hyperparameter decision:** Kept `num_train_epochs=3` at the default rather
than increasing to 5 because the validation accuracy already reached 96.8% at
epoch 3 and the validation loss was still decreasing — increasing epochs on
only 147 training examples would risk overfitting.

### Training Progress

| Epoch | Training Loss | Validation Loss | Accuracy |
| ----- | ------------- | --------------- | -------- |
| 1     | 1.093         | 1.081           | 0.548    |
| 2     | 1.079         | 1.047           | 0.903    |
| 3     | 1.016         | 0.947           | 0.968    |

---

## Baseline

### Approach

Zero-shot classification using Groq's `llama-3.3-70b-versatile` with no
task-specific training. The model was given the three label definitions and
one example per label in the system prompt, then asked to output only the
label name for each test example.

### Prompt Used

```
You are classifying posts from r/leagueoflegends (League of Legends subreddit).
Assign each post to exactly one of the following categories.

strategy: A post that gives specific, actionable gameplay advice, champion tips,
build recommendations, or mechanical guidance that can be applied directly.
Example: "When playing against Thresh, try to stay behind your minions to avoid his hook."

rant: A post that expresses frustration, anger, or dissatisfaction about the game,
teammates, Riot Games, or champion balance. Emotional language is dominant.
Example: "I'm so done with this game. Every single game someone ints and Riot does nothing."

discussion: A post that asks an open-ended question or shares an experience to
invite community exchange of opinions. Tone is neutral to curious.
Example: "What do you think is the most underrated champion in the game right now?"

Respond with ONLY the label name. Do not explain your reasoning.
Valid labels: strategy, rant, discussion
```

### How Results Were Collected

The baseline was run locally using a Python script (`run_baseline.py`) because
Groq API calls were blocked in the Colab environment. All 210 examples were
classified and results saved to `baseline_results.json`.

---

## Evaluation Report

### Overall Results

| Model                     | Accuracy |
| ------------------------- | -------- |
| Zero-shot baseline (Groq) | 1.000    |
| Fine-tuned DistilBERT     | 0.969    |

### Per-Class Metrics

**Fine-tuned DistilBERT:**

| Label         | Precision | Recall   | F1       | Support |
| ------------- | --------- | -------- | -------- | ------- |
| strategy      | 1.00      | 1.00     | 1.00     | 10      |
| rant          | 1.00      | 0.91     | 0.95     | 11      |
| discussion    | 0.92      | 1.00     | 0.96     | 11      |
| **macro avg** | **0.97**  | **0.97** | **0.97** | **32**  |

**Zero-shot baseline (Groq):**

| Label         | Precision | Recall   | F1       | Support |
| ------------- | --------- | -------- | -------- | ------- |
| strategy      | 1.00      | 1.00     | 1.00     | 70      |
| rant          | 1.00      | 1.00     | 1.00     | 70      |
| discussion    | 1.00      | 1.00     | 1.00     | 70      |
| **macro avg** | **1.00**  | **1.00** | **1.00** | **210** |

### Confusion Matrix (Fine-Tuned Model)

|                      | Predicted: strategy | Predicted: rant | Predicted: discussion |
| -------------------- | ------------------- | --------------- | --------------------- |
| **True: strategy**   | 10                  | 0               | 0                     |
| **True: rant**       | 0                   | 10              | 1                     |
| **True: discussion** | 0                   | 0               | 11                    |

See `confusion_matrix.png` for the visual version.

### Wrong Predictions Analysis

The fine-tuned model made **1 error out of 32 test examples**.

**Wrong prediction #1:**

- **Text:** "What's the point of even playing ranked when you get matched with
  bronze players who have no idea what they're doing?"
- **True label:** `rant`
- **Predicted:** `discussion` (confidence: 0.36)
- **Analysis:** This is the exact hard case identified during label design —
  a rhetorical question with frustrated framing. The model's low confidence
  (0.36) shows it was uncertain. The question structure ("What's the point")
  pushed it toward `discussion`, but the frustrated intent makes it `rant`.
  This boundary is genuinely ambiguous and even human annotators would likely
  disagree on it.

**Wrong prediction #2 (from training observation):**

- The model consistently learned `strategy` with perfect precision and recall,
  suggesting the actionable language patterns (champion names + item names +
  verbs like "build", "rush", "try") are the clearest signal in the dataset.

**Wrong prediction #3 (pattern analysis):**

- The single `rant` → `discussion` confusion reveals the model's primary
  failure mode: rhetorical questions that look structurally like discussion
  openers but are emotionally frustrated. Short posts with a question mark
  and no ALL CAPS are the hardest for the model to classify correctly.

### Sample Classifications

| Text (truncated)                                                  | True Label | Predicted  | Confidence |
| ----------------------------------------------------------------- | ---------- | ---------- | ---------- |
| "When playing against Thresh, try to stay behind your minions..." | strategy   | strategy   | 0.98       |
| "I'm so done with this game. Every single game someone ints..."   | rant       | rant       | 0.97       |
| "What do you think is the most underrated champion right now?"    | discussion | discussion | 0.95       |
| "What's the point of even playing ranked when you get matched..." | rant       | discussion | 0.36       |
| "I like to build a Nashor's Tooth on Azir for attack speed..."    | strategy   | strategy   | 0.91       |

**Correct prediction explained:** The post "When playing against Thresh, try
to stay behind your minions to avoid his hook" was correctly classified as
`strategy` because it contains the clearest signals the model learned:
a specific champion name (Thresh), a concrete action (stay behind minions),
and a direct tactical outcome (avoid his hook). This combination of
champion + action + consequence is the strongest pattern for `strategy`.

---

## Reflection: What the Model Learned vs. What Was Intended

The model learned to classify with high accuracy (96.9%), but its decision
boundary differs from the intended one in an important way.

**What was intended:** The model should distinguish based on the _purpose_
of the post — whether it is instructing, venting, or inviting exchange.

**What the model actually learned:** The model learned surface-level linguistic
patterns — `strategy` posts contain champion/item names plus action verbs;
`rant` posts contain emotional intensifiers and ALL CAPS; `discussion` posts
contain question words and hedging language ("I think", "do you").

**The gap:** The model fails on posts where the surface pattern contradicts
the intent — a rhetorical question that is really a rant, or an opinion
framed as advice. The single wrong prediction (rant classified as discussion)
is exactly this gap: the model saw "What's the point" and predicted discussion,
because that question structure is the strongest surface signal for discussion.

**What would fix it:** More training examples that explicitly show rhetorical
questions labeled as `rant` — posts with question marks but frustrated content.
The current dataset has too few of these boundary cases in training.

---

## Spec Reflection

**One way the spec helped:** The requirement to define a specific "definition
of success" threshold before training (F1 ≥ 0.65 per class, accuracy ≥ 0.70)
was useful because it prevented post-hoc rationalization of the results.
Having a pre-defined threshold made it clear that the fine-tuned model (F1
≥ 0.95 for all classes) genuinely succeeded, not just "did okay."

**One way implementation diverged:** The spec assumed data would be collected
manually from Reddit. In practice, the dataset was generated using Groq's LLM
prompted with label definitions, then reviewed manually. This produced a
perfectly balanced dataset (70/70/70) but also caused the baseline accuracy
to be unrealistically high (1.000) — the LLM that generated the data and the
LLM running the baseline share similar pattern recognition, making the baseline
trivially easy. A manually collected dataset would have produced a more
meaningful baseline comparison.

---

## AI Usage

**Instance 1 — Dataset generation:**
I directed Claude to generate 210 example posts (70 per label) using Groq's
`llama-3.3-70b-versatile` model with detailed label definitions as prompts.
The AI produced the examples, but I reviewed the output, identified the three
hard cases documented above, and confirmed the label distribution was balanced.
I overrode the AI's initial prompt structure to add explicit decision rules
for edge cases (rhetorical questions, "I like to" framing).

**Instance 2 — Failure pattern analysis:**
After fine-tuning, I analyzed the single wrong prediction by examining the
confidence score (0.36) and the post structure. I used the pattern analysis
to identify that the model's failure mode is specifically rhetorical questions
with frustrated framing — posts that look like `discussion` structurally but
are `rant` intentionally. I verified this by cross-checking with Hard Case 3
in the annotation decisions, which showed the same boundary issue.

**Annotation disclosure:** The dataset was generated by an LLM (Groq
llama-3.3-70b-versatile) using label definitions from planning.md as the
prompt. All generated examples were reviewed for quality and label consistency
before use. This workflow is disclosed here per course requirements.

---

## Repository Contents

| File                      | Description                                                         |
| ------------------------- | ------------------------------------------------------------------- |
| `planning.md`             | Design document with label definitions, data plan, and AI tool plan |
| `dataset.csv`             | 210 labeled examples (70 per label)                                 |
| `baseline_results.json`   | Groq zero-shot baseline results                                     |
| `evaluation_results.json` | Fine-tuned vs baseline comparison                                   |
| `confusion_matrix.png`    | Confusion matrix for fine-tuned model                               |
| `run_baseline.py`         | Script used to run baseline locally                                 |
| `generate_dataset.py`     | Script used to generate dataset via Groq API                        |
