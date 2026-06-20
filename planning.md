# TakeMeter — Planning Document

## Project 3 | AI201 | r/leagueoflegends Discourse Classifier

---

## 1. Community Choice

**Community:** r/leagueoflegends (Reddit)

r/leagueoflegends is one of the largest gaming subreddits with over 6 million members.
The community produces a constant stream of text-heavy posts ranging from detailed
gameplay advice to emotional venting to open-ended debates. This variety makes it
an ideal fit for a classification task — the discourse is active enough to collect
200+ examples easily, and the differences between post types are meaningful and
observable in the text itself. Unlike communities centered on image posts or short
reactions, leagueoflegends posts tend to be paragraph-length, giving the model
enough signal to learn from.

---

## 2. Label Taxonomy

### Label 1: `strategy`

**Definition:** A post or comment that shares specific, actionable gameplay advice —
including tips, builds, champion guides, or mechanical analysis — written to help
others improve. The content can stand alone as useful instruction without needing
the surrounding thread for context.

**Example 1** (real post from r/leagueoflegends):

> "Focus only on a few champs 1-2, learn the basics of your champ so you can
> better focus on what else happens in game."

**Example 2** (real post from r/leagueoflegends):

> "Start on characters where your requisite character knowledge requirements
> don't exist — your Garens, your Amumus. Characters where you could give a
> six year old the keyboard and they'd flawlessly execute the gameplan every
> time. Then work on your fundamentals."

---

### Label 2: `rant`

**Definition:** A post or comment where the primary purpose is expressing frustration,
anger, or dissatisfaction — about the game, teammates, Riot Games, or the meta.
Emotional language is the dominant tone. Even if statistics or specific examples
are cited, they serve to reinforce the complaint rather than build a constructive argument.

**Example 1** (real post from r/leagueoflegends):

> "The little joy I got from playing wasn't worth the frustration and misery
> it brought in return."

**Example 2** (real post from r/leagueoflegends):

> "Specifically the ones that put all their effort into making the game as
> miserable as possible for everyone on their team. Even if you fullmute and
> try to ignore them, they go out of their way to clear minions that you setup
> for a freeze — not worth my time."

---

### Label 3: `discussion`

**Definition:** A post or comment that opens a question or shares an experience to
invite community exchange of opinions. The tone is neutral to curious — the author
is not primarily venting frustration nor providing actionable instruction, but
seeking or contributing to a broader conversation.

**Example 1** (real post from r/leagueoflegends):

> "I recently decided to get into League. I even have some MOBA experience
> with about 20 hours in each Smite and Pokemon Unite and I am lost at how
> I'm exactly supposed to improve."

**Example 2** (real post from r/leagueoflegends):

> "What's the biggest mistake low elo players make consistently? Trying to
> improve and I feel like I'm missing something fundamental. What do you
> think holds most people back?"

---

## 3. Hard Edge Cases

**Hardest anticipated edge case:** A post that asks a question about improvement
but includes frustrated language — sitting between `discussion` and `rant`.

**Example:**

> "I'm so tired of losing — why can't anyone in low elo just play objectives?
> What's the point of winning lane if your team throws every time?"

This post complains (rant signal) but also implicitly invites discussion about
low elo behavior (discussion signal).

**Decision rule:** If the post ends with or centers on a question inviting others
to respond, label it `discussion` — even if the tone is frustrated. If the post
has no genuine question and exists purely to express anger with no invitation for
exchange, label it `rant`. The presence of a real question mark seeking input is
the deciding factor.

**Second edge case:** A comment giving advice inside a discussion thread — sitting
between `strategy` and `discussion`.

**Decision rule:** Label by the content of the individual post/comment, not the
thread. If the comment gives specific, standalone actionable advice, label it
`strategy`. If it shares an opinion or experience without actionable instruction,
label it `discussion`.

---

## 4. Data Collection Plan

**Source:** r/leagueoflegends on Reddit — public posts and top-level comments only.

**Collection method:** Manual copy-paste into a CSV file. Each row contains the
text of one post or comment and its assigned label.

**Target distribution:**

- `strategy`: ~70 examples (35%)
- `rant`: ~65 examples (32.5%)
- `discussion`: ~65 examples (32.5%)

**Where to find each label:**

- `strategy`: Search "tip", "guide", "how to", "build" on r/leagueoflegends
- `rant`: Search "frustrated", "why is", "Riot please", "I quit" on r/leagueoflegends
- `discussion`: Search "what do you think", "unpopular opinion", "thoughts on"

**If a label is underrepresented after 150 examples:** Collect an additional
targeted batch by searching keywords specific to that label before moving on.
No single label will exceed 70% of the total dataset.

---

## 5. Evaluation Metrics

**Primary metrics:**

- **Per-class F1 score** — the most important metric for this task because
  the three labels have different difficulty levels. A high overall accuracy
  could hide the model completely failing on one label. F1 balances precision
  and recall per class and surfaces those failures.
- **Overall accuracy** — used for baseline comparison only; not sufficient alone.
- **Confusion matrix** — to identify which specific label pairs the model
  confuses most (e.g., does it mix up `rant` and `discussion` more than
  `strategy` and `rant`?).

**Why accuracy alone is not enough:** If 40% of examples are `discussion`, a
model that always predicts `discussion` would achieve 40% accuracy while
learning nothing. Per-class F1 catches this immediately.

---

## 6. Definition of Success

The classifier is considered good enough for deployment in a real community
tool if it meets **all three** of the following thresholds on the test set:

- Per-class F1 ≥ 0.65 for every label (no label is completely failing)
- Overall accuracy ≥ 0.70
- Fine-tuned model outperforms the Groq zero-shot baseline on overall accuracy
  by at least 5 percentage points

If the fine-tuned model fails to beat the baseline, that is a signal to
investigate label consistency or data quality before accepting the results.

---

## 7. AI Tool Plan

### Label stress-testing

Before annotating 200 examples, I will paste my three label definitions into
Claude and ask it to generate 10 posts that sit at the boundary between two
labels. If Claude produces posts I cannot cleanly classify, I will tighten
the definitions before starting annotation.

### Annotation assistance

I will not use an LLM to pre-label examples. All 200 examples will be labeled
manually using the definitions above. This avoids introducing systematic
bias from the LLM's interpretation of the labels into the training data.

### Failure analysis

After fine-tuning, I will paste the list of misclassified test examples into
Claude and ask it to identify common patterns — such as post length, specific
label pairs being confused, or presence of sarcasm. I will then verify those
patterns by re-reading the examples myself before including the analysis in
the evaluation report.

---

## 8. Hard Annotation Decisions (updated during Milestone 3)

_(This section will be filled in during data collection with at least 3
specific examples that were genuinely difficult to label and the decisions made.)_
