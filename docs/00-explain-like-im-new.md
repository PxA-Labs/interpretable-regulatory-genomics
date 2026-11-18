# 00 — Explain Like I'm New: A Beginner-Friendly Introduction to the Project

## Welcome!

Welcome to the **Interpretable Regulatory Genomics** project.

If you're new to genomics, machine learning, or open-source research projects, don't worry—this document is your starting point.

The goal of this guide is to explain **what this project does, why it matters, and how everything fits together**, without assuming any prior biology knowledge.

---

## 1. What Problem Are We Solving?

Imagine a book with **3 billion letters**.

That book is the **human genome (DNA)**.

Only a very small portion of those letters directly contains instructions for building proteins.

The rest may look unimportant at first, but hidden within it are tiny regions that act like **control switches**.

These switches decide:
- When a gene should turn **ON**
- When a gene should turn **OFF**
- How strongly a gene should be expressed

Our project builds an Artificial Intelligence (AI) model that examines a short DNA sequence and answers one important question:

**"Is this DNA sequence a regulatory switch, or is it simply ordinary background DNA?"**

---

## 2. A Simple Analogy

Imagine standing in front of a **very long wall** covered completely with wallpaper.

Hidden behind the wallpaper are a few **light switches**.

From the outside, every part of the wall looks almost identical.

Your task is to walk along the wall and identify where the hidden switches are located without removing the wallpaper.

This is exactly what our AI model tries to do.

| Real World | Our Project |
| :--- | :--- |
| Wall | DNA |
| Hidden light switches | Regulatory DNA elements |
| Person searching | AI Model |

Most of the wall is simply wallpaper. Only a few small regions actually hide a switch. Finding those switches is the challenge.

---

## 3. What Does the AI Receive?

The model receives a short DNA sequence.

Example:
`ATCGGCTAACGTTAGCG...`

The sequence is simply a long string made from four letters: **A**, **T**, **C**, **G**.

The model studies the pattern of these letters to determine whether the sequence behaves like a regulatory element.

---

## 4. What Does the Model Predict?

The workflow is conceptually very simple.

```text
DNA Sequence
   │
   ▼
AI analyzes patterns
   │
   ▼
Is this a regulatory switch?
   │
 ┌─┴─┐
 ▼   ▼
Yes  No
```

If the answer is **Yes**, the sequence is likely involved in controlling nearby genes.  
If the answer is **No**, the sequence is most likely ordinary background DNA.

---

## 5. Why Is This Important?

Only about **2%** of the human genome directly codes for proteins. The remaining **98%** contains many regions responsible for controlling gene activity.

Researchers have discovered that many disease-associated genetic variants occur inside these regulatory regions.

Being able to identify these regions helps scientists better understand diseases such as:
- Cancer
- Diabetes
- Heart disease
- Neurological disorders

Instead of manually examining millions of DNA sequences, AI can quickly predict which regions deserve further biological investigation.

---

## 6. How Does This Repository Approach the Problem?

Rather than relying on a single model, this repository explores multiple approaches:

```text
Baseline Model
   │
   ▼
Deep Learning Model
   │
   ▼
Pretrained DNA Language Model
```

Each new model aims to improve prediction accuracy while remaining interpretable and scientifically meaningful.

---

## 7. Repository Learning Path

If you're new, we recommend reading the documentation in the following order:

1. **00 – Explain Like I'm New** *(this document)*
2. **01 – Project Charter**
3. **11 – Glossary & Project Memory**
4. **12 – Contributor Onboarding**
5. **06 – Modeling Roadmap**

This order moves from simple concepts to technical implementation.

---

## 8. Do I Need a Biology Background?

**No.**

Many valuable contributions require only software engineering or machine learning experience. You can contribute by:
- Improving documentation
- Fixing bugs
- Writing tests
- Improving Python code
- Building visualization tools
- Optimizing machine learning models
- Improving data pipelines

If you're comfortable with Python, Git, or machine learning, you already have skills that can help this project.

---

## Key Takeaways

- The project studies **non-coding DNA**.
- Some DNA regions act as **regulatory switches** that control genes.
- The AI model predicts whether a DNA sequence contains one of these switches.
- Multiple machine learning models are explored and compared.
- Contributors from software engineering, data science, and AI backgrounds are welcome—even without prior genomics experience.

---

## Welcome to the Project!

We hope this guide gives you a clear understanding of the project's purpose.

Once you're comfortable with the big picture, continue with the [Project Charter](01-project-charter.md) to explore the technical goals and implementation details.

**Happy contributing! 🚀**
