# e-GMAT Adaptive Practice Environment (P.A.C.E. Prototype)

## Project Overview

This project is a Python-based desktop application designed to simulate an **Adaptive Practice Environment** for GMAT preparation, directly inspired by **e-GMAT's P.A.C.E. (Personalized Adaptive Course Engine)** philosophy. Built in 7 days and iteratively pitched over 10 days, the prototype emulates e-GMAT's commitment to adaptive learning, behavior-aware coaching, and AI-powered feedback.

It incorporates:

* A timed, randomized, multiple-choice GMAT quiz interface.
* Real-time behavioral analytics using webcam emotion detection (via DeepFace and OpenCV).
* Hyper-personalized, question-level insights and session debriefs powered by **Google Gemini AI**.
* Visual and PDF reports with performance metrics and behavioral trend summaries.
* A scalable, modular architecture designed for future RAG-based dynamic question delivery.

---

## Architecture Overview

Inspired by multi-agent HR automation systems, the architecture fuses behavioral analysis with real-time AI feedback in a modular pipeline:

**System Flow:**

1. **Webcam Input** ➔ DeepFace Emotion Analysis (Frustration, Confusion, Confidence)
2. **Behavioral Pipeline** ➔ Aggregates engagement metrics (frame-level)
3. **Gemini AI Agent** ➔ Generates per-question and overall feedback
4. **Tkinter GUI** ➔ Presents live webcam feed, questions, and pacing visuals
5. **JSON / Pinecone** ➔ Stores learner sessions (offline or vector DB future-ready)
6. **PDF Generator** ➔ Compiles report via ReportLab

Future-ready with RAG-enhanced architecture for real-time, personalized content delivery using **Pinecone embeddings**.

---

## Core Philosophies & Alignment with e-GMAT

| e-GMAT Philosophy           | Prototype Implementation                                |
| --------------------------- | ------------------------------------------------------- |
| P.A.C.E. Adaptive Coaching  | Real-time behavior feedback, AI debriefs                |
| AI-Driven Personalization   | Gemini 2.0 insights per question                        |
| Data-Driven Decision-Making | Performance + behavioral trend visualizations           |
| Learner Confidence & Pacing | Tracks timing + emotion scores, recommends improvements |
| Continuous Innovation       | Multi-agent pipeline, future RAG integration            |

---

## Key Features

### 🔵 Learner Onboarding & Progress Persistence

* Candidate data stored in `egmat_candidate_registry.json`
* Session history stored per learner (`egmat_pacer_report_{id}.json`)

### 🔵 Adaptive Quiz Engine

* Questions loaded from `egmat.json` (fallback enabled)
* Randomized questions and answer orders
* Pacing tracked per question

### 🔵 Real-Time Emotion Analysis (Experimental)

* `DeepFace` + OpenCV detects:

  * Frustration (anger/sadness proxies)
  * Confusion (neutral/surprise proxies)
  * Confidence (happiness proxy)
* Simulated fallback if camera fails

### 🔵 Gemini AI-Powered Insights

* Per-question analysis:

  * Correctness feedback
  * Explanation and traps
  * Micro-recommendations (P.A.C.E. styled)
* Comprehensive session summary:

  * Performance snapshot
  * Behavioral metrics
  * Suggested next steps

### 🔵 Report Generation

* Visualizations using `matplotlib`:

  * Accuracy per topic
  * Pacing trends
  * Behavioral metrics over time
* Final output as downloadable PDF via `reportlab`

---

## Tech Stack

* **Language:** Python 3.12.4
* **UI:** Tkinter + ttk
* **Emotion Detection:** OpenCV, DeepFace
* **AI:** Google Gemini (`google-generativeai`)
* **Storage:** JSON (local), ready for Pinecone vector DB
* **Reporting:** Matplotlib, ReportLab
* **Env Mgmt:** `python-dotenv`

---

## Setup Instructions

1. **Install Python 3.12.4**

2. **Clone Repository**

```bash
git clone https://github.com/arshad0220/Fair-Hire/
cd Fair-Hire
```

3. **Virtual Environment (Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

4. **Install Dependencies**
   Create a `requirements.txt` like:

```txt
opencv-python
deepface
tensorflow
matplotlib
numpy
Pillow
reportlab
google-generativeai
python-dotenv
```

Then:

```bash
pip install -r requirements.txt
```

5. **Set Environment Variables**
   Create a `.env` file:

```env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

6. **Prepare Question Bank**
   Format `egmat.json` like:

```json
[
  {
    "question_id": 1,
    "question": "Sample SC question",
    "options": ["A", "B", "C", "D"],
    "answer": "A",
    "topic": "SC",
    "subtopic": "Modifiers",
    "difficulty": "Medium"
  }
]
```

Path defaults to: `C:\Users\HI\Desktop\fair hire\egmat.json`

7. **Run Application**

```bash
python your_script_name.py
```

---

## Limitations & Fallbacks

* Camera not detected ➔ Simulated behavioral metrics
* Gemini API error ➔ Placeholder AI responses
* File errors ➔ Fallback question set
* DeepFace accuracy depends on lighting and face visibility

---

## Roadmap & Future Enhancements

* ✅ Scale to 50+ questions with pacing + fallback logic
* 🔄 RAG Architecture: Pinecone + Gemini for question selection
* 📊 Dashboard: Long-term trends (Study Streak Ring style)
* 📚 Skill-linked Content: Auto-recommendation of concepts
* 🤖 AI Tutor: Interactive Gemini chatbot for doubts

---

## Disclaimer

This prototype is not affiliated with e-GMAT. It is inspired by their P.A.C.E. framework and represents a research-driven, experimental build to explore the future of adaptive AI-driven learning tools.

---

## Reference

* GitHub: [https://github.com/arshad0220/Fair-Hire](https://github.com/arshad0220/Fair-Hire)
* Author: Arshad Ahamed – AI Practitioner | Bangalore
