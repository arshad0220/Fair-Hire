import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import threading
import time
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image, ImageTk
import sys
from deepface import DeepFace
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
import google.generativeai as genai
import os
from dotenv import load_dotenv
from reportlab.lib.enums import TA_LEFT

load_dotenv()

AI_API_KEY = os.getenv("GEMINI_API_KEY")

RECOMMENDED_TIME_VERBAL = 1.96 * 60
RECOMMENDED_TIME_QUANT = 2.14 * 60
TOTAL_QUESTIONS = 20
QUESTIONS_FILE = r'C:\Users\HI\Desktop\fair hire\egmat.json'
CANDIDATES_FILE = "egmat_candidate_registry.json"
REPORT_FILE_TEMPLATE = "egmat_pacer_report_{}.json"
EMOTIONAL_STRAIN_THRESHOLD = 50
CONCEPTUAL_UNCLEARNESS_THRESHOLD = 50
ACCURACY_WEAKNESS_THRESHOLD = 0.6
ACCURACY_OPPORTUNITY_THRESHOLD = 0.7
BEHAVIORAL_CONFIDENCE_STRENGTH_THRESHOLD = 70
PERCEIVED_CONFIDENCE_RECOMMENDATION_THRESHOLD = 60
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

genai_model = None
if AI_API_KEY:
    try:
        genai.configure(api_key=AI_API_KEY)
        print("e-GMAT AI: Attempting to initialize Generative AI model...")
        try:
           genai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
           print("e-GMAT AI: Generative AI model 'gemini-1.5-flash-latest' initialized successfully for insights.")
        except Exception as e:
            print(f"e-GMAT AI: Warning: Failed to initialize 'gemini-1.5-flash-latest', trying 'gemini-2.0-flash'. Error: {e}")
            try:
                genai_model = genai.GenerativeModel('gemini-2.0-flash')
                print("e-GMAT AI: Generative AI model 'gemini-2.0-flash' initialized.")
            except Exception as e_flash:
                 print(f"e-GMAT AI: Error: Failed to initialize any suitable Gemini model. Advanced AI features will be limited or unavailable. Error: {e_flash}")
                 genai_model = None
    except Exception as e:
        print(f"e-GMAT AI: Error initializing Google Generative AI: {e}")
        print("e-GMAT AI: AI-powered features will be disabled.")
        genai_model = None
else:
    print("e-GMAT AI: Error: GEMINI_API_KEY not found in .env file or environment variables. AI features will be disabled.")
    genai_model = None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, np.ndarray):
             return obj.tolist()
        return super().default(obj)

class AppState:
    def __init__(self):
        self.page = 'home'
        self.behavioral_metrics = {"frustration": 1.0, "confusion": 1.0, "confidence": 1.0}
        self.behavioral_history = []
        self.behavioral_thread_running = False
        self.session_data = None
        self.engagement_score = 0
        self.report_registry = {}
        self.learner_id = None
        self.learner_name = None
        self.current_question = 0
        self.responses = []
        self.question_start_time = None
        self.question_set = None
        self.performance_analysis = None
        self.session_visualizations = None
        self.pdf_report_data = None
        self.camera_capture = None
        self.ai_comprehensive_analysis_text = None
        self.state_lock = threading.Lock()

state = AppState()

def initialize_camera(tried_indices=range(5)):
    print("e-GMAT Camera: Attempting to initialize learner engagement camera feed...")
    for index in tried_indices:
        print(f"e-GMAT Camera: Trying index {index}...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"e-GMAT Camera: Capture device opened successfully on index {index}.")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                return cap
            else:
                print(f"e-GMAT Camera: Device opened on index {index} but failed to capture frame.")
                cap.release()
        else:
            print(f"e-GMAT Camera: Failed to open capture device on index {index}.")
            cap.release()
    print("e-GMAT Camera: No capture device could be opened after trying all specified indices.")
    return None

def detect_behavioral_metrics():
    print("e-GMAT Behavioral: Detection thread started.")
    while state.behavioral_thread_running:
        frame = None
        if state.camera_capture and state.camera_capture.isOpened():
            ret, frame = state.camera_capture.read()
            if not ret:
                print("e-GMAT Behavioral: Failed to capture frame in detection thread.")
                frame = None
        current_metrics = None
        if frame is not None:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
                if isinstance(results, list) and results:
                    total_strain = total_uncl = total_behavioral_conf = 0
                    count = len(results)
                    for result in results:
                         emotions = result['emotion']
                         total_strain += (emotions.get('angry', 0) * 0.6 + emotions.get('disgust', 0) * 0.4)
                         total_uncl += (emotions.get('surprise', 0) * 0.5 + emotions.get('sad', 0) * 0.5)
                         total_behavioral_conf += emotions.get('happy', 0)
                    current_metrics = {
                        "frustration": round(total_strain / count, 2),
                        "confusion": round(total_uncl / count, 2),
                        "confidence": round(total_behavioral_conf / count, 2)
                    }
                else:
                    print("e-GMAT Behavioral: No face detected or DeepFace results empty. Using simulation fallback.")
                    current_metrics = None
            except Exception as e:
                print(f"e-GMAT Behavioral: DeepFace analysis error: {e}. Using simulation fallback.")
                current_metrics = None
        if current_metrics is None:
             decay_rate = 0.9
             sim_frust = state.behavioral_metrics.get('frustration', 50) * decay_rate + np.random.uniform(1, 100) * (1-decay_rate)
             sim_confu = state.behavioral_metrics.get('confusion', 50) * decay_rate + np.random.uniform(1, 100) * (1-decay_rate)
             sim_conf = state.behavioral_metrics.get('confidence', 50) * decay_rate + np.random.uniform(1, 100) * (1-decay_rate)
             current_metrics = {
                "frustration": max(0, min(100, round(sim_frust, 2))),
                "confusion": max(0, min(100, round(sim_confu, 2))),
                "confidence": max(0, min(100, round(sim_conf, 2)))
             }
             if frame is None:
                  print("e-GMAT Behavioral: (Simulating metrics due to camera unavailability or frame read failure)")
        with state.state_lock:
            state.behavioral_metrics = current_metrics.copy()
            state.behavioral_history.append(state.behavioral_metrics.copy())
            engagement_raw = state.behavioral_metrics['confidence'] - (state.behavioral_metrics['frustration'] + state.behavioral_metrics['confusion']) / 2
            engagement_score = int(((engagement_raw + 100) / 200) * 10)
            state.engagement_score = max(0, min(10, engagement_score))
        time.sleep(0.5)
    print("e-GMAT Behavioral: Detection thread stopped.")
    if state.camera_capture:
         print("e-GMAT Camera: Releasing capture device...")
         state.camera_capture.release()
         state.camera_capture = None
         print("e-GMAT Camera: Capture device released.")

def load_and_randomize_questions():
    questions = []
    try:
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        if not isinstance(questions, list) or not questions:
             print(f"e-GMAT Questions: Warning: {QUESTIONS_FILE} is empty or not a list. Using fallback questions.")
             questions = []
    except FileNotFoundError:
        print(f"e-GMAT Questions: Error: Questions file not found at {QUESTIONS_FILE}. Using fallback questions.")
        questions = []
    except json.JSONDecodeError:
        print(f"e-GMAT Questions: Error: Could not decode JSON from {QUESTIONS_FILE}. Using fallback questions.")
        questions = []
    if not questions:
        print("e-GMAT Questions: Using built-in fallback questions.")
        return [
            {
                "question_id": i + 1,
                "question": f"Sample {['SC', 'CR', 'RC', 'Algebra', 'Geometry'][i % 5]} question {i + 1}: This is a placeholder question text.",
                "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
                "answer": "A",
                "topic": ["SC", "CR", 'RC', "Algebra", "Geometry"][i % 5],
                "subtopic": ["Modifiers", "Assumption", "Main Idea", "Equations", "Triangles"][i % 5],
                "difficulty": ["Easy", "Medium", "Hard"][i % 3]
            } for i in range(TOTAL_QUESTIONS)
        ]
    random.shuffle(questions)
    processed_questions = []
    for i, q in enumerate(questions):
        if not all(key in q for key in ['question', 'options', 'answer', 'topic', 'subtopic', 'difficulty']):
            print(f"e-GMAT Questions: Warning: Skipping question {q.get('question_id', i+1)} due to missing keys.")
            continue
        q['question_id'] = i + 1
        original_options = q.get('options', [])
        original_correct_char = q.get('answer', '').strip()
        if not isinstance(original_options, list) or len(original_options) != 4:
             print(f"e-GMAT Questions: Warning: Question {q['question_id']} has invalid options list (expected 4). Skipping.")
             continue
        if original_correct_char not in ['A', 'B', 'C', 'D']:
             print(f"e-GMAT Questions: Warning: Question {q['question_id']} has invalid correct answer character '{original_correct_char}'. Expected A, B, C, or D. Skipping.")
             continue
        original_correct_text_value = None
        original_correct_index = -1
        for idx, option_string in enumerate(original_options):
            if isinstance(option_string, str) and option_string.strip().startswith(original_correct_char + '.'):
                 original_correct_text_value = option_string.strip().split('.', 1)[-1].strip()
                 original_correct_index = idx
                 break
        if original_correct_text_value is None:
             print(f"e-GMAT Questions: Warning: Question {q['question_id']} - could not find original option starting with '{original_correct_char}. '. Skipping.")
             continue
        shuffled_indices = list(range(4))
        random.shuffle(shuffled_indices)
        new_options = []
        new_correct_index = -1
        for new_idx, original_idx in enumerate(shuffled_indices):
            option_string_at_original_position = original_options[original_idx]
            cleaned_opt_value = option_string_at_original_position.strip().split('.', 1)[-1].strip()
            new_option_char = chr(65 + new_idx)
            new_option_string = f"{new_option_char}. {cleaned_opt_value}"
            new_options.append(new_option_string)
            if original_idx == original_correct_index:
                 new_correct_index = new_idx
        if new_correct_index == -1:
             print(f"e-GMAT Questions: Internal Logic Error: Question {q['question_id']} - Failed to find new correct index after shuffling. Skipping.")
             continue
        q['options'] = new_options
        q['answer'] = chr(65 + new_correct_index)
        processed_questions.append(q)
    if not processed_questions:
        print("e-GMAT Questions: Critical Error: No valid questions available after loading and processing. Cannot start test.")
        return None
    print(f"e-GMAT Questions: Loaded and randomized {len(processed_questions)} questions.")
    return processed_questions

def load_learner_report(learner_id):
    report_key = str(learner_id)
    filename = REPORT_FILE_TEMPLATE.format(report_key)
    try:
        with open(filename, "r", encoding='utf-8') as f:
            state.report_registry[report_key] = json.load(f)
            print(f"e-GMAT Report: Loaded existing report for learner ID {learner_id}")
    except FileNotFoundError:
        print(f"e-GMAT Report: No existing report found for learner ID {learner_id}. Initializing new report structure.")
        state.report_registry[report_key] = {"learner_id": report_key, "name": "", "sessions": []}
    except json.JSONDecodeError:
         print(f"e-GMAT Report: Error decoding JSON report for learner ID {learner_id}. Starting with new report structure.")
         state.report_registry[report_key] = {"learner_id": report_key, "name": "", "sessions": []}
    except Exception as e:
         print(f"e-GMAT Report: An unexpected error occurred while loading report for learner ID {learner_id}: {e}")
         state.report_registry[report_key] = {"learner_id": report_key, "name": "", "sessions": []}
    return state.report_registry[report_key]

def save_learner_report(learner_id, report):
    report_key = str(learner_id)
    filename = REPORT_FILE_TEMPLATE.format(report_key)
    try:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(report, f, indent=4, cls=NumpyEncoder)
        print(f"e-GMAT Report: Saved report data for learner ID {learner_id}")
    except Exception as e:
        print(f"e-GMAT Report: Error saving report for learner ID {learner_id}: {e}")

def load_candidate_registry():
    candidates = {"candidates": []}
    try:
        with open(CANDIDATES_FILE, "r", encoding='utf-8') as f:
            candidates = json.load(f)
            if not isinstance(candidates, dict) or "candidates" not in candidates or not isinstance(candidates["candidates"], list):
                 print(f"e-GMAT Registry: Warning: {CANDIDATES_FILE} has incorrect format. Starting with empty registry.")
                 candidates = {"candidates": []}
    except FileNotFoundError:
        print(f"e-GMAT Registry: No candidate registry file found at {CANDIDATES_FILE}.")
        candidates = {"candidates": []}
    except json.JSONDecodeError:
        print(f"e-GMAT Registry: Error decoding JSON from {CANDIDATES_FILE}. Starting with empty registry.")
        candidates = {"candidates": []}
    state.report_registry[CANDIDATES_FILE] = candidates
    return candidates.copy()

def save_candidate_registry(candidates):
    try:
        with open(CANDIDATES_FILE, "w", encoding='utf-8') as f:
            json.dump(candidates, f, indent=4)
        print(f"e-GMAT Registry: Saved candidate registry to {CANDIDATES_FILE}")
    except Exception as e:
        print(f"e-GMAT Registry: Error saving candidate registry: {e}")

def update_candidate_registry(learner_id, learner_name):
    learner_id_str = str(learner_id)
    candidates_data = load_candidate_registry()
    learner_exists = any(candidate['user_id'] == learner_id_str for candidate in candidates_data['candidates'])
    if not learner_exists:
        candidates_data['candidates'].append({
            "user_id": learner_id_str,
            "name": learner_name,
            "registration_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        print(f"e-GMAT Registry: Added new learner to registry: {learner_name} (ID: {learner_id_str})")
        save_candidate_registry(candidates_data)

def process_session_results(responses, question_set):
    report = load_learner_report(state.learner_id)
    session_id = len(report.get('sessions', [])) + 1
    session_data = {
        "session_id": session_id,
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "learner_id": state.learner_id,
        "learner_name": state.learner_name,
        "score": 0,
        "total": len(question_set),
        "questions": []
    }
    if not responses or not question_set:
        print("e-GMAT Processing: Warning: No responses or questions to process.")
        return session_data
    for i, resp in enumerate(responses):
        if i >= len(question_set):
             print(f"e-GMAT Processing: Warning: Response {i} has no corresponding question in the set. Skipping.")
             continue
        q = question_set[i]
        selected_option_text = resp.get('response', '').strip()
        correct_answer_char = q.get('answer', '').strip()
        correct = selected_option_text.startswith(correct_answer_char + '.')
        if correct:
            session_data['score'] += 1
        q_behavioral_metrics = resp.get('behavioral_metrics', {"frustration": 50.0, "confusion": 50.0, "confidence": 50.0})
        base_behavioral_confidence = q_behavioral_metrics.get('confidence', 50.0)
        correctness_boost = 100 if correct else 50
        confidence_score = min(100.0, round((base_behavioral_confidence * 0.8 + correctness_boost) / 1.3))
        time_spent = resp.get('time_spent', 0.0)
        try:
             time_spent = float(time_spent)
        except (ValueError, TypeError):
             time_spent = 0.0
        session_data['questions'].append({
            "question_id": q.get('question_id', i + 1),
            "question_text": q.get('question', 'N/A'),
            "options": q.get('options', []),
            "correct_answer": q.get('answer', 'N/A'),
            "topic": q.get('topic', 'Unknown'),
            "subtopic": q.get('subtopic', 'Unknown'),
            "difficulty": q.get('difficulty', 'Unknown'),
            "learner_response": selected_option_text,
            "time_spent": round(time_spent, 1),
            "behavioral_metrics": q_behavioral_metrics,
            "correct": correct,
            "perceived_confidence_score": float(confidence_score),
            "ai_question_insight": resp.get('ai_insight', 'No AI insight available.')
        })
    return session_data

def analyze_session_performance(session_data):
    questions_results = session_data.get('questions', [])
    if not questions_results:
        print("e-GMAT Analysis: No question results found for analysis.")
        return {'summary': {'topics': {}, 'subtopics': {}}, 'recommendations': []}
    topics_agg = {}
    subtopics_agg = {}
    for q in questions_results:
        topic = q.get('topic', 'Unknown Topic')
        subtopic = q.get('subtopic', 'Unknown Subtopic')
        correct = q.get('correct', False)
        time_spent = q.get('time_spent', 0.0)
        behavioral_metrics = q.get('behavioral_metrics', {"frustration": 50.0, "confusion": 50.0, "confidence": 50.0})
        perceived_confidence_score = q.get('perceived_confidence_score', 50.0)
        topics_agg.setdefault(topic, {'correct': 0, 'total': 0, 'time': [], 'behavioral_metrics': {'frustration': [], 'confusion': [], 'confidence': []}, 'perceived_confidence_scores': []})
        subtopics_agg.setdefault(subtopic, {'correct': 0, 'total': 0, 'time': [], 'behavioral_metrics': {'frustration': [], 'confusion': [], 'confidence': []}, 'perceived_confidence_scores': []})
        topics_agg[topic]['total'] += 1
        if correct:
            topics_agg[topic]['correct'] += 1
        topics_agg[topic]['time'].append(float(time_spent))
        topics_agg[topic]['behavioral_metrics']['frustration'].append(behavioral_metrics.get('frustration', 50.0))
        topics_agg[topic]['behavioral_metrics']['confusion'].append(behavioral_metrics.get('confusion', 50.0))
        topics_agg[topic]['behavioral_metrics']['confidence'].append(behavioral_metrics.get('confidence', 50.0))
        topics_agg[topic]['perceived_confidence_scores'].append(float(perceived_confidence_score))
        subtopics_agg[subtopic]['total'] += 1
        if correct:
            subtopics_agg[subtopic]['correct'] += 1
        subtopics_agg[subtopic]['time'].append(float(time_spent))
        subtopics_agg[subtopic]['behavioral_metrics']['frustration'].append(behavioral_metrics.get('frustration', 50.0))
        subtopics_agg[subtopic]['behavioral_metrics']['confusion'].append(behavioral_metrics.get('confusion', 50.0))
        subtopics_agg[subtopic]['behavioral_metrics']['confidence'].append(behavioral_metrics.get('confidence', 50.0))
        subtopics_agg[subtopic]['perceived_confidence_scores'].append(float(perceived_confidence_score))
    summary = {'topics': {}, 'subtopics': {}}
    for topic, data in topics_agg.items():
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        avg_time = sum(data['time']) / len(data['time']) if data['time'] else 0
        avg_behavioral = {
            'frustration': sum(data['behavioral_metrics']['frustration']) / len(data['behavioral_metrics']['frustration']) if data['behavioral_metrics']['frustration'] else 0,
            'confusion': sum(data['behavioral_metrics']['confusion']) / len(data['behavioral_metrics']['confusion']) if data['behavioral_metrics']['confusion'] else 0,
            'confidence': sum(data['behavioral_metrics']['confidence']) / len(data['behavioral_metrics']['confidence']) if data['behavioral_metrics']['confidence'] else 0
        }
        avg_perceived_confidence = sum(data['perceived_confidence_scores']) / len(data['perceived_confidence_scores']) if data['perceived_confidence_scores'] else 0
        summary['topics'][topic] = {
            'accuracy': float(accuracy),
            'avg_time': float(avg_time),
            'avg_behavioral': {k: float(v) for k, v in avg_behavioral.items()},
            'perceived_confidence_score': float(avg_perceived_confidence),
            'correct': data['correct'],
            'total': data['total']
        }
    for subtopic, data in subtopics_agg.items():
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        avg_time = sum(data['time']) / len(data['time']) if data['time'] else 0
        avg_behavioral = {
            'frustration': sum(data['behavioral_metrics']['frustration']) / len(data['behavioral_metrics']['frustration']) if data['behavioral_metrics']['frustration'] else 0,
            'confusion': sum(data['behavioral_metrics']['confusion']) / len(data['behavioral_metrics']['confusion']) if data['behavioral_metrics']['confusion'] else 0,
            'confidence': sum(data['behavioral_metrics']['confidence']) / len(data['behavioral_metrics']['confidence']) if data['behavioral_metrics']['confidence'] else 0
        }
        avg_perceived_confidence = sum(data['perceived_confidence_scores']) / len(data['perceived_confidence_scores']) if data['perceived_confidence_scores'] else 0
        summary['subtopics'][subtopic] = {
            'accuracy': float(accuracy),
            'avg_time': float(avg_time),
            'avg_behavioral': {k: float(v) for k, v in avg_behavioral.items()},
            'perceived_confidence_score': float(avg_perceived_confidence),
            'correct': data['correct'],
            'total': data['total']
        }
    recommendations = []
    for topic, stats in summary['topics'].items():
        is_quant = any(q.get('topic') == topic and q.get('topic') in ['Algebra', 'Geometry', 'Quant'] for q in questions_results)
        recommended_time = RECOMMENDED_TIME_QUANT if is_quant else RECOMMENDED_TIME_VERBAL
        if stats['accuracy'] < ACCURACY_WEAKNESS_THRESHOLD:
            recommendations.append(f"Strengthen fundamentals in {topic}.")
        if stats['avg_behavioral'].get('frustration', 0) > EMOTIONAL_STRAIN_THRESHOLD:
            recommendations.append(f"High emotional strain in {topic}. Break down complex problems or review earlier concepts.")
        if stats['avg_behavioral'].get('confusion', 0) > CONCEPTUAL_UNCLEARNESS_THRESHOLD:
             recommendations.append(f"High conceptual unclarity in {topic}. Focus on structured learning of core concepts.")
        if stats['avg_time'] > recommended_time * 1.2:
            recommendations.append(f"Improve pacing for {topic}. Avg time ({stats['avg_time']:.1f}s) is slower than recommended ({recommended_time:.1f}s). Practice timing drills.")
        if stats['perceived_confidence_score'] < PERCEIVED_CONFIDENCE_RECOMMENDATION_THRESHOLD:
            recommendations.append(f"Boost confidence in {topic}. Start with easier practice problems to build mastery.")
    if not recommendations:
         recommendations.append("Solid performance detected! Continue structured practice to reinforce mastery.")
    return {'summary': summary, 'recommendations': recommendations}

def analyze_question_with_ai(question_data):
    if genai_model is None:
        return "e-GMAT AI: Question insight unavailable due to AI service status."
    question_text = question_data.get('question_text', 'N/A')
    options = question_data.get('options', [])
    selected_option = question_data.get('learner_response', 'No response')
    correct_option = question_data.get('correct_answer', 'N/A')
    is_correct = question_data.get('correct', False)
    time_spent = question_data.get('time_spent', 0.0)
    behavioral_metrics = question_data.get('behavioral_metrics', {"frustration": 50.0, "confusion": 50.0, "confidence": 50.0})
    outcome = "Correctly identified" if is_correct else "Incorrectly identified"
    prompt = f"""
You are the e-GMAT AI Coaching System, providing personalized insights on a practice question response.
Question Details:
Question Text: {question_text}
Options: {', '.join(options)}
Learner's Response: {selected_option}
Correct Option: {correct_option}
Outcome: Learner {outcome} the correct option.
Time Spent: {time_spent:.1f} seconds
Observed Behavioral Metrics at Response (Scale 0-100): Effort={behavioral_metrics['frustration']:.2f}, Unclarity={behavioral_metrics['confusion']:.2f}, Behavioral Confidence={behavioral_metrics['confidence']:.2f}
Analyze this single question event for the learner. Focus on:
1. Briefly explain the core reason Option {correct_option} is the correct answer, referencing the concept tested (e.g., SC grammar rule, CR logic gap, RC inference).
2. Briefly explain common traps or why incorrect options might be appealing but flawed.
3. Based *only* on the response outcome ({outcome}), time spent, and observed behavioral metrics (e.g., high unclarity on a correct answer might indicate lucky guess or slow processing; low effort on an incorrect answer might indicate rushed mistake), what specific, actionable insight or micro-recommendation can you give for *this question type or underlying concept*? Keep it concise (1-3 sentences). Frame recommendations as P.A.C.E. adjustments.
"""
    try:
        response = genai_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"e-GMAT AI: Question insight generation failed for QID {question_data.get('question_id')}: {e}")
        return f"e-GMAT AI: Insight unavailable for this question ({e})."

def generate_ai_comprehensive_analysis(session_data, performance_analysis, behavioral_history, historical_report_data):
    if genai_model is None:
        return "e-GMAT AI: Comprehensive analysis unavailable due to AI service status."
    current_session_summary_str = f"""
Current Session ({session_data.get('date', 'N/A')}):
- Score: {session_data.get('score', 0)}/{session_data.get('total', len(session_data.get('questions', [])))} ({session_data.get('score', 0)/session_data.get('total', 1) * 100 if session_data.get('total', 1)>0 else 0:.1f}%)
- Overall P.A.C.E. Engagement Score (out of 10) for this session: {state.engagement_score} (Based on final behavioral reading)
- Performance Summary by Topic: {json.dumps(performance_analysis['summary']['topics'], indent=2, cls=NumpyEncoder)}
- Performance Summary by Subtopic: {json.dumps(performance_analysis['summary']['subtopics'], indent=2, cls=NumpyEncoder)}
- Automated System Recommendations: {', '.join(performance_analysis['recommendations'])}
"""
    historical_summary_str = "Historical Performance Context (Previous Sessions):\n"
    if historical_report_data and 'sessions' in historical_report_data and len(historical_report_data['sessions']) > 1:
         past_sessions = [t for t in historical_report_data['sessions'] if t.get('session_id') != session_data.get('session_id')]
         if past_sessions:
            for i, session in enumerate(past_sessions[-3:]):
                 historical_summary_str += f"""
Session {session.get('session_id', 'N/A')} ({session.get('date', 'N/A')}):
- Score: {session.get('score', 0)}/{session.get('total', len(session.get('questions', [])))} ({session.get('score', 0)/session.get('total', 1)*100 if session.get('total',1)>0 else 0:.1f}%)
- Key Performance Indicators (snapshot): {json.dumps({topic: {k:v for k,v in data.items() if k != 'behavioral_metrics' and k != 'time'} for topic, data in analyze_session_performance(session).get('summary',{}).get('topics', {}).items()}, indent=2, cls=NumpyEncoder)}
"""
         else:
             historical_summary_str += "No previous sessions available.\n"
    else:
        historical_summary_str += "No previous sessions available.\n"
    behavioral_trends_str = "Behavioral Metric Trends During Session (Snapshots over time):\n"
    history_subset = behavioral_history[::max(1, len(behavioral_history) // 20)]
    behavioral_trends_str += json.dumps(history_subset, indent=2, cls=NumpyEncoder)
    prompt = f"""
You are the e-GMAT AI Coaching System, analyzing a learner's adaptive practice session performance and behavioral metrics. Provide a comprehensive, structured, and actionable analysis for {state.learner_name}, aligned with the e-GMAT methodology (focus on fundamentals, process, pacing, combining performance with behavioral data).

Summary of Learner Data:
{current_session_summary_str}
{historical_summary_str}
{behavioral_trends_str}

Structure your analysis with e-GMAT style sections:
Performance Snapshot:
- Summarize the current session results (score, percentage, key topic mastery indicators).
- If historical data exists, provide a brief comparison showing trajectory or consistency.
Behavioral Insights:
- Describe the general behavioral trend observed during the session (e.g., consistent focus, points of unclarity or strain). Reference the overall P.A.C.E. Engagement Score.
- Connect behavioral observations to performance (e.g., high unclarity coinciding with incorrect responses in a specific subtopic).
Targeted Focus Areas:
- Identify key areas (topics/subtopics) based on combined accuracy, time, and behavioral data. Clearly state Strengths (areas of demonstrated mastery) and Areas for Refinement (weaknesses/opportunities for growth). Frame this similar to an adaptive learning engine's output.
Actionable Next Steps & P.A.C.E. Recommendations:
- Provide concrete, step-by-step advice. Recommend specific study actions aligned with e-GMAT's structured approach (e.g., review concept files for [Subtopic], practice targeted drills in [Topic], work on timing for [Question Type], apply the 3-step process for [Specific Verbal Area]).
- Explain *why* these steps are recommended based on the data (e.g., "Your high unclarity metric in [Subtopic] suggests reviewing the core logic before attempting more complex problems").
- Maintain a supportive and motivating tone, empowering the learner on their GMAT journey.

Generate the Comprehensive AI Analysis for {state.learner_name}:
"""
    try:
        response = genai_model.generate_content(prompt)
        text = response.text.strip()
        prompted_start = f"Performance Snapshot:"
        if text.startswith(prompted_start):
             text = text[len(prompted_start):].strip()
        return text
    except Exception as e:
        print(f"e-GMAT AI: Comprehensive analysis generation failed: {e}")
        return f"e-GMAT AI: Unable to generate comprehensive analysis due to an AI error: {e}"

def perform_session_breakdown_analysis(summary):
    breakdown = {'Demonstrated Strengths': [], 'Areas for Refinement': [], 'Efficiency Opportunities': [], 'Potential Challenges': []}
    subtopics_data = summary.get('subtopics', {})
    for subtopic, stats in subtopics_data.items():
        accuracy = stats.get('accuracy', 0)
        avg_time = stats.get('avg_time', float('inf'))
        frustration = stats.get('avg_behavioral', {}).get('frustration', 50)
        confusion = stats.get('avg_behavioral', {}).get('confusion', 50)
        confidence = stats.get('perceived_confidence_score', 50)
        is_quant = any(q.get('subtopic') == subtopic and q.get('topic') in ['Algebra', 'Geometry', 'Quant', 'Data Insights'] for q in state.session_data.get('questions',[]))
        recommended_time = RECOMMENDED_TIME_QUANT if is_quant else RECOMMENDED_TIME_VERBAL
        is_strength = accuracy > ACCURACY_OPPORTUNITY_THRESHOLD and confidence > BEHAVIORAL_CONFIDENCE_STRENGTH_THRESHOLD
        is_weakness = accuracy < ACCURACY_WEAKNESS_THRESHOLD or frustration > EMOTIONAL_STRAIN_THRESHOLD or confusion > CONCEPTUAL_UNCLEARNESS_THRESHOLD
        is_opportunity = avg_time < recommended_time * 0.9 and accuracy > ACCURACY_WEAKNESS_THRESHOLD * 0.8
        is_challenge = avg_time > recommended_time * 1.1
        is_emotional_challenge = frustration > EMOTIONAL_STRAIN_THRESHOLD or confusion > CONCEPTUAL_UNCLEARNESS_THRESHOLD
        if is_strength:
            breakdown['Demonstrated Strengths'].append(subtopic)
        if is_weakness:
            breakdown['Areas for Refinement'].append(f"{subtopic} (Accuracy: {accuracy*100:.0f}%, Strain: {frustration:.0f}, Unclarity: {confusion:.0f})")
        if is_opportunity:
            breakdown['Efficiency Opportunities'].append(f"{subtopic} (Avg. Time: {avg_time:.1f}s)")
        if is_challenge:
            breakdown['Potential Challenges'].append(f"{subtopic} (Pacing: {avg_time:.1f}s > Rec: {recommended_time:.1f}s)")
        if is_emotional_challenge and not is_weakness:
            breakdown['Potential Challenges'].append(f"{subtopic} (Behavioral Struggle, but Accurate)")
    for category in breakdown:
        breakdown[category] = sorted(list(set(breakdown[category])))
    if not breakdown['Demonstrated Strengths']: breakdown['Demonstrated Strengths'] = ["Specific areas of strength could not be definitively identified from this session."]
    if not breakdown['Areas for Refinement']: breakdown['Areas for Refinement'] = ["No prominent areas for refinement were definitively identified from this session."]
    if not breakdown['Efficiency Opportunities']: breakdown['Efficiency Opportunities'] = ["Specific efficiency opportunities were not identified from this session."]
    if not breakdown['Potential Challenges']: breakdown['Potential Challenges'] = ["No prominent potential challenges were definitively identified from this session."]
    return breakdown

def create_session_visualizations(summary, learner_name, session_id, behavioral_history):
    visualizations = {}
    topics_summary = summary.get('topics', {})
    if not topics_summary:
        print("e-GMAT Visuals: No topic data to generate visualizations.")
        return visualizations
    topics = list(topics_summary.keys())
    if not topics:
        print("e-GMAT Visuals: No topics available for charts.")
        return visualizations
    def save_plot_to_bytes():
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = '#1a1a1a'
    plt.rcParams['axes.facecolor'] = '#1a1a1a'
    plt.rcParams['text.color'] = '#FFFFFF'
    plt.rcParams['axes.labelcolor'] = '#FFFFFF'
    plt.rcParams['xtick.color'] = '#FFFFFF'
    plt.rcParams['ytick.color'] = '#FFFFFF'
    plt.rcParams['figure.dpi'] = 100
    try:
        accuracies = [topics_summary[t].get('accuracy', 0) * 100 for t in topics]
        plt.figure(figsize=(8, 5))
        plt.bar(topics, accuracies, color='#00FF00')
        plt.title(f'Accuracy by Topic - Learner Session ({learner_name})', color='#FFFFFF')
        plt.xlabel('Topic', color='#FFFFFF')
        plt.ylabel('Accuracy (%)', color='#FFFFFF')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        visualizations['accuracy_chart'] = save_plot_to_bytes()
    except Exception as e:
        print(f"e-GMAT Visuals: Error creating accuracy chart: {e}")
    try:
        times = [topics_summary[t].get('avg_time', 0) for t in topics]
        recommended_times = []
        for topic in topics:
             is_quant = any(q.get('topic') == topic and q.get('topic') in ['Algebra', 'Geometry', 'Quant', 'Data Insights'] for q in state.session_data.get('questions',[]))
             recommended_times.append(RECOMMENDED_TIME_QUANT if is_quant else RECOMMENDED_TIME_VERBAL)
        plt.figure(figsize=(8, 5))
        x_pos = np.arange(len(topics))
        width = 0.35
        rects1 = plt.bar(x_pos - width/2, times, width, label='Your Avg. Pacing', color='#FFA500')
        rects2 = plt.bar(x_pos + width/2, recommended_times, width, label='Recommended Pacing', color='#00FFFF')
        plt.title(f'Average Pacing per Topic ({learner_name})', color='#FFFFFF')
        plt.xlabel('Topic', color='#FFFFFF')
        plt.ylabel('Time (seconds)', color='#FFFFFF')
        plt.xticks(x_pos, topics)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        visualizations['time_chart'] = save_plot_to_bytes()
    except Exception as e:
        print(f"e-GMAT Visuals: Error creating time chart: {e}")
    try:
        perceived_confidences = [topics_summary[t].get('perceived_confidence_score', 0) for t in topics]
        plt.figure(figsize=(8, 5))
        plt.bar(topics, perceived_confidences, color='#FFFF00')
        plt.title(f'Perceived Confidence Score by Topic ({learner_name})', color='#FFFFFF')
        plt.xlabel('Topic', color='#FFFFFF')
        plt.ylabel('Confidence Score (0-100)', color='#FFFFFF')
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        visualizations['perceived_confidence_chart'] = save_plot_to_bytes()
    except Exception as e:
        print(f"e-GMAT Visuals: Error creating perceived confidence chart: {e}")
    try:
        if behavioral_history:
            timestamps = list(range(len(behavioral_history)))
            frustration = [m.get('frustration', 50) for m in behavioral_history]
            confusion = [m.get('confusion', 50) for m in behavioral_history]
            confidence = [m.get('confidence', 50) for m in behavioral_history]
            plt.figure(figsize=(8, 5))
            plt.plot(timestamps, frustration, label='Effort Indicator', color='#FF0000')
            plt.plot(timestamps, confusion, label='Focus Indicator', color='#FFA500')
            plt.plot(timestamps, confidence, label='Behavioral Confidence', color='#00FF00')
            plt.title(f'Behavioral Metric Trends During Session ({learner_name})', color='#FFFFFF')
            plt.xlabel(f'Time (approx. {0.5}s intervals)', color='#FFFFFF')
            plt.ylabel('Metric Score (0-100)', color='#FFFFFF')
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            visualizations['behavioral_trends_chart'] = save_plot_to_bytes()
        else:
             print("e-GMAT Visuals: No behavioral history data to generate trend chart.")
    except Exception as e:
        print(f"e-GMAT Visuals: Error creating behavioral trends chart: {e}")
    return visualizations

def generate_pdf_report(session_data, performance_analysis, learner_name, session_visualizations, ai_comprehensive_analysis_text, behavioral_history):
    print("e-GMAT PDF: Generating Personalized Performance Report (PDF)...")
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    base_font = "Helvetica"
    base_color = colors.black

    styles['Normal'].wordWrap = 'CJK'
    styles['Normal'].fontSize = 12
    styles['Normal'].alignment = TA_LEFT
    styles['Normal'].fontName = base_font

    styles.add(ParagraphStyle(name='Heading4_NoSpace', parent=styles['Heading4'], spaceAfter=0))
    styles.add(ParagraphStyle(name='Heading5_NoSpace', parent=styles['Heading5'], spaceAfter=0))
    styles.add(ParagraphStyle(name='Normal_Indent', parent=styles['Normal'], leftIndent=10))
    styles.add(ParagraphStyle(name='Normal_DoubleIndent', parent=styles['Normal'], leftIndent=20))

    def draw_title(text, y):
        c.setFont(base_font + "-Bold", 14)
        c.setFillColor(base_color)
        c.drawString(50, y, text)
        return y - 18
    def draw_paragraph(text, y, style='Normal', indent=0):
        p_style = styles[style]
        
        
        if indent > 0 and style == 'Normal_Indent':
             p_style = styles['Normal_Indent']
        elif indent > 0 and style == 'Normal_DoubleIndent':
             p_style = styles['Normal_DoubleIndent']
        elif indent > 0 :
             custom_style_name = f'Normal_Indent_{indent}_{style}'
             if custom_style_name not in styles:
                styles.add(ParagraphStyle(name=custom_style_name, parent=styles[style], leftIndent=indent))
             p_style = styles[custom_style_name]

        paragraph = Paragraph(text, p_style)
        available_width = letter[0] - 100
        
        current_left_indent = p_style.leftIndent if hasattr(p_style, 'leftIndent') else 0
        w, h = paragraph.wrap(available_width - current_left_indent, letter[1]) 
        
        draw_x = 50 + current_left_indent if (style.startswith('Normal_Indent') or style.startswith('Normal_DoubleIndent')) else (50 + indent)
        paragraph.drawOn(c, draw_x , y - h)
        return y - h - 10
    y = 750
    c.setFont(base_font + "-Bold", 18)
    c.drawString(50, y, f"e-GMAT Personalized Performance Report")
    y -= 24
    c.setFont(base_font, 12)
    y = draw_paragraph(f"Learner: {learner_name}", y, style='Normal', indent=0)
    y = draw_paragraph(f"Session ID: {session_data.get('session_id', 'N/A')}, Date: {session_data.get('date', 'N/A')}", y, style='Normal', indent=0)
    y = draw_paragraph(f"Session Score: {session_data.get('score', 0)}/{session_data.get('total', 1)} ({session_data.get('score', 0)/session_data.get('total',1)*100 if session_data.get('total',1)>0 else 0:.1f}%)", y, style='Normal', indent=0)
    y -= 12
    y = draw_title("Session Engagement Metrics", y)
    engagement_score_label = "N/A"
    with state.state_lock:
        engagement_score_label = f"{state.engagement_score}/10 (Overall P.A.C.E. Engagement Score)"
    y = draw_paragraph(f"Observed Behavioral State During Session: {engagement_score_label}", y, style='Normal', indent=0)
    y -= 18
    y = draw_title("Performance Breakdown by Topic", y)
    topics_data = performance_analysis.get('summary', {}).get('topics', {})
    data = [['Topic', 'Correct/Total', 'Accuracy (%)', 'Avg Pacing (s)', 'Perceived Confidence', 'Avg Effort', 'Avg Focus', 'Avg Behav. Conf']]
    for topic, stats in topics_data.items():
        is_quant = any(q.get('topic') == topic and q.get('topic') in ['Algebra', 'Geometry', 'Quant', 'Data Insights'] for q in session_data.get('questions',[]))
        recommended_time = RECOMMENDED_TIME_QUANT if is_quant else RECOMMENDED_TIME_VERBAL
        time_str = f"{stats.get('avg_time', 0):.1f} (Rec: {recommended_time:.1f})"
        data.append([
            topic,
            f"{stats.get('correct', 0)}/{stats.get('total', 0)}",
            f"{stats.get('accuracy', 0)*100:.1f}",
            time_str,
            f"{stats.get('perceived_confidence_score', 0):.1f}",
            f"{stats.get('avg_behavioral',{}).get('frustration',0):.1f}",
            f"{stats.get('avg_behavioral',{}).get('confusion',0):.1f}",
            f"{stats.get('avg_behavioral',{}).get('confidence',0):.1f}"
        ])
    table = Table(data)
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), base_font + "-Bold"),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.gray),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ])
    table.setStyle(table_style)
    table_width = letter[0] - 100
    table.wrapOn(c, table_width, letter[1])
    w, h = table.wrap(table_width, letter[1])
    table.drawOn(c, 50, y - h)
    y -= h + 12
    c.showPage()
    y = 750
    y = draw_title("Performance Breakdown by Subtopic", y)
    subtopics_data = performance_analysis.get('summary', {}).get('subtopics', {})
    data = [['Subtopic', 'Correct/Total', 'Accuracy (%)', 'Avg Pacing (s)', 'Perceived Confidence']]
    for subtopic, stats in subtopics_data.items():
        data.append([
            subtopic,
            f"{stats.get('correct', 0)}/{stats.get('total', 0)}",
            f"{stats.get('accuracy', 0)*100:.1f}",
            f"{stats.get('avg_time', 0):.1f}",
            f"{stats.get('perceived_confidence_score', 0):.1f}"
        ])
    table = Table(data)
    table.setStyle(table_style)
    table.wrapOn(c, table_width, letter[1])
    w, h = table.wrap(table_width, letter[1])
    if y - h < 50:
        c.showPage()
        y = 750
        y = draw_title("Performance Breakdown by Subtopic (cont.)", y)
        table.wrapOn(c, table_width, letter[1])
        w, h = table.wrap(table_width, letter[1])
    table.drawOn(c, 50, y - h)
    y -= h + 12
    c.showPage()
    y = 750
    y = draw_title("Session Visualizations", y)
    img_height = 180
    for chart_name in ['accuracy_chart', 'time_chart', 'perceived_confidence_chart', 'behavioral_trends_chart']:
        if chart_name in session_visualizations:
            try:
                img_data = session_visualizations[chart_name]
                img = ImageReader(io.BytesIO(img_data))
                img_width_orig, img_height_orig = img.getSize()
                img_height_display = img_height
                img_width_display = (img_height_display / img_height_orig) * img_width_orig
                if img_width_display > letter[0] - 100:
                     img_width_display = letter[0] - 100
                     img_height_display = (img_width_display / img_width_orig) * img_height_orig
                img_x = 50 + (letter[0] - 100 - img_width_display) / 2
                if y - img_height_display < 50:
                    c.showPage()
                    y = 750
                    y = draw_title(f"Session Visualizations (cont.) - {chart_name.replace('_', ' ').title()}", y)
                c.drawImage(img, img_x, y - img_height_display, width=img_width_display, height=img_height_display)
                y -= img_height_display + 12
            except Exception as e:
                print(f"e-GMAT PDF: Error embedding {chart_name} visualization: {e}")
                y = draw_paragraph(f"[Error embedding {chart_name}: {e}]", y, style='Italic')
    c.showPage()
    y = 750
    y = draw_title("Session Breakdown Analysis (Subtopic Focus)", y)
    breakdown = perform_session_breakdown_analysis(performance_analysis.get('summary', {}))
    for category in ['Demonstrated Strengths', 'Areas for Refinement', 'Efficiency Opportunities', 'Potential Challenges']:
        y -= 8
        y = draw_paragraph(f"{category}:", y, style='Heading4_NoSpace', indent=0)
        items = breakdown.get(category, ["No items identified."])
        if items:
            for item in items:
                p = Paragraph(f"- {item}", styles['Normal_Indent'])
                
                w, h = p.wrap(letter[0] - 100 - styles['Normal_Indent'].leftIndent, letter[1])
                if y - h < 50:
                     c.showPage()
                     y = 750
                     y = draw_title(f"Session Breakdown Analysis (cont.) - {category}", y)
                p.drawOn(c, 50 + styles['Normal_Indent'].leftIndent, y - h)
                y -= h + 4
        else:
             pass
    if ai_comprehensive_analysis_text:
        c.showPage()
        y = 750
        y = draw_title("e-GMAT AI Comprehensive Analysis", y)
        y -= 6
        
        current_y = y
        lines = ai_comprehensive_analysis_text.split('\n')
        
        for line in lines:
            
            line_para_style = styles['Normal']
            line_para = Paragraph(line, line_para_style)
            
            current_line_indent = 0 
            w, h = line_para.wrap(letter[0] - 100 - current_line_indent, letter[1])
            if current_y - h < 50 and line.strip():
                c.showPage()
                current_y = 750
            if line.strip():
                 line_para.drawOn(c, 50 + current_line_indent, current_y - h)
                 current_y -= h + 3
        y = current_y
    c.showPage()
    y = 750
    y = draw_title("Question-by-Question Insights", y)
    y -= 6
    questions_results = session_data.get('questions', [])
    if questions_results:
        for i, q_data in enumerate(questions_results):
            q_index = q_data.get('question_id', i+1)
            q_text = f"Question {q_index}:"
            y = draw_paragraph(q_text, y, style='Heading4_NoSpace', indent=0)
            stats_text = f"  Outcome: {'Correct' if q_data.get('correct', False) else 'Incorrect'}. Pacing: {q_data.get('time_spent', 0):.1f}s. Behavioral: Effort={q_data.get('behavioral_metrics', {}).get('frustration',0):.1f}, Focus={q_data.get('behavioral_metrics',{}).get('confusion',0):.1f}, Confidence={q_data.get('behavioral_metrics',{}).get('confidence',0):.1f}. Perceived Confidence Score: {q_data.get('perceived_confidence_score', 0):.1f}/100"
            y = draw_paragraph(stats_text, y, style='Normal_Indent', indent=0) 
            insight_text = q_data.get('ai_question_insight', 'No AI insight available.')
            y = draw_paragraph("  e-GMAT AI Insight:", y, style='Heading5_NoSpace', indent=10) 
            insight_para = Paragraph(insight_text, styles['Normal_DoubleIndent'])
            
            w, h = insight_para.wrap(letter[0] - 100 - styles['Normal_DoubleIndent'].leftIndent, letter[1])
            if y - h < 50 and i < len(questions_results) -1:
                c.showPage()
                y = 750
                y = draw_title(f"Question-by-Question Insights (cont.)", y)
            insight_para.drawOn(c, 50 + styles['Normal_DoubleIndent'].leftIndent, y - h)
            y -= h + 8
    else:
         y = draw_paragraph("No question-wise insight data available.", y, style='Normal', indent=0)
    c.save()
    buffer.seek(0)
    print("e-GMAT PDF: Personalized Performance Report (PDF) generated successfully.")
    return buffer.getvalue()

class GMATApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("e-GMAT Adaptive Practice Environment")
        self.geometry("800x600")
        self.configure(bg="#000000")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.hacking_text = tk.Label(self, text="e-GMAT System Initializing... Engaging AI Subsystems.", fg="#00FF00", bg="#000000", font=("Courier", 10))
        self.hacking_text.place(relx=0, rely=0, relwidth=1, height=20)
        self.hacking_sequence = ["e-GMAT System Initializing...", "e-GMAT System Initializing. ", "e-GMAT System Initializing..", "e-GMAT System Initializing...", "Engaging AI Subsystems.", "Engaging AI Subsystems..", "Engaging AI Subsystems..."]
        self.current_hacking_step = 0
        self.animate_hacking()
        self.container = tk.Frame(self, bg="#000000")
        self.container.place(relx=0, rely=0.035, relwidth=1, relheight=0.965)
        self.frames = {}
        for F in (HomePage, TestPage, CountdownPage, ReportPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        load_candidate_registry()
        self.show_page("HomePage")
    def animate_hacking(self):
        if self.hacking_text:
             self.hacking_text.config(text=self.hacking_sequence[self.current_hacking_step])
             self.current_hacking_step = (self.current_hacking_step + 1) % len(self.hacking_sequence)
             self.after(300, self.animate_hacking)
    def on_closing(self):
        print("e-GMAT App: Closing application instance...")
        state.behavioral_thread_running = False
        if state.camera_capture:
            state.camera_capture.release()
            state.camera_capture = None
            print("e-GMAT Camera: Capture device released on closing.")
        self.destroy()
    def show_page(self, page_name):
        print(f"e-GMAT Navigation: Switching to page: {page_name}")
        if state.behavioral_thread_running and page_name != "TestPage":
            print("e-GMAT Behavioral: Stopping detection thread as not on TestPage.")
            state.behavioral_thread_running = False
        if state.camera_capture and page_name in ["HomePage"]:
             print("e-GMAT Camera: Releasing capture device when returning to Home page.")
             state.camera_capture.release()
             state.camera_capture = None
        frame = self.frames.get(page_name)
        if frame:
            frame.tkraise()
            state.page = page_name.lower().replace("page", "")
            if page_name == "TestPage":
                print("e-GMAT Test: Setting up Adaptive Practice Session...")
                state.current_question = 0
                state.responses = []
                state.behavioral_history = []
                state.question_start_time = time.time()
                state.question_set = load_and_randomize_questions()
                if state.question_set is None or not state.question_set:
                     messagebox.showerror("e-GMAT Data Error", "Could not load question set. Please verify 'egmat.json' format.")
                     self.show_page("HomePage")
                     return
                state.camera_capture = initialize_camera()
                if not state.behavioral_thread_running:
                    state.behavioral_thread_running = True
                    threading.Thread(target=detect_behavioral_metrics, daemon=True).start()
                    print("e-GMAT Behavioral: Emotion detection thread initiated.")
                if state.camera_capture is None:
                     if isinstance(frame, TestPage):
                          frame.notify_camera_unavailable()
                if isinstance(frame, TestPage):
                     frame.update_question()
            elif page_name == "CountdownPage":
                 print("e-GMAT Processing: Setting up Report Generation Interface...")
                 state.behavioral_thread_running = False
                 if state.camera_capture:
                     print("e-GMAT Camera: Releasing capture device upon finishing session.")
                     state.camera_capture.release()
                     state.camera_capture = None
                 state.session_data = process_session_results(state.responses, state.question_set)
                 state.performance_analysis = analyze_session_performance(state.session_data)
                 report_data = load_learner_report(state.learner_id)
                 state.session_data['session_id'] = len(report_data.get('sessions', [])) + 1
                 state.ai_comprehensive_analysis_text = generate_ai_comprehensive_analysis(
                      state.session_data,
                      state.performance_analysis,
                      state.behavioral_history.copy(),
                      report_data.copy()
                  )
                 state.session_visualizations = create_session_visualizations(
                     state.performance_analysis.get('summary', {}),
                     state.learner_name,
                     state.session_data['session_id'],
                     state.behavioral_history.copy()
                 )
                 state.pdf_report_data = generate_pdf_report(
                      state.session_data,
                      state.performance_analysis,
                      state.learner_name,
                      state.session_visualizations,
                      state.ai_comprehensive_analysis_text,
                      state.behavioral_history.copy()
                 )
                 report_data.setdefault('sessions', []).append(state.session_data)
                 report_data['name'] = state.learner_name
                 save_learner_report(state.learner_id, report_data)
                 if isinstance(frame, CountdownPage):
                      frame.start_countdown()
            elif page_name == "ReportPage":
                print("e-GMAT Report: Setting up Personalized Performance Report Page...")
                if isinstance(frame, ReportPage):
                     frame.update_report()
        else:
            print(f"e-GMAT Navigation: Error: Page '{page_name}' not found in frame registry.")

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#000000")
        self.controller = controller
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#000000')
        self.style.configure('TLabel', background='#000000', foreground='#FFFFFF')
        self.style.configure('TButton', background='#1a1a1a', foreground='#00FF00', font=('Helvetica', 10, 'bold'))
        self.style.map('TButton', background=[('active', '#00FF00')], foreground=[('active', '#000000')])
        self.style.configure('TEntry', fieldbackground='#1a1a1a', foreground='#FFFFFF', insertbackground='#FFFFFF')
        tk.Label(self, text="Welcome to Your e-GMAT Adaptive Practice Session", fg="#00FFFF", bg="#000000", font=("Helvetica", 18, "bold")).pack(pady=20)
        input_frame = tk.Frame(self, bg="#000000")
        input_frame.pack(pady=10)
        tk.Label(input_frame, text="Learner ID:", fg="#00FF00", bg="#000000").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.learner_id_entry = tk.Entry(input_frame, bg="#1a1a1a", fg="#FFFFFF", insertbackground="#FFFFFF", font=('Courier', 10), width=30)
        self.learner_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.learner_id_entry.focus_set()
        tk.Label(input_frame, text="Learner Name:", fg="#00FF00", bg="#000000").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.learner_name_entry = tk.Entry(input_frame, bg="#1a1a1a", fg="#FFFFFF", insertbackground="#FFFFFF", font=('Courier', 10), width=30)
        self.learner_name_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        tk.Button(self, text="Launch Adaptive Practice", command=self.start_practice_session, bg="#1a1a1a", fg="#00FF00", activebackground="#00FFFF", activeforeground="#000000", font=("Helvetica", 12, "bold")).pack(pady=20)
        tk.Label(self, text="View Previous Performance Reports (Learner ID required):", fg="#00FFFF", bg="#000000", font=("Helvetica", 12)).pack(pady=5)
        tk.Button(self, text="Load Session History", command=self.load_and_display_history, bg="#1a1a1a", fg="#FFFF00", activebackground="#FFFF00", activeforeground="#000000").pack(pady=5)
        self.prev_sessions_text = tk.Text(self, height=8, width=60, bg="#1a1a1a", fg="#FFFFFF", insertbackground="#FFFFFF", font=('Courier', 9), state='disabled')
        self.prev_sessions_text.pack(pady=10)
    def load_and_display_history(self):
         learner_id = self.learner_id_entry.get().strip()
         if not learner_id:
            messagebox.showwarning("Input Required", "Please enter a Learner ID to view history.")
            return
         report = load_learner_report(learner_id)
         self.prev_sessions_text.config(state='normal')
         self.prev_sessions_text.delete(1.0, tk.END)
         if report and report.get('sessions'):
            self.prev_sessions_text.insert(tk.END, f"--- Adaptive Practice History for Learner ID: {learner_id} (Name: {report.get('name', 'N/A')}) ---\n\n")
            for session in sorted(report['sessions'], key=lambda x: x.get('date', ''), reverse=True):
                date_str = session.get('date', 'N/A')
                session_id_str = session.get('session_id', 'N/A')
                score = session.get('score', 0)
                total = session.get('total', 0)
                percentage = score/total*100 if total > 0 else 0
                self.prev_sessions_text.insert(tk.END, f"Session {session_id_str} on {date_str}:\n")
                self.prev_sessions_text.insert(tk.END, f"  Score: {score}/{total} ({percentage:.1f}%)\n")
                self.prev_sessions_text.insert(tk.END, "-"*20 + "\n")
         else:
            self.prev_sessions_text.insert(tk.END, f"No previous session history found for Learner ID {learner_id}.\n")
         self.prev_sessions_text.config(state='disabled')
    def start_practice_session(self):
        learner_id = self.learner_id_entry.get().strip()
        learner_name = self.learner_name_entry.get().strip()
        if not learner_id or not learner_name:
            messagebox.showerror("Input Error", "Please enter both Learner ID and Name to start the practice session.")
            return
        state.learner_id = learner_id
        state.learner_name = learner_name
        load_learner_report(learner_id)
        update_candidate_registry(learner_id, learner_name)
        self.learner_id_entry.delete(0, tk.END)
        self.learner_name_entry.delete(0, tk.END)
        self.prev_sessions_text.config(state='normal')
        self.prev_sessions_text.delete(1.0, tk.END)
        self.prev_sessions_text.config(state='disabled')
        self.controller.show_page("TestPage")

class TestPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#000000")
        self.controller = controller
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#000000')
        self.style.configure('TLabel', background='#000000', foreground='#FFFFFF')
        self.style.configure('TButton', background='#1a1a1a', foreground='#00FF00', font=('Helvetica', 10, 'bold'))
        self.style.map('TButton', background=[('active', '#00FF00')], foreground=[('active', '#000000')])
        self.style.configure('Horizontal.TProgressbar',
                             background='#00FF00', troughcolor='#1a1a1a', bordercolor='#333333', lightcolor='#00CC00', darkcolor='#006600')
        self.style.map('Horizontal.TProgressbar', background=[('disabled', '#555555')])
        self.behavioral_frame = tk.Frame(self, bg="#000000", bd=2, relief="groove")
        self.behavioral_frame.place(relx=0.02, rely=0.05, relwidth=0.38, relheight=0.9)
        tk.Label(self.behavioral_frame, text="Real-Time Behavioral Insights", fg="#00FFFF", bg="#000000", font=("Helvetica", 14, "bold")).pack(pady=10)
        self.webcam_label = tk.Label(self.behavioral_frame, bg="#1a1a1a", text="Engagement Feed\nor Status", fg="#CCCCCC", font=("Helvetica", 10))
        self.webcam_label.pack(pady=10, fill="both", expand=True)
        tk.Label(self.behavioral_frame, text="Behavioral Metrics (0-100 Scale):", fg="#00FF00", bg="#000000", font=("Helvetica", 10)).pack(pady=(10,0))
        self.strain_label = tk.Label(self.behavioral_frame, text="Effort Indicator: N/A", fg="#FFFFFF", bg="#000000", font=("Helvetica", 9))
        self.strain_label.pack(pady=2)
        self.strain_progress = ttk.Progressbar(self.behavioral_frame, style='Horizontal.TProgressbar', length=180, maximum=100.0, mode='determinate')
        self.strain_progress.pack()
        self.unclarity_label = tk.Label(self.behavioral_frame, text="Focus Indicator: N/A", fg="#FFFFFF", bg="#000000", font=("Helvetica", 9))
        self.unclarity_label.pack(pady=2)
        self.unclarity_progress = ttk.Progressbar(self.behavioral_frame, style='Horizontal.TProgressbar', length=180, maximum=100.0, mode='determinate')
        self.unclarity_progress.pack()
        self.behavioral_conf_label = tk.Label(self.behavioral_frame, text="Behavioral Confidence: N/A", fg="#FFFFFF", bg="#000000", font=("Helvetica", 9))
        self.behavioral_conf_label.pack(pady=2)
        self.behavioral_conf_progress = ttk.Progressbar(self.behavioral_frame, style='Horizontal.TProgressbar', length=180, maximum=100.0, mode='determinate')
        self.behavioral_conf_progress.pack()
        self.engagement_score_label = tk.Label(self.behavioral_frame, text="Overall P.A.C.E. Engagement Score: N/A/10", fg="#FFFF00", bg="#000000", font=("Helvetica", 10, "bold"))
        self.engagement_score_label.pack(pady=10)
        self.current_perceived_confidence_label = tk.Label(self.behavioral_frame, text="Est. Perceived Confidence: N/A/100", fg="#00FFFF", bg="#000000", font=("Helvetica", 10, "bold"))
        self.current_perceived_confidence_label.pack(pady=5)
        self.practice_frame = tk.Frame(self, bg="#000000", bd=2, relief="groove")
        self.practice_frame.place(relx=0.42, rely=0.05, relwidth=0.56, relheight=0.9)
        self.q_info_label = tk.Label(self.practice_frame, text="", fg="#00FF00", bg="#000000", font=("Helvetica", 10), wraplength=380, anchor="w", justify="left")
        self.q_info_label.pack(pady=(10, 0), padx=10, fill="x")
        self.q_number_label = tk.Label(self.practice_frame, text="Question ?/20", fg="#FFFFFF", bg="#000000", font=("Helvetica", 12, "bold"), anchor="w", justify="left")
        self.q_number_label.pack(pady=(0, 5), padx=10, fill="x")
        self.question_text_widget = tk.Text(self.practice_frame, height=8, width=50, bg="#1a1a1a", fg="#FFFFFF", wrap="word", bd=0, highlightthickness=0, insertbackground="#FFFFFF")
        self.question_text_widget.pack(pady=5, padx=10, fill="both", expand=False)
        self.question_text_widget.tag_configure("center", justify='center')
        self.pacing_timer_label = tk.Label(self.practice_frame, text="Pacing: 00:00", fg="#FFFF00", bg="#1a1a1a", font=("Courier", 12), relief="sunken", bd=1)
        self.pacing_timer_label.pack(pady=5)
        tk.Label(self.practice_frame, text="Select Your Response:", fg="#00FF00", bg="#000000", font=("Helvetica", 10)).pack(pady=(10, 0), padx=10, fill="x")
        self.options_var = tk.StringVar(value="")
        self.options_frame = tk.Frame(self.practice_frame, bg="#000000")
        self.options_frame.pack(pady=5, padx=10, fill="both", expand=True, anchor="nw")
        self.progress_label = tk.Label(self.practice_frame, text="Adaptive Practice Progress: 0%", fg="#FFFFFF", bg="#000000")
        self.progress_label.pack(pady=(10, 2))
        self.test_progress_bar = ttk.Progressbar(self.practice_frame, style='Horizontal.TProgressbar', length=300, maximum=TOTAL_QUESTIONS, mode='determinate')
        self.test_progress_bar.pack()
        tk.Button(self.practice_frame, text="Submit Response", command=self.submit_response, bg="#1a1a1a", fg="#00FF00", activebackground="#00FFFF", activeforeground="#000000", font=("Helvetica", 12, "bold")).pack(pady=15)
        self.controller.bind('<Return>', lambda event=None: self.submit_response())
        self.after_id_webcam = None
        self.after_id_behavioral = None
        self.after_id_pacing = None
        self.update_loop_started = False
    def notify_camera_unavailable(self):
        print("e-GMAT Camera: Initialization failed. Displaying 'Engagement Feed Unavailable'.")
        self.cancel_updates()
        self.webcam_label.config(text="Engagement Feed Unavailable.\nUsing simulated behavioral data.\n\nPlease check camera connection & permissions.", fg="#FF0000")
        if self.after_id_webcam:
             self.after_cancel(self.after_id_webcam)
             self.after_id_webcam = None
        if not self.update_loop_started:
            self.update_behavioral_metrics()
            self.update_loop_started = True
    def update_webcam(self):
        if not state.behavioral_thread_running or state.camera_capture is None or not state.camera_capture.isOpened():
            if self.webcam_label.cget("text") != "Engagement Feed Unavailable.\nUsing simulated behavioral data.\n\nPlease check camera connection & permissions.":
                 self.notify_camera_unavailable()
            self.after_id_webcam = self.after(500, self.update_webcam)
            return
        ret, frame = state.camera_capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk, text="")
        self.after_id_webcam = self.after(100, self.update_webcam)
    def update_behavioral_metrics(self):
        with state.state_lock:
            current_metrics = state.behavioral_metrics.copy()
            engagement_score = state.engagement_score
        strain = current_metrics.get('frustration', 1.0)
        uncl = current_metrics.get('confusion', 1.0)
        behav_conf = current_metrics.get('confidence', 1.0)
        self.strain_label.config(text=f"Effort Indicator: {strain:.2f}")
        self.strain_progress['value'] = strain
        self.unclarity_label.config(text=f"Focus Indicator: {uncl:.2f}")
        self.unclarity_progress['value'] = uncl
        self.behavioral_conf_label.config(text=f"Behavioral Confidence: {behav_conf:.2f}")
        self.behavioral_conf_progress['value'] = behav_conf
        self.engagement_score_label.config(text=f"Overall P.A.C.E. Engagement Score: {engagement_score}/10")
        if state.behavioral_thread_running:
             self.after_id_behavioral = self.after(500, self.update_behavioral_metrics)
    def update_pacing_timer(self):
        if state.current_question < len(state.question_set) and state.question_start_time is not None:
            elapsed = time.time() - state.question_start_time
            minutes, seconds = int(elapsed // 60), int(elapsed % 60)
            self.pacing_timer_label.config(text=f"Pacing: {minutes:02d}:{seconds:02d}")
            self.after_id_pacing = self.after(1000, self.update_pacing_timer)
        else:
            self.pacing_timer_label.config(text="Pacing: --:--")
    def update_question(self):
        if state.question_set is None or not state.question_set:
            self.question_text_widget.config(state='normal')
            self.question_text_widget.delete(1.0, tk.END)
            self.question_text_widget.insert(tk.END, "e-GMAT Data Error: Question set not loaded.")
            self.question_text_widget.config(state='disabled')
            for widget in self.options_frame.winfo_children():
                 widget.destroy()
            print("e-GMAT Test: Update question called with no questions available.")
            return
        if state.current_question < len(state.question_set):
            q = state.question_set[state.current_question]
            self.q_info_label.config(text=f"Area: {q.get('topic', 'N/A')} - Sub-Area: {q.get('subtopic', 'N/A')} (Difficulty: {q.get('difficulty', 'N/A')})")
            self.q_number_label.config(text=f"Question {state.current_question + 1}/{len(state.question_set)}")
            self.question_text_widget.config(state='normal')
            self.question_text_widget.delete(1.0, tk.END)
            self.question_text_widget.insert(tk.END, q.get('question', 'Error loading question text.'))
            self.question_text_widget.config(state='disabled')
            for widget in self.options_frame.winfo_children():
                widget.destroy()
            self.options_var.set("")
            options = q.get('options', [])
            if isinstance(options, list) and len(options) == 4:
                for option in options:
                    if isinstance(option, str) and len(option) > 2 and option[1:2] == '.':
                        rb = tk.Radiobutton(self.options_frame,
                                          text=option,
                                          variable=self.options_var,
                                          value=option,
                                          fg="#FFFFFF", bg="#000000",
                                          selectcolor="#1a1a1a",
                                          activebackground="#333333", activeforeground="#00FF00",
                                          font=("Helvetica", 10),
                                          justify="left", anchor="w",
                                          indicatoron=0,
                                          width=400,
                                          pady=5
                                          )
                        rb.pack(anchor="w")
                    else:
                        print(f"e-GMAT Test: Warning: Invalid option format for Q{q.get('question_id', 'N/A')}: '{option}'. Skipping option.")
            else:
                 print(f"e-GMAT Test: Warning: Question {q.get('question_id', 'N/A')} has invalid or missing options list.")
                 tk.Label(self.options_frame, text="[Error: Options missing or invalid]", fg="#FF0000", bg="#000000").pack()
            progress_value = state.current_question
            self.progress_label.config(text=f"Adaptive Practice Progress: {progress_value}/{TOTAL_QUESTIONS} Questions ({progress_value/TOTAL_QUESTIONS*100:.1f}%)")
            self.test_progress_bar['value'] = progress_value
            state.question_start_time = time.time()
            self.update_pacing_timer()
            if not self.update_loop_started:
                 print("e-GMAT Test: Starting behavioral, camera, and pacing update loops...")
                 if state.camera_capture and state.camera_capture.isOpened():
                      self.update_webcam()
                 else:
                      self.notify_camera_unavailable()
                 self.update_behavioral_metrics()
                 self.update_loop_started = True
        else:
            print("e-GMAT Test: All questions completed. Finalizing session.")
            self.cancel_updates()
            self.controller.show_page("CountdownPage")
    def submit_response(self):
        if state.current_question >= len(state.question_set) or state.question_set is None:
            print("e-GMAT Test: Submit attempted on last question or with no questions loaded.")
            return
        selected_option = self.options_var.get()
        if not selected_option:
             messagebox.showwarning("Selection Required", "Please select a response option before submitting.")
             return
        q = state.question_set[state.current_question]
        time_spent = time.time() - (state.question_start_time if state.question_start_time is not None else time.time())
        state.question_start_time = None
        with state.state_lock:
            behavioral_metrics_at_submission = state.behavioral_metrics.copy()
        q_data_for_analysis = {
             "question_id": q.get('question_id', state.current_question + 1),
             "question_text": q.get('question', 'N/A'),
             "options": q.get('options', []),
             "learner_response": selected_option,
             "correct_answer": q.get('answer', 'N/A'),
             "correct": selected_option.startswith(q.get('answer', 'N/A') + '.'),
             "time_spent": time_spent,
             "behavioral_metrics": behavioral_metrics_at_submission
        }
        print(f"e-GMAT Test: Submitting response for Q {state.current_question + 1}. Behavioral metrics at submission: {behavioral_metrics_at_submission}")
        ai_q_insight_text = analyze_question_with_ai(q_data_for_analysis)
        state.responses.append({
            "question_id": q.get('question_id', state.current_question + 1),
            "response": selected_option,
            "time_spent": time_spent,
            "behavioral_metrics": behavioral_metrics_at_submission,
            "ai_insight": ai_q_insight_text
        })
        correct = q_data_for_analysis['correct']
        state.current_question += 1
        if state.current_question < len(state.question_set):
            self.update_question()
        else:
            self.update_question()
            print("e-GMAT Test: Session complete. Proceeding to report generation.")
            self.cancel_updates()
            self.controller.show_page("CountdownPage")
    def cancel_updates(self):
         print("e-GMAT Updates: Attempting to cancel update loops...")
         if self.after_id_webcam:
              try: self.after_cancel(self.after_id_webcam); print("e-GMAT Updates: Cancelled webcam update.")
              except tk.TclError: pass
              self.after_id_webcam = None
         if self.after_id_behavioral:
              try: self.after_cancel(self.after_id_behavioral); print("e-GMAT Updates: Cancelled behavioral metrics update.")
              except tk.TclError: pass
              self.after_id_behavioral = None
         if self.after_id_pacing:
              try: self.after_cancel(self.after_id_pacing); print("e-GMAT Updates: Cancelled pacing timer update.")
              except tk.TclError: pass
              self.after_id_pacing = None
         self.update_loop_started = False

class CountdownPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#000000")
        self.controller = controller
        tk.Label(self, text="Adaptive Practice Session Completed!", fg="#00FFFF", bg="#000000", font=("Helvetica", 20, "bold")).pack(pady=40)
        tk.Label(self, text="Preparing your detailed Personalized Performance Report...", fg="#FFFFFF", bg="#000000", font=("Helvetica", 12)).pack(pady=10)
        self.processing_timer_label = tk.Label(self, text="Processing Data...", fg="#FFFF00", bg="#000000", font=("Courier", 18, "bold"))
        self.processing_timer_label.pack(pady=20)
        self.processing_message = tk.Label(self, text="Analyzing Performance Metrics...", fg="#00FF00", bg="#000000", font=("Helvetica", 10))
        self.processing_message.pack(pady=10)
    def start_countdown(self):
        print("e-GMAT Processing: Initiating report generation finalization...")
        if state.pdf_report_data is not None:
             print("e-GMAT Processing: Report processing complete. Transitioning to report display.")
             self.processing_timer_label.config(text="Processing Complete.")
             self.processing_message.config(text="Generating Report Interface...")
             self.after(2000, lambda: self.controller.show_page("ReportPage"))
        else:
             print("e-GMAT Processing: Report data not available after processing step. Error.")
             self.processing_timer_label.config(text="Processing Error")
             self.processing_message.config(text="Could not generate report data.")
             self.after(5000, lambda: self.controller.show_page("HomePage"))

class ReportPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#000000")
        self.controller = controller
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#000000')
        self.style.configure('TLabel', background='#000000', foreground='#FFFFFF')
        self.style.configure('TButton', background='#1a1a1a', fg="#00FF00", activebackground="#00FFFF", activeforeground="#000000", font=('Helvetica', 10, 'bold'))
        self.style.map('TButton', background=[('active', '#00FF00')])
        tk.Label(self, text="e-GMAT Personalized Performance Report", fg="#00FFFF", bg="#000000", font=("Helvetica", 16, "bold")).pack(pady=10)
        report_frame = tk.Frame(self, bg="#1a1a1a")
        report_frame.pack(pady=10, padx=10, fill="both", expand=True)
        self.report_text = tk.Text(report_frame, height=20, width=80, bg="#1a1a1a", fg="#FFFFFF", wrap="word", font=('Courier', 10), insertbackground="#FFFFFF")
        self.report_text.pack(side="left", fill="both", expand=True)
        report_scrollbar = ttk.Scrollbar(report_frame, command=self.report_text.yview)
        report_scrollbar.pack(side="right", fill="y")
        self.report_text.config(yscrollcommand=report_scrollbar.set)
        self.report_text.config(state='disabled')
        button_frame = tk.Frame(self, bg="#000000")
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Download Raw Session Data (JSON)", command=self.download_json).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Download Comprehensive Report (PDF)", command=self.download_pdf).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Launch Another Session", command=self.restart_session).grid(row=0, column=2, padx=5)
    def update_report(self):
        self.report_text.config(state='normal')
        self.report_text.delete(1.0, tk.END)
        session_data = state.session_data
        performance_analysis = state.performance_analysis
        ai_comprehensive_analysis_text = state.ai_comprehensive_analysis_text
        if not session_data or not performance_analysis:
             self.report_text.insert(tk.END, "e-GMAT Report: Error: Session results or analysis data not available.")
             self.report_text.config(state='disabled')
             print("e-GMAT Report: Error: Attempted to update report with no data.")
             return
        self.report_text.insert(tk.END, f"--- e-GMAT Personalized Performance Report ---\n")
        self.report_text.insert(tk.END, f"Learner: {state.learner_name}\n")
        self.report_text.insert(tk.END, f"Session ID: {session_data.get('session_id', 'N/A')}, Date: {session_data.get('date', 'N/A')}\n")
        self.report_text.insert(tk.END, f"Session Score: {session_data.get('score', 0)}/{session_data.get('total', 1)} ({session_data.get('score', 0)/session_data.get('total',1)*100 if session_data.get('total',1)>0 else 0:.1f}%)\n")
        with state.state_lock:
             self.report_text.insert(tk.END, f"Overall P.A.C.E. Engagement Score during session: {state.engagement_score}/10\n\n")
        self.report_text.insert(tk.END, "--- Performance Summary by Topic ---\n")
        topics_summary = performance_analysis.get('summary', {}).get('topics', {})
        if topics_summary:
             for topic, stats in topics_summary.items():
                recommended_time = RECOMMENDED_TIME_QUANT if topic in ['Algebra', 'Geometry', 'Quant', 'Data Insights'] else RECOMMENDED_TIME_VERBAL
                self.report_text.insert(tk.END,
                    f"Area: {topic}: Acc: {stats.get('correct', 0)}/{stats.get('total', 0)} ({stats.get('accuracy',0)*100:.1f}%), "
                    f"Pacing: {stats.get('avg_time',0):.1f}s (Rec {recommended_time:.1f}s), "
                    f"Perceived Conf: {stats.get('perceived_confidence_score',0):.1f}/100, "
                    f"Behavioral (Avg): Effort={stats.get('avg_behavioral',{}).get('frustration',0):.1f}, "
                    f"Focus={stats.get('avg_behavioral',{}).get('confusion',0):.1f}, "
                    f"Confidence={stats.get('avg_behavioral',{}).get('confidence',0):.1f}\n"
                )
        else:
             self.report_text.insert(tk.END, "No topic performance data available.\n")
        self.report_text.insert(tk.END, "\n")
        self.report_text.insert(tk.END, "--- Key P.A.C.E. Recommendations ---\n")
        recommendations = performance_analysis.get('recommendations', [])
        if recommendations:
            for rec in recommendations:
                 self.report_text.insert(tk.END, f"- {rec}\n")
        else:
             self.report_text.insert(tk.END, "No specific recommendations generated by the system.\n")
        self.report_text.insert(tk.END, "\n")
        self.report_text.insert(tk.END, "--- e-GMAT AI Comprehensive Analysis ---\n")
        if ai_comprehensive_analysis_text:
            self.report_text.insert(tk.END, ai_comprehensive_analysis_text + "\n\n")
        else:
            self.report_text.insert(tk.END, "Comprehensive AI analysis was not generated.\n\n")
        self.report_text.insert(tk.END, "--- Notes ---")
        self.report_text.insert(tk.END, "Download the Comprehensive Report (PDF) for detailed subtopic breakdown, visual charts (performance & behavioral), and question-by-question AI insights.\n")
        self.report_text.config(state='disabled')
    def download_json(self):
        if state.session_data is None:
            messagebox.showerror("e-GMAT Save Error", "No session data available to save.")
            return
        try:
            default_filename = f"egmat_session_data_{state.learner_id}_session_{state.session_data.get('session_id', 'N/A')}_{datetime.now().strftime('%Y%m%d')}.json"
            file_path = filedialog.asksaveasfilename(
                initialfile=default_filename,
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, "w", encoding='utf-8') as f:
                    json.dump(state.session_data, f, indent=4, cls=NumpyEncoder)
                messagebox.showinfo("e-GMAT Save Success", f"Raw session data saved to {file_path}")
                print(f"e-GMAT Save: JSON session data saved to {file_path}")
        except Exception as e:
            messagebox.showerror("e-GMAT Save Error", f"Failed to save JSON session data: {e}")
            print(f"e-GMAT Save: Error during JSON save: {e}")
    def download_pdf(self):
        if state.pdf_report_data is None:
            messagebox.showerror("e-GMAT Save Error", "PDF report data is not available.")
            return
        try:
            default_filename = f"egmat_personalized_report_{state.learner_id}_session_{state.session_data.get('session_id', 'N/A')}_{datetime.now().strftime('%Y%m%d')}.pdf"
            file_path = filedialog.asksaveasfilename(
                initialfile=default_filename,
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, "wb") as f:
                    f.write(state.pdf_report_data)
                messagebox.showinfo("e-GMAT Save Success", f"Comprehensive Report (PDF) saved to {file_path}")
                print(f"e-GMAT Save: PDF report saved to {file_path}")
        except Exception as e:
            messagebox.showerror("e-GMAT Save Error", f"Failed to save PDF report: {e}")
            print(f"e-GMAT Save: Error during PDF save: {e}")
    def restart_session(self):
        print("e-GMAT Navigation: Launching another Adaptive Practice Session...")
        state.behavioral_thread_running = False
        if state.camera_capture:
             state.camera_capture.release()
             state.camera_capture = None
             print("e-GMAT Camera: Capture device released during session restart.")
        state.page = 'home'
        with state.state_lock:
             state.behavioral_metrics = {"frustration": 1.0, "confusion": 1.0, "confidence": 1.0}
             state.behavioral_history = []
        state.session_data = None
        state.engagement_score = 0
        state.learner_id = None
        state.learner_name = None
        state.current_question = 0
        state.responses = []
        state.question_start_time = None
        state.question_set = None
        state.performance_analysis = None
        state.session_visualizations = None
        state.pdf_report_data = None
        state.ai_comprehensive_analysis_text = None
        self.controller.show_page("HomePage")

if __name__ == "__main__":
    app = GMATApp()
    app.mainloop()