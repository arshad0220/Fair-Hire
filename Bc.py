import json
import random
import datetime
import math
import time
import os
import re
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import google.generativeai as genai

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.units import inch
from reportlab.lib import colors

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import sys

REPORT_FOLDER = r"C:\Users\HI\Desktop\fair hire"

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("INFO: GEMINI_API_KEY not found. Continuing with simulated API calls...")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        print("Continuing with simulated API calls...")

class ReportParser:
    def parse(self, report_input, input_type='pdf'):
        if input_type == 'pdf':
            print("Attempting PDF processing...")
            if report_input is None or report_input == "simulate":
                print("No specific report path provided or simulation requested. Using simulated data.")
                return self._simulate_pdf_parsing()

            try:
                pdf_file_path = report_input
                if not os.path.exists(pdf_file_path):
                     print(f"Error: PDF file not found at {pdf_file_path}")
                     print("Falling back to simulated data.")
                     return self._simulate_pdf_parsing()

                print(f"Processing Report: {pdf_file_path}")
                print("Simulating Gemini Vision for PDF processing...")
                extracted_data = self._simulate_pdf_parsing()
                print("Simulated PDF processing complete.")
                return extracted_data

            except Exception as e:
                print(f"Error during PDF processing: {e}")
                print("Falling back to basic simulated data.")
                return self._simulate_pdf_parsing()
        else:
            print(f"Error: Unsupported input type '{input_type}'. Only PDF is currently processed for full report.")
            return None

    def _simulate_pdf_parsing(self):
         return {
                "student_id": f"student_{random.randint(1000, 9999)}",
                "report_date": datetime.date.today().isoformat(),
                "overall_score": 650 + random.randint(-30, 30),
                "sections": {
                    "Quant": {
                        "score": 75 + random.randint(-8, 8),
                        "subtopics": {
                            "Algebra": {"accuracy": round(random.uniform(0.4, 0.9), 2) , "avg_time_s": 110 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(8,12)},
                            "Arithmetic": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 130 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(8,12)},
                            "Problem Solving (General)": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 100 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(6,10)},
                        },
                        "total_questions": 21,
                        "total_time_m": 45
                    },
                    "Verbal": {
                        "score": 75 + random.randint(-8, 8),
                        "subtopics": {
                            "Critical Reasoning": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 140 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(11,15)},
                            "Reading Comprehension": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 120 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(8,10)},
                        },
                        "total_questions": 23,
                        "total_time_m": 45
                    },
                    "Data Insights": {
                        "score": 75 + random.randint(-8, 8),
                        "subtopics": {
                            "Data Sufficiency": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 115 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(7,9)},
                            "Multi-Source Reasoning": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 150 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(3,5)},
                             "Table Analysis": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 105 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(3,5)},
                              "Graphics Interpretation": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 95 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(3,5)},
                               "Two-Part Analysis": {"accuracy": round(random.uniform(0.4, 0.9), 2), "avg_time_s": 135 + random.randint(-20, 20), "confidence": round(random.uniform(0.3, 0.95), 2), "frustration": round(random.uniform(0.05, 0.5), 2), "questions_attempted": random.randint(3,5)},
                        },
                         "total_questions": 20,
                         "total_time_m": 45
                    }
                }
            }

    def extract_features(self, parsed_data):
        if not parsed_data: return None
        features = {
            "report_date": parsed_data.get("report_date", 'N/A'),
            "overall_score": parsed_data.get("overall_score"),
            "sections": {},
            "behavioral": { "pacing": {}, "emotional_flags": {} },
            "swot": { "strengths": [], "weaknesses": [], "opportunities": [], "threats": [] }
        }
        for section, data in parsed_data.get("sections", {}).items():
            total_questions = data.get("total_questions", 1)
            total_time_m = data.get("total_time_m", 0)
            total_time_s = total_time_m * 60 if total_time_m > 0 else 0

            total_time_calc_s = 0
            total_q_attempted_calc = 0
            subtopic_data = data.get("subtopics", {})
            for st, d_item in subtopic_data.items():
                 avg_t = d_item.get("avg_time_s")
                 q_att = d_item.get("questions_attempted")
                 if avg_t is not None and q_att is not None and q_att > 0:
                     total_time_calc_s += avg_t * q_att
                     total_q_attempted_calc += q_att

            if total_q_attempted_calc > 0:
                 avg_time_per_q = total_time_calc_s / total_q_attempted_calc
            elif total_time_s > 0 and total_questions > 0:
                 avg_time_per_q = total_time_s / total_questions
            else:
                 avg_time_per_q = 135

            section_score = data.get("score")
            subtopic_accuracies_processed = {}
            for st, d_item in subtopic_data.items():
                acc = d_item.get("accuracy")
                subtopic_accuracies_processed[st] = float(acc) if acc is not None else None

            features["sections"][section] = {
                "score": section_score,
                "subtopic_accuracies": subtopic_accuracies_processed,
                "average_time_per_q_s": avg_time_per_q,
                "subtopic_details": subtopic_data
            }
            features["behavioral"]["pacing"][section] = avg_time_per_q
            features["behavioral"]["emotional_flags"][section] = []

            ideal_time_s = 135

            for subtopic, st_data in subtopic_data.items():
                accuracy = subtopic_accuracies_processed.get(subtopic)
                confidence = float(st_data.get("confidence", 0.5)) if st_data.get("confidence") is not None else 0.5
                frustration = float(st_data.get("frustration", 0)) if st_data.get("frustration") is not None else 0
                avg_time_s = int(st_data.get("avg_time_s", 135)) if st_data.get("avg_time_s") is not None else 135
                topic_id = f"{section} - {subtopic}"


                if accuracy is not None:
                    if accuracy >= 0.80 and confidence >= 0.8: features["swot"]["strengths"].append(topic_id)
                    elif accuracy < 0.60:
                        features["swot"]["weaknesses"].append(topic_id)
                        if frustration > 0.4: features["behavioral"]["emotional_flags"][section].append(f"{subtopic}: High Frustration ({frustration:.1f})")
                        if confidence < 0.5: features["behavioral"]["emotional_flags"][section].append(f"{subtopic}: Low Confidence ({confidence:.1f})")
                    elif accuracy < 0.75: features["swot"]["opportunities"].append(topic_id)

                if avg_time_s > ideal_time_s * 1.3:
                    features["swot"]["threats"].append(f"{topic_id} (Slow Pacing: {avg_time_s}s)")
                    features["behavioral"]["emotional_flags"][section].append(f"{subtopic}: Pacing Issue (Slow: {avg_time_s}s)")
                elif avg_time_s < ideal_time_s * 0.7:
                     features["swot"]["threats"].append(f"{topic_id} (Fast Pacing: {avg_time_s}s)")
                     features["behavioral"]["emotional_flags"][section].append(f"{subtopic}: Pacing Issue (Fast: {avg_time_s}s)")

        return features

class ScoringEngine:
    def __init__(self):
        self.percentile_map = {
            805: 100, 785: 99.9, 765: 99.9, 755: 99.7, 745: 99.4, 735: 98.9, 725: 98.4, 715: 97.4,
            705: 96.1, 695: 94.8, 685: 92.6, 675: 89.9, 665: 86.9, 655: 83.5, 645: 78.8, 635: 74.3,
            625: 69.9, 615: 64.7, 605: 59.4, 595: 54.1, 585: 49.7, 575: 44.5, 565: 39.9, 555: 36.1,
            545: 31.9, 535: 28.2, 525: 25.1, 515: 21.9, 505: 19.0, 495: 16.6, 485: 14.4, 475: 12.0,
            465: 10.4, 455: 8.9, 445: 7.3, 435: 6.0, 425: 4.9, 405: 3.3, 395: 2.6, 375: 1.9, 355: 1.3,
            315: 0.4, 205: 0
        }
        self.sectional_score_map_qvd = {i: i for i in range(60, 91)}

    def get_percentile_for_score(self, target_score_str):
        target_score_num = int(str(target_score_str).replace('+', ''))
        closest_lower_score = max([k for k in self.percentile_map if k <= target_score_num] or [min(self.percentile_map.keys())])
        percentile = self.percentile_map.get(closest_lower_score, 0)

        if str(target_score_str).endswith('+') and percentile < 100:
             pass

        return percentile

    def _map_overall_percentile_to_sectional_target(self, overall_percentile):
         target_score = 60 + (overall_percentile / 100) * 30
         return max(60, min(90, math.ceil(target_score)))

    def calculate_sectional_targets(self, target_score_str, current_features):
        print(f"\nCalculating Sectional Targets for GMAT Focus Score {target_score_str}...")
        target_overall_percentile = self.get_percentile_for_score(target_score_str)
        print(f"Target Score: {target_score_str} -> Target Overall Percentile: ~{target_overall_percentile:.1f}%")

        base_target = self._map_overall_percentile_to_sectional_target(target_overall_percentile)
        base_q = base_target
        base_v = base_target
        base_di = base_target
        print(f"Base targets from percentile: Q={base_q}, V={base_v}, DI={base_di}")

        current_q = current_features['sections'].get('Quant',{}).get('score') or 60
        current_v = current_features['sections'].get('Verbal',{}).get('score') or 60
        current_di = current_features['sections'].get('Data Insights',{}).get('score') or 60

        q_needed = max(0, base_q - current_q)
        v_needed = max(0, base_v - current_v)
        di_needed = max(0, base_di - current_di)

        q_target = math.ceil(min(90, max(current_q, base_q, current_q + q_needed * 1.1)))
        v_target = math.ceil(min(90, max(current_v, base_v, current_v + v_needed * 1.1)))
        di_target = math.ceil(min(90, max(current_di, base_di, current_di + di_needed * 1.1)))

        print(f"Final Adjusted Targets -> Quant: {q_target}, Verbal: {v_target}, Data Insights: {di_target}")
        return {"Quant": q_target, "Verbal": v_target, "Data Insights": di_target, "target_overall_percentile": round(target_overall_percentile, 1)}

class BehavioralAnalyzer:
    def analyze(self, features):
         print("\nAnalyzing Behavioral Data...")
         analysis = { "pacing_summary": {}, "emotional_hotspots": [] }
         ideal_time_s = 135

         pacing_data = features.get("behavioral", {}).get("pacing", {})
         if not pacing_data: return analysis

         for section, avg_time_s in pacing_data.items():
             if avg_time_s is None or avg_time_s <= 0 :
                  analysis["pacing_summary"][section] = "N/A (Missing time data)"
                  continue

             if avg_time_s > ideal_time_s * 1.25:
                 analysis["pacing_summary"][section] = f"Considerably Slow ({avg_time_s:.0f}s vs ~{ideal_time_s}s ideal)"
             elif avg_time_s < ideal_time_s * 0.75:
                 analysis["pacing_summary"][section] = f"Potentially Fast ({avg_time_s:.0f}s vs ~{ideal_time_s}s ideal)"
             else:
                 analysis["pacing_summary"][section] = f"Generally Well-Paced ({avg_time_s:.0f}s vs ~{ideal_time_s}s ideal)"

         all_flags = []
         emotional_flags = features.get("behavioral", {}).get("emotional_flags", {})
         if not emotional_flags: analysis["emotional_hotspots"].append("No significant emotional variances noted.")
         else:
             for section, flags in emotional_flags.items():
                 if flags: all_flags.extend([f"{section}: {flag}" for flag in flags])
             if all_flags: analysis["emotional_hotspots"] = sorted(list(set(all_flags)))
             else: analysis["emotional_hotspots"].append("No significant emotional variances noted.")
         print("Behavioral Analysis Complete.")
         return analysis

class AccuracyPredictor:
    def __init__(self):
        self.target_improvement_factor = 1.15
        self.hard_question_threshold = 0.75
        self.minimum_target_accuracy = 0.70
        self.maximum_target_accuracy = 0.90
        self.avg_current_focus_accuracy = {}

    def predict_hard_question_accuracy(self, features, sectional_targets):
        print("\nForecasting Target Accuracy for Challenging Questions...")
        predictions = {}
        self.avg_current_focus_accuracy = {}
        current_sections = features.get("sections", {})
        swot = features.get("swot", {})

        for section in ["Quant", "Verbal", "Data Insights"]:
            current_section_data = current_sections.get(section, {})
            subtopic_accuracies = current_section_data.get("subtopic_accuracies", {})
            self.avg_current_focus_accuracy[section] = 0.6

            if not subtopic_accuracies:
                predictions[section] = f"~{self.maximum_target_accuracy:.0%}"
                continue

            focus_accuracies_list = []
            for subtopic, acc in subtopic_accuracies.items():
                if acc is None: continue
                topic_id = f"{section} - {subtopic}"
                is_weakness = topic_id in swot.get("weaknesses", [])
                is_opportunity = topic_id in swot.get("opportunities", [])
                if is_weakness or is_opportunity:
                     focus_accuracies_list.append(acc)
                elif acc < self.hard_question_threshold:
                     focus_accuracies_list.append(acc)

            if not focus_accuracies_list:
                 predictions[section] = f"Maintain >{self.hard_question_threshold:.0%}"
                 self.avg_current_focus_accuracy[section] = self.hard_question_threshold
                 continue

            avg_current_focus_acc = sum(focus_accuracies_list) / len(focus_accuracies_list)
            self.avg_current_focus_accuracy[section] = avg_current_focus_acc

            sectional_target_score = sectional_targets.get(section, 75)
            improvement_scale = 1.0 + ((sectional_target_score - 75) / (90 - 75)) * 0.15 if (90-75) > 0 else 1.0
            adjusted_improvement_factor = self.target_improvement_factor * improvement_scale

            target_hard_accuracy = avg_current_focus_acc * adjusted_improvement_factor
            target_hard_accuracy = max(self.minimum_target_accuracy, min(self.maximum_target_accuracy, target_hard_accuracy))
            target_hard_accuracy = max(target_hard_accuracy, avg_current_focus_acc * 1.05)

            predictions[section] = f"{target_hard_accuracy:.0%}"
            print(f"  {section}: Current Avg. ({len(focus_accuracies_list)} focus areas) at {avg_current_focus_acc:.0%}. Recommended Target: {predictions[section]}.")
        print("Hard Question Accuracy Forecast Complete.")
        return predictions

class PracticePlanGenerator:
    study_pref_modifiers = {
        'A': {"weakness": 1.0, "opportunity": 1.0, "strength_maintenance": 0.15, "label": "Balanced Growth"},
        'B': {"weakness": 1.3, "opportunity": 0.8, "strength_maintenance": 0.05, "label": "Weakness Mitigation"},
        'C': {"weakness": 0.8, "opportunity": 1.2, "strength_maintenance": 0.25, "label": "Strength/Opportunity Focus"}
    }

    def __init__(self):
        self.base_drill_count = 20
        self.time_per_drill_s = 135
        self.topic_weights = { "weakness": 1.5, "opportunity": 1.0, "threat": 0.8, "emotional_flag": 0.7 }


    def generate_plan(self, features, target_score_str, study_prefs, target_accuracies, behavioral_analysis):
        print("\nConstructing Hyper-Personalized Practice Blueprint...")
        plan = { "prioritized_topics": [], "weekly_schedule": [], "overall_focus": [] }
        focus_areas = {}
        swot = features.get("swot", {})
        emotional_flags_flat = behavioral_analysis.get("emotional_hotspots", [])

        study_preference = study_prefs.get("study_preference", "A")
        pref_modifiers = self.study_pref_modifiers.get(study_preference, self.study_pref_modifiers['A'])
        pref_label = pref_modifiers["label"]

        all_subtopics = {}
        for sec, sec_data in features.get('sections', {}).items():
            for sub, details in sec_data.get('subtopic_details', {}).items():
                topic_id = f"{sec} - {sub}"
                all_subtopics[topic_id] = {"accuracy": sec_data['subtopic_accuracies'].get(sub), "score": 0}


        for topic_type, base_weight in self.topic_weights.items():
            modifier = pref_modifiers.get(topic_type, 1.0)
            weight = base_weight * modifier
            if topic_type == "emotional_flag":
                for flag_desc in emotional_flags_flat:
                     match = re.match(r"(\w+):?\s*([\w\s\(\)]+):.*", flag_desc)
                     if match:
                         section, topic_part = match.groups()
                         topic_part_clean = topic_part.split('(')[0].strip()
                         topic_id = f"{section} - {topic_part_clean}"
                         if topic_id in all_subtopics:
                             all_subtopics[topic_id]["score"] += weight
                             all_subtopics[topic_id]["reason"] = all_subtopics[topic_id].get("reason", []) + ["Emotional Pattern"]
            else:
                 topics_in_category = swot.get(topic_type, [])
                 for topic_id_raw in topics_in_category:
                     topic_id = topic_id_raw
                     reason_label = topic_type.capitalize()
                     if topic_type == 'threat':
                        if '(Slow Pacing:' in topic_id_raw: reason_label = 'Pacing (Slow)'
                        elif '(Fast Pacing:' in topic_id_raw: reason_label = 'Pacing (Fast)'
                        topic_id = topic_id_raw.split(' (')[0]

                     if topic_id in all_subtopics:
                         all_subtopics[topic_id]["score"] += weight
                         all_subtopics[topic_id]["reason"] = all_subtopics[topic_id].get("reason", []) + [reason_label]


        for topic_id, data in all_subtopics.items():
            accuracy = data.get("accuracy")
            if accuracy is not None and accuracy < 0.9:
                 deficit_score = (1.0 - accuracy) * 1.2
                 all_subtopics[topic_id]["score"] += deficit_score
                 if accuracy < 0.6 and "Weakness" not in all_subtopics[topic_id].get("reason",[]):
                      all_subtopics[topic_id]["reason"] = all_subtopics[topic_id].get("reason", []) + ["Low Accuracy"]
                 elif accuracy < 0.75 and "Opportunity" not in all_subtopics[topic_id].get("reason",[]) and "Weakness" not in all_subtopics[topic_id].get("reason",[]):
                      all_subtopics[topic_id]["reason"] = all_subtopics[topic_id].get("reason", []) + ["Accuracy Improvement"]


        strength_mod = pref_modifiers.get("strength_maintenance", 0)
        if strength_mod > 0:
            for topic_id in swot.get("strengths", []):
                 if topic_id in all_subtopics:
                      all_subtopics[topic_id]["score"] += strength_mod
                      all_subtopics[topic_id]["reason"] = all_subtopics[topic_id].get("reason", []) + ["Strength Maintenance"]

        sorted_topics_list = []
        for topic_id, data in all_subtopics.items():
             if data["score"] > 0.1:
                  unique_reasons = sorted(list(set(data.get("reason", ["Accuracy Delta"]))))
                  sorted_topics_list.append({"topic": topic_id, "priority_score": data["score"], "reason": unique_reasons})

        sorted_topics_list.sort(key=lambda item: item["priority_score"], reverse=True)

        total_study_hours_week = study_prefs.get("availability_hours", 15)
        weeks_to_test = study_prefs.get("weeks_to_test", 12)
        if weeks_to_test <= 0: weeks_to_test = 1
        total_study_hours = total_study_hours_week * weeks_to_test
        drill_hours_available = total_study_hours * 0.65
        total_priority_score = sum(item["priority_score"] for item in sorted_topics_list) + 1e-6

        for topic_item in sorted_topics_list:
            proportion = topic_item["priority_score"] / total_priority_score
            allocated_hours = drill_hours_available * proportion

            topic_target_time_s = self.time_per_drill_s
            num_drills = 0
            if topic_target_time_s > 0:
                num_drills = max(5, round((allocated_hours * 3600) / topic_target_time_s))
            num_drills = min(num_drills, 150)

            target_time_str = f"~{int(topic_target_time_s // 60)}m {int(topic_target_time_s % 60):02d}s"

            topic_item["recommended_drills"] = num_drills
            topic_item["target_timing_per_q"] = target_time_str
            topic_item["estimated_hours"] = round(allocated_hours, 1) if allocated_hours >= 0.1 else 0.1

        plan["prioritized_topics"] = sorted_topics_list

        total_est_hours_plan = sum(t['estimated_hours'] for t in plan['prioritized_topics'])
        planned_hours_per_week = (total_est_hours_plan / weeks_to_test) if weeks_to_test > 0 else total_est_hours_plan
        target_drill_hours_per_week = total_study_hours_week * 0.65
        actual_weekly_hours = min(target_drill_hours_per_week, planned_hours_per_week * 1.1)

        current_topic_idx = 0
        for week_num in range(weeks_to_test):
            week_focus_topics, week_hours_so_far = [], 0
            while current_topic_idx < len(plan['prioritized_topics']) and week_hours_so_far < actual_weekly_hours :
                topic = plan['prioritized_topics'][current_topic_idx]
                if week_hours_so_far == 0 or (week_hours_so_far + topic['estimated_hours'] < actual_weekly_hours * 1.2):
                    week_focus_topics.append(topic['topic'])
                    week_hours_so_far += topic['estimated_hours']
                    current_topic_idx += 1
                else:
                    break

            if not week_focus_topics:
                 if current_topic_idx < len(plan['prioritized_topics']):
                      topic = plan['prioritized_topics'][current_topic_idx]
                      week_focus_topics.append(topic['topic'])
                      current_topic_idx +=1
                 else:
                      break

            goal_desc = f"Target ~{week_hours_so_far:.1f} hrs practice. Focus on execution & accuracy for assigned topics."
            if not week_focus_topics:
                 goal_desc = f"Focus on overall review and timed practice sets. Target {target_drill_hours_per_week:.1f} practice hours."

            plan["weekly_schedule"].append({
                "week": week_num + 1, "focus": week_focus_topics or ["General Review"],
                "goals": goal_desc
            })

        targets_dict = study_prefs.get('targets', {})
        plan["overall_focus"].append(f"<b>Primary Objective:</b> Achieve GMAT Focus Score of <b>{target_score_str}</b> by targeting Q:{targets_dict.get('Quant','N/A')}, V:{targets_dict.get('Verbal','N/A')}, DI:{targets_dict.get('Data Insights','N/A')}.")
        plan["overall_focus"].append(f"<b>Personalized Strategy ('{study_preference}'):</b> {pref_label}. Plan tailored for {study_prefs.get('availability_hours', 'N/A')} hrs/wk over {study_prefs.get('weeks_to_test', 'N/A')} weeks.")

        pacing_issues = [f"{s} ({details})" for s, details in behavioral_analysis.get("pacing_summary", {}).items() if "Well-Paced" not in details and "N/A" not in details]
        if pacing_issues: plan["overall_focus"].append(f"<b>Critical Pacing Alert:</b> Prioritize timed practice, especially in {', '.join(pacing_issues)}.")
        else: plan["overall_focus"].append("<b>Pacing Analysis:</b> Current pacing seems generally adequate, maintain consistency with timed practice.")

        hotspots = behavioral_analysis.get("emotional_hotspots", [])
        significant_hotspots = [h for h in hotspots if 'No significant' not in h]
        if significant_hotspots:
            hotspot_summary = "; ".join(significant_hotspots[:3])
            if len(significant_hotspots) > 3: hotspot_summary += "..."
            plan["overall_focus"].append(f"<b>Performance Psychology Note:</b> Address patterns noted ({hotspot_summary}). Focus on resilience and strategy under pressure.")
        else:
             plan["overall_focus"].append(f"<b>Performance Psychology Note:</b> Maintain emotional stability during study and practice tests.")

        print("Practice Blueprint Constructed.")
        return plan

class VideoScriptGenerator:
    def generate_tts_output(self, text, filename):
        print(f"Simulating TTS generation for '{filename}'...")
        try:
             with open(filename, "w") as f: f.write(f"Simulated TTS audio content for: {text[:100]}...")
             print(f"Simulated TTS placeholder saved as {filename}")
        except Exception as fe: print(f"Could not write placeholder TTS file: {fe}")

    def generate_architect_script(self, inputs, features, targets, accuracy_preds, plan):
        print("\nGenerating Architect Briefing Script...")
        student_id=inputs.get('student_id','Unknown')
        report_date=inputs.get('report_date','N/A')
        target_score_str=inputs.get('target_score','N/A')
        overall_s=features.get('overall_score','N/A')
        q_s=features.get('sections',{}).get('Quant',{}).get('score','N/A')
        v_s=features.get('sections',{}).get('Verbal',{}).get('score','N/A')
        di_s=features.get('sections',{}).get('Data Insights',{}).get('score','N/A')
        s_engine=inputs.get('scoring_engine')
        target_overall_p = targets.get('target_overall_percentile', 'N/A')

        q_target=targets.get('Quant','N/A')
        v_target=targets.get('Verbal','N/A')
        di_target=targets.get('Data Insights','N/A')

        q_hard=accuracy_preds.get('Quant','N/A')
        v_hard=accuracy_preds.get('Verbal','N/A')
        di_hard=accuracy_preds.get('Data Insights','N/A')

        top_topic=plan.get('prioritized_topics',[{}])[0].get('topic','N/A') if plan.get('prioritized_topics') else 'N/A'
        diag_ans = inputs.get('diagnostic_answers',{})
        pref_code = diag_ans.get('study_preference','A')

        pref_desc = PracticePlanGenerator.study_pref_modifiers.get(pref_code,{}).get('label','Unknown')

        acc_predictor = inputs.get("accuracy_predictor")
        q_avg_acc_str = f"{acc_predictor.avg_current_focus_accuracy.get('Quant', 0):.0%}" if acc_predictor else "N/A"
        v_avg_acc_str = f"{acc_predictor.avg_current_focus_accuracy.get('Verbal', 0):.0%}" if acc_predictor else "N/A"
        di_avg_acc_str = f"{acc_predictor.avg_current_focus_accuracy.get('Data Insights', 0):.0%}" if acc_predictor else "N/A"


        script = f"""
--- E-GMAT AI SUCCESS PATH: ARCHITECT BRIEFING ---
CLIENT: {student_id} | DATE: {datetime.date.today().isoformat()} | Report Basis: {report_date} Data | GOAL: GMAT Focus {target_score_str} (~{target_overall_p}th %ile)
CURRENT STANDING (from report): Overall {overall_s} (Q:{q_s}, V:{v_s}, DI:{di_s})

RECOMMENDED SECTIONAL TARGETS:
- Quant: {q_target} | Verbal: {v_target} | Data Insights: {di_target}
REASONING: Calculated distribution based on goal percentile ({target_overall_p}th) and current performance profile ({overall_s}, Q:{q_s}, V:{v_s}, DI:{di_s}) to maximize probability of reaching {target_score_str}.

KEY PERFORMANCE INDICATORS (DIFFICULT QUESTIONS ACCURACY TARGETS):
- Quant Target: {q_hard}
- Verbal Target: {v_hard}
- Data Insights Target: {di_hard}
RATIONALE: Forecasted accuracy needed on higher-difficulty questions to achieve target sectional scores, derived from current accuracy in focus areas (Quant avg: ~{q_avg_acc_str}, Verbal avg: ~{v_avg_acc_str}, DI avg: ~{di_avg_acc_str}).

PRACTICE BLUEPRINT SUMMARY (Hyper-Personalized):
- Prioritization Focus: Key leverage points identified (Top: {top_topic}, others in plan). Score weightings adjusted by strategy '{pref_code}' ({pref_desc}).
- Cadence: Tailored for {diag_ans.get('availability_hours','N/A')} hrs/wk over {diag_ans.get('weeks_to_test','N/A')} wks. Drill hours allocated proportionally to priority scores.
- Behavioral Factors: Pacing adjustments and emotional resilience flags incorporated into plan implicitly through topic prioritization and overall guidance.

OUTPUTS: Detailed PDF Client Report | Client-Facing Video Script (TTS Simulated) | This Briefing
--- END OF BRIEFING ---"""
        print("Architect Briefing Generated.")
        try:
            with open("architect_script.txt", "w", encoding='utf-8') as f: f.write(script)
            print("Architect Script saved.")
        except Exception as e: print(f"Error saving architect script: {e}")
        return script

    def generate_student_script(self, inputs, features, targets, accuracy_preds, plan, diagnostic_answers):
        print("\nGenerating Personalized Student Video Script...")
        student_name=diagnostic_answers.get("name","Valued Student")
        target_score_str=inputs.get('target_score','your GMAT goal')
        current_score=features.get('overall_score','your recent score')
        q_t=targets.get('Quant','target Q')
        v_t=targets.get('Verbal','target V')
        di_t=targets.get('Data Insights','target DI')
        q_acc=accuracy_preds.get('Quant','target Q Acc')
        v_acc=accuracy_preds.get('Verbal','target V Acc')
        di_acc=accuracy_preds.get('Data Insights','target DI Acc')
        top_focus=plan.get('prioritized_topics',[{}])[0].get('topic','key areas') if plan.get('prioritized_topics') else 'key areas'

        script = f"""
Hi {student_name}, this is your e-GMAT AI, Alex! We've processed your latest GMAT Focus report data to create your personalized success path towards your goal of {target_score_str}.

You're starting from a score around {current_score}. To efficiently reach {target_score_str}, our analysis suggests focusing on these sectional targets: aiming for approximately {q_t} in Quant, {v_t} in Verbal, and {di_t} in Data Insights.

Achieving these scores, especially on the trickier questions, means boosting your accuracy. We recommend striving for around {q_acc} accuracy in challenging Quant questions, {v_acc} in Verbal, and {di_acc} in Data Insights. Your detailed plan will guide you here.

Your hyper-personalized study blueprint is outlined in the PDF report. It highlights specific areas needing attention, like {top_focus}, and provides estimated drills and timings. It's built around your {diagnostic_answers.get('availability_hours','available')} study hours per week and your preferred approach.

{student_name}, commitment and strategy are crucial. Follow your personalized blueprint, leverage the e-GMAT platform resources, and keep track of your progress. We're confident you can reach your {target_score_str} goal! Please review your detailed PDF report now for the full plan. Good luck!
"""
        print("Student Video Script Generated.")
        tts_filename = "student_tts_output.mp3"
        try:
            with open("student_script.txt", "w", encoding='utf-8') as f: f.write(script)
            print("Student Script saved.")
            self.generate_tts_output(script, tts_filename)
        except Exception as e: print(f"Error saving student script/TTS: {e}")
        return script

class DiagnosticForm:
    def __init__(self, initial_report_path=None):
        self.root = tk.Tk()
        self.root.title("e-GMAT AI Success Path Diagnostic")
        self.name_var = tk.StringVar()
        self.target_score_var = tk.StringVar(value="705")
        self.report_path_var = tk.StringVar()
        self.hours_var = tk.IntVar(value=15)
        self.weeks_var = tk.IntVar(value=12)
        self.style_var = tk.StringVar(value='A')
        self.status_var = tk.StringVar()
        self.results = None

        if initial_report_path and os.path.exists(initial_report_path):
            self.report_path_var.set(initial_report_path)
            self.status_var.set(f"Auto-detected report: {os.path.basename(initial_report_path)}. Verify or select simulation.")
        else:
            self.report_path_var.set("")
            self.status_var.set("No report auto-detected. Select simulation or Browse.")

        self._create_widgets()

    def _create_widgets(self):
        frame = ttk.Frame(self.root, padding="25 25 25 25")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        row_num = 0

        ttk.Label(frame, text="Client First Name:").grid(column=0, row=row_num, sticky=tk.W, pady=6)
        name_entry = ttk.Entry(frame, width=45, textvariable=self.name_var)
        name_entry.grid(column=1, row=row_num, sticky=(tk.W, tk.E), pady=6, columnspan=2)
        row_num += 1

        ttk.Label(frame, text="Target GMAT Focus Score:").grid(column=0, row=row_num, sticky=tk.W, pady=6)
        score_entry = ttk.Entry(frame, width=18, textvariable=self.target_score_var)
        score_entry.grid(column=1, row=row_num, sticky=tk.W, pady=6)
        ttk.Label(frame, text="(e.g., 655, 715+) Range: 205-805").grid(column=2, row=row_num, sticky=tk.W, padx=6)
        row_num += 1

        ttk.Label(frame, text="GMAT Focus Report PDF:").grid(column=0, row=row_num, sticky=tk.W, pady=6)
        path_entry = ttk.Entry(frame, width=45, textvariable=self.report_path_var, state='readonly')
        path_entry.grid(column=1, row=row_num, sticky=(tk.W, tk.E), pady=6, columnspan=2)
        browse_button = ttk.Button(frame, text="Browse...", command=self._browse_file)
        browse_button.grid(column=3, row=row_num, sticky=tk.W, padx=6)
        simulate_button = ttk.Button(frame, text="Use Simulated Report Data", command=self._simulate_report_path)
        simulate_button.grid(column=1, row=row_num + 1, columnspan=2, sticky=tk.W, padx=0, pady=(2,8))
        row_num += 2

        ttk.Label(frame, text="Study Commitment:").grid(column=0, row=row_num, sticky=tk.W, pady=6)
        hours_spinbox = ttk.Spinbox(frame, from_=1, to=100, width=6, textvariable=self.hours_var)
        hours_spinbox.grid(column=1, row=row_num, sticky=tk.W, pady=6)
        ttk.Label(frame, text="hours/week for approx.").grid(column=2, row=row_num, sticky=tk.W, padx=6)
        weeks_spinbox = ttk.Spinbox(frame, from_=1, to=52, width=6, textvariable=self.weeks_var)
        weeks_spinbox.grid(column=3, row=row_num, sticky=tk.W, pady=6)
        row_num += 1

        ttk.Label(frame, text="Strategic Approach:").grid(column=0, row=row_num, sticky=tk.W, pady=6)
        style_frame = ttk.Frame(frame)
        style_frame.grid(column=1, row=row_num, columnspan=3, sticky=tk.W)
        ttk.Radiobutton(style_frame, text="Balanced (A)", variable=self.style_var, value='A').pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(style_frame, text="Weakness Focus (B)", variable=self.style_var, value='B').pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(style_frame, text="Strength Focus (C)", variable=self.style_var, value='C').pack(side=tk.LEFT, padx=6)
        row_num += 1

        submit_button = ttk.Button(frame, text="Generate Personalized Success Path", command=self._submit, style="Accent.TButton")
        ttk.Style().configure("Accent.TButton", font=('Helvetica', 10, 'bold'))
        submit_button.grid(column=1, row=row_num, columnspan=2, pady=25)
        row_num += 1

        status_label = ttk.Label(frame, textvariable=self.status_var, relief=tk.GROOVE, anchor=tk.W, padding=5)
        status_label.grid(column=0, row=row_num, columnspan=4, sticky=(tk.W, tk.E), pady=6)
        name_entry.focus()
        self.root.bind('<Return>', lambda event: self._submit())

    def _browse_file(self):
        filepath = filedialog.askopenfilename(title="Select GMAT Focus Report PDF", filetypes=[("PDF files", "*.pdf")], initialdir=REPORT_FOLDER)
        if filepath:
            self.report_path_var.set(filepath)
            self.status_var.set(f"Report selected: {os.path.basename(filepath)}")

    def _simulate_report_path(self):
         self.report_path_var.set("simulate")
         self.status_var.set("SUCCESS PATH: Simulation mode activated for report data.")

    def _validate_inputs(self):
        name=self.name_var.get().strip()
        score_str=self.target_score_var.get().strip()
        report_path=self.report_path_var.get().strip()
        hours=self.hours_var.get()
        weeks=self.weeks_var.get()

        if not name: messagebox.showerror("Validation Error", "Client First Name is required."); return False
        if not score_str: messagebox.showerror("Validation Error", "Target GMAT Focus Score is required."); return False
        try:
            score_val = int(score_str.replace('+', ''))
            if not (205 <= score_val <= 805): messagebox.showerror("Validation Error", "Target score must be 205-805 for GMAT Focus."); return False
            if score_str.endswith('+') and score_val == 805: messagebox.showerror("Validation Error", "Target cannot be '805+'. Use '805'."); return False
            if score_val < 400: messagebox.showwarning("Input Check", f"Target score {score_str} seems low. Ensure this is correct for GMAT Focus Edition.")
            if score_val > 785 and not score_str.endswith('+'): messagebox.showwarning("Input Check", f"Target score {score_str} is very high. Ensure this is correct.")

        except ValueError: messagebox.showerror("Validation Error", "Invalid target score. Use numbers (e.g., 655) or number+ (e.g., 705+)."); return False

        if not report_path: messagebox.showerror("Validation Error", "Please select a GMAT Focus report PDF, use the auto-detected one, or activate Simulation Mode."); return False
        if report_path!="simulate" and not os.path.exists(report_path): messagebox.showerror("Validation Error", f"Report file not found: {report_path}"); return False
        if report_path!="simulate" and not report_path.lower().endswith(".pdf"): messagebox.showerror("Validation Error", "Selected file must be a PDF."); return False

        if hours<=0 or weeks<=0: messagebox.showerror("Validation Error", "Study hours and weeks must be positive values."); return False
        if hours * weeks < 10: messagebox.showwarning("Input Check", f"Total study time ({hours*weeks} hours) is very low. Ensure this is sufficient.")
        if hours > 40: messagebox.showwarning("Input Check", f"Weekly study ({hours} hours) is very high. Ensure this is sustainable.")

        return True

    def _submit(self):
        if self._validate_inputs():
            self.status_var.set("Inputs validated. Generating Success Path...")
            self.results = {"name": self.name_var.get().strip(), "target_score": self.target_score_var.get().strip(),
                            "report_path": self.report_path_var.get().strip(), "report_type": 'pdf',
                            "availability_hours": self.hours_var.get(), "weeks_to_test": self.weeks_var.get(),
                            "study_preference": self.style_var.get()}
            self.root.quit()
            self.root.destroy()
        else:
            self.status_var.set("Validation Error. Please review input fields and messages.")

    def run(self):
        self.root.mainloop()
        return self.results

class PDFReportGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.doc = SimpleDocTemplate(filename, pagesize=(8.5*inch, 11*inch), leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
        self.styles = getSampleStyleSheet()
        self.story = []
        self.chart_image_path = None
        self._define_styles()

    def _define_styles(self):
        self.styles.add(ParagraphStyle(name='ReportTitle', parent=self.styles['h1'], alignment=TA_CENTER, fontSize=20, spaceBefore=0, spaceAfter=0.25*inch, textColor=colors.HexColor("#003366")))
        self.styles.add(ParagraphStyle(name='SubTitle', parent=self.styles['h2'], alignment=TA_CENTER, fontSize=12, spaceAfter=0.2*inch, textColor=colors.HexColor("#4F4F4F")))
        self.styles.add(ParagraphStyle(name='SectionHeading', parent=self.styles['h2'], fontSize=16, spaceBefore=0.3*inch, spaceAfter=0.15*inch, textColor=colors.HexColor("#005A9C"), borderPadding=4, leading=18))
        self.styles.add(ParagraphStyle(name='SubSectionHeading', parent=self.styles['h3'], fontSize=13, spaceAfter=0.1*inch, textColor=colors.HexColor("#333333"), leading=15))
        self.styles.add(ParagraphStyle(name='CustomBodyText', parent=self.styles['Normal'], fontSize=10.5, leading=14, spaceAfter=0.1*inch, alignment=TA_JUSTIFY))
        self.styles.add(ParagraphStyle(name='ListItem', parent=self.styles['Normal'], fontSize=10.5, leading=14, leftIndent=0.25*inch, spaceBefore=2, spaceAfter=2))
        self.styles.add(ParagraphStyle(name='ReasoningText', parent=self.styles['Italic'], fontSize=10, leading=13, spaceAfter=0.1*inch, leftIndent=0.15*inch, textColor=colors.dimgray))
        self.styles.add(ParagraphStyle(name='ChartCaption', parent=self.styles['Normal'], fontSize=9, alignment=TA_CENTER, spaceBefore=0, spaceAfter=0.15*inch, textColor=colors.darkslategray))

    def _add(self, text, style_name='CustomBodyText'):
         self.story.append(Paragraph(text.replace('\n', '<br/>'), self.styles[style_name]))

    def _add_spacer(self, height=0.2*inch): self.story.append(Spacer(1, height))

    def _add_list(self, items):
        list_items = []
        for item in items:
            if isinstance(item, Paragraph):
                 list_items.append(ListItem(item))
            else:
                 list_items.append(ListItem(Paragraph(str(item), self.styles['ListItem'])))

        self.story.append(ListFlowable(list_items, bulletType='bullet', start='circle', leftIndent=10, bulletFontSize=10))

    def _add_image_with_caption(self, img_path, caption_text, width=6.5*inch):
        if not img_path or not os.path.exists(img_path):
            self._add(f"[Visualization Error: Image not found or could not be generated - {os.path.basename(img_path or 'Unknown')}]", 'ChartCaption')
            return
        try:
            img = Image(img_path)
            img_width = img.imageWidth
            img_height = img.imageHeight
            aspect = img_height / float(img_width) if img_width else 1
            final_width = min(width, img_width)
            img.drawWidth = final_width
            img.drawHeight = final_width * aspect
            self.story.append(img)
            self._add(caption_text, 'ChartCaption')
        except Exception as e: self._add(f"[Visualization Error: {e}]", 'ChartCaption')

    def _cleanup_chart(self):
         if self.chart_image_path and os.path.exists(self.chart_image_path):
             try: os.remove(self.chart_image_path); self.chart_image_path = None
             except Exception as e: print(f"Warning: Could not remove temp chart file {self.chart_image_path}: {e}")

    def _create_sectional_score_targets_chart(self, current_scores_data, target_scores_data):
        self._cleanup_chart()
        temp_img_path = f"temp_sectional_targets_chart_{int(time.time())}.png"
        sections = ['Quant', 'Verbal', 'Data Insights']
        current_s = [current_scores_data.get(sec, {}).get('score', 60) for sec in sections]
        target_s = [target_scores_data.get(sec, 60) for sec in sections]

        x = np.arange(len(sections)); width = 0.35
        fig, ax = plt.subplots(figsize=(7, 4))
        rects1 = ax.bar(x - width/2, current_s, width, label='Current Score', color="#A0CBE8")
        rects2 = ax.bar(x + width/2, target_s, width, label='Target Score', color="#FFB547")
        ax.set_ylabel('GMAT Focus Section Score (60-90)')
        ax.set_title('Sectional Score Pathway: Current vs. Target', fontsize=12, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(sections, fontsize=10)
        ax.set_ylim(55, 95)
        ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(int(height)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout(pad=1.5)
        try:
            plt.savefig(temp_img_path, dpi=250)
            self.chart_image_path = temp_img_path
            plt.close(fig)
            return temp_img_path
        except Exception as e:
            print(f"Chart generation error (Sectional): {e}")
            plt.close(fig)
            return None

    def _create_accuracy_targets_chart(self, accuracy_preds):
        self._cleanup_chart()
        temp_img_path = f"temp_accuracy_targets_chart_{int(time.time())}.png"
        sections = ['Quant', 'Verbal', 'Data Insights']
        targets = []
        for sec in sections:
            pred_str = accuracy_preds.get(sec, "0%")
            try:
                 num_str = pred_str.replace('~', '').replace('%', '').strip()
                 targets.append(float(num_str) / 100.0)
            except ValueError:
                 targets.append(0.0)

        x = np.arange(len(sections))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.bar(x, targets, width=0.5, color="#77DD77")
        ax.set_ylabel('Target Accuracy on Difficult Questions (%)')
        ax.set_title('Accuracy Uplift Strategy for Challenging Questions', fontsize=12, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(sections, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(axis='y', linestyle=':', alpha=0.7)

        for bar_item in bars:
            height = bar_item.get_height()
            ax.annotate(f'{height:.0%}',
                        xy=(bar_item.get_x() + bar_item.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        fig.tight_layout(pad=1.2)
        try:
            plt.savefig(temp_img_path, dpi=250)
            self.chart_image_path = temp_img_path
            plt.close(fig)
            return temp_img_path
        except Exception as e:
            print(f"Chart generation error (Accuracy): {e}")
            plt.close(fig)
            return None

    def generate_report(self, student_name, features, targets_data, accuracy_preds, behavioral_analysis, plan, diagnostic_answers_input):
        report_start_time = time.time()
        try:
            self._add(f"Strategic Success Path for {student_name}", 'ReportTitle')
            self._add(f"Prepared by e-GMAT AI Engine | Date: {datetime.date.today().strftime('%B %d, %Y')}", 'SubTitle')

            self._add_spacer(0.1*inch)
            self._add(f"<b>Dear {student_name},</b><br/>This hyper-personalized report outlines your strategic path to achieving your GMAT Focus score goal of <b>{diagnostic_answers_input.get('target_score','N/A')}</b>. It leverages AI-driven insights from your provided diagnostic report (Source Date: {features.get('report_date', 'N/A')}) and your study preferences. Our analysis focuses on identifying key leverage points for maximum score improvement.", style_name='CustomBodyText')
            self._add_spacer(0.1*inch)

            self._add("I. Executive Summary: Your GMAT Focus Score Blueprint", 'SectionHeading')
            pref_code = diagnostic_answers_input.get('study_preference','A')
            pref_label = PracticePlanGenerator.study_pref_modifiers.get(pref_code,{}).get('label','Unknown Strategy')

            overall_summary = f"Your recent GMAT Focus score profile (Overall: <b>{features.get('overall_score', 'N/A')}</b> | Q: {features.get('sections',{}).get('Quant',{}).get('score','N/A')} | V: {features.get('sections',{}).get('Verbal',{}).get('score','N/A')} | DI: {features.get('sections',{}).get('Data Insights',{}).get('score','N/A')}) has been analyzed. To achieve your target of <b>{diagnostic_answers_input.get('target_score','N/A')}</b> (targeting the <b>~{targets_data.get('target_overall_percentile','N/A')}th percentile</b>), this plan recommends specific sectional score targets (Q: {targets_data.get('Quant','N/A')}, V: {targets_data.get('Verbal','N/A')}, DI: {targets_data.get('Data Insights','N/A')}) and critical accuracy improvements needed, particularly in challenging question types. Key performance areas are highlighted based on a SWOT analysis derived from your diagnostic."
            self._add(overall_summary)
            self._add_spacer(0.05*inch)
            self._add(f"Your plan is tailored for your '<b>{pref_label}</b>' strategic approach and your commitment of approximately <b>{diagnostic_answers_input.get('availability_hours','N/A')} hours/week</b> for <b>{diagnostic_answers_input.get('weeks_to_test','N/A')} weeks</b>.")
            self._add_spacer()

            self._add("II. Strategic Score Targets & Rationale", 'SectionHeading')
            self._add("<b>A. Target Sectional Scores Pathway:</b>")
            target_chart_path = self._create_sectional_score_targets_chart(features.get('sections',{}), targets_data)
            if target_chart_path:
                self._add_image_with_caption(target_chart_path, f"Chart 1: {student_name}'s Current vs. Target GMAT Focus Sectional Scores (Range 60-90)")
            else:
                self._add("[Sectional Target Chart could not be generated]", 'ChartCaption')

            self._add("<b>B. Rationale for Sectional Targets:</b>", 'SubSectionHeading')
            reasoning_text = f"These targets (<b>Q: {targets_data.get('Quant','N/A')}, V: {targets_data.get('Verbal','N/A')}, DI: {targets_data.get('Data Insights','N/A')}</b>) represent an optimized path towards your overall goal of <b>{diagnostic_answers_input.get('target_score','N/A')}</b>. They consider:\n" \
                            f"  - The score distribution needed for the <b>{targets_data.get('target_overall_percentile','N/A')}th percentile</b> overall.\n" \
                            f"  - Your current sectional performance profile to identify areas with the most efficient improvement potential.\n" \
                            f"  - A balanced uplift strategy designed to maximize your probability of success within the target timeframe."
            self._add(reasoning_text, 'ReasoningText')

            self._add("III. Accuracy Uplift for Challenging Questions", 'SectionHeading')
            self._add("<b>A. Target Accuracy Focus:</b>")
            accuracy_chart_path = self._create_accuracy_targets_chart(accuracy_preds)
            if accuracy_chart_path:
                self._add_image_with_caption(accuracy_chart_path, f"Chart 2: {student_name}'s Target Accuracy (%) for Difficult Questions by Section")
            else:
                self._add("[Accuracy Target Chart could not be generated]", 'ChartCaption')

            self._add("<b>B. Rationale and Execution Guidance:</b>", 'SubSectionHeading')
            accuracy_rationale = f"The accuracy targets above (<b>Q: {accuracy_preds.get('Quant','N/A')}, V: {accuracy_preds.get('Verbal','N/A')}, DI: {accuracy_preds.get('Data Insights','N/A')}</b>) signify the proficiency needed when encountering difficult questions (typically in your identified Weakness/Opportunity areas or complex question types). Achieving this requires:\n" \
                                "  - <b>Mastery & Process:</b> Deepen understanding in focus subtopics and refine methodical problem-solving approaches.\n" \
                                "  - <b>Strategic Practice:</b> Utilize timed sets incorporating a higher proportion of challenging questions from prioritized areas.\n" \
                                "  - <b>Rigorous Error Analysis:</b> Dedicate significant time to reviewing incorrect answers on difficult questions, identifying root causes (concept, process, speed, silly mistake) and logging takeaways.\n" \
                                "Consistent improvement in hard-question accuracy is a critical driver for reaching top-tier GMAT Focus scores."
            self._add(accuracy_rationale, 'ReasoningText')

            self.story.append(PageBreak())
            self._add("IV. Hyper-Personalized Practice Blueprint", 'SectionHeading')
            pref_label_report = PracticePlanGenerator.study_pref_modifiers.get(pref_code,{}).get('label','Unknown Strategy')
            self._add(f"This blueprint integrates insights from your diagnostic data, score goals, and study preferences ('<b>{pref_label_report}</b>') into an actionable plan:")

            self._add_list([Paragraph(item, self.styles['ListItem']) for item in plan.get('overall_focus', ['No specific overall focus points generated.'])])

            self._add("A. Prioritized Subtopic Deep-Dive (Top Focus Areas):", 'SubSectionHeading')
            topic_data = [["Rank", "Focus Area (Section - Subtopic)", "Key Metric & Rationale", "Suggested Drills", "Target Time/Q", "Est. Hours"]]
            prioritized_topics = plan.get('prioritized_topics', [])

            max_topics_in_table = 12
            for i, topic in enumerate(prioritized_topics[:max_topics_in_table]):
                topic_id = topic.get('topic', 'N/A')
                section_name, subtopic_name = topic_id.split(" - ") if " - " in topic_id else (topic_id, "")
                current_acc = features.get('sections', {}).get(section_name, {}).get('subtopic_accuracies', {}).get(subtopic_name, None)
                current_acc_str = f"{current_acc:.0%}" if isinstance(current_acc, float) else "N/A"
                reason_list = topic.get('reason',[])
                reason_str = ', '.join(reason_list[:2])
                if len(reason_list) > 2: reason_str += ", ..."

                metric_rationale = f"Acc: {current_acc_str}. Focus: {reason_str} (Score: {topic.get('priority_score', 0):.1f})"

                topic_paragraph = Paragraph(topic_id, self.styles['CustomBodyText'])
                rationale_paragraph = Paragraph(metric_rationale, self.styles['CustomBodyText'])

                topic_data.append([
                    f"{i+1}.",
                    topic_paragraph,
                    rationale_paragraph,
                    str(topic.get('recommended_drills', 'N/A')),
                    topic.get('target_timing_per_q', 'N/A'),
                    f"{topic.get('estimated_hours', 'N/A'):.1f}"
                ])

            if len(prioritized_topics) >= 1:
                topic_table = Table(topic_data, colWidths=[0.4*inch, 1.8*inch, 2.7*inch, 0.6*inch, 0.8*inch, 0.7*inch])
                table_style_def = [
                   ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#4A4A4A")), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                   ('ALIGN', (0,0), (-1,0), 'CENTER'),
                   ('ALIGN', (0,1), (0,-1), 'CENTER'),
                   ('ALIGN', (3,1), (-1,-1), 'CENTER'),
                   ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                   ('ALIGN', (1,1), (2,-1), 'LEFT'),
                   ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 9),
                   ('FONTSIZE', (0,1), (-1,-1), 8),
                   ('BOTTOMPADDING', (0,0), (-1,0), 8), ('TOPPADDING', (0,0), (-1,0), 8),
                   ('BOTTOMPADDING', (0,1), (-1,-1), 5), ('TOPPADDING', (0,1), (-1,-1), 5),
                   ('GRID', (0,0), (-1,-1), 0.5, colors.darkgrey),
                   ('LEFTPADDING', (0,0), (-1,-1), 4), ('RIGHTPADDING', (0,0), (-1,-1), 4)
                ]
                topic_table.setStyle(TableStyle(table_style_def))
                self.story.append(topic_table)
                if len(prioritized_topics) > max_topics_in_table:
                     self._add(f"... Additional {len(prioritized_topics)-max_topics_in_table} lower-priority topics are included in your online plan.", style_name='ReasoningText')
            else:
                self._add("No specific subtopics identified for prioritized deep-dive based on current data and settings.")

            self._add_spacer()
            self._add("B. Indicative Weekly Milestone Focus:", 'SubSectionHeading')
            schedule_data = [["Week", "Primary Focus Areas (Examples)", "Strategic Objective"]]
            max_weeks_in_table = 8
            for week in plan.get('weekly_schedule', [])[:max_weeks_in_table]:
                 focus_list = week.get('focus',[])
                 focus_str = ", ".join(focus_list[:3])
                 if len(focus_list) > 3: focus_str += ", ..."
                 if not focus_list or focus_list == ["General Review"]: focus_str = "General Review & Mixed Practice"

                 focus_paragraph = Paragraph(focus_str, self.styles['CustomBodyText'])
                 goals_paragraph = Paragraph(week.get('goals','N/A'), self.styles['CustomBodyText'])

                 schedule_data.append([ str(week.get('week', 'N/A')), focus_paragraph, goals_paragraph])

            if len(plan.get('weekly_schedule', [])) > 0:
                schedule_table = Table(schedule_data, colWidths=[0.5*inch, 3.5*inch, 3.0*inch])
                table_style_def_sched = table_style_def[:]
                table_style_def_sched.append(('FONTSIZE', (0,1), (-1,-1), 8))
                table_style_def_sched.append(('BOTTOMPADDING', (0,1), (-1,-1), 5))
                table_style_def_sched.append(('TOPPADDING', (0,1), (-1,-1), 5))
                schedule_table.setStyle(TableStyle(table_style_def_sched))

                self.story.append(schedule_table)
                if len(plan.get('weekly_schedule', [])) > max_weeks_in_table:
                     self._add(f"... Schedule continues for the full {len(plan.get('weekly_schedule',[]))} weeks. Access your online plan for details.", style_name='ReasoningText')

            else:
                self._add("Weekly milestones require defined study duration. Please ensure weeks to test date is set.")

            self._add_spacer(0.3*inch)
            self._add("V. Concluding Remarks & Next Steps", 'SectionHeading')
            self._add(f"{student_name}, this AI-generated strategic plan provides a clear, actionable path. Success hinges on diligent execution, consistent self-assessment using e-GMAT tools (like Scholaranium), and adapting based on your progress. Use the platform's detailed analytics and learning resources alongside this high-level strategy. For deeper insights or support, consider e-GMAT coaching options. Let's work together to achieve your target score of <b>{diagnostic_answers_input.get('target_score','N/A')}</b>!")

            self.doc.build(self.story)
            report_duration = time.time() - report_start_time
            print(f"PDF report building completed in {report_duration:.2f}s.")

        except Exception as e:
            print(f"CRITICAL ERROR during PDF report generation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_chart()

def find_latest_report(directory):
    latest_pdf = None
    latest_mtime = 0

    if not os.path.isdir(directory):
        print(f"Warning: Report directory not found: {directory}")
        return None

    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(directory, filename)
                try:
                    if os.path.isfile(filepath):
                        mtime = os.path.getmtime(filepath)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_pdf = filepath
                except OSError as e:
                    print(f"Warning: Could not access file {filepath}: {e}")
    except OSError as e:
         print(f"Warning: Could not list directory contents {directory}: {e}")
         return None

    if latest_pdf:
        print(f"Auto-detected latest PDF report: {os.path.basename(latest_pdf)}")
    else:
        print(f"No PDF reports found in directory: {directory}")

    return latest_pdf

if __name__ == "__main__":
    main_start_time = time.time()
    print(f"--- e-GMAT AI Success Path Recommender Initializing ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")

    parser = ReportParser()
    scorer = ScoringEngine()
    behavior_analyzer = BehavioralAnalyzer()
    acc_predictor = AccuracyPredictor()
    plan_gen = PracticePlanGenerator()
    script_gen = VideoScriptGenerator()

    print(f"Searching for latest report in: {REPORT_FOLDER}")
    latest_report_path_auto = find_latest_report(REPORT_FOLDER)

    print("Launching Diagnostic Form...")
    diagnostic_form = DiagnosticForm(initial_report_path=latest_report_path_auto)
    diagnostic_answers = diagnostic_form.run()

    if diagnostic_answers is None:
        print("\nDiagnostic window closed or cancelled. Exiting.")
        exit()

    print("Diagnostic complete. Inputs received.")
    student_name_input = diagnostic_answers["name"]
    report_path_input = diagnostic_answers["report_path"]
    target_score_input_str = diagnostic_answers["target_score"]
    report_file_basename = os.path.basename(report_path_input) if report_path_input != "simulate" else "Simulated Data"

    print(f"\nInitiating analysis for: {student_name_input}")
    print(f"Target Score: {target_score_input_str}")
    print(f"Report Input: {report_file_basename}")
    print(f"Study Plan: {diagnostic_answers['availability_hours']} hrs/wk for {diagnostic_answers['weeks_to_test']} wks, Strategy '{diagnostic_answers['study_preference']}'")


    print(f"\n>>> Step 1: Parsing Report Data...")
    parsing_start_time = time.time()
    parsed_report = parser.parse(report_path_input, input_type='pdf')
    print(f"<<< Parsing Complete ({time.time()-parsing_start_time:.2f}s)")


    if parsed_report:
        pipeline_start_time = time.time()
        print(f"\n>>> Step 2: Extracting Features...")
        feature_start_time = time.time()
        features = parser.extract_features(parsed_report)
        print(f"<<< Feature Extraction Complete ({time.time()-feature_start_time:.2f}s)")

        if features:
            print(f"\n>>> Step 3: Analyzing Behavior...")
            behavior_start_time = time.time()
            behavioral_summary = behavior_analyzer.analyze(features)
            print(f"<<< Behavior Analysis Complete ({time.time()-behavior_start_time:.2f}s)")

            print(f"\n>>> Step 4: Calculating Targets...")
            target_start_time = time.time()
            sectional_targets = scorer.calculate_sectional_targets(target_score_input_str, features)
            diagnostic_answers['targets'] = sectional_targets
            print(f"<<< Target Calculation Complete ({time.time()-target_start_time:.2f}s)")


            print(f"\n>>> Step 5: Predicting Accuracy...")
            accuracy_start_time = time.time()
            hard_q_accuracy_targets = acc_predictor.predict_hard_question_accuracy(features, sectional_targets)
            print(f"<<< Accuracy Prediction Complete ({time.time()-accuracy_start_time:.2f}s)")


            print(f"\n>>> Step 6: Generating Practice Plan...")
            plan_start_time = time.time()
            practice_plan = plan_gen.generate_plan(features, target_score_input_str, diagnostic_answers, hard_q_accuracy_targets, behavioral_summary)
            print(f"<<< Practice Plan Generation Complete ({time.time()-plan_start_time:.2f}s)")


            print(f"\n>>> Step 7: Generating Scripts...")
            script_start_time = time.time()
            script_inputs = {
                "student_id": parsed_report.get("student_id", f"Client_{student_name_input[:3].upper()}"),
                "report_date": features.get("report_date", 'N/A'),
                "report_type": 'pdf',
                "target_score": target_score_input_str,
                "behavioral_analysis": behavioral_summary,
                "scoring_engine": scorer,
                "accuracy_predictor": acc_predictor,
                "plan_generator": plan_gen,
                "diagnostic_answers": diagnostic_answers
            }
            architect_script = script_gen.generate_architect_script(script_inputs, features, sectional_targets, hard_q_accuracy_targets, practice_plan)
            student_script = script_gen.generate_student_script(script_inputs, features, sectional_targets, hard_q_accuracy_targets, practice_plan, diagnostic_answers)
            print(f"<<< Script Generation Complete ({time.time()-script_start_time:.2f}s)")


            print(f"\n>>> Step 8: Generating PDF Report...")
            pdf_gen_start_time = time.time()
            safe_student_name = re.sub(r'[\\/*?:"<>|]', "", student_name_input)
            pdf_filename_base = f"GMAT_Focus_SuccessPath_{safe_student_name}_{datetime.date.today().strftime('%Y%m%d')}.pdf"
            pdf_filename = os.path.join(REPORT_FOLDER, pdf_filename_base)

            pdf_report_gen = PDFReportGenerator(pdf_filename)
            pdf_report_gen.generate_report(
                student_name_input, features, sectional_targets, hard_q_accuracy_targets,
                behavioral_summary, practice_plan, diagnostic_answers
            )
            print(f"<<< PDF Report Generation Attempt Finished ({time.time()-pdf_gen_start_time:.2f}s)")

            pipeline_duration = time.time() - pipeline_start_time
            total_duration = time.time() - main_start_time
            print(f"\n--- Processing Summary ---")
            print(f"Report generated for: {student_name_input}")
            print(f"Overall Score (Report): {features.get('overall_score','N/A')}")
            print(f"Target Score: {target_score_input_str}")
            print(f"Recommended Sectional Targets: Q:{sectional_targets.get('Quant')}, V:{sectional_targets.get('Verbal')}, DI:{sectional_targets.get('Data Insights')}")
            print(f"Analysis & Report Pipeline Duration: {pipeline_duration:.2f} seconds")
            print(f"Total Script Execution Time: {total_duration:.2f} seconds")
            if os.path.exists(pdf_filename):
                print(f"Output Files Generated:")
                print(f"  - PDF Report: {pdf_filename}")
                print(f"  - Architect Script: {os.path.abspath('architect_script.txt')}")
                print(f"  - Student Script: {os.path.abspath('student_script.txt')}")
                print(f"  - Student TTS (Simulated): {os.path.abspath('student_tts_output.mp3')}")
            else:
                 print(f"!! PDF Report generation failed or file not found at expected location: {pdf_filename} !!")


            try:
                if os.path.exists(pdf_filename):
                    print(f"\nAttempting to open generated PDF: {pdf_filename}")
                    if os.name == 'nt':
                        os.startfile(pdf_filename)
                    elif sys.platform == 'darwin':
                         os.system(f'open "{pdf_filename}"')
                    else:
                        try:
                             os.system(f'xdg-open "{pdf_filename}"')
                        except Exception:
                             print("Could not automatically open PDF (xdg-open not found or failed). Please open it manually.")

                else: print(f"\nCould not auto-open PDF: File not found at {pdf_filename}")
            except Exception as e: print(f"\nWarning: Could not auto-open PDF ({e}). Please locate it at: {pdf_filename}")

        else: print("\nProcessing halted: Feature extraction failed (likely due to issues in parsed data).")
    else: print("\nProcessing halted: Report loading/parsing failed. Check report file or simulation logic.")

    print(f"\n--- Run Finished ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")