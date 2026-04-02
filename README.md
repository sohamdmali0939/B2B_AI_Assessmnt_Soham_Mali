# B2B AI System – Crop & Traffic Intelligence

## Overview
This project implements an end-to-end AI system for:
- Crop detection and segmentation
- Traffic analysis and object detection
- Agent-based orchestration with QC, memory, and self-optimization

## Setup Instructions

```bash
git clone <your-repo-link>             #THis is my Github Repo Link: https://github.com/sohamdmali0939/B2B_AI_Assessmnt_Soham_Mali
cd B2B_AI_Assessment
pip install -r requirements.txt



Download .tiff images and aerial traffic video for detection.

Place files inside:   data/raw/...


How to Run:
First run teh command:   "python run_crop.py"
Next run:   "python run_traffic.py"
THen at last run: "python run_agent.py"


The Tasks that get completed are: Tasks Implemented 

Task 1: Crop Detection
Crop segmentation pipeline

Task 2: Traffic Analysis
Object detection + tracking
Density estimation and heatmaps

Task 3: Agent System
OrchestratorAgent
QC Agent
Memory + Drift Detection

Task 4: System Design
Architecture diagram
MLOps plan
Scalability analysis



Outputs:
Heatmaps
GeoJSON files
Detection logs


Notes
Fallback logic implemented for LLM failures.
Modular architecture for scalability.


Architecture Diagram is in Root folder.
Heatmap, Crop detection, Tiling images, Reports of Agent Pipeline in the Output folder