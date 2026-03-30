# eye_tracking_study_assistance
Eye tracking enables real-time analysis of visual attention and information processing, providing reliable, objective data on attention allocation and offering unique insights into these cognitive dynamics in educational contexts. Visual perception is crucial for both teachers and students to distill relevant information in complex classroom scenarios, making temporal analysis of attention patterns essential for understanding learning processes.
This research provides evidence-based insights for curriculum design, instructional timing, and adaptive learning systems that respond to real-time attention patterns, ultimately enhancing educational effectiveness through precision analysis of visual attention dynamics.
Overview

This project focuses on:

1. Processing raw eye-tracking data
2. Identifying valid/invalid gaze points
3. Computing similarity metrics between gaze patterns
4. Supporting quick study analysis workflows
 
It is useful for:

1. Research in Human-Computer Interaction (HCI)
2. Study pattern analysis
3. Cognitive behavior analysis

Features
1. Eye-tracking data parsing (TSV format)
2. Noise filtering & invalid point removal
3. Gaze similarity computation
4. Quick analysis pipeline (Python + MATLAB)
5. Modular scripts for flexible experimentation

Tech Stack
1. MATLAB – Core data processing and analysis
2. Python – Data parsing and quick execution scripts
3. TSV Data Format – Input eye-tracking datasets

eye_tracking_study_assistance/

│

├── parse_files.py            # Parses raw eye-tracking data

├── quick_study.py            # Runs quick analysis pipeline

├── testingFormat.tsv         # Sample dataset

│

├── avg_farm.m                # Average computation logic

├── read_data.m               # Reads eye-tracking data

├── invalidPoints.m           # Filters invalid gaze points

├── tylers_similarity.m       # Similarity calculation

├── tylers_version.m          # Alternate similarity method

│

└── getfiles.m                # File handling utility
