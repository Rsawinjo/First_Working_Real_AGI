#!/usr/bin/env python3
"""
Test autonomous learning research functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_core.autonomous_learner import AutonomousLearner
from utils.research_assistant import ResearchAssistant
import time

def test_autonomous_research():
    print('Testing autonomous learning research functionality...')

    # Create research assistant
    research_assistant = ResearchAssistant()

    # Create autonomous learner (without full initialization)
    learner = AutonomousLearner(research_assistant=research_assistant)

    # Test the deep research method
    test_topic = 'machine learning'
    print(f'Testing deep research on: {test_topic}')

    start_time = time.time()
    try:
        research_results = learner._deep_research(test_topic)
        end_time = time.time()

        print('.2f')
        print(f'Research completed successfully!')
        print(f'Sources found: {len(research_results.get("sources", []))}')
        print(f'Key concepts: {len(research_results.get("key_concepts", []))}')

        if research_results.get('sources'):
            print(f'First source preview: {research_results["sources"][0].get("content", "")[:100]}...')

    except Exception as e:
        end_time = time.time()
        print('.2f')
        print(f'Error during research: {e}')

if __name__ == '__main__':
    test_autonomous_research()