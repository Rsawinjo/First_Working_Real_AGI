#!/usr/bin/env python3
"""
Focused test of AGI research functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_core.autonomous_learner import AutonomousLearner
from utils.research_assistant import ResearchAssistant
import time
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_agi_research_only():
    print('Testing AGI research functionality only...')

    # Create research assistant
    research_assistant = ResearchAssistant()
    print('✓ ResearchAssistant created')

    # Create autonomous learner with minimal initialization
    learner = AutonomousLearner(research_assistant=research_assistant)
    print('✓ AutonomousLearner created')

    # Test the deep research method
    test_topic = 'quantum computing'
    print(f'Testing deep research on: {test_topic}')

    start_time = time.time()
    try:
        research_results = learner._deep_research(test_topic)
        end_time = time.time()

        duration = end_time - start_time
        print('.2f')

        if duration > 60:
            print('❌ Research took too long - possible hang!')
            return False
        else:
            print('✅ Research completed within timeout!')

        sources = research_results.get('sources', [])
        print(f'Sources found: {len(sources)}')

        if sources:
            print('✅ Research returned data successfully')
            return True
        else:
            print('⚠️ No sources found, but research completed')
            return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print('.2f')
        print(f'❌ Error during research: {e}')
        return False

if __name__ == '__main__':
    success = test_agi_research_only()
    if success:
        print('\n🎉 AGI research timeout fix appears to be working!')
    else:
        print('\n💥 AGI research still has issues')