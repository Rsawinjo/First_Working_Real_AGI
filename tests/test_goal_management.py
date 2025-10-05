import pytest
from ai_core.autonomous_learner import AutonomousLearner, LearningGoal
from datetime import datetime

def test_goal_selection():
    learner = AutonomousLearner()
    learner.learning_goals.clear()
    learner.learning_goals.append(LearningGoal(id='1', topic='A', priority=1.0, knowledge_gap='', target_depth=1, created_at=datetime.now(), estimated_duration=1, prerequisites=[]))
    selected = learner._select_next_goal()
    assert selected.topic == 'A'

def test_dynamic_goal_management():
    learner = AutonomousLearner()
    learner.learning_goals.clear()
    learner.mastered_topics.add('A')
    learner.learning_goals.append(LearningGoal(id='1', topic='A', priority=1.0, knowledge_gap='', target_depth=1, created_at=datetime.now(), estimated_duration=1, prerequisites=[]))
    selected = learner._select_next_goal()
    assert selected is None or selected.topic != 'A'
