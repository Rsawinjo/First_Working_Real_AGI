import pytest
from ai_core.autonomous_learner import AutonomousLearner

def test_topic_filtering():
    learner = AutonomousLearner()
    assert learner._topics_are_similar('AI', 'ai')
    assert not learner._topics_are_similar('AI', 'Biology')
