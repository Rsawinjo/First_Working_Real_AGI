#!/usr/bin/env python3
"""
Test script to verify dyna    # Check for AI/computing contamination (more specific check)
    ai_terms = ["ai", "artificial intelligence", "machine learning", "neural network", "computer", "computing", "algorithm", "data science", "deep learning", "quantum computing"]
    ai_contamination = 0
    for topic in generated_topics:
        topic_lower = topic.lower()
        # More specific check - avoid false positives with words like "science" in "political science"
        contaminated = False
        for term in ai_terms:
            if term in topic_lower:
                # Additional check: if it's just "science" in a compound word, don't count it
                if term == "science":
                    # Check if it's part of a legitimate scientific field
                    science_fields = ["political science", "computer science", "data science", "cognitive science", "neuroscience", "biological science", "physical science"]
                    if any(field in topic_lower for field in science_fields):
                        continue
                ai_contamination += 1
                contaminated = True
                break generation is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_core.autonomous_learner import AutonomousLearner

def test_dynamic_topic_generation():
    """Test that the new dynamic topic generation creates truly novel topics"""

    print("🧪 Testing Dynamic Topic Generation System")
    print("=" * 50)

    # Initialize the learner (without full GUI)
    learner = AutonomousLearner()

    # Test the dynamic discovery goal generation
    print("\n🚀 Generating dynamic discovery goals...")

    # Generate several topics to test diversity
    generated_topics = []
    for i in range(5):
        goal = learner._force_discovery_goal()
        topic = goal.topic
        generated_topics.append(topic)
        print(f"Topic {i+1}: {topic}")

    print("\n📊 Analysis:")
    print(f"Generated {len(generated_topics)} topics")

    # Check for uniqueness
    unique_topics = set(generated_topics)
    print(f"Unique topics: {len(unique_topics)}")

    # Check that topics contain diverse domains
    diverse_domains = [
        "biology", "chemistry", "physics", "mathematics", "medicine", "psychology",
        "sociology", "economics", "history", "geology", "astronomy", "ecology",
        "genetics", "neuroscience", "cardiology", "oncology", "microbiology"
    ]

    domain_counts = {}
    for topic in generated_topics:
        topic_lower = topic.lower()
        for domain in diverse_domains:
            if domain in topic_lower:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

    print(f"Domain diversity: {len(domain_counts)} different scientific domains found")
    print(f"Domains used: {list(domain_counts.keys())}")

    # Check that no AI/computing terms appear (more specific check)
    ai_terms = ["ai", "artificial intelligence", "machine learning", "neural network", "computer", "computing", "algorithm", "data science", "deep learning", "quantum computing"]
    ai_contamination = 0
    for topic in generated_topics:
        topic_lower = topic.lower()
        for term in ai_terms:
            if term in topic_lower:
                # Additional check: if it's just "science" in a compound word, don't count it
                if term == "science":
                    # Check if it's part of a legitimate scientific field
                    science_fields = ["political science", "computer science", "data science", "cognitive science", "neuroscience", "biological science", "physical science"]
                    if any(field in topic_lower for field in science_fields):
                        continue
                ai_contamination += 1
                break

    print(f"AI term contamination: {ai_contamination} topics (should be 0)")

    # Check similarity to mastered topics
    similar_to_mastered = 0
    for topic in generated_topics:
        for mastered in learner.mastered_topics:
            if learner._topics_are_similar(topic, mastered):
                similar_to_mastered += 1
                break

    print(f"Similar to mastered topics: {similar_to_mastered} topics (should be 0)")

    print("\n✅ Test Results:")
    if len(unique_topics) == len(generated_topics):
        print("✅ All topics are unique")
    else:
        print("❌ Some topics are duplicates")

    if len(domain_counts) >= 3:
        print("✅ Good domain diversity")
    else:
        print("❌ Low domain diversity")

    if ai_contamination == 0:
        print("✅ No AI/computing contamination")
    else:
        print("❌ AI/computing terms detected")

    if similar_to_mastered == 0:
        print("✅ All topics are novel (not similar to mastered)")
    else:
        print("❌ Some topics similar to already mastered")

    print("\n🎯 Dynamic topic generation test completed!")

if __name__ == "__main__":
    test_dynamic_topic_generation()