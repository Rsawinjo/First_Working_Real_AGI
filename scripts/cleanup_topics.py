import sqlite3

def is_valid_learning_topic(topic: str) -> bool:
    """Validate that a topic is a legitimate learning subject"""
    if not topic or len(topic.strip()) < 3:
        return False

    # Reject obvious non-topics
    invalid_indicators = [
        "i'm", "i am", "i'm learning", "i encountered an error",
        "cuda", "error:", "compile with", "for debugging",
        "stacktrace", "device-side", "assertion triggered",
        "could you tell me", "thinking about that",
        "learning from this", "feedback", "response",
        "sorry", "unfortunately", "however"
    ]

    topic_lower = topic.lower()
    for indicator in invalid_indicators:
        if indicator in topic_lower:
            return False

    # Must contain at least one noun-like word (basic heuristic)
    words = topic.split()
    if len(words) < 2:
        return False

    # Should not be a complete sentence (look for sentence structure)
    if topic.endswith('.') or topic.endswith('!') or topic.endswith('?'):
        return False

    return True

conn = sqlite3.connect('data/autonomous_learning.db')
cursor = conn.cursor()

# Get all topics
cursor.execute("SELECT rowid, topic FROM mastered_topics")
all_topics = cursor.fetchall()

invalid_count = 0
for topic_id, topic in all_topics:
    clean_topic = topic.split('[')[0].strip()
    if not is_valid_learning_topic(clean_topic):
        print(f"Removing invalid topic: {clean_topic}")
        cursor.execute("DELETE FROM mastered_topics WHERE rowid = ?", (topic_id,))
        invalid_count += 1

conn.commit()
print(f"Removed {invalid_count} invalid topics")

# Check remaining valid topics
cursor.execute("SELECT COUNT(*) FROM mastered_topics")
remaining = cursor.fetchone()[0]
print(f"Remaining valid topics: {remaining}")

conn.close()