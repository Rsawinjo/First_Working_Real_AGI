import sqlite3

conn = sqlite3.connect('data/autonomous_learning.db')
cursor = conn.cursor()

# Get all topics and clean them
cursor.execute("SELECT topic FROM mastered_topics")
all_topics = cursor.fetchall()

cleaned_topics = set()
for (topic,) in all_topics:
    # Clean topic the same way the code does
    clean_topic = topic.split('[')[0].strip()
    cleaned_topics.add(clean_topic)

print(f'Total raw entries in DB: {len(all_topics)}')
print(f'Unique cleaned topics: {len(cleaned_topics)}')
print('Sample cleaned topics:')
for topic in sorted(list(cleaned_topics))[:10]:
    print(f'  - {topic}')

conn.close()