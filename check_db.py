import sqlite3

conn = sqlite3.connect('data/autonomous_learning.db')
cursor = conn.cursor()

# Count total mastered topics
cursor.execute('SELECT COUNT(*) FROM mastered_topics')
count = cursor.fetchone()[0]
print(f'Total mastered topics in DB: {count}')

# Get sample topics
cursor.execute('SELECT topic FROM mastered_topics LIMIT 10')
print('Sample topics:')
for row in cursor.fetchall():
    print(f'  - {row[0]}')

conn.close()