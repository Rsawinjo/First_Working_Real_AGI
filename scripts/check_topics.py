import sqlite3

conn = sqlite3.connect('data/autonomous_learning.db')
cursor = conn.cursor()

# Clean topics (no brackets)
cursor.execute("SELECT topic FROM mastered_topics WHERE topic NOT LIKE '%[%' LIMIT 5")
print('Clean topics (no brackets):')
for row in cursor.fetchall():
    print(f'  - {row[0]}')

# Topics with brackets
cursor.execute("SELECT topic FROM mastered_topics WHERE topic LIKE '%[%' LIMIT 5")
print('Topics with brackets:')
for row in cursor.fetchall():
    print(f'  - {row[0]}')

conn.close()