import sqlite3
import os

db = r"C:\Users\alexa\OneDrive\Documents\Academia\Naissance\naissance.db"

print(os.path.exists(db))

conn = sqlite3.connect(db)

print("opened")

conn.close()