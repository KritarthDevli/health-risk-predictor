import sqlite3
import os
import csv
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'health_portal.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            age INTEGER NOT NULL,
            systolic INTEGER NOT NULL,
            diastolic INTEGER NOT NULL,
            bmi REAL NOT NULL,
            smoker TEXT NOT NULL,
            physactivity TEXT NOT NULL,
            diabetes_risk REAL NOT NULL,
            hypertension_risk REAL NOT NULL,
            stroke_risk REAL NOT NULL,
            ai_summary TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def get_or_create_user(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    clean_name = name.strip()
    cursor.execute("SELECT id FROM users WHERE name = ?", (clean_name,))
    user = cursor.fetchone()
    
    if user:
        user_id = user['id']
    else:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("INSERT INTO users (name, created_at) VALUES (?, ?)", (clean_name, now))
        conn.commit()
        user_id = cursor.lastrowid
        
    conn.close()
    return user_id

def save_scan_results(user_id, age, systolic, diastolic, bmi, smoker, activity, risks, ai_summary=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    cursor.execute('''
        INSERT INTO scans (
            user_id, age, systolic, diastolic, bmi, smoker, physactivity, 
            diabetes_risk, hypertension_risk, stroke_risk, ai_summary, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, age, systolic, diastolic, bmi, smoker, activity,
        risks.get('Diabetes', 0.0), risks.get('Hypertension', 0.0), risks.get('Stroke', 0.0),
        ai_summary, now
    ))
    
    conn.commit()
    conn.close()

def get_user_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
    history = cursor.fetchall()
    conn.close()
    return history

def export_db_to_csv():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT users.name, scans.age, scans.systolic, scans.diastolic, 
               scans.bmi, scans.smoker, scans.physactivity, 
               scans.diabetes_risk, scans.hypertension_risk, scans.stroke_risk, scans.timestamp 
        FROM scans 
        JOIN users ON scans.user_id = users.id
        ORDER BY scans.timestamp DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    csv_path = os.path.join(os.path.dirname(__file__), 'patient_records_log.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Patient Name', 'Age', 'Systolic BP', 'Diastolic BP', 'BMI', 'Smoker', 'Physically Active', 'Diabetes Risk %', 'Hypertension Risk %', 'Stroke Risk %', 'Timestamp'])
        for row in rows:
            writer.writerow(list(row))
