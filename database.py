import sqlite3
import os
from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, 'resume_ranker.db')

def get_connection():
    c = sqlite3.connect(DATABASE)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = get_connection()
    c.execute('''CREATE TABLE IF NOT EXISTS candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, candidate_name TEXT, email TEXT, phone TEXT,
        resume_text TEXT, clean_text TEXT, tfidf_score REAL DEFAULT 0, semantic_score REAL DEFAULT 0, overall_score REAL DEFAULT 0,
        status TEXT DEFAULT 'Pending', uploaded_at TEXT, skills TEXT DEFAULT '', matched_skills TEXT DEFAULT '', missing_skills TEXT DEFAULT '',
        experience_years REAL DEFAULT 0, qualification TEXT DEFAULT '', job_description TEXT DEFAULT '', recruiter_decision TEXT DEFAULT 'Pending', feedback_notes TEXT DEFAULT '')''')
    cols = {r[1] for r in c.execute('PRAGMA table_info(candidates)').fetchall()}
    additions = {'skills':'TEXT DEFAULT ""','matched_skills':'TEXT DEFAULT ""','missing_skills':'TEXT DEFAULT ""','experience_years':'REAL DEFAULT 0','qualification':'TEXT DEFAULT ""','job_description':'TEXT DEFAULT ""','recruiter_decision':'TEXT DEFAULT "Pending"','feedback_notes':'TEXT DEFAULT ""'}
    for col, typ in additions.items():
        if col not in cols:
            c.execute(f'ALTER TABLE candidates ADD COLUMN {col} {typ}')
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id INTEGER, resume_text TEXT,
        decision TEXT, notes TEXT, created_at TEXT)''')
    c.commit(); c.close()

def save_candidate(**kw):
    c = get_connection(); kw.setdefault('uploaded_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    fields = ['filename','candidate_name','email','phone','resume_text','clean_text','tfidf_score','semantic_score','overall_score','status','uploaded_at','skills','matched_skills','missing_skills','experience_years','qualification','job_description']
    vals = [kw.get(x, '') for x in fields]
    c.execute(f"INSERT INTO candidates ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", vals)
    c.commit(); c.close()

def get_candidates():
    c=get_connection(); rows=c.execute('SELECT * FROM candidates ORDER BY overall_score DESC, id DESC').fetchall(); c.close(); return rows

def get_candidate(i):
    c=get_connection(); r=c.execute('SELECT * FROM candidates WHERE id=?',(i,)).fetchone(); c.close(); return r

def search_candidates(search='', min_score=0, max_score=100, status='', min_experience=0, skills='', qualification=''):
    c=get_connection(); q='SELECT * FROM candidates WHERE overall_score BETWEEN ? AND ?'; p=[min_score,max_score]
    if search:
        q += ' AND (candidate_name LIKE ? OR email LIKE ? OR filename LIKE ? OR skills LIKE ? OR resume_text LIKE ?)'
        v=f'%{search}%'; p += [v,v,v,v,v]
    if status: q += ' AND status=?'; p.append(status)
    if min_experience: q += ' AND experience_years>=?'; p.append(min_experience)
    if skills:
        for s in [x.strip() for x in skills.split(',') if x.strip()]: q += ' AND skills LIKE ?'; p.append(f'%{s}%')
    if qualification: q += ' AND qualification LIKE ?'; p.append(f'%{qualification}%')
    q += ' ORDER BY overall_score DESC, id DESC'
    rows=c.execute(q,p).fetchall(); c.close(); return rows

def update_status(i,status):
    c=get_connection(); c.execute('UPDATE candidates SET status=? WHERE id=?',(status,i)); c.commit(); c.close()

def update_candidate(i,name,email,phone,status,decision,notes):
    c=get_connection(); c.execute('UPDATE candidates SET candidate_name=?,email=?,phone=?,status=?,recruiter_decision=?,feedback_notes=? WHERE id=?',(name,email,phone,status,decision,notes,i)); c.commit(); c.close()

def save_feedback(i,decision,notes):
    c=get_connection(); row=c.execute('SELECT resume_text FROM candidates WHERE id=?',(i,)).fetchone(); text=row['resume_text'] if row else ''
    c.execute('UPDATE candidates SET recruiter_decision=?,feedback_notes=? WHERE id=?',(decision,notes,i))
    c.execute('INSERT INTO feedback(candidate_id,resume_text,decision,notes,created_at) VALUES(?,?,?,?,?)',(i,text,decision,notes,datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    c.commit(); c.close()

def get_feedback():
    c=get_connection(); rows=c.execute("SELECT * FROM feedback WHERE decision IN ('Hired','Rejected') ORDER BY id").fetchall(); c.close(); return rows

def delete_candidate(i):
    c=get_connection(); c.execute('DELETE FROM feedback WHERE candidate_id=?',(i,)); c.execute('DELETE FROM candidates WHERE id=?',(i,)); c.commit(); c.close()

def clear_candidates():
    c=get_connection(); c.execute('DELETE FROM feedback'); c.execute('DELETE FROM candidates'); c.commit(); c.close()

init_db()
