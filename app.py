from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, make_response
from werkzeug.utils import secure_filename
from docx import Document
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os, re, io
from datetime import datetime
from database import *

app=Flask(__name__)
app.secret_key=os.environ.get('RESUMERANK_SECRET','change-this-secret')
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER=os.path.join(BASE_DIR,'uploads')
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER,MAX_CONTENT_LENGTH=25*1024*1024)
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
ALLOWED_RESUME={'docx','pdf'}; ALLOWED_JD={'docx','pdf','txt'}
SKILLS=['python','java','javascript','typescript','c++','c#','sql','mysql','sqlite','postgresql','mongodb','flask','django','fastapi','rest api','rest apis','api','html','css','react','angular','vue','node.js','nodejs','pandas','numpy','scikit-learn','sklearn','tensorflow','pytorch','keras','nlp','natural language processing','deep learning','machine learning','data science','artificial intelligence','docker','kubernetes','aws','azure','gcp','git','github','linux','spark','hadoop','power bi','tableau','excel','communication','leadership','agile','scrum']
QUALIFICATIONS=['Bachelor','Master','PhD','B.Tech','M.Tech','B.E','M.E','MBA','Computer Science','Artificial Intelligence','Data Science','Information Technology','Software Engineering']

def ext(name): return name.rsplit('.',1)[-1].lower() if '.' in name else ''
def allowed_file(name,kind='resume'): return ext(name) in (ALLOWED_RESUME if kind=='resume' else ALLOWED_JD)
def extract_docx_text(path):
    doc=Document(path); out=[]
    for p in doc.paragraphs:
        if p.text.strip(): out.append(p.text.strip())
    for table in doc.tables:
        for row in table.rows:
            vals=[c.text.strip() for c in row.cells if c.text.strip()]
            if vals: out.append(' | '.join(vals))
    return '\n'.join(out)
def extract_pdf_text(path):
    reader=PdfReader(path); return '\n'.join((p.extract_text() or '') for p in reader.pages).strip()
def extract_text(path):
    return extract_docx_text(path) if ext(path)=='docx' else extract_pdf_text(path) if ext(path)=='pdf' else open(path,encoding='utf8',errors='ignore').read() if ext(path)=='txt' else ''
def preprocess_text(text):
    text=re.sub(r'[^a-zA-Z0-9+#.\s]',' ',(text or '').lower()); return re.sub(r'\s+',' ',text).strip()
def extract_email(text):
    m=re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',text or ''); return m.group(0) if m else ''
def extract_phone(text):
    for m in re.finditer(r'(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}',text or ''):
        if 10<=len(re.sub(r'\D','',m.group(0)))<=15: return m.group(0).strip()
    return ''
def extract_candidate_name(filename,text):
    base=re.sub(r'[_\-]+',' ',os.path.splitext(filename)[0]); base=re.sub(r'\b(resume|cv|curriculum vitae|profile)\b','',base,flags=re.I).strip()
    if base and len(base)<80:return base
    for line in (text or '').splitlines():
        line=line.strip()
        if line and len(line)<80 and not re.search(r'@|phone|email|resume',line,re.I):return line
    return 'Unknown Candidate'
def extract_skills(text):
    low=(text or '').lower(); found=[]
    for s in SKILLS:
        if re.search(r'(?<![a-z0-9])'+re.escape(s)+r'(?![a-z0-9])',low):found.append(s)
    return sorted(set(found))
def extract_experience_years(text):
    low=(text or '').lower(); vals=[]
    for pat in [r'(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s*)?(?:professional\s*)?(?:experience|exp)',r'(\d+(?:\.\d+)?)\s*yrs?']:
        vals += [float(x) for x in re.findall(pat,low)]
    return round(max(vals),1) if vals else 0.0
def extract_qualification(text):
    low=(text or '').lower(); out=[]
    patterns=[('PhD',r'phd|doctorate'),("Master's",r'master|m\.tech|mtech|m\.e|mba'),("Bachelor's",r'bachelor|b\.tech|btech|b\.e|be\b')]
    for label,pat in patterns:
        if re.search(pat,low):out.append(label)
    for q in ['computer science','artificial intelligence','data science','information technology','software engineering']:
        if q in low:out.append(q.title())
    return ', '.join(dict.fromkeys(out)) or 'Not detected'
def tfidf_scores(job,resumes):
    docs=[preprocess_text(job)]+[preprocess_text(x) for x in resumes]
    try:
        v=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=10000); m=v.fit_transform(docs)
        return [round(float(x)*100,2) for x in cosine_similarity(m[0:1],m[1:]).flatten()]
    except ValueError:return [0.0]*len(resumes)
_embedding_model=None
def semantic_scores(job,resumes):
    global _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        if _embedding_model is None:_embedding_model=SentenceTransformer('all-MiniLM-L6-v2')
        je=_embedding_model.encode(job,normalize_embeddings=True); re_= _embedding_model.encode(resumes,normalize_embeddings=True,show_progress_bar=False)
        return [round(float(x)*100,2) for x in cosine_similarity(je.reshape(1,-1),re_).flatten()]
    except Exception:
        jw=set(preprocess_text(job).split()); return [round(len(jw & set(preprocess_text(r).split()))/len(jw)*100,2) if jw else 0 for r in resumes]
def status_for(s):return 'Shortlisted' if s>=70 else 'Under Review' if s>=50 else 'Rejected'
def prepare(r,rank=0):
    d=dict(r); return {'id':d['id'],'name':d.get('candidate_name') or 'Unknown Candidate','filename':d.get('filename',''),'email':d.get('email',''),'phone':d.get('phone',''),'resume_text':d.get('resume_text',''),'tfidf_score':float(d.get('tfidf_score') or 0),'semantic_score':float(d.get('semantic_score') or 0),'final_score':float(d.get('overall_score') or 0),'status':d.get('status','Pending'),'uploaded_at':d.get('uploaded_at',''),'rank':rank,'skills':d.get('skills',''),'matched_skills':d.get('matched_skills',''),'missing_skills':d.get('missing_skills',''),'experience_years':float(d.get('experience_years') or 0),'qualification':d.get('qualification',''),'job_description':d.get('job_description',''),'recruiter_decision':d.get('recruiter_decision','Pending'),'feedback_notes':d.get('feedback_notes','')}
def safe_float(value, default):
    try: return float(value)
    except (TypeError, ValueError): return default

def filters():
    return (request.args.get('search','').strip(), request.args.get('status','').strip(), safe_float(request.args.get('min_score'),0), safe_float(request.args.get('max_score'),100), safe_float(request.args.get('min_experience'),0), request.args.get('skills','').strip(), request.args.get('qualification','').strip())

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        jd=request.form.get('job_description','').strip(); jf=request.files.get('job_description_file')
        if jf and jf.filename:
            if not allowed_file(jf.filename,'jd'):return render_template('index.html',error='JD file must be DOCX, PDF or TXT.',job_description=jd)
            p=os.path.join(UPLOAD_FOLDER,'jd_'+secure_filename(jf.filename)); jf.save(p)
            try:jd=extract_text(p).strip()
            finally:
                if os.path.exists(p):os.remove(p)
        files=[f for f in request.files.getlist('resumes') if f and f.filename]
        valid=[f for f in files if allowed_file(f.filename)]
        if not jd:return render_template('index.html',error='Please enter or upload a job description.',job_description=jd)
        if not valid:return render_template('index.html',error='Please upload at least one DOCX or PDF resume.',job_description=jd)
        texts=[]; metas=[]
        for f in valid:
            safe=secure_filename(f.filename); unique=datetime.now().strftime('%Y%m%d%H%M%S%f')+'_'+safe; p=os.path.join(UPLOAD_FOLDER,unique); f.save(p)
            try:
                t=extract_text(p)
                if len(t.strip())>=30:texts.append(t);metas.append((unique,t))
            except Exception as e:print('Parse error:',e)
        if not texts:return render_template('index.html',error='No readable resumes were found.',job_description=jd)
        tfs=tfidf_scores(jd,texts); sems=semantic_scores(jd,texts); required=extract_skills(jd)
        for (fn,t),tf,sem in zip(metas,tfs,sems):
            skills=extract_skills(t); matched=[s for s in required if s in skills]; missing=[s for s in required if s not in skills]; score=round(tf*.5+sem*.5,2)
            save_candidate(filename=fn,candidate_name=extract_candidate_name(fn,t),email=extract_email(t),phone=extract_phone(t),resume_text=t,clean_text=preprocess_text(t),tfidf_score=tf,semantic_score=sem,overall_score=score,status=status_for(score),skills=', '.join(skills),matched_skills=', '.join(matched),missing_skills=', '.join(missing),experience_years=extract_experience_years(t),qualification=extract_qualification(t),job_description=jd)
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    search,status,min_s,max_s,min_exp,skills,qualification=filters(); rows=search_candidates(search,min_s,max_s,status,min_exp,skills,qualification); candidates=[prepare(r,i) for i,r in enumerate(rows,1)]; allr=get_candidates(); total=len(allr)
    return render_template('dashboard.html',candidates=candidates,total_candidates=total,shortlisted=sum(r['status']=='Shortlisted' for r in allr),under_review=sum(r['status']=='Under Review' for r in allr),rejected=sum(r['status']=='Rejected' for r in allr),average_score=round(sum(float(r['overall_score'] or 0) for r in allr)/total,2) if total else 0,search=search,status=status,min_score=min_s,max_score=max_s,min_experience=min_exp,skills=skills,qualification=qualification,skill_options=SKILLS,qualification_options=QUALIFICATIONS,feedback_count=len(get_feedback()))

@app.route('/candidate/<int:candidate_id>')
def candidate(candidate_id):
    r=get_candidate(candidate_id)
    if not r:return render_template('candidate.html',candidate=None)
    score=float(r['overall_score'] or 0); rank=1+sum(float(x['overall_score'] or 0)>score for x in get_candidates()); return render_template('candidate.html',candidate=prepare(r,rank))

@app.route('/candidate/<int:candidate_id>/edit',methods=['GET','POST'])
def edit_candidate(candidate_id):
    r=get_candidate(candidate_id)
    if not r:return redirect(url_for('dashboard'))
    if request.method=='POST':
        update_candidate(candidate_id,request.form.get('candidate_name','').strip(),request.form.get('email','').strip(),request.form.get('phone','').strip(),request.form.get('status','Pending'),request.form.get('recruiter_decision','Pending'),request.form.get('feedback_notes','').strip()); return redirect(url_for('candidate',candidate_id=candidate_id))
    return render_template('edit_candidate.html',candidate=prepare(r))
@app.route('/candidate/<int:candidate_id>/status',methods=['POST'])
def change_status(candidate_id):
    s=request.form.get('status');
    if s in ['Shortlisted','Under Review','Rejected']:update_status(candidate_id,s)
    return redirect(url_for('candidate',candidate_id=candidate_id))
@app.route('/candidate/<int:candidate_id>/feedback',methods=['POST'])
def feedback(candidate_id):
    d=request.form.get('decision','Pending');n=request.form.get('notes','').strip()
    if d in ['Hired','Rejected','Maybe','Pending']:save_feedback(candidate_id,d,n)
    return redirect(url_for('candidate',candidate_id=candidate_id))
@app.route('/candidate/<int:candidate_id>/delete',methods=['POST'])
def remove_candidate(candidate_id):
    r=get_candidate(candidate_id)
    if r:
        delete_candidate(candidate_id);p=os.path.join(UPLOAD_FOLDER,r['filename'])
        if os.path.exists(p):os.remove(p)
    return redirect(url_for('dashboard'))
@app.route('/clear-candidates',methods=['POST'])
def clear_all_candidates():
    for r in get_candidates():
        p=os.path.join(UPLOAD_FOLDER,r['filename'])
        if os.path.exists(p):
            try:os.remove(p)
            except OSError:pass
    clear_candidates();return redirect(url_for('dashboard'))
@app.route('/candidate/<int:candidate_id>/resume')
def view_resume(candidate_id):
    r=get_candidate(candidate_id);return redirect(url_for('dashboard')) if not r else send_from_directory(UPLOAD_FOLDER,r['filename'],as_attachment=False)
@app.route('/candidate/<int:candidate_id>/download')
def download_resume(candidate_id):
    r=get_candidate(candidate_id);return redirect(url_for('dashboard')) if not r else send_from_directory(UPLOAD_FOLDER,r['filename'],as_attachment=True)
@app.route('/export/excel')
def export_excel():
    from openpyxl import Workbook
    wb=Workbook();ws=wb.active;ws.title='Ranked Candidates';ws.append(['Rank','Candidate','Email','Phone','TF-IDF %','Semantic %','Overall %','Status','Experience Years','Qualification','Skills','Matched Skills','Missing Skills','Decision','Uploaded At'])
    for i,r in enumerate(get_candidates(),1):ws.append([i,r['candidate_name'],r['email'],r['phone'],r['tfidf_score'],r['semantic_score'],r['overall_score'],r['status'],r['experience_years'],r['qualification'],r['skills'],r['matched_skills'],r['missing_skills'],r['recruiter_decision'],r['uploaded_at']])
    out=io.BytesIO();wb.save(out);out.seek(0);return make_response(out.getvalue(),200,{'Content-Type':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet','Content-Disposition':'attachment; filename=ResumeRank_Results.xlsx'})
@app.route('/export/pdf')
def export_pdf():
    from reportlab.platypus import SimpleDocTemplate,Table,TableStyle,Paragraph,Spacer
    from reportlab.lib.pagesizes import landscape,A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    out=io.BytesIO();doc=SimpleDocTemplate(out,pagesize=landscape(A4),rightMargin=20,leftMargin=20,topMargin=20,bottomMargin=20);styles=getSampleStyleSheet();story=[Paragraph('ResumeRank — Ranked Candidate Report',styles['Title']),Spacer(1,10)];data=[['Rank','Candidate','Score','Status','Experience','Qualification','Decision']]
    for i,r in enumerate(get_candidates(),1):data.append([i,r['candidate_name'] or 'Unknown',f"{float(r['overall_score'] or 0):.2f}%",r['status'],f"{float(r['experience_years'] or 0):g} yrs",r['qualification'] or 'Not detected',r['recruiter_decision'] or 'Pending'])
    t=Table(data,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#10182f')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#dfe3eb')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8)]));story.append(t);doc.build(story);out.seek(0);return make_response(out.getvalue(),200,{'Content-Type':'application/pdf','Content-Disposition':'attachment; filename=ResumeRank_Results.pdf'})
@app.route('/retrain',methods=['POST'])
def retrain():
    rows=get_feedback()
    if len(rows)>=4 and len({r['decision'] for r in rows})>=2:
        model=make_pipeline(TfidfVectorizer(stop_words='english',ngram_range=(1,2)),LogisticRegression(max_iter=1000));model.fit([r['resume_text'] or '' for r in rows],[r['decision'] for r in rows]);import joblib;joblib.dump(model,os.path.join(BASE_DIR,'feedback_model.joblib'));flash(f'Feedback model retrained using {len(rows)} decisions.','success')
    else:flash('Add at least 4 Hired/Rejected decisions with both classes before retraining.','info')
    return redirect(url_for('dashboard'))

if __name__=='__main__':init_db();app.run(debug=True)
