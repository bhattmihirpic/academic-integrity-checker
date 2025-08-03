# These are like importing different tools into our workshop
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import os
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

# Import our custom detectives
from text_processor import extract_text_from_file, clean_up_text, get_text_info
from plagiarism_detector import PlagiarismDetector
from ai_detector import AIDetector

# Create our main app (like starting up our restaurant)
app = Flask(__name__)

# Give our app some settings (like house rules)
app.config['SECRET_KEY'] = 'my-super-secret-password-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///integrity.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Create our database helper (like a filing cabinet)
db = SQLAlchemy(app)

# Make sure our uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Wake up our detectives!
print("🔄 Waking up our detectives...")
plagiarism_detective = PlagiarismDetector()
ai_detective = AIDetector()
print("✅ All detectives are ready!")

# Create a database table to remember all our analyses
class Analysis(db.Model):
    """This keeps track of every document we've analyzed"""
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    assignment_name = db.Column(db.String(255), nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    plagiarism_score = db.Column(db.Float, default=0.0)
    ai_score = db.Column(db.Float, default=0.0)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    analysis_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))

# Create a form for people to upload files
class UploadForm(FlaskForm):
    """This creates the upload form people will see"""
    
    file = FileField('Choose Document', validators=[
        FileRequired('Please select a file!'),
        FileAllowed(['txt', 'pdf', 'docx'], 'Only TXT, PDF, and DOCX files are allowed!')
    ])
    assignment_name = StringField('Assignment Name', validators=[DataRequired()], 
                                default='My Assignment')
    submit = SubmitField('🔍 Analyze This Document!')

# This is what happens when someone visits our website
@app.route('/', methods=['GET', 'POST'])
def home_page():
    """The main page where people upload files"""
    
    form = UploadForm()
    
    # If someone submitted a file
    if form.validate_on_submit():
        try:
            print("📄 Someone uploaded a file!")
            
            # Get the uploaded file
            file = form.file.data
            filename = secure_filename(file.filename)  # Make filename safe
            
            # Create a unique filename so files don't overwrite each other
            name, extension = os.path.splitext(filename)
            unique_filename = f"{name}_{uuid.uuid4().hex[:8]}{extension}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file temporarily
            file.save(filepath)
            print(f"💾 Saved file as: {unique_filename}")
            
            # Extract text from the file
            print("📖 Reading the file...")
            original_text = extract_text_from_file(filepath)
            
            # Check if we got enough text to analyze
            if len(original_text.strip()) < 100:
                flash('Sorry! Your document is too short to analyze properly. Please upload a document with at least 100 characters.', 'error')
                os.remove(filepath)  # Clean up
                return redirect(url_for('home_page'))
            
            # Clean up the text for better analysis
            processed_text = clean_up_text(original_text)
            
            # Get basic information about the text
            text_stats = get_text_info(original_text)
            
            print("🕵️ Starting plagiarism analysis...")
            # Check for plagiarism
            plagiarism_result = plagiarism_detective.check_for_plagiarism(processed_text)
            
            print("🤖 Starting AI detection...")
            # Check for AI content
            ai_result = ai_detective.detect_ai_content(processed_text)
            
            # Save everything to our database
            analysis = Analysis(
                filename=filename,
                assignment_name=form.assignment_name.data,
                original_text=original_text,
                plagiarism_score=plagiarism_result['overall_score'],
                ai_score=ai_result['ai_probability']
            )
            
            db.session.add(analysis)
            db.session.commit()
            
            # Clean up the temporary file
            os.remove(filepath)
            
            print(f"✅ Analysis complete! Results saved with ID: {analysis.analysis_id}")
            flash('Analysis completed successfully!', 'success')
            return redirect(url_for('show_results', analysis_id=analysis.analysis_id))
            
        except Exception as e:
            error_message = f'Oops! Something went wrong: {str(e)}'
            print(f"❌ Error: {error_message}")
            flash(error_message, 'error')
            
            # Clean up if file exists
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
                
            return redirect(url_for('home_page'))
    
    # Show recent analyses on the home page
    recent_analyses = Analysis.query.order_by(Analysis.timestamp.desc()).limit(5).all()
    
    # System status information
    system_info = {
        'ai_detective_working': ai_detective.model_working,
        'total_analyses_done': Analysis.query.count()
    }
    
    return render_template('index.html', form=form, recent_analyses=recent_analyses, 
                         system_info=system_info)

# This shows the results of an analysis
@app.route('/results/<analysis_id>')
def show_results(analysis_id):
    """Show detailed results for a specific analysis"""
    
    # Find the analysis in our database
    analysis = Analysis.query.filter_by(analysis_id=analysis_id).first_or_404()
    
    print(f"📊 Showing results for: {analysis.filename}")
    
    # Re-run the analysis to get detailed results
    processed_text = clean_up_text(analysis.original_text)
    plagiarism_result = plagiarism_detective.check_for_plagiarism(processed_text)
    ai_result = ai_detective.detect_ai_content(processed_text)
    text_stats = get_text_info(analysis.original_text)
    
    return render_template('results.html', 
                         analysis=analysis,
                         plagiarism_result=plagiarism_result,
                         ai_result=ai_result,
                         text_stats=text_stats)

# Show history of all analyses
@app.route('/history')
def show_history():
    """Show all previous analyses"""
    
    page = request.args.get('page', 1, type=int)
    analyses = Analysis.query.order_by(Analysis.timestamp.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('history.html', analyses=analyses)

# About page
@app.route('/about')
def about_page():
    """Information about our tool"""
    return render_template('about.html')

# Handle file too large errors
@app.errorhandler(413)
def file_too_large(e):
    flash("File is too large! Maximum size is 32MB.", "error")
    return redirect(url_for('home_page'))

# Handle page not found errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Start our app!
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("🗄️ Database is ready!")
    
    print("🚀 Starting our Academic Integrity Tool...")
    print("🌐 Open your web browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)