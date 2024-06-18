from flask import Flask, render_template, session, flash, redirect, request, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import or_
from sqlalchemy.orm import joinedload
from datetime import datetime, timedelta
import joblib
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import sys
import warnings
from sqlalchemy.sql import func

warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

app = Flask(__name__)

sys.stdout.reconfigure(encoding='utf-8')

# model = joblib.load('Project_ML.pkl')
l1 = [ 'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 
                  'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 
                  'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 
                  'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 
                  'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 
                  'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 
                  'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 
                  'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 
                  'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 
                  'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 
                  'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 
                  'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 
                  'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
                  'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 
                  'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 
                  'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 
                  'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 
                  'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort','foul_smell_of urine', 
                  'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 
                  'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 
                  'belly_pain', 'abnormal_menstruation','dischromic _patches', 'watering_from_eyes', 
                  'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 
                  'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 
                  'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 
                  'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 
                  'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 
                  'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 
                  'red_sore_around_nose', 'yellow_crust_ooze','increased_thirst']

# List of diseases
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 
                'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine', 
                'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 
                'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 
                'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 
                'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 
                'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 
                'Urinary tract infection', 'Psoriasis', 'Impetigo','Covid']

model1 = joblib.load('mental_health_model.pkl')

questions = [
    "How often do you have trouble falling asleep or staying asleep?",
    "How often do you experience changes in your appetite, such as eating significantly more or less than usual?",
    "How often do you lose interest in hobbies or activities that used to bring you joy?",
    "How frequently do you feel physically or mentally exhausted? ",
    "How often do you struggle with feelings of low self-worth or inadequacy? ",
    "How often do you find it difficult to concentrate or focus on tasks?",
    "How frequently do you feel restless or on edge?",
    "Have you had thoughts of harming yourself or ending your life?",
    "How often do you experience disturbances in your sleep, such as nightmares or night terrors?",
    "How frequently do you experience aggressive behavior towards others?",
    "Have you experienced sudden, intense feelings of fear or panic?",
    "How often do you feel a sense of despair or hopelessness about the future?",
    "How frequently do you feel restless or unable to relax?",
    "How often do you experience a lack of energy or motivation to engage in activities?"
]

clf_tree = joblib.load('models/DecisionTree.pkl')
knn = joblib.load('models/KNN.pkl')
gnb = joblib.load('models/NaiveBayes.pkl')

app.secret_key = "Deepubhai"
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///users.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['UPLOAD_FOLDER'] = 'D:/22f2000876/Menty/Photos'
db = SQLAlchemy(app)
# Models Below 
class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
class Review(db.Model):
	_id = db.Column("id", db.Integer, primary_key=True)
	user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
	rating = db.Column(db.Integer, nullable=False)
	feedback = db.Column(db.Text, nullable=True)
	date_reviewed = db.Column(db.DateTime, nullable=False, default=datetime.now())


	user = db.relationship('User', backref=db.backref('feedback', lazy=True))
class Section(db.Model):
	id = db.Column("id", db.Integer, primary_key=True)
	name = db.Column(db.String(100), unique=True, nullable=False)
	date_created = db.Column(db.DateTime, default=datetime.now())
	description = db.Column(db.Text, nullable=True)
class Document(db.Model):
	id = db.Column("id", db.Integer, primary_key=True)
	title = db.Column("title", db.String(100), nullable=False)
	date_added = db.Column(db.DateTime, nullable=False, default=datetime.now())
	description = db.Column("description", db.Text, nullable=True)
	file_path = db.Column(db.String(255), nullable=False)
	cover_path = db.Column(db.String(255), nullable=False)
	section_id = db.Column(db.Integer, db.ForeignKey('section.id'))

	section = db.relationship('Section', backref=db.backref('book', lazy=True))
class RatingFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doc_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    username = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    date_reviewed = db.Column(db.DateTime, nullable=False, default=datetime.now())

    document = db.relationship('Document', backref=db.backref('ratings_feedback', lazy=True))
    user = db.relationship('User', backref=db.backref('ratings_feedback', lazy=True))



with app.app_context():
    db.create_all()

model = load_model('my_model.h5')

# Preprocess the uploaded image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image
@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image and make prediction
            image = preprocess_image(file_path)
            if image is not None:
                predictions = model.predict(image)
                results = [round(float(pred), 2) for pred in predictions[0]]

                # Define thresholds
                thresholds = {
                    "Acne": 0.10,
                    "Pigmentation": 0.10,
                    "Redness": 0.15,
                    "Sagging": 0.10,
                    "Hydration": 0.15,  # Lower is poor
                    "Translucency": 0.20,  # Lower is poor
                    "Uniformness": 0.20,  # Lower is poor
                    "Wrinkles": 0.15,
                    "Pores": 0.15,
                    "Oily": 0.15
                }

                # Define product recommendations
                product_recommendations = {
                    "Acne": "Acne removal cream",
                    "Pigmentation": "Pigmentation reducing serum",
                    "Redness": "Anti-redness lotion",
                    "Sagging": "Skin firming cream",
                    "Hydration": "Moisturizing cream",
                    "Translucency": "Brightening essence",
                    "Uniformness": "Tone correcting treatment",
                    "Wrinkles": "Anti-wrinkle cream",
                    "Pores": "Pore minimizing toner",
                    "Oily": "Oil control gel"
                }

                # Check if any condition is above or below the threshold and recommend products
                recommendations = {}
                for key, result in zip(thresholds.keys(), results):
                    if key in ["Hydration", "Translucency", "Uniformness"]:
                        if result < thresholds[key]:  # Lower is poor for these conditions
                            recommendations[key] = product_recommendations[key]
                    else:
                        if result > thresholds[key]:
                            recommendations[key] = product_recommendations[key]

                if not recommendations:
                    recommendations["Overall"] = "Your skin condition is good. No products needed."

                return render_template('result1.html', results=results, recommendations=recommendations)
    return render_template('skin.html')




@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    if request.method == 'POST':
        responses = [int(request.form.get(question)) for question in questions]
        # Assuming model.predict is a placeholder for your actual prediction function
        prediction = model1.predict([responses])[0]
        
        return render_template('result.html', prediction=prediction)
    return render_template('predict.html', questions=questions)



@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
                symptoms = request.form.getlist('symptom')
                input_data = [1 if symptom in symptoms else 0 for symptom in l1]
                
                prediction_tree = clf_tree.predict([input_data])[0]
                prediction_knn = knn.predict([input_data])[0]
                prediction_gnb = gnb.predict([input_data])[0]

                result_tree = disease[prediction_tree]
                result_knn = disease[prediction_knn]
                result_gnb = disease[prediction_gnb]
                flash("Please note that the results provided by Menty may not always be accurate and should not be considered a substitute for professional medical advice. If you have serious symptoms or concerns, please consult a doctor immediately.","warning")


                return render_template('ml.html', symptoms=l1, result_tree=result_tree, result_knn=result_knn, result_gnb=result_gnb)
    return render_template('ml.html', symptoms=l1)



@app.route('/quiz')
def quiz_redirect():
    if 'username' in session:
        return redirect('/symptoms')
    else:
        flash('Please login before continuing...','warning')
        return redirect('/login')
@app.route('/symptoms')
def symptoms1():
    if 'username' in session:
        flash("Please note that the results provided by Menty may not always be accurate and should not be considered a substitute for professional medical advice. If you have serious symptoms or concerns, please consult a doctor immediately.","warning")
        return render_template('quiz.html')
    else:
        flash('Please login before continuing...','warning')
        return redirect('/login')
@app.route('/adminlogin', methods=['GET','POST'])
def adminlogin():
    if request.method == 'POST':
        admin_username = request.form['username']
        admin_password = request.form['password']

        admin = Admin.query.filter_by(username=admin_username, password=admin_password).first()

        if admin:
            session['admin_id'] = admin.id
            return redirect("admindashboard")
        else:
            flash('Invalid username or password. Please try again.', 'error')
           
    return render_template('adminlogin.html')
@app.route("/admindashboard")
def admin():
    return render_template("admindashboard.html")
@app.route('/doc')
def admin_dashboard():

    docs = Document.query.all()

    return render_template('doc.html', docs=docs)
@app.route('/edit_doc/<int:doc_id>', methods=['GET', 'POST'])
def edit_book(doc_id):
    doc = Document.query.get(doc_id)
    sections = Section.query.all()
    if not doc:
        flash('doc not found.', 'error')
        return redirect('/admindashboard')

    if request.method == 'POST':
        
        title = request.form['title'].strip() 
        pdf_path = request.form['pdf_path'].strip()
        content = request.form['Content'].strip()
        section_id = request.form['section'].strip()
        cover=request.form['cover'].strip()
        if title:
            doc.title = title
        
        if pdf_path:
            doc.file_path = pdf_path
        if content:
            doc.description = content
        if section_id:
            if section_id == 'Unassigned':
                doc.section_id=0
            else:
                doc.section_id=section_id
        if cover:
            doc.cover_path = cover
      
        db.session.commit()
        flash('Document details updated successfully.', 'success')
        return redirect(f'/edit_doc/{doc_id}')
    return render_template('edit_doc.html', doc=doc, sections=sections)

@app.route('/edit_sec/<int:section_id>', methods=['GET', 'POST'])
def edit_section(section_id):
    section = Section.query.get(section_id)
    if not section:
        flash('section not found.', 'error')
        return redirect('/admindashboard')

    if request.method == 'POST':
        name = request.form['name'].strip() 
        description = request.form['description'].strip()

        if name:
            section.name = name
        if description:
            section.description = description
        db.session.commit()
        flash('Section details updated successfully.', 'success')
        return redirect(f'/edit_sec/{section_id}')
    return render_template('edit_sec.html', section=section)

@app.route('/sec')
def sec():

    secs = Section.query.all()

    return render_template('sec.html', secs=secs)

@app.route('/add_sec', methods=['GET', 'POST'])
def add_section():
    if request.method == 'POST':
        name = request.form['name']
        description= request.form['Description']

        new_section = Section(name=name, description=description)

        try:
            db.session.add(new_section)
            db.session.commit()
            flash('Section Added Successfully','success')
            return redirect('/add_sec')
        except IntegrityError:
            db.session.rollback()
            flash('Error adding the section. Please try again.','warning')

    return render_template('add_sec.html')



@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user:
            app.logger.info("User found: %s", user.username)
            if user.check_password(password):
                session["username"] = username
                flash("Logged in successfully", "success")
                return redirect("/home")
            else:
                app.logger.error("Password mismatch for user: %s", username)
                flash("Incorrect username or password. Please try again.", "warning")
        else:
            app.logger.error("User not found: %s", username)
            flash("User not found. Please check your username.", "warning")

    elif "username" in session:
        flash("Already logged in!", "warning")
        return redirect("/home")

    return render_template("login.html")


@app.route("/sign_up", methods=["POST", "GET"])
def sign_up():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        email = request.form['email']
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists', 'warning')
            return redirect("/sign_up")

        new_user = User(username=username, email=email)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! You can now login.", "success")
        return redirect("/login")

    elif "username" in session:
        flash("Already logged in!", "warning")
        return redirect("/home")

    return render_template("signup.html")

@app.route('/services')
def services():
    if 'username' in session:
        return render_template("service.html")
    else:
        flash('Please login before continuing...','warning')
        return redirect('/login')
@app.route('/product')
def product():
    if 'username' in session:
        return render_template("product.html")
    else:
        flash('Please login before continuing...','warning')
        return redirect('/login')

@app.route('/medication')
def medication():
    if 'username' in session:
        user = User.query.get(session['username'])
        sections = Section.query.all()
        documents = Document.query.all()
        recent_docs = Document.query.filter(Document.date_added >= datetime.now() - timedelta(days=5)).all()

        top_rated_docs = db.session.query(
            Document, func.avg(RatingFeedback.rating).label('average_rating')
        ).join(RatingFeedback).group_by(Document.id).order_by(func.avg(RatingFeedback.rating).desc()).all()

        return render_template('medication.html', documents=documents, sections=sections, current_datetime=datetime.now(), recent_docs=recent_docs, top_rated_docs=top_rated_docs)
    else:
        flash('Please login before continuing...', 'warning')
        return redirect('/login')



@app.route('/')
@app.route('/home')
def home():
    good_reviews = Review.query.filter(Review.rating > 3).order_by(Review.date_reviewed.desc()).limit(3).all()
    users=User.query.all()
    return render_template('home.html',reviews=good_reviews, users=users)
@app.route('/userprofile')
def userp():
    username = session.get('username')
    user = User.query.filter_by(username=username).first()
    return render_template("userp.html", user=user)
# Route to handle changing password
@app.route('/change_password', methods=['POST'])
def change_password():
    new_password = request.form['new_password']
    username = session.get('username')
    user = User.query.filter_by(username=username).first()
    if user:
        user.set_password(new_password)
        db.session.commit()
        flash("Password successfully updated.", "success")
    else:
        flash("User not found.", "danger")

    return redirect('/userprofile')

# Route to handle changing email
@app.route('/change_email', methods=['POST'])
def change_email():
    new_email = request.form['new_email']
    username = session.get('username')
    user = User.query.filter_by(username=username).first()
    if user:
        user.email = new_email
        db.session.commit()
        flash("Email successfully updated.", "success")
    else:
        flash("User not found.", "danger")

    return redirect('/userprofile')


# Route to handle logout
@app.route('/logout')
def logout():
    # Code to clear session and log out user
    session.clear()
    return redirect('/login')


@app.route('/review', methods=['GET','POST'])
def submit_review():
    if request.method=='POST':
        username=session.get("username")
        user = User.query.filter_by(username=username).first()
        user_id=user.id
        existing_review = Review.query.filter_by(user_id=user_id).first()
        if existing_review:
            flash('You have already submitted a review.', 'warning')
            return redirect('/home')

        rating = int(request.form['rating'])
        feedback = request.form['feedback']
        review = Review(user_id=user_id, rating=rating, feedback=feedback)
        db.session.add(review)
        db.session.commit()
        flash('Review submitted successfully!', 'success')
        return redirect('/home')
    reviews=Review.query.all()
    users=User.query.all()
    return render_template('review.html', reviews=reviews,users=users)
# Not for use:
@app.route('/add_doc', methods=['GET','POST'])
def add_book_to_section():
    if request.method == 'POST':
        title = request.form['title']
        section_id = request.form['section']  
        section = Section.query.get(section_id)
        description= request.form['content']
        file_path=request.form['pdf_path']
        cover_path=request.form['cover']
        
        
        if not section:
            flash('Section does not exist.', 'error')
            return redirect('/add_book')
        new_doc = Document(title=title, section_id=section_id,description=description,file_path=file_path,cover_path=cover_path)

        db.session.add(new_doc)
        db.session.commit()
        flash('Doc added successfully!', 'success')
        return redirect("/add_doc")
    sections = Section.query.all()
    return render_template('add_doc.html', sections=sections)
@app.route('/view_pdf', methods=['POST'])
def view_books():
    doc_id = request.form.get("doc_id")
    doc = Document.query.filter_by(id=doc_id).first()
    if not doc:
        return "Document not found", 404
    file_path = doc.file_path
    return render_template('pdf_viewer.html', file_path=file_path)
@app.route("/search", methods=["GET"])
def search():
    if "username" in session:
        query = request.args.get("query")
        search_type = request.args.get("search_type")
        
        if not query:
            flash("Please enter a search query.", "warning")
            return redirect("/medication")
        else:
            documents = Document.query.outerjoin(Section).filter(
                or_(
                    Document.title.ilike(f"%{query}%"), 
                    Section.name.ilike(f"%{query}%")
                )
            ).all()
            return render_template("srb.html", documents=documents, query=query)
        
    flash("You need to login to search for documents!", "warning")
    return redirect("/login")
@app.route("/rate_doc/<int:doc_id>", methods=["GET", "POST"])
def rate_redirect(doc_id):
    return redirect(url_for('review_doc', doc_id=doc_id))

@app.route("/review_doc", methods=["GET", "POST"])
def review_doc():
    doc_id = request.args.get("doc_id") or request.form.get("doc_id")
    doc = Document.query.filter_by(id=doc_id).first()
    username = session.get('username')
    ratingandfeedbacks = RatingFeedback.query.filter_by(doc_id=doc_id).all()

    if not doc:
        flash('Document not found.', 'warning')
        return redirect('/medication')

    if request.method == 'POST':
        ratingandfeedbacks = RatingFeedback.query.filter_by(doc_id=doc_id, username=username).all()
        # Check if the user has already submitted a review for this document
        if ratingandfeedbacks:
            flash('You have already submitted a review for this document.', 'error')
            return redirect('/medication')  # Redirect to appropriate page
        else:
            rating = request.form['rating']
            feedback = request.form['feedback']
            new_rf = RatingFeedback(doc_id=doc_id, username=username, rating=rating, feedback=feedback)
            try:
                db.session.add(new_rf)
                db.session.commit()
                flash('Thank you for your rating and feedback!', 'success')
                return redirect('/medication')
            except IntegrityError:
                db.session.rollback()
                flash('Unable to rate/feedback', 'error')
                return redirect('/medication')

    return render_template('review_doc.html', doc=doc, ratingandfeedbacks=ratingandfeedbacks)



if __name__ == '__main__':
    app.run(host="192.168.1.37",debug=True)
