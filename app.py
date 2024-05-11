from flask import Flask,url_for,render_template,redirect,request
# import sqlite3 as SQL
app = Flask(__name__)
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os

import glob
import subprocess
from werkzeug.utils import secure_filename

global graph


app = Flask(__name__)
app.config['SECRET_KEY'] = '1233444545555'  # Replace with a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # SQLite database URI
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'static/uploader/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SIZE = 24

DATA_FOLDER = 'data/'
app.config['DATA_FOLDER'] = DATA_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed file extensions for images
# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User class for Flask-Login
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create the database tables before running the app
with app.app_context():
    db.create_all()

@app.route("/", methods=['GET'])
def hello():
    return render_template('home.html')


@app.route("/index", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose another username.', 'error')
            return redirect(url_for('signup'))

        # Create a new user
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template("signup.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username and password are valid
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            login_user(user)
            flash('Login successful.', 'success')
            return redirect(url_for('choice'))
        else:
            flash('Invalid username or password. Please try again.', 'error')

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash('Logout successful.', 'success')
    return redirect(url_for('hello'))

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        # Get uploaded files and disease name
        images = request.files.getlist('images[]')
        disease_name = request.form['disease_name']

        # Check if disease name is provided
        if not disease_name:
            flash('Please enter a disease name', 'error')
            return redirect(request.url)

        # Create a folder for the disease if it doesn't exist
        disease_folder = os.path.join(app.config['DATA_FOLDER'], disease_name)
        if not os.path.exists(disease_folder):
            os.makedirs(disease_folder)

        # Save uploaded images
        for image in images:
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image_path = os.path.join(disease_folder, filename)
                image.save(image_path)

        # Check if images were uploaded
        if not os.listdir(disease_folder):
            flash('No images uploaded', 'error')
            return redirect(request.url)

        # Call train.py script to initiate model training
   #     result = subprocess.run(['python', 'train1.py'], capture_output=True, text=True)
    #    output = result.stdout
     #   return render_template('train.html', output=output)
        
        return redirect(url_for('train'))  # Redirect to choice page after training
    return render_template('data.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
      try:
         subprocess.run(['python', 'train.py'])
         flash('Training initiated successfully', 'success')
      except Exception as e:
            flash(f'Error: {str(e)}', 'error')
      return redirect(url_for('train_success'))


@app.route('/loading')
def loading():
    flash("Training completed successfully!")
    return render_template("loading.html")

@app.route('/train_success')
def train_success():
    flash("Training completed successfully!")
    return render_template('train_success.html')

@app.route('/Predict',methods =['POST','GET']  )
def Upload():
    if request.method == 'POST':
        file = request.files['image']
        print(file) 
        if not file:
            flash('No file uploaded', 'error')
            return redirect(request.url)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],'1.png'))
        model_files = glob.glob('model/*.h5')
        # Sort the list of model files by modification time to get the latest model
        latest_model_file = max(model_files, key=os.path.getmtime)
        # Load the latest model
        print(latest_model_file)
        model = keras.models.load_model(latest_model_file)
        print(model)
     #  model = keras.models.load_model(r'model\Sequential.h5')
        categories = [folder for folder in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, folder))]
        nimage = cv2.imread(r"static\uploader\1.png", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(nimage,(SIZE,SIZE))
        image = image/255.0
        prediction = model.predict(np.array(image).reshape(-1,SIZE,SIZE,1))
        pclass = np.argmax(prediction)
        pValue = "Predict: {0}".format(categories[int(pclass)])
        print(pValue)
        realvalue = "Real Value 1"
        print('success')
        img = "/uploader/1.png"
        return render_template('result.html',value=pValue)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
    
    