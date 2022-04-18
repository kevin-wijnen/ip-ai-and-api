import json
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import sqlite3
import subprocess
from threading import Thread

UPLOAD_FOLDER = "./input"
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Overige functies, voor afhandelen van proces
def filetype_check(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/start', methods=['POST'])
def start_ai():
    # Starten van AI, zodat het proces kan beginnen
    # print("Nog niet geimplementeerd, vanwege background thread.")
    thread = Thread(target=start_ai_script())
    return "App started successfully!"
    thread.start()



@app.route('/status', methods=['GET'])
def check_ai():
    # Check SQLite database voor status!
    con = sqlite3.connect('API/AI_Status')
    cur = con.cursor()
    # Eerste rij pakken vanwege eerste AI
    status = cur.execute("SELECT * FROM info").fetchone()
    con.close()
    return "Status: " + status[1]


@app.route('/image', methods=['POST'])
def image_input():
    # Invoeren van afbeelding in API afbeelding map!
    print(request)
    print(request.headers)
    print(request.files)
    if 'file' not in request.files:
        return "Bestand niet gevo   nden"
    file = request.files['file']

    # Wanneer geen bestand geselecteerd is
    if file.filename == '':
        return "Geen bestand geselecteerd"
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return "Bestand ontvangen"


# Resultaten app route GET
# Dictionary terug geven met resultaten en anchor filenamen!
# Dit invullen wanneer AI resultaten

@app.route('/results', methods=['GET'])
def check_results():
    with open("API/results.json", "r") as results_file:
        results = json.load(results_file)
    return results


def start_ai_script():
    subprocess.call(["python", "siamese_network.py"])
