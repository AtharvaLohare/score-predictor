# Student Exam Performance Predictor

This project predicts a student's math score based on their background and test scores using a machine learning model. It has a web interface built with Flask where you can enter details and get a prediction.

---

## How it works

- Fill out a form on the website with info like gender, ethnicity, parental education, lunch type, test prep, reading and writing scores.
- The app processes your input, runs it through a trained ML model, and shows you the predicted math score.

---

## Setup & Usage

1. **Clone the repo and set up the environment**
    ```
    git clone https://github.com/yourusername/student-exam-performance-predictor.git
    cd student-exam-performance-predictor
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

2. **Train the model**
    - Run the scripts in `src/components/` to process the data and train the model.
    - This will create `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

3. **Start the Flask app**
    ```
    python app.py
    ```
    - Go to [http://localhost:5000](http://localhost:5000) in your browser.

---

## Project Structure


---

## What's inside

- **Flask app:** Handles the website and user input.
- **ML pipeline:** Code for data processing, training, and prediction.
- **Templates:** HTML files for the web pages.
- **Artifacts:** The trained model and preprocessor.

---

## Tech stack

- Python
- Flask
- pandas, numpy
- scikit-learn, xgboost, catboost

---

## Why this project?

I wanted to learn how to connect machine learning with a web interface. I'm new to Flask and web stuff, so this project helped me understand how to organize code and make a simple ML web app.

---

## Credits

Thanks to the open-source community and various tutorials for inspiration!

---

