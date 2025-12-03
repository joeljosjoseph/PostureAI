📌 PostureAI – Intelligent Workout \& Posture Detection App



Welcome to PostureAI, an AI-powered fitness assistant that uses computer vision, MediaPipe pose landmarks, and machine learning to help users track workouts, count reps, detect posture quality, and estimate calories burned — all in real time through a webcam.



This project was fully developed by Arjun Bindhu Suresh as part of a fitness-based AI application initiative.



🚀 Features

🔍 1. Real-Time Webcam Workout Detection



Detect major exercises such as:



Push Ups



Squats



Lunges



Pull Ups



Bench Press



Jumping Jacks



Wall Pushups



And 20+ more workouts (from your curated dataset)



Also includes Auto-Detect mode powered by a trained ML model.



🔢 2. Smart Rep Counter



Automatically counts repetitions based on joint angle patterns.



Supports customizable target reps.



Stops automatically when reps are completed.



🧍‍♂️ 3. Posture Quality Feedback



Analyses body angles using MediaPipe.



Displays visual real-time feedback such as:



“Good Form!”



“Fix your back”



“Straighten elbows”



Uses clean, professional overlay UI.



🔔 4. Notification-Style Alerts



No sounds needed.



Subtle pop-up bubbles that do not clutter the screen.



🧮 5. Calories \& Session Summary



After a workout session, users get:



Total reps completed



Calories burned (per exercise-based estimation)



Total workout duration



Exercise type detected



🎯 6. Full-Screen HD Camera Interface



Professional layout



No black boxes



Clean color-coded UI



High readability on both light and dark backgrounds



🧠 Tech Stack

Computer Vision



MediaPipe Pose (33 landmark detection)



OpenCV (real-time rendering)



Machine Learning



Scikit-learn RandomForestClassifier



Custom pose dataset created from:



UCF101 Exercise dataset



Kaggle Workout Videos dataset



.joblib model saved for fast inference



Backend Logic



Python 3.10



Real-time angle calculations



Session tracker \& CSV logging



📁 Project Structure

PostureAI/

│

├── posture\_main.py            # Main application (UI + webcam + detection)

├── posture\_detector.py        # Pose \& angle extraction utilities

├── utils.py                   # Helper functions (angles, notifications, UI)

│

├── model/

│   └── workout\_pose\_model.joblib      # Trained ML model

│

├── dataset/

│   ├── workout\_pose\_dataset.csv       # Pose landmark dataset

│   └── workout\_subset\_summary.csv     # Summary of workouts used

│

├── scripts/

│   └── train\_pose\_model.py            # Training script

│

├── session\_logs.csv            # Optional: logs of your sessions

└── README.md                   # Project documentation



▶️ How to Run the App Locally

1\. Clone the Repository

git clone https://github.com/Arjun-Bindhu-Suresh/PostureAI.git

cd PostureAI



2\. Activate Virtual Environment

.\\venv310\\Scripts\\activate



3\. Install Dependencies (if needed)

pip install -r requirements.txt





(If you want, I can generate a requirements.txt for you — just say “create requirements.txt”)



4\. Run the Application

python posture\_main.py



🎥 Demo (Optional)



If you want, I can help you create:



GIF previews



Demo video instructions



UI screenshots

Just say: “Add demo section to README”



📊 Dataset Sources



UCF101 Exercise Subset



Kaggle – Workout Videos Dataset (652 videos)



Custom extracted frames → pose dataset → ML model



📌 Future Enhancements



Mobile app version



Personalized workout plan generator



Diet recommendation module



Form violation detection using DL models



Integration with GCP/AWS backend



Progress dashboard with charts



👤 Author



Arjun Bindhu Suresh

Canada 🇨🇦

Cloud Computing \& AI Specialist



GitHub: Arjun-Bindhu-Suresh

