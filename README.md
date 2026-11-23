# Face Recognition Attendance Management System

## Project Description

The **Face Recognition Attendance Management System** is a web-based application developed using Django and OpenCV to automate the attendance management process through facial recognition.

The system uses computer vision and machine learning techniques to detect and recognize registered student faces, allowing attendance to be recorded automatically. It aims to reduce manual attendance work, minimize errors, and provide an efficient way to manage student attendance records.

The application also provides separate interfaces for administrators and students to manage attendance records, submit leave requests, and monitor attendance information.

## Key Features

- **Face Detection:** Detects human faces using OpenCV Haar Cascade Classifier.
- **Face Recognition:** Identifies registered students using the LBPH (Local Binary Patterns Histograms) face recognition algorithm.
- **Image Preprocessing:** Uses grayscale conversion, Gamma Correction, CLAHE, and Weighted Local Descriptor (WLD) techniques to improve facial image representation.
- **Automated Attendance:** Records student attendance after successful face recognition.
- **Student Management:** Allows administrators to manage student records.
- **Leave Management:** Students can submit sick leave, casual leave, and on-duty requests.
- **Admin Dashboard:** Provides attendance monitoring and leave request approval or rejection.
- **Attendance Tracking:** Maintains student attendance records, including check-in and check-out information.

## Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, Django |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Computer Vision | OpenCV |
| Face Detection | Haar Cascade Classifier |
| Face Recognition | LBPH Face Recognizer |
| Image Processing | CLAHE, Gamma Correction, WLD |
| Database | SQLite |
| Development Environment | Visual Studio Code |
| Version Control | Git and GitHub |

## Face Recognition Workflow

1. Student facial images are collected and stored in the dataset.
2. Images undergo preprocessing, including grayscale conversion, Gamma Correction, and CLAHE enhancement.
3. Haar Cascade detects the face region.
4. The detected face is resized and processed using the WLD descriptor.
5. The LBPH algorithm is trained using the processed facial images.
6. During attendance, the system captures a face and applies the same preprocessing pipeline.
7. The trained model compares the captured face with registered student data.
8. If a student is recognized, attendance is recorded in the database.

## Project Structure

```text
Face-Recognition-Attendance-System/
│
├── faceweb/
│   ├── settings.py
│   ├── urls.py
│   └── ...
│
├── recognition/
│   ├── models.py
│   ├── views.py
│   ├── opencv_face.py
│   ├── templates/
│   └── migrations/
│
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
├── haarcascade_smile.xml
├── manage.py
├── requirements.txt
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Face-Recognition-Attendance-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Face-Recognition-Attendance-System
```

### 3. Create a Virtual Environment

```bash
python -m venv env
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
env\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create a `.env` file in the project root directory and add:

```env
DJANGO_SECRET_KEY=your-secret-key
```

Generate a new secret key locally and keep it private.

### 7. Apply Database Migrations

```bash
python manage.py migrate
```

### 8. Create an Administrator Account

```bash
python manage.py createsuperuser
```

### 9. Run the Development Server

```bash
python manage.py runserver
```

Open your browser and visit:

```text
http://127.0.0.1:8000/
```

## Future Enhancements

- Improve face recognition accuracy under different lighting conditions.
- Integrate real-time camera-based attendance monitoring.
- Generate downloadable attendance reports.
- Enhance the dashboard with advanced attendance analytics.
- Explore additional facial recognition techniques.

## Author

**Dinesh T.**

## License

This project is developed for academic and educational purposes.
