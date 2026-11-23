import cv2
import os
import numpy as np
from django.conf import settings
from recognition.models import Student


DATASET_PATH = os.path.join(
    settings.BASE_DIR, "media", "students"
)

MODEL_PATH = os.path.join(
    settings.BASE_DIR, "media", "trained", "enhanced_model.yml"
)

CASCADE_PATH = os.path.join(
    settings.BASE_DIR, "haarcascade_frontalface_default.xml"
)

FACE_SIZE = (100, 100)


# Load the face detector once instead of loading it for
# every training image and recognition request.
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def adjust_gamma(image, gamma=1.2):
    """
    Helps normalize illumination before CLAHE.
    Uses the same transformation during training
    and recognition.
    """

    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")

    inverse_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inverse_gamma) * 255
        for i in range(256)
    ]).clip(0, 255).astype(np.uint8)

    return cv2.LUT(image, table)


def compute_wld_descriptor(face_roi):
    """
    Computes the eight-neighbor Weber Local Descriptor.

    The descriptor is kept consistent with your original
    implementation.
    """

    img_f = face_roi.astype(np.float64)

    padded = np.pad(
        img_f,
        pad_width=1,
        mode="edge"
    )

    h, w = img_f.shape

    differential_excitation = np.zeros(
        (h, w),
        dtype=np.float64
    )

    for i in range(1, h + 1):
        for j in range(1, w + 1):

            center = padded[i, j]

            if center == 0:
                continue

            neighbors = [
                padded[i - 1, j - 1],
                padded[i - 1, j],
                padded[i - 1, j + 1],
                padded[i, j - 1],
                padded[i, j + 1],
                padded[i + 1, j - 1],
                padded[i + 1, j],
                padded[i + 1, j + 1]
            ]

            ratio_sum = sum(
                (neighbor - center) / center
                for neighbor in neighbors
            )

            differential_excitation[i - 1, j - 1] = (
                np.arctan(ratio_sum)
            )

    wld_normalized = cv2.normalize(
        differential_excitation,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return wld_normalized.astype(np.uint8)


def preprocess_image(image_path):
    """
    Applies the same preprocessing pipeline to training
    and recognition images.

    Pipeline:
        Grayscale
        -> Gamma correction
        -> CLAHE
        -> Haar face detection
        -> Largest face crop
        -> Resize to 100 x 100
        -> WLD descriptor

    Returns None if a face cannot be detected.
    """

    img = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        return None

    if img.size == 0:
        return None

    if face_cascade.empty():
        print(
            f"ERROR: Haar Cascade could not be loaded: "
            f"{CASCADE_PATH}"
        )
        return None

    # Step 1: Normalize illumination.
    gamma_img = adjust_gamma(img, gamma=1.2)

    # Step 2: Improve local contrast.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_img = clahe.apply(gamma_img)

    # Step 3: Detect faces on both the original grayscale
    # image and the enhanced image.
    detection_images = [enhanced_img]

    if not np.array_equal(img, enhanced_img):
        detection_images.append(img)

    detected_faces = []

    for detection_img in detection_images:

        faces = face_cascade.detectMultiScale(
            detection_img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            detected_faces = list(faces)
            break

    # Do not silently treat the whole photograph as a face.
    if not detected_faces:
        print(
            f"WARNING: No face detected in: {image_path}"
        )
        return None

    # Step 4: Select the largest detected face.
    x, y, w, h = max(
        detected_faces,
        key=lambda face: face[2] * face[3]
    )

    # Step 5: Crop from the same enhanced image used
    # for normalization.
    face_roi = enhanced_img[y:y + h, x:x + w]

    if face_roi.size == 0:
        return None

    # Step 6: Standardize the face dimensions.
    face_roi = cv2.resize(
        face_roi,
        FACE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Step 7: Compute the WLD descriptor.
    descriptor = compute_wld_descriptor(face_roi)

    return descriptor


def train_model():
    """
    Trains the LBPH recognizer using student images
    stored in media/students/<student_name>/.

    Existing student label IDs are preserved.
    """

    faces = []
    labels = []

    if not os.path.isdir(DATASET_PATH):
        print(
            f"ERROR: Dataset folder not found: {DATASET_PATH}"
        )
        return None

    if face_cascade.empty():
        print("ERROR: Haar Cascade failed to load.")
        return None

    # Continue from the highest existing student label.
    existing_students = Student.objects.all()

    if existing_students.exists():
        label_id = (
            existing_students
            .order_by("-label_id")
            .first()
            .label_id + 1
        )
    else:
        label_id = 0

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    # Process each student's dataset folder.
    for person_name in sorted(os.listdir(DATASET_PATH)):

        person_path = os.path.join(
            DATASET_PATH,
            person_name
        )

        if (
            not os.path.isdir(person_path)
            or not person_name.strip()
        ):
            continue

        student_name = person_name.strip()

        person_faces = []

        # Process supported image files only.
        for img_name in sorted(os.listdir(person_path)):

            if not img_name.lower().endswith(
                valid_extensions
            ):
                continue

            img_path = os.path.join(
                person_path,
                img_name
            )

            processed_face = preprocess_image(img_path)

            if processed_face is None:
                continue

            person_faces.append(processed_face)

        # Do not train on students without valid face images.
        if not person_faces:
            print(
                f"WARNING: No valid face images for "
                f"{student_name}. Skipping."
            )
            continue

        # Reuse the existing database label if available.
        student_obj = Student.objects.filter(
            name=student_name
        ).first()

        if student_obj:
            student_label = student_obj.label_id
        else:
            student_label = label_id

            Student.objects.create(
                name=student_name,
                label_id=student_label
            )

            label_id += 1

        faces.extend(person_faces)

        labels.extend(
            [student_label] * len(person_faces)
        )

        print(
            f"Training data: {student_name} | "
            f"Label: {student_label} | "
            f"Valid images: {len(person_faces)}"
        )

    if not faces:
        print(
            "ERROR: No valid face images available "
            "for training."
        )
        return None

    # Keep the existing LBPH recognition algorithm.
    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(
        faces,
        np.array(labels, dtype=np.int32)
    )

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    model.save(MODEL_PATH)

    print(
        f"Model trained successfully using "
        f"{len(faces)} valid face images."
    )

    return model


def recognize_face(image_path):
    """
    Predicts a student label and returns the LBPH
    confidence/distance score.

    Returns (None, None) when recognition cannot run.
    The attendance view retains responsibility for
    applying the confidence threshold.
    """

    if not os.path.isfile(MODEL_PATH):
        print(
            f"ERROR: Trained model not found: {MODEL_PATH}"
        )
        return None, None

    processed_face = preprocess_image(image_path)

    if processed_face is None:
        return None, None

    try:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(MODEL_PATH)

        label, confidence = model.predict(processed_face)

        print(
            f"LBPH prediction: label={label}, "
            f"distance={confidence:.2f}"
        )

        return label, confidence

    except (cv2.error, OSError) as error:
        print(f"ERROR during face recognition: {error}")
        return None, None
