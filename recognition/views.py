import base64
import os
import cv2
import numpy as np
from collections import Counter
from django.shortcuts import render, redirect, get_object_or_404
from datetime import timedelta, datetime, time
from django.utils import timezone
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from .opencv_face import train_model, recognize_face
from .models import Student, Attendance, LeaveRequest
import csv
import calendar
from django.views.decorators.http import require_POST

def save_base64_image(data, path):
    """Decode a base64 data URL and save it to disk."""
    if not data:
        raise ValueError("No image data received")

    # Accept both a full data URL and the base64 payload itself.
    if ";base64," in data:
        _, encoded_image = data.split(";base64,", 1)
    else:
        encoded_image = data

    import binascii
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image data") from exc

    if not image_bytes:
        raise ValueError("Decoded image is empty")

    with open(path, "wb") as image_file:
        image_file.write(image_bytes)


def save_debug_stage(image_np, stage_name):
    """Save intermediate OpenCV stages to media/temp/ for project demonstration."""
    if image_np is None or image_np.size == 0:
        return

    temp_dir = settings.BASE_DIR / "media" / "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = temp_dir / f"{stage_name}.jpg"
    if not cv2.imwrite(str(file_path), image_np):
        print(f"WARNING: Could not save debug image: {file_path}")


def adjust_gamma(image, gamma=1.2):
    """Apply gamma correction before CLAHE normalization."""
    if image is None or image.size == 0:
        return image
    if gamma <= 0:
        raise ValueError("gamma must be greater than zero")

    inverse_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inverse_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def compute_wld(gray_img):
    """
    Debug helper for WLD visualization.
    The recognition model uses compute_wld_descriptor() in opencv_face.py;
    this helper remains available for compatibility with the debug pipeline.
    """
    if gray_img is None or gray_img.size == 0:
        return None

    from .opencv_face import compute_wld_descriptor

    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    return compute_wld_descriptor(gray_img)


def register(request):
    if request.method != "POST":
        return JsonResponse({"msg": "POST request required"}, status=405)

    name = (request.POST.get("student_name") or "").strip()
    image = request.POST.get("image")

    if not name:
        return JsonResponse({"msg": "Student name missing"}, status=400)
    if not image:
        return JsonResponse({"msg": "No image data received"}, status=400)

    # Prevent a name from being used to create folders outside media/students.
    if name in {".", ".."} or "/" in name or "\\\\" in name:
        return JsonResponse({"msg": "Invalid student name"}, status=400)

    folder = settings.BASE_DIR / "media" / "students" / name
    os.makedirs(folder, exist_ok=True)
    img_path = folder / f"{datetime.now().timestamp()}.jpg"

    try:
        save_base64_image(image, img_path)
    except ValueError as exc:
        return JsonResponse({"msg": str(exc)}, status=400)

    # Preserve the existing registration behavior: save the sample as grayscale.
    img = cv2.imread(str(img_path))
    if img is None:
        img_path.unlink(missing_ok=True)
        return JsonResponse({"msg": "Could not read captured image"}, status=400)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not cv2.imwrite(str(img_path), gray):
        return JsonResponse({"msg": "Could not save captured image"}, status=500)

    return JsonResponse({"msg": "Image saved. Capture samples then train."})


def train(request):
    if request.method != "POST":
        return JsonResponse({"msg": "POST request required"}, status=405)

    model = train_model()
    if model is None:
        return JsonResponse({"msg": "No training data found"}, status=400)
    return JsonResponse({"msg": "Model trained successfully"})

def create_face_debug_image(image):
    """
    Draw face, eye, and approximate mouth-region boxes
    for visual debugging. Does not modify recognition logic.
    """
    if image is None or image.size == 0:
        return None

    debug_image = image.copy()
    gray = cv2.cvtColor(debug_image, cv2.COLOR_BGR2GRAY)

    base_dir = str(settings.BASE_DIR)

    face_cascade = cv2.CascadeClassifier(
        os.path.join(base_dir, "haarcascade_frontalface_default.xml")
    )
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(base_dir, "haarcascade_eye.xml")
    )
    mouth_cascade = cv2.CascadeClassifier(
        os.path.join(base_dir, "haarcascade_smile.xml")
    )

    if face_cascade.empty():
        print("WARNING: Face cascade could not be loaded.")
        return debug_image

    # Try both original grayscale and enhanced grayscale images.
    gamma_img = adjust_gamma(gray, gamma=1.2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gamma_img)

    faces = face_cascade.detectMultiScale(
        normalized,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

    for (x, y, w, h) in faces:

        # Green: whole face.
        cv2.rectangle(
            debug_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        face_gray = normalized[y:y + h, x:x + w]

        # Red: detected eyes.
        if not eye_cascade.empty():
            eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(10, 10)
            )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    debug_image,
                    (x + ex, y + ey),
                    (x + ex + ew, y + ey + eh),
                    (0, 0, 255),
                    2
                )

        # Blue: approximate mouth region.
        if not mouth_cascade.empty():
            lower_start = int(h * 0.45)
            lower_face = face_gray[lower_start:, :]

            mouths = mouth_cascade.detectMultiScale(
                lower_face,
                scaleFactor=1.5,
                minNeighbors=15,
                minSize=(15, 10)
            )

            if len(mouths) > 0:
                mx, my, mw, mh = max(
                    mouths,
                    key=lambda item: item[2] * item[3]
                )

                cv2.rectangle(
                    debug_image,
                    (x + mx, y + lower_start + my),
                    (x + mx + mw, y + lower_start + my + mh),
                    (255, 0, 0),
                    2
                )

    return debug_image
    
def recognize(request):
    if request.method != "POST":
        return JsonResponse(
            {"msg": "POST request required"},
            status=405
        )

    # Collect burst frames from the frontend.
    image_keys = ["image1", "image2", "image3"]
    burst_images = [
        request.POST.get(key)
        for key in image_keys
        if request.POST.get(key)
    ]

    # Support the original single-image request.
    if not burst_images:
        single_img = request.POST.get("image")

        if single_img:
            burst_images = [single_img]
        else:
            return JsonResponse(
                {"msg": "No image data received"},
                status=400
            )

    predicted_labels = []

    for idx, img_data in enumerate(burst_images):
        temp_path = (
            settings.BASE_DIR / "media" / f"temp_{idx}.jpg"
        )

        try:
            save_base64_image(img_data, temp_path)
        except ValueError as exc:
            print(
                f"WARNING: Could not save recognition frame "
                f"{idx}: {exc}"
            )
            continue

        # 1. Load the captured image.
        img = cv2.imread(str(temp_path))

        if img is None or img.size == 0:
            print(f"WARNING: Could not read image: {temp_path}")
            continue

        if idx == 0:
            save_debug_stage(img, "1_raw_capture")

            # Save the face, eye, and mouth visualization.
            face_debug = create_face_debug_image(img)

            if face_debug is not None:
                save_debug_stage(
                    face_debug,
                    "6_face_detection_boxes"
                )

        # 2. Convert to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if idx == 0:
            save_debug_stage(gray, "2_grayscale")

        # 3. Gamma correction.
        gamma_img = adjust_gamma(gray, gamma=1.2)

        if idx == 0:
            save_debug_stage(gamma_img, "2_gamma_corrected")

        # 4. CLAHE normalization.
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        normalized = clahe.apply(gamma_img)

        if idx == 0:
            save_debug_stage(
                normalized,
                "3_clahe_normalized"
            )

        # 5. Detect a face for debug crop generation.
        # Try enhanced grayscale first, then original grayscale.
        face_cascade = cv2.CascadeClassifier(
            str(
                settings.BASE_DIR
                / "haarcascade_frontalface_default.xml"
            )
        )

        face_box = None

        if not face_cascade.empty():
            for detection_img in [normalized, gray]:
                faces = face_cascade.detectMultiScale(
                    detection_img,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(faces) > 0:
                    face_box = max(
                        faces,
                        key=lambda box: box[2] * box[3]
                    )
                    break
        else:
            print("WARNING: Face cascade could not be loaded.")

        # 6. Save detected face crop and WLD debug image.
        if face_box is not None:
            x, y, w, h = face_box

            face_crop = normalized[y:y + h, x:x + w]

            if face_crop.size > 0:
                face_resized_debug = cv2.resize(
                    face_crop,
                    (200, 200),
                    interpolation=cv2.INTER_AREA
                )

                if idx == 0:
                    save_debug_stage(
                        face_resized_debug,
                        "4_detected_face_crop"
                    )

                face_for_model = cv2.resize(
                    face_crop,
                    (100, 100),
                    interpolation=cv2.INTER_AREA
                )

                wld_img = compute_wld(face_for_model)

                if idx == 0:
                    save_debug_stage(
                        wld_img,
                        "5_wld_processed"
                    )
        else:
            print(f"WARNING: No face detected in: {temp_path}")

            if idx == 0:
                save_debug_stage(
                    compute_wld(normalized),
                    "5_wld_processed"
                )

        # 7. Run actual recognition using opencv_face.py.
        try:
            label, conf = recognize_face(str(temp_path))
        except Exception as exc:
            print(
                f"WARNING: Recognition failed for "
                f"{temp_path}: {exc}"
            )
            continue

        if label is not None and conf is not None:
            print(
                f"LBPH prediction: label={label}, "
                f"distance={conf:.2f}"
            )

            # Preserve the existing confidence threshold.
            if conf < 85:
                predicted_labels.append(label)

    # 8. Burst voting.
    if not predicted_labels:
        return JsonResponse({
            "msg": "Unknown Face (Burst Confidence Threshold Exceeded)"
        })

    winning_label, count = Counter(
        predicted_labels
    ).most_common(1)[0]

    try:
        student = Student.objects.get(
            label_id=winning_label
        )
    except Student.DoesNotExist:
        return JsonResponse({"msg": "Unknown Face"})

    # 9. Existing attendance logic.
    today = timezone.localtime().date()

    attendance = Attendance.objects.filter(
        student=student,
        date=today
    ).first()

    now = timezone.localtime()
    late_time = time(12, 0)

    # First scan: check-in.
    if not attendance:
        check_in_status = timezone.now()

        status_msg = (
            f"{student.name} Checked In (Late)"
            if now.time() > late_time
            else f"{student.name} Checked In"
        )

        Attendance.objects.create(
            student=student,
            check_in=check_in_status,
            check_out=None
        )

        return JsonResponse({"msg": status_msg})

    # Second scan: check-out.
    elif attendance and not attendance.check_out:
        has_approved_od = LeaveRequest.objects.filter(
            student=student,
            date=today,
            leave_type="od",
            status="approved"
        ).exists()

        if has_approved_od:
            attendance.check_out = timezone.now()
            attendance.save()

            return JsonResponse({
                "msg": (
                    f"{student.name} Checked OUT "
                    "(Approved OD Verified)"
                )
            })

        else:
            time_difference = now - attendance.check_in
            cooldown_period = timedelta(minutes=1)

            if time_difference < cooldown_period:
                remaining_secs = int(
                    (
                        cooldown_period - time_difference
                    ).total_seconds()
                )

                return JsonResponse({
                    "msg": (
                        f"Too early to checkout! "
                        f"Wait {remaining_secs}s"
                    )
                })

            attendance.check_out = timezone.now()
            attendance.save()

            return JsonResponse({
                "msg": f"{student.name} Checked OUT"
            })

    # Attendance already completed.
    else:
        return JsonResponse({
            "msg": (
                f"{student.name}: Attendance already "
                "completed for today"
            )
        })


def attendance_status(request):
    today = timezone.localtime().date()
    attendances = Attendance.objects.filter(date=today)
    total_students = Student.objects.count()
    present_count = attendances.count()
    absent_count = max(0, total_students - present_count)
    
    late_time = time(12, 0)
    data = []
    for a in attendances:
        check_in_time = timezone.localtime(a.check_in) if a.check_in else None
        check_out_time = timezone.localtime(a.check_out) if a.check_out else None

        if check_out_time:
            status = "Checked Out"
        elif check_in_time:
            status = "Checked In (Late)" if check_in_time.time() > late_time else "Checked In"
        else:
            status = "-"
            
        data.append({
            "student_name": a.student.name,
            "check_in": check_in_time.strftime("%H:%M:%S") if a.check_in else None,
            "check_out": check_out_time.strftime("%H:%M:%S") if a.check_out else None,
            "status": status
        })
    
    return JsonResponse({
        "attendance": data,
        "stats": {
            "total": total_students,
            "present": present_count,
            "absent": absent_count
        }
    })

def export_attendance_csv(request):
    today = timezone.localtime().date()
    attendances = Attendance.objects.filter(date=today)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="attendance_{today}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Student Name', 'Check-In', 'Check-Out', 'Status'])
    
    late_time = time(12, 0)
    for a in attendances:
        check_in_time = timezone.localtime(a.check_in).strftime("%H:%M:%S") if a.check_in else "-"
        check_out_time = timezone.localtime(a.check_out).strftime("%H:%M:%S") if a.check_out else "-"
        
        if a.check_out:
            status = "Checked Out"
        elif a.check_in and timezone.localtime(a.check_in).time() > late_time:
            status = "Checked In (Late)"
        elif a.check_in:
            status = "Checked In"
        else:
            status = "-"
            
        writer.writerow([a.student.name, check_in_time, check_out_time, status])
        
    return response

def admin_dashboard(request):
    today = timezone.localtime().date()
    total_students = Student.objects.count()
    attendances = Attendance.objects.filter(date=today)
    present_count = attendances.count()
    absent_count = max(0, total_students - present_count)
    
    late_time = time(12, 0)
    late_count = sum(1 for a in attendances if a.check_in and timezone.localtime(a.check_in).time() > late_time)
    pending_leaves = LeaveRequest.objects.filter(status='pending')
    
    context = {
        'total_students': total_students,
        'present_count': present_count,
        'absent_count': absent_count,
        'late_count': late_count,
        'pending_leaves': pending_leaves,
        'students': Student.objects.all(), # Added this so the button works smoothly
        'today': today,
    }
    return render(request, "admin_dashboard.html", context)

def request_leave(request):
    if request.method == "POST":
        student_id = request.POST.get("student_id")
        leave_type = request.POST.get("leave_type")
        date = request.POST.get("date")
        reason = request.POST.get("reason")
        
        student = get_object_or_404(Student, id=student_id)
        LeaveRequest.objects.create(
            student=student,
            leave_type=leave_type,
            date=date,
            reason=reason
        )
        return redirect('camera_view')
    
    students = Student.objects.all()
    return render(request, "request_leave.html", {"students": students})

@require_POST
def handle_leave(request, leave_id, action):
    leave = get_object_or_404(LeaveRequest, id=leave_id)
    if action == 'approve':
        leave.status = 'approved'
    elif action == 'reject':
        leave.status = 'rejected'
    else:
        return redirect('admin_dashboard')
    leave.save()
    return redirect('admin_dashboard')

def student_calendar(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    year = int(request.GET.get('year', timezone.now().year))
    month = int(request.GET.get('month', timezone.now().month))
    
    # Get attendance & leaves for this month
    cal = calendar.monthcalendar(year, month)
    attendances = {a.date.day: a for a in Attendance.objects.filter(student=student, date__year=year, date__month=month)}
    leaves = {l.date.day: l for l in LeaveRequest.objects.filter(student=student, date__year=year, date__month=month, status='approved')}
    
    late_time = time(12, 0)
    calendar_data = []
    for week in cal:
        week_data = []
        for day in week:
            if day == 0:
                week_data.append({'day': '', 'status': 'empty'})
                continue
            
            status = 'absent'
            details = 'Absent'
            if day in attendances:
                att = attendances[day]
                in_t = timezone.localtime(att.check_in).strftime("%H:%M") if att.check_in else "-"
                out_t = timezone.localtime(att.check_out).strftime("%H:%M") if att.check_out else "-"
                if att.check_in and timezone.localtime(att.check_in).time() > late_time:
                    status = 'late'
                    details = f"Late | In: {in_t} | Out: {out_t}"
                else:
                    status = 'present'
                    details = f"Present | In: {in_t} | Out: {out_t}"
            elif day in leaves:
                status = 'leave'
                details = f"Approved Leave ({leaves[day].get_leave_type_display()})"
            
            week_data.append({'day': day, 'status': status, 'details': details})
        calendar_data.append(week_data)
        
    context = {
        'student': student,
        'calendar_data': calendar_data,
        'year': year,
        'month': month,
        'month_name': calendar.month_name[month],
        'students': Student.objects.all()
    }
    return render(request, "student_calendar.html", context)

def student_profile(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    year = int(request.GET.get('year', timezone.now().year))
    month = int(request.GET.get('month', timezone.now().month))
    
    # Get attendance & leaves for this specific student for this month
    cal = calendar.monthcalendar(year, month)
    attendances = {a.date.day: a for a in Attendance.objects.filter(student=student, date__year=year, date__month=month)}
    leaves = {l.date.day: l for l in LeaveRequest.objects.filter(student=student, date__year=year, date__month=month, status='approved')}
    
    # Monthly stats counters
    present_count = 0
    late_count = 0
    leave_count = len(leaves)
    absent_count = 0
    
    late_time = time(12, 0)
    calendar_data = []
    
    for week in cal:
        week_data = []
        for day in week:
            if day == 0:
                week_data.append({'day': '', 'status': 'empty'})
                continue
            
            # Check if this day is in the future
            is_future = datetime(year, month, day).date() > timezone.localtime().date()
            
            status = 'absent'
            details = 'Absent'
            
            if day in attendances:
                att = attendances[day]
                in_t = timezone.localtime(att.check_in).strftime("%H:%M") if att.check_in else "-"
                out_t = timezone.localtime(att.check_out).strftime("%H:%M") if att.check_out else "-"
                if att.check_in and timezone.localtime(att.check_in).time() > late_time:
                    status = 'late'
                    details = f"Late | In: {in_t} | Out: {out_t}"  # <-- Updated here
                    late_count += 1
                else:
                    status = 'present'
                    details = f"Present | In: {in_t} | Out: {out_t}"  # <-- Updated here
                    present_count += 1
            elif day in leaves:
                status = 'leave'
                details = f"Approved Leave"
            elif is_future:
                status = 'empty'
                details = '-'
            else:
                absent_count += 1
                
            week_data.append({'day': day, 'status': status, 'details': details})
        calendar_data.append(week_data)
        
    student_leaves = LeaveRequest.objects.filter(student=student).order_by('-date')

    context = {
        'student': student,
        'calendar_data': calendar_data,
        'year': year,
        'month': month,
        'month_name': calendar.month_name[month],
        'present_count': present_count,
        'late_count': late_count,
        'leave_count': leave_count,
        'absent_count': absent_count,
        'student_leaves': student_leaves,
    }
    return render(request, "student_profile.html", context)

def student_apply_leave(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    error = None
    
    if request.method == "POST":
        leave_type = request.POST.get("leave_type") # 'leave' or 'od'
        date_str = request.POST.get("date")
        reason = request.POST.get("reason")
        
        leave_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        
        # Check if an attendance record exists for this date
        attendance = Attendance.objects.filter(student=student, date=leave_date).first()
        
        if leave_type == 'leave':
            # Rule 1: For Leave, student must NOT be checked in
            if attendance and attendance.check_in:
                error = "Cannot apply for Leave: You have already checked in on this date!"
        elif leave_type == 'od':
            # Rule 2: For OD/Permission, student MUST have checked in, but NOT checked out
            if not attendance or not attendance.check_in:
                error = "Cannot apply for OD/Permission: You have not checked in on this date yet!"
            elif attendance.check_out:
                error = "Cannot apply for OD/Permission: You have already checked out for the day!"
                
        if error:
            return render(request, "student_apply_leave.html", {"student": student, "error": error})
        
        LeaveRequest.objects.create(
            student=student,
            leave_type=leave_type,
            date=leave_date,
            reason=reason
        )
        return redirect('student_profile', student_id=student.id)
    
    return render(request, "student_apply_leave.html", {"student": student})

def camera_view(request):
    students = Student.objects.all()
    return render(request, 'camera.html', {'students': students})

def admin_student_list(request):
    students = Student.objects.all()
    return render(request, "admin_student_list.html", {"students": students})