import os
import json
import numpy as np
from django.shortcuts import render, redirect
from django.conf import settings
from .models import Student
from .forms import StudentForm
import cv2
from django.http import JsonResponse

# Paths
RECOGNIZER_PATH = os.path.join(settings.MEDIA_ROOT, 'recognizer.yml')
LABELS_PATH = os.path.join(settings.MEDIA_ROOT, 'label_map.json')

LBPH_CONFIDENCE_THRESHOLD = 70.0  # Tune this

# ----------------- Helpers -----------------
def read_image_from_path_or_file(path_or_file):
    if hasattr(path_or_file, 'read'):
        file_bytes = np.frombuffer(path_or_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        try: path_or_file.seek(0)
        except: pass
        return img
    return cv2.imread(path_or_file)

def detect_face_cascade(image):
    if image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(40,40))
    if len(faces)==0: return None
    x,y,w,h = faces[0]; face = gray[y:y+h, x:x+w]
    return cv2.resize(face, (160,160))

def train_recognizer():
    students = Student.objects.exclude(photo='').exclude(photo__isnull=True)
    faces, labels = [], []
    for s in students:
        img = read_image_from_path_or_file(s.photo.path)
        face = detect_face_cascade(img)
        if face is not None:
            faces.append(face)
            labels.append(int(s.id))
    if not faces:
        if os.path.exists(RECOGNIZER_PATH): os.remove(RECOGNIZER_PATH)
        if os.path.exists(LABELS_PATH): os.remove(LABELS_PATH)
        return False
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    recognizer.write(RECOGNIZER_PATH)
    label_map = {str(int(l)): int(l) for l in labels}
    with open(LABELS_PATH, 'w') as f: json.dump(label_map, f)
    return True

def predict_face_with_recognizer(face):
    if face is None or not os.path.exists(RECOGNIZER_PATH): return None, None
    recognizer = cv2.face.LBPHFaceRecognizer_create(); recognizer.read(RECOGNIZER_PATH)
    try: label, confidence = recognizer.predict(face)
    except: return None, None
    if confidence <= LBPH_CONFIDENCE_THRESHOLD:
        try: return Student.objects.get(id=int(label)), float(confidence)
        except Student.DoesNotExist: return None, None
    return None, float(confidence)

# ----------------- Views -----------------
def students_list(request):
    return render(request, 'matches/students_list.html', {'students': Student.objects.all()})

def submit_student(request):
    if request.method == "POST":
        form = StudentForm(request.POST, request.FILES)
        if form.is_valid():
            photo = form.cleaned_data.get('photo')
            img = read_image_from_path_or_file(photo)
            face = detect_face_cascade(img)
            if face is None:
                return render(request, 'matches/forms.html', {'match_result': {'status':'not_matched','message':'No face detected'}, 'form': form})

            matched_student, confidence = predict_face_with_recognizer(face)
            if matched_student:
    # Update all fields except 'photo'
                for field in ['first_name','last_name','middle_initial','id_number','year_section','e_signature']:
                    setattr(matched_student, field, form.cleaned_data.get(field))
                
                # Save form photo separately for left-side comparison
                matched_student.form_photo = form.cleaned_data.get('photo')
                
                matched_student.status = 'completed'
                matched_student.save()
                match_result = {'status':'matched','student':matched_student,'distance':confidence}

            else:
                match_result = {'status':'not_matched','message':'No match found','confidence':confidence}

            return render(request, 'matches/forms.html', {'match_result': match_result, 'form': form})

    else:
        form = StudentForm()
    return render(request, 'matches/forms.html', {'form': form})

def upload_photos(request):
    if request.method=='POST' and request.FILES.getlist('photos'):
        for f in request.FILES.getlist('photos'):
            Student.objects.create(photo=f, status='no')
        train_recognizer()
    return redirect('students_list')

def delete_students(request):
    if request.method=='POST':
        ids = request.POST.getlist('selected_students')
        Student.objects.filter(id__in=ids).delete()
        train_recognizer()
    return redirect('students_list')

def match_detail(request, student_id):
    try: student = Student.objects.get(id=student_id)
    except Student.DoesNotExist: return JsonResponse({"error": "Student not found"})
    form_photo_url = student.form_photo.url if student.form_photo else ""
    clicked_photo_url = student.photo.url if student.photo else ""
    accuracy = "N/A"
    if student.photo and os.path.exists(RECOGNIZER_PATH):
        face = detect_face_cascade(cv2.imread(student.photo.path))
        _, confidence = predict_face_with_recognizer(face)
        if confidence is not None:
            accuracy = f"{max(0, min(100, round(100-confidence,2)))}%"
    return JsonResponse({
        "form_photo": form_photo_url,
        "clicked_photo": clicked_photo_url,
        "name": f"{student.first_name} {student.last_name}",
        "section": student.year_section,
        "status": "Match Successfully ✅",
        "accuracy": accuracy
    })
