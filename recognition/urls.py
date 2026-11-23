from django.urls import path
from . import views

urlpatterns = [
    path('', views.camera_view, name='camera_view'),
    path('register/', views.register, name='register'),
    path('train/', views.train, name='train'),
    path('recognize/', views.recognize, name='recognize'),
    path('attendance_status/', views.attendance_status, name='attendance_status'),
    path('export-csv/', views.export_attendance_csv, name='export_csv'),
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('request-leave/', views.request_leave, name='request_leave'),
    path('leave/<int:leave_id>/<str:action>/', views.handle_leave, name='handle_leave'),
    path('calendar/<int:student_id>/', views.student_calendar, name='student_calendar'),
    path('student/<int:student_id>/', views.student_profile, name='student_profile'),
    path('student/<int:student_id>/apply-leave/', views.student_apply_leave, name='student_apply_leave'),
    path('admin-dashboard/students/', views.admin_student_list, name='admin_student_list'),
]