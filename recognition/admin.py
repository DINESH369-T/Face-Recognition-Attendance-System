from django.contrib import admin
from .models import Student, Attendance, LeaveRequest

admin.site.register(Student)
admin.site.register(Attendance)
admin.site.register(LeaveRequest)