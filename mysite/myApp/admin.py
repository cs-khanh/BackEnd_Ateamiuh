from django.contrib import admin
from .models import User,PreLab,InLab
# Register your models here.
admin.site.register(User)
admin.site.register(PreLab)
admin.site.register(InLab)