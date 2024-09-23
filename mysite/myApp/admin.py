from django.contrib import admin
from .models import User,PreLab,InLab,QuestionData,FinalData,predictFinalUser
# Register your models here.
admin.site.register(User)
admin.site.register(PreLab)
admin.site.register(InLab)
admin.site.register(QuestionData)
admin.site.register(FinalData)
admin.site.register(predictFinalUser)