from django.db import models

# Create your models here.
   
class User(models.Model):
    id = models.AutoField(primary_key=True)  # Khóa chính
    predict_score=models.FloatField(default=0)

class PreLab(models.Model):
    id = models.AutoField(primary_key=True)  # Khóa chính
    user = models.ForeignKey(User, related_name='prelabs', on_delete=models.CASCADE)
    name_object = models.CharField(max_length=7)
    max_score = models.FloatField(default=0)
    min_score = models.FloatField(default=0)
    attempts = models.PositiveIntegerField()
    number_of_question = models.PositiveIntegerField()

    def __str__(self):
        return self.name_object
    
class InLab(models.Model):
    id = models.AutoField(primary_key=True)  # Khóa chính
    user = models.ForeignKey(User, related_name='inlabs', on_delete=models.CASCADE)
    name_object = models.CharField(max_length=7)
    max_score = models.FloatField(default=0)
    min_score = models.FloatField(default=0)
    attempts = models.PositiveIntegerField()
    number_of_question = models.PositiveIntegerField()

    def __str__(self):
        return self.name_object
class QuestionData(models.Model):
    id = models.AutoField(primary_key=True)  # Khóa chính
    mssv= models.CharField(max_length=8)
    question=models.CharField()
    numberOfQuestion=models.PositiveIntegerField(default=1)
    scorePredict=models.FloatField(default=0)
    
class FinalData(models.Model):
    studentID =models.CharField(primary_key=True,max_length=7)
    prelab1=models.FloatField(default=0)
    inlab1=models.FloatField(default=0)
    prelab2=models.FloatField(default=0)
    inlab2=models.FloatField(default=0)
    prelab3=models.FloatField(default=0)
    inlab3=models.FloatField(default=0)
    prelab4=models.FloatField(default=0)
    inlab4=models.FloatField(default=0)
    
    def __str__(self):
        return self.studentID
    
class predictFinalUser(models.Model):
    id = models.AutoField(primary_key=True)  # Khóa chính
    prelab1=models.FloatField(default=0)
    inlab1=models.FloatField(default=0)
    prelab2=models.FloatField(default=0)
    inlab2=models.FloatField(default=0)
    prelab3=models.FloatField(default=0)
    inlab3=models.FloatField(default=0)
    prelab4=models.FloatField(default=0)
    inlab4=models.FloatField(default=0)
    scorePredict=models.FloatField(default=0)