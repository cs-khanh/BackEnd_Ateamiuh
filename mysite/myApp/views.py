from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import User,PreLab,InLab,FinalData
from .serializers import UserSerializer,InLabSerializer,PrelabSerializer
import pandas as pd
import numpy as np
import joblib
import csv
from django.http import HttpResponse
# Create your views here.
def import_csv(request):
    try:
        csv_path="D:/HK1_2024-2025/AI_HCM/BaiToan3/data_final.csv"
        
        with open(csv_path,newline='',encoding='utf-8') as csvfile:
            reader=csv.reader(csvfile,delimiter=',')
            
            next(reader)
            
            for row in reader:
                if not FinalData.objects.filter(studentID=str(row[0])).exists():
                    FinalData.objects.create(
                        studentID=str(row[0]),
                        prelab1=float(row[1]),
                        inlab1=float(row[2]),
                        prelab2=float(row[3]),
                        inlab2=float(row[4]),
                        prelab3=float(row[5]),
                        inlab3=float(row[6]),
                        prelab4=float(row[7]),
                        inlab4=float(row[8])
                    )
        return HttpResponse('<h4>CSV file imported successfully!</h4>')
    except FileNotFoundError: #Không tìm thấy file CSV
        return HttpResponse('<h4>Error: CSV file not found!</h4>')
    except csv.Error as e: #Lỗi khi đọc file CSV
        return HttpResponse(f'<h4>Error reading CSV file: {str(e)}</h4>')
    except Exception as e:
        return HttpResponse(f'<h4>CSV file import error: {str(e)}</h4>')

def predictInlab1(data):
    #df = pd.DataFrame([data], columns=['Prelab','Prelab-growths', 'Prelab-attempts','Prelab-questions'])
    load_model=joblib.load('D:/HK1_2024-2025/AI_HCM/ModelsBaiToan1/Model1/best_rf_model1_with_scaler.pkl')
    load_scaler=joblib.load('D:/HK1_2024-2025/AI_HCM/ModelsBaiToan1/Model1/scaler1.pkl')
    df=np.array(data).reshape(1,-1)
    data_scaler=load_scaler.transform(df)
    predictions=load_model.predict(data_scaler)
    return predictions
def predictInlab2(data):
    load_model=joblib.load('D:/HK1_2024-2025/AI_HCM/ModelsBaiToan1/Model2/best_rf_model2_with_scaler.pkl')
    load_scaler=joblib.load('D:/HK1_2024-2025/AI_HCM/ModelsBaiToan1/Model2/scaler2.pkl')
    df=np.array(data).reshape(1,-1)
    data_scaler=load_scaler.transform(df)
    predictions=load_model.predict(data_scaler)
    return predictions
def predictInlab3(data):
    load_model=joblib.load('D:/HK1_2024-2025/AI_HCM/ModelsBaiToan1/Model3/best_rf_model3_with_scaler.pkl')
    load_scaler=joblib.load('D:/HK1_2024-2025/AI_HCM/ModelsBaiToan1/Model3/scaler3.pkl')
    df=np.array(data).reshape(1,-1)
    data_scaler=load_scaler.transform(df)
    predictions=load_model.predict(data_scaler)
    return predictions
def predictInlab4(data):
    load_model=joblib.load('D:/HK1_2024-2025/AI_HCM/ModelsBaiToan1/Model4/best_rf_model4_with_scaler.pkl')
    load_scaler=joblib.load('D:/HK1_2024-2025/AI_HCM/ModelsBaiToan1/Model4/scaler4.pkl')
    df=np.array(data).reshape(1,-1)
    data_scaler=load_scaler.transform(df)
    predictions=load_model.predict(data_scaler)
    return predictions
def predictLabFinal(mssv):
    try: 
        student_data=FinalData.objects.get(studentID=mssv)
        arr=[   student_data.prelab1,
                student_data.inlab1,
                student_data.prelab2,
                student_data.inlab2,
                student_data.prelab3,
                student_data.inlab3,
                student_data.prelab4,
                student_data.inlab4,
            ]
        load_model=joblib.load('D:/HK1_2024-2025/AI_HCM/BaiToan3/rf_model.pkl')
        data=np.array(arr).reshape(1,-1)
        predictions=load_model.predict(data)
        return predictions
    except FinalData.DoesNotExist:
        return -1
    
class UserLabDataAPIView(APIView):
    def post(self, request):
        data_s = request.data 
        task_type=data_s.get('task_type')
        if task_type=='predictInlab':
            data=data_s.get('data')
            user = User.objects.create()  # Tạo user 
            # Lặp qua từng object trong danh sách data
            for entry in data:
                name_object = entry.get('nameObject', '')

                # Kiểm tra nếu là Prelab
                if 'Prelab' in name_object:
                    PreLab.objects.update_or_create(
                        user=user,
                        name_object=name_object,
                        defaults={
                            'max_score': entry['maxScore'],
                            'min_score': entry['minScore'],
                            'attempts': entry['attempts'],
                            'number_of_question': entry['numberOfQuestion']
                        }
                    )
                # Kiểm tra nếu là Inlab
                elif 'Inlab' in name_object:
                    InLab.objects.update_or_create(
                        user=user,
                        name_object=name_object,
                        defaults={
                            'max_score': entry['maxScore'],
                            'min_score': entry['minScore'],
                            'attempts': entry['attempts'],
                            'number_of_question': entry['numberOfQuestion']
                        }
                    )
            data_arr=[]
            for item in data:
                growths=float(item["maxScore"])-float(item["minScore"])
                row = [
                #item["nameObject"],
                item["maxScore"],
                item["attempts"],
                item["numberOfQuestion"],
                growths
                ]
                data_arr.append(row)
            flat_array = [element for row in data_arr for element in row]
            diem_pre=-1
            lenght=len(data)
            if(lenght==1):
                diem_pre=predictInlab1(flat_array)
            if(lenght==3):
                diem_pre=predictInlab2(flat_array)
            
            if(lenght==5):
                diem_pre=predictInlab3(flat_array)
            if(lenght==7):
                diem_pre=predictInlab4(flat_array)
            user.predict_score=diem_pre
            user.save()
            return Response(diem_pre, status=status.HTTP_200_OK)
        if task_type=='predictFinal':
            mssv=str(data_s.get('data')[0])
            diem_pre=predictLabFinal(mssv)
            return Response(diem_pre, status=status.HTTP_200_OK)
               