from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import User,PreLab,InLab
from .serializers import UserSerializer,InLabSerializer,PrelabSerializer
import pandas as pd
import numpy as np
import joblib
# Create your views here.
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
class UserLabDataAPIView(APIView):
    def post(self, request):
        data = request.data 
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
        #Trả về thông tin user sau khi cập nhật dữ liệu PreLab/InLab
        #user_data = UserSerializer(user).data
        return Response(diem_pre, status=status.HTTP_200_OK)