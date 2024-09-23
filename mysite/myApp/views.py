from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import User,PreLab,InLab,QuestionData,FinalData,predictFinalUser
from .serializers import UserSerializer,InLabSerializer,PrelabSerializer
import pandas as pd
import numpy as np
import joblib
import csv
from django.http import HttpResponse
from BaiToan2.processing import text_clean,text_sementic,addFeature,Words2Text,Text2Words
path_root="C:/Users/admin/Desktop/WebsiteAteamIUH/BackEnd_Ateamiuh/"
# Create your views here.
def import_csv(request):
    try:
        csv_path=path_root+"BaiToan3/data_final.csv"
        
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
    load_model=joblib.load(path_root+'ModelsBaiToan1/Model1/best_rf_model1_with_scaler.pkl')
    load_scaler=joblib.load(path_root+'ModelsBaiToan1/Model1/scaler1.pkl')
    df=np.array(data).reshape(1,-1)
    data_scaler=load_scaler.transform(df)
    predictions=load_model.predict(data_scaler)
    return predictions


def predictInlab2(data):
    load_model=joblib.load(path_root+'ModelsBaiToan1/Model2/best_rf_model2_with_scaler.pkl')
    load_scaler=joblib.load(path_root+'ModelsBaiToan1/Model2/scaler2.pkl')
    df=np.array(data).reshape(1,-1)
    data_scaler=load_scaler.transform(df)
    predictions=load_model.predict(data_scaler)
    return predictions


def predictInlab3(data):
    load_model=joblib.load(path_root+'ModelsBaiToan1/Model3/best_rf_model3_with_scaler.pkl')
    load_scaler=joblib.load(path_root+'ModelsBaiToan1/Model3/scaler3.pkl')
    df=np.array(data).reshape(1,-1)
    data_scaler=load_scaler.transform(df)
    predictions=load_model.predict(data_scaler)
    return predictions


def predictInlab4(data):
    load_model=joblib.load(path_root+'ModelsBaiToan1/Model4/best_rf_model4_with_scaler.pkl')
    load_scaler=joblib.load(path_root+'ModelsBaiToan1/Model4/scaler4.pkl')
    df=np.array(data).reshape(1,-1)
    data_scaler=load_scaler.transform(df)
    predictions=load_model.predict(data_scaler)
    return predictions


def predictLabFinal(arrScore):
    arr=[]
    for point in arrScore:
        arr.append(point)
    load_model=joblib.load(path_root+'BaiToan3/rf_model.pkl')
    data=np.array(arr).reshape(1,-1)
    predictions=load_model.predict(data)
    try:
        predictFinalUser.objects.create(
            prelab1=float(arr[0]),
            inlab1=float(arr[1]),
            prelab2=float(arr[2]),
            inlab2=float(arr[3]),
            prelab3=float(arr[4]),
            inlab3=float(arr[5]),
            prelab4=float(arr[6]),
            inlab4=float(arr[7]),
            scorePredict=float(predictions)
        )
        return predictions
    except Exception as e:
        return -1
    
def meanLab(lab):
    full=FinalData.objects.values_list(lab,flat=True)
    full=np.array(full)
    meanfull=np.sum(full)/len(full)
    meanfull=round(meanfull,2)
    return meanfull
    

def analysisFinal():
    meanp1=meanLab('prelab1')
    meani1=meanLab('inlab1')
    meanp2=meanLab('prelab2')
    meani2=meanLab('inlab3')
    meanp3=meanLab('prelab3')
    meani3=meanLab('inlab3')
    meanp4=meanLab('prelab4')
    meani4=meanLab('inlab4')
    analysisData={'dataAll':[meanp1,meani1,meanp2,meani2,meanp3,meani3,meanp4,meani4,]}
    return analysisData
    
def create_or_update_lab(lab_model, user, entry):
    lab_model.objects.update_or_create(
        user=user,
        name_object=entry['nameObject'],
        defaults={
            'max_score': entry['maxScore'],
            'min_score': entry['minScore'],
            'attempts': entry['attempts'],
            'number_of_question': entry['numberOfQuestion']
        }
    )
    
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
                    create_or_update_lab(PreLab, user, entry)
                # Kiểm tra nếu là Inlab
                elif 'Inlab' in name_object:
                    create_or_update_lab(InLab, user, entry)
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
            arrScore=data_s.get('data')
            diem_pre=predictLabFinal(arrScore)
            return Response(diem_pre, status=status.HTTP_200_OK)
        if task_type=='analysisFinal':
            return  Response(analysisFinal(), status=status.HTTP_200_OK)
        if task_type=='predictQuestionScore':
            mssv=int(data_s.get('data')[0])
            question=str(data_s.get('data')[1])
            numberoftimes=int(data_s.get('data')[2])
            studenID=mssv
            questionSave=question
            text=text_clean(question)
            process=text_sementic(text,'T',False)
            tfidf_loaded = joblib.load(path_root+'mysite/BaiToan2/tfidf.pkl')
            df_test=addFeature(tfidf_loaded,process,mssv,numberoftimes)
            ch2_loaded = joblib.load(path_root+'mysite/BaiToan2/selectkbest_model_chi2.pkl')
            feature_names = df_test.columns
            feature_names_chi = [feature_names[i] for i in ch2_loaded.get_support(indices=True)] 
            feature_names_chi = feature_names_chi[1:(len(feature_names_chi)-1)]
            feature_names_chi = Words2Text(feature_names_chi)
            feature_names_chi = text_sementic(feature_names_chi, 'T', False)
            feature_names_chi = Text2Words(feature_names_chi)
            tfidf_chi2 = joblib.load(path_root+'mysite/BaiToan2/tfidf1.pkl')
            df=addFeature(tfidf_chi2,process,mssv,numberoftimes)
            df = df.drop(df.columns[22], axis=1)
            df['effort']=1/numberoftimes
            model_loaded=joblib.load(path_root+"mysite/BaiToan2/model2.pkl")
            prediction=model_loaded.predict(df)
            QuestionData.objects.create(
                mssv=studenID,
                question=questionSave,
                numberOfQuestion=numberoftimes,
                scorePredict=prediction
            )
            return Response(prediction,status=status.HTTP_200_OK)
            
               