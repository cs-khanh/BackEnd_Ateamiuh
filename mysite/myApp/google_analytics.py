from google.oauth2 import service_account
from googleapiclient.discovery import build

# Đường dẫn tới file JSON của service account
SERVICE_ACCOUNT_FILE = 'C:/Users/Administrator/Desktop/UIT_HCM/client_secret.json'

# Các scope mà bạn cần truy cập (ở đây là Analytics)
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']

# ID của View trong Google Analytics mà bạn muốn truy vấn
VIEW_ID = '460488076'

def initialize_analyticsreporting():
    # Xác thực bằng cách sử dụng file JSON của service account
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    
    # Xây dựng đối tượng để gọi API
    analytics = build('analyticsreporting', 'v4', credentials=credentials)
    
    return analytics

def get_analytics_data():
    analytics = initialize_analyticsreporting()
    
    # Gọi API để lấy dữ liệu từ Google Analytics
    response = analytics.reports().batchGet(
        body={
            'reportRequests': [
                {
                    'viewId': VIEW_ID,
                    'dateRanges': [{'startDate': '7daysAgo', 'endDate': 'today'}],
                    'metrics': [{'expression': 'ga:sessions'}],
                    'dimensions': [{'name': 'ga:country'}]
                }]
        }
    ).execute()
    
    return response
