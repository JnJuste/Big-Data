from django.shortcuts import render
import joblib
# Create your views here.


def index(request):
    return render(request,'index.html')

def about(request):
    return render(request,'about.html')

def result(request):
    comb_code = int(request.POST.get('comb_code'))
    grade = int(request.POST.get('grade'))
    
    data_input = [comb_code,grade]

    model = joblib.load('./recomendation-system.joblib')

    prediction = model.predict([data_input])[0]

    context = {
        'comb_code':comb_code,
        'grade':grade,
        'prediction':prediction
    }
    return render(request,'index.html',context)