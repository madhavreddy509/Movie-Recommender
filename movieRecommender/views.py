from django.shortcuts import render,get_object_or_404
from model import get_recommendations
import joblib
from model import df2



def home (request):
    # print(get_recommendations('The Dark Knight Rises'))
    # data=get_recommendations('The Dark Knight Rises')
    titles=df2['title']
    titles=list(titles)
    return render(request,'index.html',{'titles':titles})

def result(request):
    movie_name=request.POST['movie']
    titles=df2['title']
    titles=list(titles)
    if movie_name not in titles:
        data={"sorry":"NO MOvies FoUnd"}
    else:
        data =get_recommendations(movie_name)
        f=list(data['title'])
        s=list(data['overview'])
        data=dict(zip(f,s))
    
    
    return render(request,'results.html',{'data':data,'movie_name':movie_name})