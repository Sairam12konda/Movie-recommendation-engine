import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv("tmdb_5000_movies.csv")
df.head(5)

userinput = [{'title':'avatar','rating':7.2},
             {'title':'Spectre','rating':6.9},
             {'title':'John Carter','rating':6.1},
             {'title':'tangled','rating':7.4}]
inputMovies = pd.DataFrame(userinput)
print(inputMovies.shape)
inputMovies

movie_genere = ["action","romance","animation","suspense"]
movie = [["avatar",1,0,1,0],["Spectre",1,0,0,1],["John Carter",1,1,0,0],["tangled",0,1,1,0]]
movie_df = pd.DataFrame(movie,columns=['Title']+movie_genere)
movie_df

user = np.array([[7.2,6.9,6.1,7.4]])
c = movie_df[["action","romance","animation","suspense"]].to_numpy()
print(f"User profile vector shape {user.shape} and course genre matrix shape {c.shape}")

user_weight =  np.matmul(user, c)
user_weight

sum_userweights = sum(user_weight[0])
sum_userweights

weighted_mat =[]
for i in user_weight:
    for j in i:
      weighted_mat.append(j/sum_userweights)
weighted_mat
w_mat = np.array([[0.36594202898550726, 0.24456521739130432, 0.2644927536231884, 0.12512345678]])
print(w_mat)

suggest_movie_genere = ["action","anime","fiction","drama"]
suggest_movie = [["Skyfall",1,0,1,0],["Cars 2",1,1,0,0],["Iron Man",1,0,1,0],["Hugo",0,1,1,1]]
suggest_movie_df = pd.DataFrame(suggest_movie,columns=['Title']+suggest_movie_genere)
suggest_movie_df

sug_c = suggest_movie_df[["action","anime","fiction","drama"]].to_numpy()
print(f"weighted matrix profile vector shape {w_mat.shape} and new movies genere  {sug_c.shape}")
sug_c

new_weights = np.matmul(w_mat,sug_c)
new_weights

new_movie_rating=[]
for i in new_weights:

    new_movie_rating.append(10*i)
new_movie_rating

z = np.array(new_movie_rating)
z.sort()
print(z)

print("The movies that will be recommended are:")
print(f"1.{suggest_movie[0][0]}-->{new_movie_rating[0][0]}")