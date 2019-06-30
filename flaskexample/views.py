from flask import render_template
from flaskexample import app
from flaskexample.perfume_recommendation import Fragrance_Retrieve_Model
import pandas as pd
from flask import request
import requests

# my homepage
@app.route('/')
def homepage():
   return render_template("model_input.html")

@app.route('/model_output')
def perfume_recommand_output():
   # pull 'original_test' from input field and store it
   original_text = request.args.get('original_text')
   unlike_message = request.args.get('hate_message')

   gre = Fragrance_Retrieve_Model()

   # recs is a dataframe which only contains the product name and similarity score
   recs = gre.query_similar_perfumes(original_text, unlike_message)
   if recs is not None:
      # need to convert to dictionary to display on html
      rec_dic = []
      for i in range(0, recs.shape[0]):
         single_title = recs.index.tolist()[i]
         single_perfume = gre.df.query('name==@single_title')
         p_name = single_perfume.name.values[0]
         p_description = single_perfume.display.values[0]
         p_brand = single_perfume.brand.values[0]
         p_rating = single_perfume.rating.values[0]
         p_price = single_perfume.price.values[0]
         rec_dic.append(dict(name=p_name,brand=p_brand,rating=p_rating,price=p_price,
                        description=p_description,product_image="static/images/{}.jpg".format(p_name).replace(" ", "_")))
  
      return render_template("model_output.html", recommendations=rec_dic)
   
   else:
      return render_template("error.html")
