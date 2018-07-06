from keras.models import load_model,model_from_json
from keras.preprocessing.text import text_to_word_sequence
from keras.datasets import imdb
from keras.preprocessing import sequence
from flask import Flask
from flask import request
from flask import jsonify

import numpy as np
import tensorflow as tf

#INSTRUCTIONS
#set FLASK_APP=sentiment_app.py
#flask run --host=0.0.0.0

import speech_recognition as sr
sample_rate=10000 #sample rate is how often value are reccorded
chunk_size=4096 #number of byte (size) to be taken at a time (2 pow n)
r = sr.Recognizer()
#generate a list of all audio cards / microphones
mic_list= sr.Microphone.list_microphone_names()
device_id=0


app = Flask(__name__)

global loaded_model, graph
def get_model():

	json_file = open('sentiment_model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("sentiment_model.h5")
	
	#compile and evaluate loaded model
	loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	print(" Model loaded! ")
	graph = tf.get_default_graph()

	return loaded_model,graph

	

global word_index
word_index = imdb.get_word_index()
def preprocess(words):

	global word_index
	
	words=text_to_word_sequence(words)
	input_word=[[word_index[word] if word in word_index else 0 for word in words]]
	input_word = sequence.pad_sequences(input_word, maxlen=100)
	
	vector = np.array([input_word.flatten()])
	
	return vector

	
	
print("Loading keras model")
loaded_model,graph=get_model()

@app.route('/predict',methods=['POST'])
def predict():

	with sr.Microphone(device_index = device_id , sample_rate= sample_rate,chunk_size= chunk_size) as source:
		r.adjust_for_ambient_noise(source)
		print("say something")
		#listen to the user's input
		audio= r.listen(source)
		
		try:
			text=r.recognize_google(audio)   #RECOGNIZE AUDIO FROM GOOGLE
			#r.recognize_google(audio,key="GOOGLE_SPEECH_RECOGNITION_API_KEY")
			print("you said : "+text)
		except sr.UnknownValueError:        #ERROR OF  speech recognition could not understand audio
			print("google speech recognition could not understand audio")
		except sr.RequestError as e:        # ERROR OF INTERNAT PROBLEM
			print("could not request result from google".format(e))

	

	with graph.as_default():
		global loaded_model
		vector= preprocess(text)
		result_class=loaded_model.predict_classes(vector)[0][0] 
		result_output=loaded_model.predict(vector)[0][0]
		
		outcome=''
		
		if result_class == 1:
			outcome='positive'
		else:
			outcome='negative'
		
		response={
		'input': text,
		'greeting': outcome+ " " + str(round((result_output*100),2))+"%",
		'accuracy': str(result_output)
		}
		
		return jsonify(response)

