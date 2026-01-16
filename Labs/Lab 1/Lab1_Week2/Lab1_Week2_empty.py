import pandas as pd 
import spacy
from spacy.matcher import Matcher
import requests

#Observe the default natural language processing (NLP) pipeline
nlp = spacy.blank("en")
print("Pipeline components:", nlp.pipe_names) #Tokenizer is always available (even if the output shows blank)

text = "On a foggy Tuesday morning, Eleanor Whitmore met Daniel Alvarez outside the old Briarwood Train Station in Portsmouth, New Hampshire, clutching a leather-bound notebook and a chipped blue ceramic mug."
doc = nlp(text)

#GOAL: Add in the following components of the NLP pipeline
#       - PoS-tagging
#       - Parser
#       - Named-entity recognition

#HINT: Use the built-in add_pipe(...) command, which is a method of the Language object created by spacy.blank("en")
#HINT: You can use the built-in pipe_names command (a method of the Language object created by spacy.blank("en")) to view the current NLP pipeline


#GOAL: Test each component and observe their results (if the results are blank, then please explain why). The following components you are considering are:
#       - Tokenization
#       - PoS-tagging
#       - Parsing
#       - Named-Entity Recognition


#The following works with a pre-trained NLP pipeline
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

#GOAL: Test each component of the pre-trained model and observe their results (take a quick skim through and verify them)
#HINT: For PoS-tagging, consider the "pos_" attribute of a token
#      For Parsing, consider the "dep_" attribute of a token
#      For NER, consider the "label_" attribute of an entity



    
