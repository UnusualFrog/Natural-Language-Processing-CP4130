# Import required libraries
import pandas as pd 
import spacy
from spacy.matcher import Matcher
import requests

#INS. COMM. Observe the default natural language processing (NLP) pipeline
nlp = spacy.blank("en")
print("Pipeline components:", nlp.pipe_names) #INS. COMM. Tokenizer is always available (even if the output shows blank)

#INS. COMM. HINT: Use the built-in add_pipe(...) command, which is a method of the Language object created by spacy.blank("en")
#INS. COMM. HINT: You can use the built-in pipe_names command (a method of the Language object created by spacy.blank("en")) to view the current NLP pipeline
#INS. COMM. GOAL: Add in the following components of the NLP pipeline

#NOTE: Source had to be included for add_pipe, otherwise code throws initialization errors for added pipeline components
source_nlp = spacy.load("en_core_web_sm")

#INS. COMM.        - PoS-tagging
nlp.add_pipe("tagger", source=source_nlp)
#INS. COMM.        - Parser
nlp.add_pipe("parser", source=source_nlp)
#INS. COMM.        - Named-entity recognition
nlp.add_pipe("ner", source=source_nlp)

text = "On a foggy Tuesday morning, Eleanor Whitmore met Daniel Alvarez outside the old Briarwood Train Station in Portsmouth, New Hampshire, clutching a leather-bound notebook and a chipped blue ceramic mug."
doc = nlp(text)

#INS. COMM. GOAL: Test each component and observe their results (if the results are blank, then please explain why). The following components you are considering are:
#INS. COMM.        - Tokenization
tokens = []
for token in doc:
    tokens.append(token)
print("Tokenization Results: ")
print(tokens)
print()

#INS. COMM.        - PoS-tagging
# token.pos__ prints blank because spacy.blank("en") is a blank model 
# with no POS library to reference for different speech part types
pos_tokens = []
for token in doc:
    pos_tokens.append(f"{token}: {token.pos_}")
print("POS Results: ")
print(pos_tokens)
print()

#INS. COMM.        - Parsing
# Parser fails to correctly parse the tokens due to the empty model 
# lacking the vocabulary normally provided to the parser by a pre-trained model such as en_core_web_sm
# resulting in most tokens being incorrectly identified as conj
parsed_tokens = []
for token in doc:
    parsed_tokens.append(f"{token}: {token.tag_}, {token.dep_}")
print("Parsing Results: ")
print(parsed_tokens)
print()

#INS. COMM.        - Named-Entity Recognition
ner_tokens = []
for ent in doc.ents:
    ner_tokens.append(f"{ent.text}: {ent.label_}")
print("NER Results: ")
print(ner_tokens)
print("\n\n")