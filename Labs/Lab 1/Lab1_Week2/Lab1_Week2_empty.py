# Import required libraries
import pandas as pd
import spacy
from spacy.matcher import Matcher
import requests

# INS. COMM. Create a blank English NLP pipeline
nlp = spacy.blank("en")

print("Pipeline components before adding anything:", nlp.pipe_names)

# INS. COMM. Add untrained components
tagger = nlp.add_pipe("tagger")
parser = nlp.add_pipe("parser")
ner = nlp.add_pipe("ner")

print("Pipeline components after adding pipes:", nlp.pipe_names)
print()

#NOTE: Added dummy labels so spaCy will not crash
tagger.add_label("NOUN")
parser.add_label("dep")
ner.add_label("PERSON")

#NOTE: Added initialize so spacy will not crash
nlp.initialize()

text = "On a foggy Tuesday morning, Eleanor Whitmore met Daniel Alvarez outside the old Briarwood Train Station in Portsmouth, New Hampshire, clutching a leather-bound notebook and a chipped blue ceramic mug."
doc = nlp(text)

# INS. COMM. Tokenization
tokens = [token.text for token in doc]
print("Tokenization Results:")
print(tokens)
print()

# INS. COMM. POS tagging 
pos_tokens = [f"{token.text}: {token.pos_}" for token in doc]
print("POS Results:")
print(pos_tokens)
print()

# INS. COMM. Dependency parsing 
parsed_tokens = [f"{token.text}: {token.tag_}, {token.dep_}" for token in doc]
print("Parsing Results:")
print(parsed_tokens)
print()

# INS. COMM. Named-Entity Recognition 
ner_tokens = [f"{ent.text}: {ent.label_}" for ent in doc.ents]
print("NER Results:")
print(ner_tokens)
print("\n\n")


#============================ Pre-Trained Model ==================================
#INS. COMM. The following works with a pre-trained NLP pipeline
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
print("Components of en_core_web_sm: ")
print(nlp.pipe_names)
print()

#INS. COMM. GOAL: Test each component of the pre-trained model and observe their results (take a quick skim through and verify them)
#INS. COMM. HINT: For PoS-tagging, consider the "pos_" attribute of a token
#INS. COMM.       For Parsing, consider the "dep_" attribute of a token
#INS. COMM.       For NER, consider the "label_" attribute of an entity

# Tokenizer/tok2vec
# Each word and punctuation is treated as its own token
tokens = []
for token in doc:
    tokens.append(token)
print("Tokenization results: ")
print(tokens)
print()

# POS Tagger/tagger
# Each token is matched correctly to its part of speech such as 'morning' -> noun or 'Whitmore' -> proper noun
pos_tokens = []
for token in doc:
    pos_tokens.append(f"{token}: {token.pos_}")
print("POS results: ")
print(pos_tokens)
print()

# Parser/parser
# Each token's POS tag and dependency relationship is displayed, ex. "foggy" -> JJ=adjective amod=adjective modifier
parsed_tokens = []
for token in doc:
    parsed_tokens.append(f"{token}: {token.tag_}, {token.dep_}")
print("Parsing Results: ")
print(parsed_tokens)
print()

# Attribute Ruler/attribute_ruler
# Forces the text "Hampshire" to be recognized by POS tagging as an adjective, rather than a proper noun
ruler = nlp.get_pipe("attribute_ruler")
ruler.add(
    patterns=[[{"TEXT": "Hampshire"}]],
    attrs={
        "POS": "ADJ"
    }
)

doc_att_ruler = nlp(text)
att_tokens = []
for token in doc_att_ruler:
    att_tokens.append(f"{token}: {token.pos_}")
print("Attribute Rule results: ")
print(att_tokens)
print()

# Lemmatizer/lemmatizer
# Words are correctly reduced to their base forms ex. "met" -> meet
lemma = [token.lemma_ for token in doc]
print("Lemmatization Results: ")
print(lemma)
print()

# Named-Entity Recognition/ner
# Correctly identifes tokens as their correct entity type ex. "Tuesday" -> DATE, "Daniel Alvarez" => PERSON
ner_tokens = []
for ent in doc.ents:
    ner_tokens.append(f"{ent.text}: {ent.label_}")
print("NER Results: ")
print(ner_tokens)





    
