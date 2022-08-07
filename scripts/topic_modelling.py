import time
startTime = time.time()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import spacy
# import pyLDAvis.gensim_models
# pyLDAvis.enable_notebook()# Visualise inside a notebook
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import LdaModel
from gensim.models import CoherenceModel

import sys
lang = sys.argv[1]

# lang = "en"
print("Reading in data now.")
df_comb = pd.read_excel("./files/input/combined_data.xlsx")

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

df_comb["actual_lang"] = df_comb["content"].apply(detect)

df = df_comb[df_comb["actual_lang"] == str(lang)].reset_index(drop=True)

import warnings
warnings.filterwarnings('ignore')
lang_models = {"en": spacy.load("en_core_web_md"), "it": spacy.load("it_core_news_md"), "ru": spacy.load("ru_core_news_md")}

# Our spaCy model:
nlp = lang_models[str(lang)]

print("Loading " + str(lang) + " model")
# Tags I want to remove from the text

print("Tokenizing, lemmatizing and removing stopwords.")
removal= ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
tokens = []
for summary in nlp.pipe(df['content']):
    proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal and not token.is_stop and token.is_alpha]
    tokens.append(proj_tok)
    
df['tokens'] = tokens

dictionary = Dictionary(df['tokens'])

dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)

corpus = [dictionary.doc2bow(doc) for doc in df['tokens']]


# topics = []
# score = []
# for i in range(1,10,1):
#     print("Running iteration number " + str(i))
#     lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers = 4, passes=10, random_state=100)
#     cm = CoherenceModel(model=lda_model, texts = df['tokens'], corpus=corpus, dictionary=dictionary, coherence='c_v')
#     topics.append(i)
#     score.append(cm.get_coherence())
# num_topics = topics[score.index(max(score))]

topic_lang = {"en": 7, "it": 6, "ru": 5} 

num_topics = topic_lang[lang]

print("Number of topics with highest coherence score for " + str(lang) + " model is: " + str(num_topics))

# lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=100, num_topics=num_topics, workers = 4, passes=100)
lda_model = LdaModel(corpus=corpus, id2word=dictionary, iterations=100, num_topics=num_topics, passes=100)

# lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
# pyLDAvis.save_html(lda_display, "./files/output/" + str(lang) + "/lda_" + str(lang) + ".html")

def format_topics_sentences(ldamodel=None, corpus=corpus, texts=df['tokens']):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=df['tokens'])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index(drop=True)
df_dominant_topic.columns = ['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']
df_dominant_topic = pd.concat([df,df_dominant_topic], axis=1)

writer = pd.ExcelWriter('./files/output/' +str(lang) + '/topic_modelling_output_' + str(lang) + '.xlsx' , engine='xlsxwriter')
workbook = writer.book
workbook.formats[0].set_font_size(12)
df_dominant_topic.to_excel(writer, sheet_name = 'Output', index=False)

for idx, col in enumerate(df_dominant_topic):  # loop through all columns
    series = df_dominant_topic[col]
    max_len = 20
    writer.sheets['Output'].set_column(idx, idx, max_len)  # set column width
    
writer.close()

print("Outputted excel file has been saved in ./files/output/"+str(lang)+" folder")
print("Visualisation data has been saved in ./files/output/"+str(lang)+" folder")
print("Coherence plot has been saved in ./files/output/"+str(lang)+" folder")

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))