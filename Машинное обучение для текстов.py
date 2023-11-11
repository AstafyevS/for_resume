#!/usr/bin/env python
# coding: utf-8

# <font color='blue' size=5><b>Комментарий ревьюера</b></font>
# 
# <font color='blue'>Привет, Сергей! Меня зовут Павел Григорьев, и я буду проверять этот проект.<br>Моя основная цель - не указать на совершённые тобой ошибки, а поделиться своим опытом и помочь тебе совершенствоваться как профессионалу.<br>Спасибо за проделанную работу! Предлагаю общаться на «ты».</font>
# <details>
# 	<summary><u>Инструкция по организационным моментам (кликабельно)</u></summary>
# <font color='blue'>Я буду использовать различные цвета, чтобы было удобнее воспринимать мои комментарии:</font>
# 
# 
# ---
# 
# 
# <font color='blue'>синий текст - просто текст комментария</font>
# 
# <font color='green'>✔️ и зеленый текст - все отлично</font>
# 
# <font color='orange'>⚠️ и оранжевый текст - сделано все правильно, однако есть рекомендации, на что стоит обратить внимание</font>
# 
# <font color='red'>❌ и красный текст - есть недочеты</font>
# 
# 
# </details>    
#     </br>
# <font color='blue'>Пожалуйста, не удаляй мои комментарии в случае возврата работы, так будет проще разобраться, какие были недочеты, а также сразу увидеть исправленное. </font>
# 
# Ответы на мои комментарии лучше тоже помечать.
# Например: <font color='purple'><b>Комментарий студента</b></font>
# 
# <font color='blue'><b>Давай смотреть, что получилось!</b></font>

# <font color='blue' size=3><b>Комментарий ревьюера 2</b></font>
# 
# <font color='blue'>Привет еще раз. Спасибо, за исправления. Оформление комментариев по работе сохраняется. Только обозначим, что это вторая итерация.</font>

# <font color='blue' size=3><b>Комментарий ревьюера 3</b></font>
# 
# <font color='blue'>Привет еще раз. Спасибо, за исправления. Оформление комментариев по работе сохраняется.</font>

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#LogisticRegression" data-toc-modified-id="LogisticRegression-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>LogisticRegression</a></span></li><li><span><a href="#DecisionTreeClassifier" data-toc-modified-id="DecisionTreeClassifier-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>DecisionTreeClassifier</a></span></li></ul></li><li><span><a href="#Выводы" data-toc-modified-id="Выводы-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Выводы</a></span></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li></ul></div>

# # Проект для «Викишоп»

# Интернет-магазин «Викишоп» запускает новый сервис. Теперь пользователи могут редактировать и дополнять описания товаров, как в вики-сообществах. То есть клиенты предлагают свои правки и комментируют изменения других. Магазину нужен инструмент, который будет искать токсичные комментарии и отправлять их на модерацию. 
# 
# Обучите модель классифицировать комментарии на позитивные и негативные. В вашем распоряжении набор данных с разметкой о токсичности правок.
# 
# Постройте модель со значением метрики качества *F1* не меньше 0.75. 
# 
# **Инструкция по выполнению проекта**
# 
# 1. Загрузите и подготовьте данные.
# 2. Обучите разные модели. 
# 3. Сделайте выводы.
# 
# Для выполнения проекта применять *BERT* необязательно, но вы можете попробовать.
# 
# **Описание данных**
# 
# Данные находятся в файле `toxic_comments.csv`. Столбец *text* в нём содержит текст комментария, а *toxic* — целевой признак.

# ## Подготовка

# In[1]:


get_ipython().run_line_magic('autosave', '10')


# In[2]:


get_ipython().system('pip install spacy')


# In[3]:


import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score
import numpy as np
from sklearn import metrics
import spacy
import en_core_web_sm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️ <br>
# <font color='green'>Здорово, что все импорты находятся в одной ячейке.<br>
# Если хочешь сделать ещё лучше, соблюдай стандарт PEP-8, разбей импорты на функциональные группы, раздели группы пустой строкой. Можно в каждой группе отсортировать импорты в алфавитном порядке.<br>
# Ссылка для ознакомления: https://pythonworld.ru/osnovy/pep-8-rukovodstvo-po-napisaniyu-koda-na-python.html
# </font>

# In[4]:


comments = pd.read_csv('/datasets/toxic_comments.csv')
comments.head()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Данные загружены корректно, первичный осмотр проведен.</font>

# In[5]:


comments = comments.drop(columns='Unnamed: 0')


# Очистим тексты от символов и приведем все в нижнему регистру:

# In[6]:


def clean_data(row):
    row = re.sub(r"(?:\n|\r)", " ", row)
    row = re.sub(r"[^a-zA-Z ]+", "", row).strip()
    row = row.lower()
    return row

comments['text'] = comments['text'].apply(clean_data)
comments.head()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Очистка сделана верно! Мы оставили только символы Латинского алфавита нижнего регистра.</font>

# In[7]:


comments.info()


# In[8]:


m = spacy.load("en_core_web_sm", disable=['parser', 'ner'])


# In[9]:


comments_corpus = comments['text'].values
def lemmatize(text):
    doc = m(text)
    return " ".join([token.lemma_ for token in doc])

for i in range(len(comments_corpus)):
    comments_corpus[i] = lemmatize(comments_corpus[i])
comments_corpus


# In[12]:


target = comments['toxic']


# In[15]:


features_train_valid, features_test, target_train_valid, target_test = train_test_split(comments_corpus, target, test_size = 0.1, random_state = 12345)
features_train, features_valid, target_train, target_valid = train_test_split(features_train_valid, target_train_valid, shuffle=False, test_size=0.25, random_state = 12345)
print(features_train.shape, features_valid.shape, features_test.shape,target_train.shape, target_valid.shape, target_test.shape)


# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'> Здорово , что у нас есть три выборки!</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s>Приводить тексты к юникоду не имеет смысла, так как после очистки осталась только латинница. Юникод  сильно увеличивает  количество потребляемой памяти.</s></font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Обрати внимание, лемматизация не выполнена. pymystem работает только для русского языка, и здесь бесполезна. Про другие инструменты для лемматизации (Wordnet Lemmatizer или Spacy Lemmatizer) можно почитать [тут](https://webdevblog.ru/podhody-lemmatizacii-s-primerami-v-python/). Обрати внимание на POS-теги. Для некоторых лемматизаторов они необходимы.</s></font>

# <font color='blue'><b>Комментарий студента: </b></font> \
# <font color='purple'> Cпасибо поправил.</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ❌\
# <font color='red'><s>Обрати внимание, Лемматизация проведена только для превой строчки корпуса (lemmatize(train_corpus[0]),\
# И слова склеились в одно. Нужно Джойнить отлемматизированные слова по пробелу `" ".join([token.lemma_ for token in doc])`.</s></font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ⚠️\
# <font color='darkorange'> Лемматизацию можно проводить на всём корпусе текстов, до разбиения на выборке. Там обрабатывается каждое слово по отдельности, предобученной моделью, поэтому лемматизация не зависит от соседних слов.</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font>\
# <font color='blue'>Присмотрись к spaCy - работает быстрее и кода меньше.</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Лемматизацию можно делать на всех данных. Внутри идёт обработка каждого слова по отдельности.</font>

# <font color='blue'><b>Комментарий студента 2: </b></font> \
# <font color='purple'> Cпасибо поправил.</font>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ✔️\
# <font color='green'> Отлично! Теперь видно, что слова в начальных формах.</font>

# In[17]:


stopwords = set(nltk_stopwords.words('english'))
count_tf_idf = TfidfVectorizer(stop_words=stopwords, ngram_range=(1,1))
tf_idf_train = count_tf_idf.fit_transform(features_train)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Убрали частые неинформативные слова!</font>

# In[18]:


tf_idf_valid = count_tf_idf.transform(features_valid)


# In[19]:


tf_idf_test = count_tf_idf.transform(features_test)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Классно, что векторизатор был обучен только на тренировочной части данных!</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Можно объединить Векторизатор с моделью через Pipeline. Так можно избежать утечек даже при кроссвалидации моделей.<br> Материалы по Pipeline:<br> [О Пайплайн](https://www.helenkapatsa.ru/paiplain/)<br>
# 
# [Примеры работы с текстами](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html)</font>

# ## Обучение

# ### LogisticRegression

# In[20]:


lr = LogisticRegression(random_state=1, solver='liblinear', max_iter=100)
params = {
   'penalty':['l1', 'l2'],        
   'C':list(range(1,15,3)) 
}



lr_gs = GridSearchCV(lr, params, cv=3, scoring='f1', verbose=True).fit(tf_idf_train, target_train)

print ("Best Params", lr_gs.best_params_)
print ("Best Score", lr_gs.best_score_)


# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'>Подбор гиперпараметров проведён верно.</font>

# In[21]:


lr_best = LogisticRegression(random_state=1, class_weight = 'balanced', C = 4, penalty = 'l1', solver='liblinear', max_iter=100)
lr_best.fit(tf_idf_train, target_train)


# Проверим на валидационной выборке

# In[22]:


pred1 = lr_best.predict(tf_idf_valid)
f1_score(target_valid, pred1)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# ### DecisionTreeClassifier

# In[ ]:


tree = DecisionTreeClassifier(random_state = 123)
params = {
   'criterion':['gini', 'entropy'],        
   'max_depth':list(range(1,15,5)) 
}



tree_gs = GridSearchCV(tree, params, cv=3, scoring='f1', verbose=True).fit(tf_idf_train, target_train)

print ("Best Params", tree_gs.best_params_)
print ("Best Score", tree_gs.best_score_)


# Также проверим на валидационной выборке

# In[ ]:


tree_best = DecisionTreeClassifier(random_state = 123, criterion='gini', max_depth=11)
tree_best.fit(tf_idf_train, target_train)
pred2 = tree_best.predict(tf_idf_valid)
f1_score(target_valid, pred2)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# Найдем предсказания на тестовой выборке LogisticRegression(её результаты лучше)

# In[ ]:


pred1 = lr_best.predict(tf_idf_test)      
f1_lr = f1_score(target_test, pred1)     
f1_lr


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Тестирование проведено верно.</font>

# ## Выводы

# Значение F1 больше 0.75, что значит, что с заданием наша модель справилась! Лучшая модель - LogisticRegression с гиперпараметрами C = 4, penalty = 'l1'.

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Очень приятно видеть вывод в конце проекта!\
# Приведены ответы на главные вопросы проекта.</font>

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>\
# <font color='green'>Сергей, хороший проект получился!
# Большое спасибо за проделанную работу. Видно, что приложено много усилий.
# </font>
# 
# <font color='blue'>Что нужно исправить:</font>
# <ul><font color='red'>Не применяй Юникод для текстов на латинице.</font></ul>
# <ul><font color='red'>Не применяй pymystem для текстов на Английском языке.</font></ul>
# 
# <font color='blue'>Что можно сделать лучше:</font>
# <font color='orange'>В работе я оставил несколько советов. Буду рад, если ты учтешь их.</font></ul>
# 
# <font color='blue'><b>Жду новую версию проекта :)</b></font>

# <font color='blue'><b>Комментарий студента: </b></font> \
# <font color='purple'> Критические замечания подобрал очень жду проверки.</font>

# <font color='blue'><b>Итоговый комментарий ревьюера 2</b></font>\
# <font color='green'>Сергей, большое спасибо за доработку. Убрали Юникод, применили SpaCy.
# </font>
# 
# <font color='blue'>Что нужно исправить:</font>
# <ul><font color='red'>Поправь Лемматизацию.</font></ul>
# 
# <font color='blue'>Что можно сделать лучше:</font>
# <font color='orange'>В работе я оставил несколько советов. Буду рад, если ты учтешь их.</font></ul>
# 
# <font color='blue'><b>Жду новую версию проекта :)</b></font>

# <font color='blue'><b>Итоговый комментарий ревьюера 2</b></font>\
# <font color='green'> Сергей, проект принят! \
# Все этапы пройдены. Все рекомендации учтены.\
# Надеюсь, тебе понравился процесс выполнения и результат.</font> \
# <font color='blue'><b>Спасибо, удачи в освоении профессии!</b></font>

# <font color='green'><b>Полезные (и просто интересные) материалы:</b> \
# Для работы с текстами используют и другие подходы. Например, сейчас активно используются RNN (LSTM) и трансформеры (BERT и другие с улицы Сезам, например, ELMO). НО! Они не являются панацеей, не всегда они нужны, так как и TF-IDF или Word2Vec + модели из классического ML тоже могут справляться. \
# BERT тяжелый, существует много его вариаций для разных задач, есть готовые модели, есть надстройки над библиотекой transformers. Если, обучать BERT на GPU (можно в Google Colab или Kaggle), то должно быть побыстрее.\
# https://huggingface.co/transformers/model_doc/bert.html \
# https://colah.github.io/posts/2015-08-Understanding-LSTMs/ - Про LSTM \
# https://web.stanford.edu/~jurafsky/slp3/10.pdf - про энкодер-декодер модели, этеншены\
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html - официальный гайд
# по трансформеру от создателей pytorch\
# https://transformer.huggingface.co/ - поболтать с трансформером \
# Библиотеки: allennlp, fairseq, transformers, tensorflow-text — множествореализованных
# методов для трансформеров методов NLP \
# Word2Vec https://radimrehurek.com/gensim/models/word2vec.html
# 
# <font color='green'>Пример BERT с GPU:
# ```python
# %%time
# from tqdm import notebook
# batch_size = 2 # для примера возьмем такой батч, где будет всего две строки датасета
# embeddings = []
# for i in notebook.tqdm(range(input_ids.shape[0] // batch_size)):
#         batch = torch.LongTensor(input_ids[batch_size*i:batch_size*(i+1)]).cuda() # закидываем тензор на GPU
#         attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)]).cuda()
# 
#         with torch.no_grad():
#             model.cuda()
#             batch_embeddings = model(batch, attention_mask=attention_mask_batch)
# 
#         embeddings.append(batch_embeddings[0][:,0,:].cpu().numpy()) # перевод обратно на проц, чтобы в нумпай кинуть
#         del batch
#         del attention_mask_batch
#         del batch_embeddings
# 
# features = np.concatenate(embeddings)
# ```
# Можно сделать предварительную проверку на наличие GPU.\
# Например, так: ```device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")```\
# Тогда вместо .cuda() нужно писать .to(device)
# 
# Если понравилась работа с текстами, то можешь посмотреть очень интересный (но очень-очень сложный) курс лекций: https://github.com/yandexdataschool/nlp_course .
# 
#     
# NLP от Samsung https://stepik.org/course/54098/promo \
# NLP от Huawei https://ods.ai/tracks/nlp-course-spring-23
# </font>

# ## Чек-лист проверки

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Данные загружены и подготовлены
# - [x]  Модели обучены
# - [x]  Значение метрики *F1* не меньше 0.75
# - [x]  Выводы написаны
