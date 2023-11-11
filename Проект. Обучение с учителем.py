#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# <b> Сергей, привет!👋</b> 
# 
# Меня зовут Кирилл Васильев, я буду ревьюером твоего проекта. Я предлагаю общаться на «ты», но если привычнее на «вы», просто скажи об этом! 
# 
# Я буду оставлять комментарии в твоем проекте. Пожалуйста, не удаляй их, они могут быть полезны в случае повторной проверки.
# 
# Ты можешь реагировать на комментарии как просто написав их в Markdown-ячейках, так и выделив их цветом. Например, <font color='blue'>синим</font>. Второй способ, на мой взгляд, удобнее.
# 
# Свои комментарии я буду обозначать <font color='green'>зеленым</font>, <font color='gold'>желтым</font> и <font color='red'>красным</font> цветами, например:
#  
# <br/>
# 
# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# <b>Все отлично!👍:</b> В случае, если решение на отдельном шаге является полностью правильным. Здесь же я могу давать советы и предложения.
# </div>
# 
# <br/>
# 
# <div class="alert alert-warning">
#     <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> В случае, когда решение на отдельном шаге станет еще лучше, если внести небольшие коррективы.
# </div>
# 
# 
# <br/>
# <div class="alert alert-block alert-danger">
# <h2> Комментарий ревьюера <a class="tocSkip"></h2>
# 
#     
# <b>На доработку🤔:</b>
#  В случае, когда решение на отдельном шаге требует существенной переработки и внесения правок. Напоминаю, что проект не может быть принят с первого раза, если ревью содержит комментарии, рекомендующие доработать шаги.
# </div>
#     
# Увидев у тебя неточность, в первый раз я лишь укажу на ее наличие и дам тебе возможность самому найти и исправить ее. На реальной работе твой руководитель будет поступать также. Но если ты пока не справишься с такой задачей - при следующей проверке я дам более точную подсказку! 

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Проверим-есть-ли-пропуски-в-данных" data-toc-modified-id="Проверим-есть-ли-пропуски-в-данных-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Проверим есть ли пропуски в данных</a></span></li></ul></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Исследование задачи</a></span></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span><ul class="toc-item"><li><span><a href="#" data-toc-modified-id="-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span><div class="alert alert-warning"></div></a></span></li></ul></li><li><span><a href="#Тестирование-модели" data-toc-modified-id="Тестирование-модели-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование модели</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте *F1*-меру на тестовой выборке самостоятельно.
# 
# Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# 👍 Отлично. Есть описание проекта. Ясны цели и задачи. Видны обязательные метрики.
# </div>

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# Эта строка точно должна быть в markdown?
#        
# </div>
# 

# In[1]:


get_ipython().system('pip install imblearn')


# In[2]:


import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    f1_score, 
    mean_squared_error, 
    roc_auc_score,
    mean_absolute_error,
    precision_score,
    precision_recall_curve,
    recall_score
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.datasets import make_circles, make_classification, make_regression

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from collections import Counter
from sklearn.datasets import make_classification


# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
#     
# Все ли из импортов выше мы используем. Если что-то не используем, нужно удалить. GridSearchCV? 
#     
# Если загружаем несколько функций из одного пакета, то удобнее записать это через запятую
#     
# И в таком случае здесь было бы идеально привести оформление в соответствие с <a href="https://pythonworld.ru/osnovy/pep-8-rukovodstvo-po-napisaniyu-koda-na-python.html">pep8</a> - не оставлять длинных строк. 
# 
# Такое оформление предпочтительнее:    
#     
# `from sklearn.metrics import (
#     accuracy_score, 
#     confusion_matrix, 
#     f1_score, 
#     mean_squared_error, 
#     roc_auc_score
# )`
#     
#         
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера #2 <a class="tocSkip"> </h2>
# 
# 👍 Здесь все отлично.
# </div>

# In[3]:


pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
pd.options.mode.chained_assignment = None

rcParams['figure.figsize'] = 10,10
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')
factor = 1
default_dpi = rcParamsDefault['figure.dpi']
rcParams['figure.dpi'] = default_dpi*factor


# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# А эта строка тоже не код?
#        
# </div>
# 
# 

# ## Подготовка данных

# In[4]:


data = pd.read_csv('/datasets/Churn.csv')
display(data.info())
display(data)


# ### Проверим есть ли пропуски в данных

# In[5]:


data.isna().sum()


# В столбце "Tenure" указано сколько лет клиент обсулживается в этом банке. Заменим отсутствующие значения на нулевые, предположив, что это клиенты которые стали польховаться этим банком меньше года назад.

# In[6]:


data=data.fillna(data['Tenure'].median())


# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Предположение, что пропуски там, где клиент пришел недавно достаточно неплохое. Но подтвердить его сейчас мы не можем. Поэтому лучше следовать принципу, что следующие данные будут такими же по распределению, как текущие, и лучшей тактикой минимизации ошибки будет значение из области наибольшей плотности распределения, не изменяющую структуры распределения признака. 
#     
# Доля пропусков не критично велика, возможно и решение с удалением. Однако данных у нас не много, я бы остановился на чем-то вроде медианы.
# 
# </div>
# 
# 
# 

# In[7]:


data.describe(include='all')


# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# 👍 Отлично. Здесь можно передать аргументы, чтобы информация отображалась и по переменным с нечисленными типами данных.
#       
# `describe(include='all')`
#     
# Возможно, удобнее будет смотреть, если транспонировать таблицу.
# </div>
# 

# In[8]:


fig, ax = plt.subplots(figsize=(12, 7))
_ = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True, ax=ax)


# In[9]:


data = data.drop(columns=['Surname', 'CustomerId', 'RowNumber'])


# In[10]:


data_ohe = pd.get_dummies(data, drop_first=True)


# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# 👍 Согласен. В дамми-ловушку не попались.
# </div>

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Подобные строки нужно либо удалить, либо раскомментировать (превратить в код).
# 
# </div>
# 

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Необходимо удалить три лишних колонки. Если этого не сделать, то у нас будет примерно 3 тысячи столбцов из-за энкодинга фамилий. Айдишники и номера модели тоже не помогут. На данных, которые она видит один раз, она не способна сделать обобщение.
# 
# </div>
# 
# 
# 

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Нужен промежуточный вывод по предобработке.
# 
# </div>
# 
# 
# 

# В данном разделе мы заполнили пропущенные данные медианным значением, удалили 3 колонки которые не помогут нашей модели в определении уйдет ли клиент из банка. А также визуализировали зависимость данных и вывели краткую информацию по датафрейму.

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
#  
# Здесь по проекту мы должны выбрать метрики (ROC-AUC за нас выбрали по умолчанию, ее необходимо считать везде. Про f1-меру также сказано, что ее нужно подсчитать и довести до требуемого показателя). По остальным нужно прокомментировать, что нам важно посчитать для бизнеса: найти всех клиентов, которые могут захотеть уйти или максимально точно отделить тех, кто хочет уйти от тех, кто бы остался? Нужно обоснование выбора метрик и того, какие из них важнее.
#     
# Здесь же можно проверить модели на адекватность.
#     
# Здесь можно легко посчитать все в уме, можно использовать «дамми-классифайер» https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html .
# 
# </div>
# 

# ## Исследование задачи

# In[11]:


# Разделим данные на целевой признак и тестовую выборку
target = data_ohe['Exited']
features = data_ohe.drop(['Exited'] , axis=1)
# Разделим выборки на тестовую и валидационную
features_train, features_temp, target_train, target_temp = train_test_split(
    features, target, test_size=0.5, random_state=12345)
features_valid, features_test, target_valid, target_test rm= train_test_split(
    features_temp, target_temp, test_size=0.5, random_state=12345)


# In[12]:


numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
scaler = StandardScaler()
scaler = scaler.fit(features_train[numeric])


# In[13]:


features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_test[numeric] = scaler.transform(features_test[numeric])


# 
# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера #2 <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Кажется, забыли масштабировать тестовую выборку!
# 
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# 👍 С разбиением выборок согласен. Не увидел пока тестовую, но посмотрим дальше.
# </div>
# 

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Далее при подборе гиперпараметров используем валидационную выборку. Соответственно, тестировать будет не на чем. Нужно выделить тренировочную, валидационную и тестовую выборки.
# 
# </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Разбил добавив тестовую.</div>

# Узнаем эфективность различных моделей на несбалансированных данных

# In[14]:


model_libl = LogisticRegression(random_state=12345, solver='liblinear')
model_libl.fit(features_train, target_train)
predictions_valid_libl = model_libl.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_libl))
probabilities_valid = model_libl.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# <div class="alert alert-block alert-danger">
# <h2> <s>Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Здесь по заданию смотрим на модели без борьбы с дисбалансом. class_weight='balanced' относится к методам борьбы с дисбалансом. Здесь использовать такой аргумент будет неверно, лучше оставить это на потом.
# 
# </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Удалил балансировку.</div>

# In[15]:


model_log = LogisticRegression(random_state=12345, solver='liblinear')
model_log.fit(features_train, target_train)
predictions_valid_log = model_log.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_log))
probabilities_valid = model_log.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# In[16]:


model_sag = LogisticRegression(random_state=12345, solver='sag')
model_sag.fit(features_train, target_train)
predictions_valid_sag = model_sag.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_sag))
probabilities_valid = model_sag.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# От изменения сольвера результат не меняется

# In[17]:


valid_scores = []
train_scores = []

best_max_depth = 1
best_score = 0
param_range = list(range(1,50))

for n in param_range:
    model = DecisionTreeClassifier(max_depth=n, random_state=12345)
    model.fit(features_train, target_train)
    valid_preds = model.predict(features_valid)
    train_preds = model.predict(features_train)

    train_scores.append(f1_score(target_train, train_preds))
    valid_scores.append(f1_score(target_valid, valid_preds))
    
    if f1_score(target_valid, valid_preds) > best_score:
        best_score = f1_score(target_valid, valid_preds)
        best_max_depth = n
        
print(f'Лучшая глубина дерева: {best_max_depth}')
print(f'Лучшая метрика: {best_score}')

model_d = DecisionTreeClassifier(max_depth = best_max_depth, random_state=12345, class_weight='balanced')
model_d.fit(features_train, target_train)
predictions_valid_d = model_d.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_d))
probabilities_valid = model_d.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# Комментарий выше необходимо либо перевести в код, либо удалить.
#        
# </div>
# 

# In[18]:


best_model_r=0
best_result_r=0
for depth in range(1, 50, 1):
    model_r = RandomForestClassifier(n_estimators=175, max_depth=depth, random_state=12345)
    model_r.fit(features_train, target_train)
    predictions_valid_r=model_r.predict(features_valid)
    mae_r=f1_score(predictions_valid_r, target_valid)
    if mae_r > best_result_r:
        best_result_r=mae_r
        best_model_r=model_r
print(best_result_r)
print(best_model_r)
model_r = best_model_r
model_r.fit(features_train, target_train)
predictions_valid_r = model_r.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_r))
probabilities_valid = model_r.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# Модель DecisionTreeClassifier продемонстрировала наилучший результат с глубиной дерева = 6 и наилучший результат вообще

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# В дереве и лесе важно подобрать гиперпараметры. В дереве - глубину, в лесе - минимум глубину и количество оценщиков (эстиматоров, деревьев).
# 
# </div>
# 
# 
# 

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# Аналогично: комментарий выше необходимо либо перевести в код, либо удалить.
#        
# </div>

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Необходимо прокомментировать, какая модель с какими гиперпараметрами продемонстрировала какой лучший результат F1
# </div>
# 
# 
# 

# ## Борьба с дисбалансом

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Скалирование необходимо провести перед обучением моделей, в том числе и тех, которые обучаем без учета дисбаланса.
# 
# </div>
# 

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# Нужно кратко обозначить, какую методику борьбы с дисбалансом здесь применяем. Как видно дальше − взвешенные классы.
#        
# </div>

# Используем параметр балансировки классов

# In[19]:


model_libl = LogisticRegression(random_state=12345, solver='liblinear', class_weight='balanced')
model_libl.fit(features_train, target_train)

preds_libl = model_libl.predict(features_valid)
print("F1:", f1_score(target_valid, preds_libl))
probabilities_valid = model_libl.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# In[20]:


model_lbf = LogisticRegression(random_state=12345, solver='lbfgs', class_weight='balanced')
model_lbf.fit(features_train, target_train)

preds_lbf = model_lbf.predict(features_valid)
print("F1:", f1_score(target_valid, preds_lbf))
probabilities_valid = model_lbf.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# In[21]:


model_sag = LogisticRegression(random_state=12345, solver='sag', class_weight='balanced')
model_sag.fit(features_train, target_train)

preds = model_sag.predict(features_valid)
print("F1:", f1_score(target_valid, preds))
probabilities_valid_sag= model_sag.predict_proba(features_valid)
probabilities_one_valid_sag = probabilities_valid_sag[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid_sag)
print('AUC-ROC:', auc_roc)


# In[22]:


valid_scores = []
train_scores = []

best_max_depth = 1
best_score = 0
param_range = list(range(1,50))

for n in param_range:
    model = DecisionTreeClassifier(max_depth=n, random_state=12345, class_weight='balanced')
    model.fit(features_train, target_train)
    valid_preds = model.predict(features_valid)
    train_preds = model.predict(features_train)

    train_scores.append(f1_score(target_train, train_preds))
    valid_scores.append(f1_score(target_valid, valid_preds))
    
    if accuracy_score(target_valid, valid_preds) > best_score:
        best_score = f1_score(target_valid, valid_preds)
        best_max_depth = n
        
print(f'Лучшая глубина дерева: {best_max_depth}')
print(f'Лучшая метрика: {best_score}')

model_d = DecisionTreeClassifier(max_depth = best_max_depth, random_state=12345, class_weight='balanced')
model_d.fit(features_train, target_train)
predictions_valid_d = model_d.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_d))
probabilities_valid = model_d.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# In[23]:


best_model_r=0
best_result_r=0
for depth in range(1, 50, 1):
    model_r = RandomForestClassifier(n_estimators=80, max_depth=depth, random_state=12345, class_weight='balanced')
    model_r.fit(features_train, target_train)
    predictions_valid_r=model_r.predict(features_valid)
    mae_r=f1_score(predictions_valid_r, target_valid)
    if mae_r > best_result_r:
        best_result_r=mae_r
        best_model_r=model_r
print(best_result_r)
print(best_model_r)
model_r = best_model_r
model_r.fit(features_train, target_train)
predictions_valid_r = model_r.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_r))
probabilities_valid = model_r.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# Использование балансировки классов сильно улучшило качество моделей. При данных условиях лучшая модель это RandomForestClassifier с максимальной глубиной-12 и количеством эстиматоров-80.

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# Здесь также нужно перебрать гиперпараметры, использовать все те же модели, что и выше, описать лучшую модель.
#        
# </div>

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Здесь крайне важно для чистоты эксперимента и корректности сравнения унифицировать работу с моделями. Предлагаю такой план:
#     
# Выбрали модели: три - регрессия, дерево, лес. Посмотрели на них без дисбаланса, подобрали гиперпараметры, прокомментировали лучшую модель.
#     
# Дальше тот же порядок действий, только при борьбе с дисбалансом: сделали апсемплинг, посмотрели с ним три модели, выбрали лучшую, прокомментировали, обязательно с гиперпараметрами (нужно для воспроизводимости модели).
#     
# Все эти действия необходимо со всеми моделями проделать для апсемплинга, даунсемплинга и взвешеных классов.
#     
# Везде измеряем ROC-AUC.
# 
# 
# </div>

# Исследуем вариант борьбы с дизбалансом с помощью Upsampling

# In[24]:


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train, target_train, 4)


# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера #2 <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Апсемплинг делаем до достижения баланса классов 1:1. У нас было 1:4, мы умножили на 10, теперь дисбаланс в обратную сторону. Проверь параметр, который ты подаешь в функцию, вместо 10 нужно 4. При даунсемплинге такая же ошибка, там вместо 0.1 нужно 0.25.
# 
# </div>
# 

# <div class="alert alert-success">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
# 
# 👍 Отлично, что используешь рукописную функцию, но здесь можно все оптимизировать, используя <a href="https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html">SMOTE</a> для увеличения выборки и <a href="https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html">RandomUnderSampler</a>  для уменьшения. 
#     
# Апсемплинг делается очень просто:
#     
# `oversample = SMOTE(random_state=RANDOM_STATE)`
# 
# `features_train_up, target_train_up = oversample.fit_resample(features_train_enc, target_train)`
#     
# *предварительно импотрировав `from imblearn.over_sampling import SMOTE`
# 
# **Чтобы импортировать imblearn его сначала требуется проинсталлировать. через `!pip install imblearn`
#     
# </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> не получилось импортировать SMOTE</div>

# <div style="border:solid Lime 2px; padding: 40px">
# <font color=Blue>
# 
# 
# <div class="alert alert-success"  style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Комментарий ревьюера #2 <a class="tocSkip"> </h2>    
# 
# 👍 Ок, тогда без него пока.
# </div>
# 

# <div style="border:solid Lime 2px; padding: 40px">
# <font color=Blue>
# 
# 
# <div class="alert alert-success"  style="border-radius: 15px; box-shadow: 4px 4px 4px; border: 1px solid ">
# <h2> Комментарий ревьюера #3 <a class="tocSkip"> </h2>    
# 
# 👍 UPD Видимо, проблема в платформе. За сутки несколько человек написали, что не получается. До этого не было таких проблем.
# </div>

# In[25]:


valid_scores = []
train_scores = []

best_max_depth = 1
best_score = 0
param_range = list(range(1,50))

for n in param_range:
    model = DecisionTreeClassifier(max_depth=n, random_state=12345)
    model.fit(features_upsampled, target_upsampled)
    valid_preds = model.predict(features_valid)
    train_preds = model.predict(features_upsampled)

    train_scores.append(f1_score(target_upsampled, train_preds))
    valid_scores.append(f1_score(target_valid, valid_preds))
    
    if f1_score(target_valid, valid_preds) > best_score:
        best_score = f1_score(target_valid, valid_preds)
        best_max_depth = n
        
print(f'Лучшая глубина дерева: {best_max_depth}')
print(f'Лучшая метрика: {best_score}')

model_d = DecisionTreeClassifier(max_depth = best_max_depth, random_state=12345)
model_d.fit(features_upsampled, target_upsampled)
predictions_valid_d = model_d.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_d))
probabilities_valid = model_d.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# In[26]:


model_libl = LogisticRegression(random_state=12345, solver='liblinear')
model_libl.fit(features_upsampled, target_upsampled)

preds_libl = model_libl.predict(features_valid)
print("F1:", f1_score(target_valid, preds_libl))
probabilities_valid = model_libl.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# In[40]:


best_model_r=0
best_result_r=0
for depth in range(1, 50, 1):
    model_r = RandomForestClassifier(n_estimators=115, max_depth=depth, random_state=12345)
    model_r.fit(features_upsampled, target_upsampled)
    predictions_valid_r=model_r.predict(features_valid)
    mae_r=f1_score(predictions_valid_r, target_valid)
    if mae_r > best_result_r:
        best_result_r=mae_r
        best_model_r=model_r
print(best_result_r)
print(best_model_r)
model_r = best_model_r
model_r.fit(features_upsampled, target_upsampled)
predictions_valid_r = model_r.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_r))
probabilities_valid = model_r.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# при использовании Upsampling качество модели немного ухудшилось, но экспериментируя с количеством эстиматоров в RandomForestClassifier удалось также достич метрики F1=0.59

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Будет ошибочно использовать одновременно апсеплинг или даунсемплинг и взвешенные классы class_weight='balanced'. Мы и так доводим соотношение классов к 1:1, так что, class_weight работать не будет. Технически это не приведет к падению кода или ухудшению результатов (но и к улучшению не приведет аналитически), фактически же такой код говорит о том, что мы не понимаем механизма работы и назначения этого параметра и, соответственно, техники борьбы с дибсалансом.
# 
# </div>
# 

# Исследуем вариант борьбы с дизбалансом с помощью Downsampling

# In[28]:


def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)
    
    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, 0.25)


# In[29]:


model_libl = LogisticRegression(random_state=12345, solver='liblinear')
model_libl.fit(features_downsampled, target_downsampled)

preds_libl = model_libl.predict(features_valid)
print("F1:", f1_score(target_valid, preds_libl))
probabilities_valid = model_libl.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# In[30]:


valid_scores = []
train_scores = []

best_max_depth = 1
best_score = 0
param_range = list(range(1,50))

for n in param_range:
    model = DecisionTreeClassifier(max_depth=n, random_state=12345)
    model.fit(features_downsampled, target_downsampled)
    valid_preds = model.predict(features_valid)
    train_preds = model.predict(features_downsampled)

    train_scores.append(f1_score(target_downsampled, train_preds))
    valid_scores.append(f1_score(target_valid, valid_preds))
    
    if f1_score(target_valid, valid_preds) > best_score:
        best_score = f1_score(target_valid, valid_preds)
        best_max_depth = n
        
print(f'Лучшая глубина дерева: {best_max_depth}')
print(f'Лучшая метрика: {best_score}')

model_d = DecisionTreeClassifier(max_depth = best_max_depth, random_state=12345)
model_d.fit(features_downsampled, target_downsampled)
predictions_valid_d = model_d.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_d))
probabilities_valid = model_d.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# In[31]:


best_model_r=0
best_result_r=0
for depth in range(1, 50, 1):
    model_r = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=12345)
    model_r.fit(features_downsampled, target_downsampled)
    predictions_valid_r=model_r.predict(features_valid)
    mae_r=f1_score(predictions_valid_r, target_valid)
    if mae_r > best_result_r:
        best_result_r=mae_r
        best_model_r=model_r
print(best_result_r)
print(best_model_r)
model_r = best_model_r
model_r.fit(features_downsampled, target_downsampled)
predictions_valid_r = model_r.predict(features_valid)
print("F1:", f1_score(target_valid, predictions_valid_d))
probabilities_valid = model_r.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# Downsampling ухудшил качество всех моделей

# ### <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# Везде подбираем гиперпараметры.
#        
# </div>

# In[32]:


for threshold in np.arange(0, 1, 0.1):
    predicted_valid = probabilities_one_valid > threshold 
    precision = precision_score(target_valid, predicted_valid)
    recall = recall_score(target_valid, predicted_valid)
    ff1_score = f1_score(target_valid, predicted_valid)
    print("Порог = {:.2f} | Точность = {:.3f} | Полнота = {:.3f} | F1 = {:.3f}".format(
          threshold, precision, recall, ff1_score))
print('\n')


# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
#     
# Здесь по коду все более или менее верно. Попробуем исправить остальное, посмотрим, что изменится здесь. Попробуй еще взять больше диапазон, не только в три десятых.
# 
# `for threshold in np.arange(0, 1, 0.1):
#     predicted_valid = probabilities_one_valid > threshold 
#     precision = precision_score(target_valid, predicted_valid)
#     recall = recall_score(target_valid, predicted_valid)
#     ff1_score = f1_score(target_valid, predicted_valid)
#     print("Порог = {:.2f} | Точность = {:.3f} | Полнота = {:.3f} | F1 = {:.3f}".format(
#           threshold, precision, recall, ff1_score))
# print('\n')`
# 
#        
# </div>

# Наибольшая точность при пороге в 0.90, а наибольшая F1 при пороге 0.70

# In[33]:


model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])

plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Кривая Precision-Recall')
plt.show() 


# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# Здесь необходимо описать лучшую модель из найденных, которую возьмем на тестирование.
# 
# </div>
# 

# ## Тестирование модели

# In[34]:


model_r = RandomForestClassifier(class_weight='balanced', max_depth=12, n_estimators=80,
                       random_state=12345)
model_r.fit(features_train, target_train)
predictions_valid_r = model_r.predict(features_test)
print("F1:", f1_score(target_test, predictions_valid_r))
probabilities_valid = model_r.predict_proba(features_test)
probabilities_one_valid = probabilities_valid[:, 1]
auc_roc = roc_auc_score(target_test, probabilities_one_valid)
print('AUC-ROC:', auc_roc)


# <div class="alert alert-warning">
# <h2> Комментарий ревьюера  #2 <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# Не пугайся сейчас низкого результата, это из-за отсутствия масштабирования на тестовой. Также думаю стоит попробовать на тест апсемплинг после его доработки (до нужного баланса).
#        
# </div>

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#  
# При тестировании мы уже смотрим на одну лучшую модель с лучшими гиперпараметрами. Здесь ничего не подбираем. Тестирование - это применение чистых данных, которые наша модель еще не видела. Таким образом мы можем получить представление о том, какое качество модели мы можем ожидать в проде, в реальной жизни. 
#     
# Важно в тесте посмотреть на одну модель на тестовой выборке (которую нужно выделить) и с лучшими гиперпараметрами, подобранными ранее.
# 
# </div>

# Наилучшее качество модели (F1 = 0.5908 мы получили со следующими гиперпараметрами модели: RandomForestClassifier(class_weight='balanced', max_depth=12, n_estimators=80,random_state=12345)
# 
# А в использовании на тестовой выборке получили всего лишь F1 = 0.606.

# ## Чек-лист готовности проекта

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Выполнен шаг 1: данные подготовлены
# - [x]  Выполнен шаг 2: задача исследована
#     - [x]  Исследован баланс классов
#     - [x]  Изучены модели без учёта дисбаланса
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 3: учтён дисбаланс
#     - [x]  Применено несколько способов борьбы с дисбалансом
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 4: проведено тестирование
# - [x]  Удалось достичь *F1*-меры не менее 0.59
# - [x]  Исследована метрика *AUC-ROC*

# Подскажите пожалуйста какими еще способами можно добиться улучшения качества. И почему у RandomForestRegressor не получается найти метрику F1

# <div class="alert alert-warning">
# <h2> Комментарий ревьюера <a class="tocSkip"> </h2>
#     
# <b>Некоторые замечания и рекомендации💡:</b> 
# 
# По второму вопросу - у нас задача классификации, нам нужен не регрессор, а классифайер RandomForestClassifier.
#        
# </div>
# 
#  

# <div class="alert alert-block alert-danger">
# <h2><s> Комментарий ревьюера <a class="tocSkip"></h2>
#   
# <b>На доработку🤔:</b>
#     
# Метрику на тестовой выборке не измерили, нельзя считать, что качество модели достигнуто.
#  
# В целом, думаю, если ранжировать комментарии по полезности, то самое важное − удалить ненужные колонки (три штуки), затем провести энкодинг, разделить на выборки и провести масштабирование.
#     
# Важно в деревянных моделях подбирать параметры, выводя результат f1 меры, в лесе- минимум два гиперпараметра подбираются, удобнее в двойном цикле.
#     
# 
# 
# </div>

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# **Общий вывод по проекту**
#     
# Спасибо за твой проект! Мне было интересно следить за ходом мыслей и твоими решениями. Видны основательная работа и интерес к проекту. Есть несколько важных моментов, которые необходимо улучшить. Их ты найдешь в разноцветных комментариях.
#     
# **Отмечу положительные моменты проекта🚀**
#     
# * Все шаги проекта пройдены.
# * Мы используем три модели, три способа борьбы с дисбалансом.
# * Есть комментарии и пояснения к коду (не везде, правда, но все равно это сильно помогает).
#     
# **На что стоит обратить внимание🤔**
#     
# * Нужно удалить лишние колонки до энкодинга.
# * Везде оставляем промежуточные выводы и пояснения.
# * Модели смотрим сначала без дисбаланса, подбирая гиперпараметры, потом с разными техниками борьбы с дисбалансом.
# * Необходимо выделить и использовать в тестировании тестовую выборку.
# * В тестировании нельзя подбирать гиперпараметры.
# * Эти и остальные замечания я также написал по ходу проекта.
#     
# **Желаю удачи и жду твой проект на повторное ревью!**😉
#     
#     
# </div>

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# **Комментарий ко второй проверке**
#     
# Хорошая доработка, почти все сделано, два штриха и все заработает.
#     
# 1) Забыли сделать масштабирование после добавления тестовой выборки. Это частая ошибка, когда тест добавляем позже.
#     
# 2) Используем функцию из тренажера, в ней коэффициент для апсемплинга 10, нам нужен 4, если у нас редкий класс в четыре раза реже встречается. Мы доводим баланс до 1:1. 
#     
# По остальному, кажется, все нормально. Надеюсь, ничего не упустил.
# 
# 
# **Желаю удачи и жду твой проект на повторное ревью!**😉
# </div>

# <div style="border:solid Chocolate 2px; padding: 40px">
# 
# **Комментарий к финальной проверке**
# 
# Спасибо за доработку. Критичных комментариев нет. 
# 
# Желаю удачи в дальнейшей учебе!
# 
# </div>
# 

# In[ ]:




