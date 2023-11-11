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

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Подготовка" data-toc-modified-id="Подготовка-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Подготовка</a></span></li><li><span><a href="#Анализ" data-toc-modified-id="Анализ-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Анализ</a></span></li><li><span><a href="#Обучение" data-toc-modified-id="Обучение-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Обучение</a></span><ul class="toc-item"><li><span><a href="#LGBMRegressor" data-toc-modified-id="LGBMRegressor-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>LGBMRegressor</a></span></li><li><span><a href="#CatBoost" data-toc-modified-id="CatBoost-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>CatBoost</a></span></li><li><span><a href="#RandomForestRegressor" data-toc-modified-id="RandomForestRegressor-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>RandomForestRegressor</a></span></li></ul></li><li><span><a href="#Тестирование" data-toc-modified-id="Тестирование-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тестирование</a></span></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li></ul></div>

# #  Прогнозирование заказов такси

# Компания «Чётенькое такси» собрала исторические данные о заказах такси в аэропортах. Чтобы привлекать больше водителей в период пиковой нагрузки, нужно спрогнозировать количество заказов такси на следующий час. Постройте модель для такого предсказания.
# 
# Значение метрики *RMSE* на тестовой выборке должно быть не больше 48.
# 
# Вам нужно:
# 
# 1. Загрузить данные и выполнить их ресемплирование по одному часу.
# 2. Проанализировать данные.
# 3. Обучить разные модели с различными гиперпараметрами. Сделать тестовую выборку размером 10% от исходных данных.
# 4. Проверить данные на тестовой выборке и сделать выводы.
# 
# 
# Данные лежат в файле `taxi.csv`. Количество заказов находится в столбце `num_orders` (от англ. *number of orders*, «число заказов»).

# ## Подготовка

# In[1]:


get_ipython().run_line_magic('autosave', '10')


# In[2]:


get_ipython().system(' pip install -q optuna')


# In[3]:


import pandas as pd
import time
import numpy as np
import os
import optuna
import time
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import make_scorer
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Хорошее оформление импортов! \
# Импорты собраны в одной ячейке, разделены на функциональные группы пустой строкой.</font>

# In[4]:


data = pd.read_csv('/datasets/taxi.csv', index_col=[0], parse_dates=[0])
data.head()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Здорово, что время сразу в индексе!</font>

# In[5]:


data.info()


# Проверим на пропуски в данных

# In[6]:


data.isna().sum()


# Пропусков нет

# In[7]:


print(data.index.is_monotonic)


# Данные расположены в хронологическом порядке

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Данные загружены корректно, первичный осмотр проведен.</font>

# ## Анализ

# In[8]:


data.plot()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Есть график временного ряда! </font>

# In[9]:


data = data.resample('1H').sum()
data.plot()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s>Здесь предпологается ресемплирование по сумме. Требования к метрике в RMSE > 48 расчитаны из этого предположения.</s></font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Если ресемлировать ряд по дню и вывести интервал около месяца, то можно увидеть сезонность внутри недели.</font>

# In[10]:


data_an = data['2018-03-01':'2018-03-31'].resample('1D').sum()
data_an.plot()


# In[11]:


decomposed = seasonal_decompose(data_an)


# In[12]:


plt.figure(figsize=(6, 20))
plt.subplot(311)

decomposed.trend.plot(ax=plt.gca())
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuals')
plt.tight_layout()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Нужно рассмотреть компоненты ряда на уменьшенных интервалах. Так можно увидеть особенности временного ряда , на основании которых подготовить нужные признаки.</s></font>

# <font color='blue'><b>Комментарий сутдента: </b></font> \
# <font color='purple'> Поправил</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>Здорово, нашли сезонность внутри недели!</font>

# ## Обучение

# In[13]:


def make_features(data, max_lag, rolling_mean_size):
    data['dayofweek'] = data.index.dayofweek
    for lag in range(1, max_lag +1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)
        
    data['rolling_mean'] = data['num_orders'].shift().rolling(rolling_mean_size).mean()
    
make_features(data,50,30)
print(data.head())


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s>На мой взгляд: не все признаки подходят, чисто логически. Выскажу свою позицию:<br>
# 
# <font color='red'>1) Признак год: НЕТ, однозначно не подходит. У нас данных меньше года. Поэтому будет просто константа.<br>
# 2) Признак месяц: НЕТ, аналогично году, у нас в истории всего один март, май.... Да и горизонт прогноза - часы. Поэтому сезонность времени года будет просто в тренде уже заложена.</font><br>
# <font color='darkorange'>3) Признак день: НЕТ, также не подходит, у нас нет сезонности внутри месяца, будет просто последовательность чисел.</s><br></font>
# <font color='green'>4) Признак день недели: ДА, т.к. у нас есть недельная сезонность.<br>
# 5) Признак час: ДА, т.к. есть часовая (суточная) сезонность.<br>
# 6) Признак lag: ДА, это параметр авторегрессии - учёт влияния вчерашних событий на сегодняшние.<br>
# 7) Признак rolling: ДА, это аналог тренда. Чем больше окно сглаживания, тем ровнее линия тренда.</font> <br>
# 
# <font color='red'>Как ты думаешь?</font>

# <font color='blue'><b>Комментарий сутдента: </b></font> \
# <font color='purple'> Согласен.Поправил</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'> Отлично! </font>

# In[14]:


train, valid1 = train_test_split(data, shuffle=False, test_size=0.2)
valid, test = train_test_split(valid1, shuffle=False, test_size=0.5)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Верно, что запрещаешь перемешивание отчётов.</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Во втором разбиении делятся опять полные данные</s></font>

# <font color='blue'><b>Комментарий студента: </b></font> \
# <font color='purple'> Поправил</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# In[15]:


len(test)/len(data)


# In[16]:


print(train.index.min(), train.index.max())
print(valid.index.min(), valid.index.max())
print(test.index.min(), test.index.max())


# In[17]:


train = train.dropna()


# In[18]:


# Напишем функцию определения RMSE
def rmse(actual, predictions):
    rmse = sqrt(mean_squared_error(actual, predictions))
    return rmse


# In[20]:


y = data['num_orders']
X = data.drop(columns='num_orders')


# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Это нам зачем?</font>

# <font color='blue'><b>Комментарий студента: </b></font> \
# <font color='purple'> Для Optuna потом не пригодилось</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> \
# <font color='blue'>Если будешь применять короссвалидацию, то учти что при кроссвалидации также нельзя разбивать выборки так, что бы проверочные данные были раньше обучающих. К сожалению стандартное разбиение Kfold такого условия не обеспечивает. Нужно передать специальный Сплитер - [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> \
# <font color='blue'>Если мы исследуем временной ряд, то в каждом сплите тестовые индексы должны быть позже по времени, чем обучающие, поэтому перетасовка обычным способом неуместна. Как раз TimeSeriesSplit решает эту проблему.\
# Иллюстрация работы:\
# </font>
# ![image.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_cv_indices_013.png)
# 
# Код реализации:
# ```python
# tscv = TimeSeriesSplit(n_splits=3)
# gsearch = GridSearchCV(estimator=model, cv=tscv,
#                       scoring = 'neg_root_mean_squared_error',
#                       param_grid=param_search)
# gsearch.fit(X, y)
# ```

# In[21]:


train_target = train['num_orders']
train_features = train.drop(columns='num_orders')
valid_target = valid['num_orders']
valid_features = valid.drop(columns='num_orders')
test_target = test['num_orders']
test_features = test.drop(columns='num_orders')


# In[34]:


model_LR=LinearRegression()
model_LR.fit(train_features, train_target)


# In[35]:


predictions_train = model_LR.predict(train_features)
predictions_valid = model_LR.predict(valid_features)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Оценивать модели нужно, либо кроссвлидацией, либо на валидационных данных.\
#     Тестовую выборку нужно оставить для финального тестирвания Лучшей модели.</s></font>

# <font color='blue'><b>Комментарий студента: </b></font> \
# <font color='purple'> Поправил</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# In[36]:


print("RMSE обучающей выборки:", rmse(train_target, predictions_train))
print("RMSE валидационной выборки: ", rmse(valid_target, predictions_valid))


# ### LGBMRegressor

# In[26]:


lgbm=LGBMRegressor()
lgb_params = {'boosting_type' : 'gbdt',
              'objective' : 'regression',
              'metric' : 'rmse',
              'num_iterations' : 120,
              'learning_rate' : 0.5,
              'num_leaves' : 40,
              'num_threads' : 20,
              'verbosity' : 0
             }
train_data = lgb.Dataset(train_features, 
                         train_target, 
                         )


valid_data = lgb.Dataset(valid_features, 
                         valid_target, 
                         )


# In[27]:


learned_lgd = lgb.train(lgb_params, 
          train_data, 
          valid_sets=[valid_data], 
          verbose_eval=20, 
          num_boost_round=1200)


# In[28]:


valid_pred_lgbm = learned_lgd.predict(valid_features)
print('RMSE_valid=', rmse(valid_target, valid_pred_lgbm))


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Отлично, что используешь валидационную выборку для оценки моделей.</font>

# ### CatBoost

# In[22]:


train_dataset = cb.Pool(train_features,train_target)                                                      
valid_dataset = cb.Pool(valid_features,valid_target)


# In[23]:


model_cbr = cb.CatBoostRegressor(loss_function='RMSE',  
                              eval_metric='RMSE')


# In[24]:


grid = {'learning_rate': [0.03, 0.5],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5,],
        'iterations': [50, 100, 150]}


# In[25]:


model_cbr.grid_search(grid,train_dataset)


# In[29]:


model_cbr.best_score_


# In[30]:


pred_cbr_valid = model_cbr.predict(valid_features)
print('RMSE CatBoost=', rmse(valid_target, pred_cbr_valid))


# ### RandomForestRegressor

# In[36]:


model_rnd = RandomForestRegressor(random_state = 42, n_jobs = -1)


# In[37]:


param_search = { 'n_estimators': range (10, 31, 10),
              'max_depth': range (1,13, 2),
              'min_samples_leaf': range (1,8),
              'min_samples_split': range (2,10,2) }


# In[38]:


data_wo_na = data.dropna()


# In[39]:


X_wo_na = data_wo_na.drop(columns='num_orders')
y_wo_na = data_wo_na['num_orders']


# In[40]:


tscv = TimeSeriesSplit(n_splits=3)
gsearch = GridSearchCV(estimator=model_rnd, cv=tscv,
                      scoring = 'neg_root_mean_squared_error',
                      param_grid=param_search)
gsearch.fit(X_wo_na, y_wo_na)


# In[41]:


gsearch.best_estimator_


# In[ ]:


# зададим распрпделение параметров для optuna
param_distributions_randomforest_optuna = {
    'n_estimators': optuna.distributions.IntDistribution(10,1000,10),
    'max_depth': optuna.distributions.IntDistribution(2,40,1),
    'min_samples_split': optuna.distributions.IntDistribution(20,60,5),
    'min_samples_leaf': optuna.distributions.IntDistribution(1, 201, 10),
    'max_features': optuna.distributions.CategoricalDistribution(['sqrt','log2']),
    'bootstrap' : optuna.distributions.CategoricalDistribution([True, False])}


# In[42]:


best_model_rnd = gsearch.best_estimator_
best_model_rnd.fit(train_features, train_target)


# In[43]:


gsearch.best_score_


# In[44]:


pred_RFR_valid = best_model_rnd.predict(valid_features)
print('RMSE_RandomForrestRegressor =', rmse(valid_target, pred_RFR_valid))


# CatBoost выдает лучший результат. Проверим на тестовой выборке

# ## Тестирование

# In[32]:


test_pred_cbr = model_cbr.predict(test_features)
print('RMSE_test =', rmse(test_target, test_pred_cbr))


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Тестирование проведено верно.</font>

# На тестовой выборке получили RMSE = 10.85, что сильно меньше 48, думаю что это победа)

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Очень приятно видеть вывод в конце проекта! В выводах можно приводить полученные ранее значения. Также можно расписать все, что было сделано в ходе проведения работы.</font>

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>\
# <font color='green'> Сергей, хороший проект получился!
# Большое спасибо за проделанную работу. Видно, что приложено много усилий.
# </font>
# 
# <font color='blue'>Что нужно исправить:</font>
# <ul><font color='red'>Проведи рессемплирование по сумме.</font></ul>
# <ul><font color='red'>Рассмотри компоненты на уменьшенных интервалах.</font></ul>
# <ul><font color='red'>Не используй признаки "год" и "месяц" на данных меньше года.</font></ul>
# <ul><font color='red'>Попарвь разбиение на выборки.</font></ul>
# <ul><font color='red'>Не оценивай модели на тестовых данных.</font></ul>
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

# <font color='green'>Дополнительные материалы:\
# Е.Рябенко - Прогнозирование временных рядов https://www.youtube.com/watch?v=u433nrxdf5k \
# Б.Б. Демешев - временные ряды https://disk.yandex.ru/i/LiDHB-B3A6Lz5A \
# Канторович - Анализ временных рядов https://yadi.sk/i/IOkUOS3hTXf3gg \
# 
# Базовое применение ARIMA - https://colab.research.google.com/drive/17RnG91Eq8JBKyxToNzvCvjibfxum-oPj?usp=sharing \
# Prophet модель от известной компании https://facebook.github.io/prophet/ \
# https://facebook.github.io/prophet/docs/quick_start.html#python-api \
# https://nbviewer.jupyter.org/github/miptgirl/habra_materials/blob/master/prophet/habra_data.ipynb \
# Или [Предсказываем будущее с помощью библиотеки Facebook Prophet](https://habr.com/ru/company/ods/blog/323730/)
# </font> 

# ## Чек-лист проверки

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Данные загружены и подготовлены
# - [x]  Данные проанализированы
# - [x]  Модель обучена, гиперпараметры подобраны
# - [x]  Качество моделей проверено, выводы сделаны
# - [x]  Значение *RMSE* на тестовой выборке не больше 48
