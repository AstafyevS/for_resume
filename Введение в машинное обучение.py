#!/usr/bin/env python
# coding: utf-8

# <div style="border:solid green 2px; padding: 20px">
# <b>Привет, Сергей!</b>
# 
# Меня зовут Александр Пономаренко, и я буду проверять твой проект. Предлагаю общаться на «ты» :) Но если это не удобно - дай знать, и мы перейдем на "вы". 
# 
# Моя основная цель — не указать на совершенные тобою ошибки, а поделиться своим опытом и помочь тебе стать data science. Ты уже проделал большую работу над проектом, но давай сделаем его еще лучше. Ниже ты найдешь мои комментарии - **пожалуйста, не перемещай, не изменяй и не удаляй их**. Увидев у тебя ошибку, в первый раз я лишь укажу на ее наличие и дам тебе возможность самой найти и исправить ее. На реальной работе твой начальник будет поступать так же, а я пытаюсь подготовить тебя именно к работе аналитиком. Но если ты пока не справишься с такой задачей - при следующей проверке я дам более точную подсказку. Я буду использовать цветовую разметку:
# 
# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> Так выделены самые важные замечания. Без их отработки проект не будет принят. </div>
# 
# <div class="alert alert-warning">
# <b>Комментарий ревьюера ⚠️:</b> Так выделены небольшие замечания. Я надеюсь, что их ты тоже учтешь - твой проект от этого станет только лучше. Но настаивать на их отработке не буду.
# 
# </div>
# 
# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Так я выделяю все остальные комментарии.</div>
# 
# Давай работать над проектом в диалоге: **если ты что-то меняешь в проекте или отвечаешь на мои комменатри — пиши об этом.** Мне будет легче отследить изменения, если ты выделишь свои комментарии:
# <div class="alert alert-info"> <b>Комментарий студента:</b> Например, вот так.</div>
# 
# Всё это поможет выполнить повторную проверку твоего проекта оперативнее. 

# # Исследование рекомендации пользователям

# ## Откроем файл с данными и изучим его

# In[3]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


# In[4]:


df = pd.read_csv('/datasets/users_behavior.csv')

display(df)


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Да, хорошо, данные на месте:)
# 
# 
# P.S Только импорт библиотек и чтение из файла лучше разделить. Для читаемости и удобства 
# </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Done</div>

# В данном случае целевым признаком является 'is_ultra'

# In[5]:


#Извлечем из данных выборку и целевой признак
features = df.drop('is_ultra', axis=1)
target = df['is_ultra']


# ## Разделим исходные данные на обучающую, валидационную и тестовую выборки.

# In[6]:


features_train, features_testing, target_train, target_testing = train_test_split(features, target, test_size=0.4, random_state=12345)
features_test, features_valid, target_test, target_valid = train_test_split(features_testing, target_testing, test_size=0.5, random_state=12345)


# ## Исследуем качество разных моделей, меняя гиперпараметры.

# Использовать будем 3 типа моделей:
# 
# -DecisionTreeClassifier
# 
# -RandomTreeClassifier
# 
# -LogisticRegression

# In[12]:


# Проведем исследования для DecisionTreeClassifier
best_model_tree = None
best_result_tree = 0
for depth in range(1, 50):
    model_tree = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    model_tree.fit(features_train, target_train)
    predictions = model_tree.predict(features_valid)
    result_tree = accuracy_score(target_valid, predictions)
    if result > best_result_tree:
        best_model_tree = model_tree
        best_result_tree = result_tree
        
print('Максимальное качество модели типа DecisionTreeClassifier:', best_result_tree)
print('Лучшие параметры модели:', best_model_tree)


# <div class="alert alert-danger">
#     <b>Комментарий ревьюера ❌:</b> Подправь, пожалуста, названия моделей </div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Done</div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Да, так лучше
# 
# 
# </div>

# In[10]:


# Проведем исследования для LogisticRegression
best_model_log = None
best_result_log = 0
for iter in range(500, 100000, 1000):
    model_log = LogisticRegression(random_state=12345, solver = 'lbfgs', max_iter = iter)
    model_log.fit(features_train, target_train)
    predictions = model_log.predict(features_valid)
    result_log = accuracy_score(target_valid, predictions)
    if result > best_result_log:
        best_model_log = model_log
        best_result_log = result_log
        
print('Максимальное качество модели типа LogisticRegression:', best_result_log)
print('Лучшие параметры модели:', best_model_log)


# In[11]:


# Проведем исследования для RandomForestClassifier
best_model_forest = None
best_result_forest = 0
for est in range(10, 101, 10):
    for depth in range(1000, 10000, 1000):
        model_forest = RandomForestClassifier(random_state=12345, max_depth = depth, n_estimators = est)
        model_forest.fit(features_train, target_train)
        predictions = model_forest.predict(features_valid)
        result_forest = accuracy_score(target_valid, predictions)
    if result > best_result_forest:
        best_model_forest = model_forest
        best_result_forest = result_forest
        
print('Максимальное качество модели типа RandomForestClassifier:', best_result_forest)
print('Лучшие параметры модели:', best_model_forest)


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> 
# 
# 
# ```max_depth``` глубина дерева. Чем оно глубже, тем больше в нем разбиений и оно собирает больше информации о данных. обычно использует не супер большую глубину ( от 1 до 32 порой хватает, может быть более) 
# </div>

# ### Лучше всех показала себя модель RandomForestClassifier с параметрами (max_depth=9000, n_estimators=10, random_state=12345)

# ## Проверим качество моделей на тестовой выборке

# In[14]:


b_pred_forest = best_model_forest.predict(features_test)
result_forest = accuracy_score(target_test, b_pred_forest)
print('Качество модели RandomForestClassifier на тестовой выборке:', result_forest)


# <div class="alert alert-danger">
# <b>Комментарий ревьюера ❌:</b> Да, огонь. Только мы проводим тестирование для самой лучшей модели, тем самым имитируя продакшен</div>

# <div class="alert alert-info"> <b>Комментарий студента:</b> Удалил лишнее</div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Да, хорошо
# 
# 
# </div>

# ### А с тестовой выборокой модели справились одинаково

# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b>Когда мы говорим о проверке на вменяемость, мы имеем ввиду то, что наша обученная модель работает лучше, чем что-то простое. В случае с задачей классификации чем-то простым может быть, например, случайное угадывание класса или, как в нашем случае, всегда предсказывание большего класса. Если метрика на нашей модели хуже, чем на простой, то очевидно наша модель не годится. В качестве метрики в этой задаче мы используем accuracy - долю правильных ответов. Если мы будем на всех элементах предсказывать 0, то accuracy будет равно доле нулей в столбце is_ultra. То есть суть проверки на вменяемость в нашем случае - это убедиться, что accuracy нашей модели больше, чем доля нулей в is_ultra.
# 
# Статья по Dummy model:
# +  https://towardsdatascience.com/why-using-a-dummy-classifier-is-a-smart-move-4a55080e3549
# 
# </div>
# 
# 

# In[7]:


#КОД РЕВЬЮЕРА
# from sklearn.dummy import DummyClassifier
# dummy = DummyClassifier(strategy='most_frequent', random_state=123).fit(features_train, target_train)
# print('Accuracy Dummy-модели на тестовой выборке:', accuracy_score(target_test, dummy.predict(features_test)))


# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b> Оставлю немного дополнительного материала, все это в дальнейшем вы будете изучать:
# 
# +  https://towardsdatascience.com/cross-validation-430d9a5fee22
# +  https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85
# 
# А вот неплохой учебник:
# +  https://academy.yandex.ru/handbook/ml - можешь глянуть там главу Оценка качества моделей -> кросс валидация;)
# 
# 
# </div>

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>
# <div class="alert alert-success">
# <b>Комментарий ревьюера ✔️:</b>Сергей, получился хороший проект! Подправь чуть чуть и я приму проект:)
#     
# Если есть  если есть какие либо вопросы я с удовольствием на них отвечу:) <br> Исправь, пожалуйста, замечания и жду проект на следующую проверку:) </div>

# <div class="alert alert-success">
# <b>Комментарий ревьюера V2✔️:</b> Удачи в следующих проектах!!! Пожалуйста, пиши выводы в конце проекта, хотя бы в кратце
# 
# 
# </div>
