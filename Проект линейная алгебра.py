#!/usr/bin/env python
# coding: utf-8

# <h1> Комментарий ревьюера </h1>
# 
# Привет, Сергей!
# Меня зовут Виктория Ильина, и я буду проверять твой проект. Предлагаю общаться на "ты", но если тебе будет не комфортно, обязательно сообщи и мы перейдем на "вы". 
# 
# Моя основная цель - не указать на совершённые тобой ошибки, а поделиться своим опытом и помочь тебе стать специалистом по Data Science. Ты уже проделал большую работу над проектом, но давай сделаем его ещё лучше. Ниже ты найдёшь мои комментарии - пожалуйста, не перемещай, не изменяй и не удаляй их. Я буду использовать различные цвета, чтобы было легче воспринимать мои комментарии:
# 
# <div class="alert alert-block alert-danger">
# <b>Комментарий ревьюера❌:</b> Так выделены самые важные замечания. Без их отработки проект не будет принят.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Комментарий ревьюера🤔:</b> Так выделены небольшие замечания. Одно-два таких замечания в проекте допустимы, но если их будет больше - тебе будет необходимо внести исправления. Однако, если перед нами стоит цель не просто сдать проект и забыть, а приобрести новые знания и успешно применять их в будущем, я бы всё-таки рекомендовала тебе прорабатывать эти недочёты.
# </div>
#     
# <div class="alert alert-block alert-success">
# <b>Комментарий ревьюера👍:</b> Так я выделю все остальные комментарии.
# </div>
#     
# Давай работать над проектом в диалоге: если ты что-то меняешь в проекте или отвечаешь на мои комментарии - пиши об этом. Мне будет легче отследить изменения, если ты выделишь свои комментарии:
# 
# <div class="alert alert-block alert-info">
# <b>Комментарий студента:</b> Например, вот так.
# </div>
# 
# Давай посмотрим, что получилось!

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Загрузка-данных" data-toc-modified-id="Загрузка-данных-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Загрузка данных</a></span></li><li><span><a href="#Умножение-матриц" data-toc-modified-id="Умножение-матриц-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Умножение матриц</a></span></li><li><span><a href="#Алгоритм-преобразования" data-toc-modified-id="Алгоритм-преобразования-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Алгоритм преобразования</a></span></li><li><span><a href="#Проверка-алгоритма" data-toc-modified-id="Проверка-алгоритма-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Проверка алгоритма</a></span><ul class="toc-item"><li><span><a href="#Вывод:-Перемножение-на-обратимую-матрицу-не-влияет-на-качество-предсказания-линейной-регрессии." data-toc-modified-id="Вывод:-Перемножение-на-обратимую-матрицу-не-влияет-на-качество-предсказания-линейной-регрессии.-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Вывод: Перемножение на обратимую матрицу не влияет на качество предсказания линейной регрессии.</a></span></li></ul></li><li><span><a href="#Чек-лист-проверки" data-toc-modified-id="Чек-лист-проверки-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Чек-лист проверки</a></span></li></ul></div>

# # Защита персональных данных клиентов

# Вам нужно защитить данные клиентов страховой компании «Хоть потоп». Разработайте такой метод преобразования данных, чтобы по ним было сложно восстановить персональную информацию. Обоснуйте корректность его работы.
# 
# Нужно защитить данные, чтобы при преобразовании качество моделей машинного обучения не ухудшилось. Подбирать наилучшую модель не требуется.

# ## Загрузка данных

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[2]:


data = pd.read_csv('/datasets/insurance.csv')


# In[3]:


print(data.head())


# Проверим есть ли пропущенные значения в нашей табличке

# In[4]:


data.isna().sum()


# Пропущенных данных не выявлено приступим к расчетам

# Разделим выборку на признаки и целевой признак

# In[5]:


features = data.drop('Страховые выплаты', axis=1)
target = data['Страховые выплаты']


# <div class="alert alert-block alert-success">
# <b>Комментарий ревьюера👍:</b> Хорошо, продолжаем
# </div>

# ## Умножение матриц

# В этом задании вы можете записывать формулы в *Jupyter Notebook.*
# 
# Чтобы записать формулу внутри текста, окружите её символами доллара \\$; если снаружи —  двойными символами \\$\\$. Эти формулы записываются на языке вёрстки *LaTeX.* 
# 
# Для примера мы записали формулы линейной регрессии. Можете их скопировать и отредактировать, чтобы решить задачу.
# 
# Работать в *LaTeX* необязательно.

# Обозначения:
# 
# - $X$ — матрица признаков (нулевой столбец состоит из единиц)
# 
# - $y$ — вектор целевого признака
# 
# - $P$ — матрица, на которую умножаются признаки
# 
# - $w$ — вектор весов линейной регрессии (нулевой элемент равен сдвигу)

# Предсказания:
# 
# $$
# a = Xw
# $$
# 
# Задача обучения:
# 
# $$
# w = \arg\min_w MSE(Xw, y)
# $$
# 
# 
# Формула обучения:
# 
# $$
# w = (X^T X)^{-1} X^T y
# $$

# Учитывая следующие свойства обратных и обратимых матриц
# $$
# (A^T)^{-1} = (A^{-1})^T
# $$
# 
# $$
# (AB)^{-1} = B^{-1}A^{-1}
# $$
# 
# $$
# (AB)^T = B^TA^T
# $$
# 
# 
# 
# 
# 
# 

# Из этих свойств следует, что раскрывая скобки
# $$
# w = (X^-{1})^T X^{-1} X^T y
# $$

# А вот формула получения w через умножение на обратную матрицу:
#     $$
#     w = (XP^-{1})^T XP^{-1} XP^T y
#     $$  
#     $$
#     w = ((XP^{-1})^T (XP))^{-1} (XP)^T y
#     $$
#     $$
#     w = (P^T X^T XP)^{-1} (XP)^T y
#     $$
#     $$
#     w = (P^TWP)^{-1} (XP)^T y
#     $$
#     $$
#     w = P^{-1}W^{-1}(P^T)^{-1}P^TX^Ty = P^{-1}W^{-1}X^Ty
#     $$
#     $$
#     xPw = xPP^{-1}W^{-1}X^Ty = xW^{-1}X^Ty
#     $$
#     $$
#     xPw = x(X^TX)^{-1}X^Ty = xw
#     $$

# **Ответ:** При замене координат XA=X', где A обратимая матрица качество предсказания линейной регрессии не изменится
# 
# **Обоснование:** Свойства обратимых и обратных матриц

# <div class="alert alert-block alert-danger">
# <b>Комментарий ревьюера❌:</b> Мы домножаем матрицу признаков на случайную обратимую матрицу, т.е. домножение происходит справа
#     
# Так же мне нехватило применения формул выше, пожалуйста, домножь и раскрой скобки    
# </div>

# <div class="alert alert-block alert-success">
# <b>Комментарий ревьюера V.2👍:</b> Теперь верно =)
# </div>

# In[6]:


X = features
y = target
X_matrix = features.values
print(X_matrix)


# In[7]:


w0 = y.mean()


# In[8]:


print(w0)


# In[9]:


class LinearRegression:
    def fit(self, train_features, train_target):
        X = np.concatenate((np.ones((train_features.shape[0], 1)), train_features), axis=1)
        y = train_target
        w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.w = w[1:]
        self.w0 = w[0]
        
    def predict(self, test_features):
        return test_features.dot(self.w)+self.w0


# In[10]:


model = LinearRegression()
model.fit(features, target)
predictions = model.predict(features)
print(predictions.head)
print(r2_score(target, predictions))


# ## Алгоритм преобразования

# **Алгоритм**

# In[11]:



while True:
    P = np.random.rand((X.shape[1]), (X.shape[1]))
    P_inv = np.linalg.pinv(P)
    is_product_eye = np.allclose(np.dot(P, P_inv), np.eye(P.shape[0]))
    if is_product_eye:
        break
print(P)
    


# In[12]:


secret_features = X.dot(P)
unsecret_features = secret_features.dot(P_inv)
print(secret_features)
print(unsecret_features)


# **Обоснование**
# 
# Создаем случайную матрицу проверяем ее на обратимость и сохраняем в переменной P. Произведение которой с основными признаками находим после.

# ## Проверка алгоритма

# In[15]:


model = LinearRegression()
model.fit(X, target)
predictions = model.predict(X)
print(predictions.head)
print(r2_score(target, predictions))


# In[16]:


model = LinearRegression()
model.fit(secret_features, target)
predictions = model.predict(secret_features)
print(predictions.head)
print(r2_score(target, predictions))


# ### Вывод: Перемножение на обратимую матрицу не влияет на качество предсказания линейной регрессии.

# <div class="alert alert-block alert-success">
# <b>Комментарий ревьюера👍:</b> Верно, таким образом в случае домножения матрицы признаков на случайную обратимую матрицу можно обеспечить защиту данных, не особо потеряв в качестве модели. (Незначительное отличие вызвано особенностью преобразования матриц и чисел с плавающей точкой, это норма https://docs-python.ru/tutorial/operatsii-chislami-python/problemy-chisel-plavajuschej-zapjatoj/)
# </div>

# ## Чек-лист проверки

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Выполнен шаг 1: данные загружены
# - [x]  Выполнен шаг 2: получен ответ на вопрос об умножении матриц
#     - [x]  Указан правильный вариант ответа
#     - [x]  Вариант обоснован
# - [x]  Выполнен шаг 3: предложен алгоритм преобразования
#     - [x]  Алгоритм описан
#     - [x]  Алгоритм обоснован
# - [x]  Выполнен шаг 4: алгоритм проверен
#     - [x]  Алгоритм реализован
#     - [x]  Проведено сравнение качества моделей до и после преобразования

# <div class="alert alert-block alert-danger">
# <b>Комментарий ревьюера❌:</b> Спасибо тебе за проделанную работу, к сожалению, сейчас я не могу принять твой проект, поэтому придётся ещё немного поработать - исправить ошибку в доказательстве.
#     
# До скорой встречи, если у тебя есть какие-то вопросы, то обязательно пиши =)
#     
# </div>

# <div class="alert alert-block alert-success">
# <b>Комментарий ревьюера V.2👍:</b> Дополнительные материалы:
# 
# Обзоры по применению криптографии в машинном обучении: https://arxiv.org/pdf/1902.04109.pdf, https://www.researchgate.net/publication/342746786_Some_Applications_of_Machine_Learning_in_Cryptography
# 
# Небольшой обзор на Шифр Хилла: https://habr.com/ru/post/595281/
#     
# Ещё пара статей на эту тему:
#     
# -	https://medium.com/intuit-engineering/machine-learning-on-encrypted-data-no-longer-a-fantasy-58e37e9f31d7
# -	https://towardsdatascience.com/cryptography-crash-course-for-the-intimidated-892b7889528a
# -	https://towardsdatascience.com/homomorphic-encryption-machine-learning-new-business-models-2ba6a4f185d
# 
#    
# Примеры использования линейной алгебры в машинном обучении:
# https://machinelearningmastery.com/examples-of-linear-algebra-in-machine-learning/
#     
# Для чего нужна математика в машинном обучении?
# https://proglib.io/p/dlya-chego-nuzhen-matematicheskiy-analiz-v-mashinnom-obuchenii-2021-02-15
# 
# Большая статья по векторам:
# https://neptune.ai/blog/understanding-vectors-from-a-machine-learning-perspective  
# 
# Можно ещё дополнительно посмотреть лекции от МФТИ https://www.youtube.com/playlist?list=PLk4h7dmY2eYHHTyfLyrl7HmP-H3mMAW08
#     
# Классная серия роликов по линейной алгебре на английском (можно включить автоматические русские субтитры): https://www.youtube.com/playlist?list=PLRDl2inPrWQW1QSWhBU0ki-jq_uElkh2a
# 
# 
# Удачи в будущих проектах и работе, всё обязательно получится!
# </div>

# In[ ]:




