#!/usr/bin/env python
# coding: utf-8

# <font color='blue' size=5><b>Комментарий ревьюера</b></font>
# 
# <font color='blue'>Привет, Сергей! Меня зовут Павел Григорьев, и я буду проверять этот проект.<br>Моя основная цель - не указать на совершённые тобой ошибки, а поделиться своим опытом и помочь тебе совершенствоваться как профессионалу.<br>Спасибо за проделанную работу! Предлагаю общаться на «ты».</font>
# <details>
# 	<summary><u>Инструкция по организационным мометнам (кликабельно)</u></summary>
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
# 

# <font color='blue' size=3><b>Комментарий ревьюера 2</b></font>
# 
# <font color='blue'>Привет еще раз. Спасибо, за исправления. Оформление комментариев по работе сохраняется. Только обозначим, что это вторая итерация.</font> 

# <font color='blue' size=3><b>Комментарий ревьюера 3</b></font>
# 
# <font color='blue'>Привет еще раз. Спасибо, за исправления. Оформление комментариев по работе сохраняется.</font> 

# # Определение стоимости автомобилей

# Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 
# 
# Заказчику важны:
# 
# - качество предсказания;
# - скорость предсказания;
# - время обучения.

# ## Подготовка данных

# In[1]:


get_ipython().run_line_magic('autosave', '10')


# !pip install lightgbm

# In[2]:


get_ipython().system(' pip install -q optuna')


# In[3]:


#Загрузим необходимые библиотеки
import pandas as pd
import numpy as np
import seaborn as sns
import os
import optuna
import time
import catboost as cb
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from math import sqrt


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Хорошее оформление импортов! \
# Импорты собраны в одной ячейке, сгрупированы по функциональности. Можно ещё разделить группы пустой строкой.</font>

# In[4]:


data = pd.read_csv('/datasets/autos.csv')
data.head(50)


# In[5]:


data.info()


# ### Обработаем пропуски в данных

# In[6]:


data.isna().sum()


# Итого получили:
# 1. Около 10 процентов пропусков по типу автомобильного кузова.
# 2. 6 процентов пропущенных данных по коробке
# 3. 6 процентов пропущено по модели
# 4. 10 процентов пропущено по типу топлива
# 5. И у целых 20 процентов пропущена информация о ремонте. (при продаже скорее всего не хотят сообщать эту информацию сами владельцы)

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Данные загружены корректно, первичный осмотр проведен.</font>

# In[7]:


#Заполним пропуски в столбце с типом автомобильного кузова самым частно встречающимся значением
data['VehicleType'].value_counts()


# In[8]:


data['VehicleType'] = data['VehicleType'].fillna('sedan')


# In[9]:


#Заполним пропуски по типу топлива предыдущим значением
data['FuelType'] = data['FuelType'].fillna(method='ffill')


# In[10]:


#Заполним пропуски по ремонту значением 'no'
data['Repaired'] = data['Repaired'].fillna('no')


# In[11]:


#Удалим пропуски в данных менее 10 процентов
data = data.dropna(subset=['Gearbox', 'Model'])


# In[12]:


data.isna().sum()


# In[13]:


data.info()


# В общей сложности потеряли чуть менее 10 процентов данных

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'> Пропуски заполнены достаточно безопасно.
# </font>

# In[14]:


data.duplicated().sum()


# Удалим признаки которые не повлияют на цену автомобиля.

# In[15]:


data = data.drop(columns=['PostalCode','LastSeen','NumberOfPictures','DateCrawled','DateCreated'])
data.head()


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'> Отлично!</font>

# Найдем нереальные значения

# In[16]:


sns.boxplot(data['Price'])


# Удалим данные больше 15000 и меньше 100

# In[17]:


data = data.loc[data['Price'] < 15000]
data = data.loc[data['Price'] >100]


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ⚠️\
# <font color='darkorange'> Обрати внимание на цену = 0.</font>

# In[18]:


sns.boxplot(data['Power'])


# Удалим все что больше 2500

# In[19]:


data = data.loc[data['Power'] < 2500]


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# In[20]:


sns.boxplot(data['Kilometer'])


# С километрами вроде все норм

# In[21]:


sns.boxplot(data['RegistrationYear'])


# In[22]:


data = data.loc[data['RegistrationYear'] < 2023]
data = data.loc[data['RegistrationYear'] > 1885]


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ⚠️\
# <font color='darkorange'> Обрати внимание на даты в DateCrawled в сравнении с RegistrationYear</font>

# In[23]:


data.info()


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Нужно обрботать нереальные значения в количественных признаках.</s></font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s>ё нужно решить, а все ли признаки несут информацию о цене авто. Лишний шум усложняет работу модели.</s></font>

# <font color='purple'> Сделано.</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# Задублировались 4 значения не будем их удалять или менять на качество моделей не повлияет

# In[24]:


# Разобьтем выборку на features и target
features = data.drop(columns='Price')
target = data['Price']
X = features
y = target


# In[25]:


X.size


# In[26]:


X.info()


# In[27]:


y.size


# ### Приведение выборки к числовым данным

# In[28]:


cat_features = ['VehicleType', 'Gearbox', 'Model', 'FuelType', 'Brand', 'Repaired']


# In[29]:


# Обработаем выборку по методу One-Hot-Encoder
Xo = pd.get_dummies(features, drop_first=True)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'>Хороший инструмент для быстрого анализа, но для новых данных если набор категорий будет отличаться, получить данные с такой-же размерностью будет нельзя. Для возможности работы модели с новыми данными нужен OneHotEncoder.</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Энкодер (как и Скалер) нельзя обучать на полных данных.\
# Для вычисления трансформаций они используют информацию о всех поданных в них данных и обучение на полных данных приводит к утечке из обучающей выборки во все остальные. Тестовые и валидационные выборки можно только трансформировать, предобработчиками обученными на train.</s></font>

# In[30]:


#Подготовим данны для Ordinal Encoder
features_ordinal = features.drop(columns=['Model','Brand'])


# In[31]:


# Разобьем features и target на валидационную, обучающую, а затем и тестовую выборки
X_or_train, X_or_valid, y_or_train, y_or_valid = train_test_split(features_ordinal, target, test_size=0.5, random_state=12345)
X_or_valid, X_or_test, y_or_valid, y_or_test = train_test_split(X_or_valid,y_or_valid, test_size=0.3, random_state=12345)


# In[32]:


# Обработаем выборку OrdinalEncoder
ordinal = OrdinalEncoder()
ordinal.fit(X_or_train)
features_or_train = pd.DataFrame(ordinal.transform(X_or_train),columns=X_or_train.columns)
print(features_or_train.head())


# In[33]:


features_or_valid = pd.DataFrame(ordinal.set_params(handle_unknown='use_encoded_value', unknown_value=666).transform(X_or_valid),columns=X_or_valid.columns)
features_or_test =  pd.DataFrame(ordinal.set_params(handle_unknown='use_encoded_value', unknown_value=666).transform(X_or_test),columns=X_or_test.columns)


# In[34]:


features_or_train.info()
y_or_train.size


# In[35]:


features_or_valid.info()
y_or_valid.size


# In[36]:


features_or_test.info()
y_or_test.size


# 
# ## Обучение моделей

# ### Подготовка переменных и формул

# In[37]:


# Напишем формулу определения RMSE
def rmse(actual, predictions):
    rmse = sqrt(mean_squared_error(actual, predictions))
    return rmse


# In[38]:


# Напишем скорер
scorer = make_scorer(rmse, greater_is_better=False)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ⚠️\
# <font color='darkorange'> Для кроссвалидации есть готовая метрика 'neg_root_mean_squared_error'. Метрики можно выбирать [здесь](https://scikit-learn.org/stable/modules/model_evaluation.html) </font>

# ### Работаем с LightGBM

# In[39]:


kf = KFold(n_splits=3, shuffle=False)
for train_index, valid1_index in kf.split(X):
    print(train_index)
    print(valid1_index)
    


# In[40]:


valid_index, test_index = np.array_split(valid1_index, 2)


# In[41]:


valid_index


# In[42]:


test_index


# <font color='purple'><b>Комментарий студента: </b></font> \
# <font color='red'> Как раз здесь разбиваем фичи на 3 выборки. Тренировочную, валидационную и тестовую.</font>

# In[43]:


for feature in cat_features:
            X[feature] = pd.Series(X[feature], dtype="category")


# In[44]:



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
train_data = lgb.Dataset(X.iloc[train_index], 
                         y.iloc[train_index], 
                         categorical_feature=cat_features, 
                         free_raw_data=False)


valid_data = lgb.Dataset(X.iloc[valid_index], 
                         y.iloc[valid_index], 
                         categorical_feature=cat_features, 
                         free_raw_data=False)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>LGBM может сам кодировать категории, если их перевести в тип данных 'category', но эта кодровка занимает дополнительное время. Если кодировать OE, то получатся быстрее.</font>

# In[45]:


start = time.time()
learned_lgd = lgb.train(lgb_params, 
          train_data, 
          valid_sets=[valid_data], 
          verbose_eval=20, 
          categorical_feature=cat_features, 
          num_boost_round=1200)

end = time.time()
study_lgbm = end-start
print('Время выполнения', study_lgbm, 'c')


# In[46]:


valid_pred_lgbm = learned_lgd.predict(X.iloc[valid_index])
print('RMSE_valid=', (rmse(y.iloc[valid_index], valid_pred_lgbm)))


# In[47]:


start = time.time()
test_pred_lgbm = learned_lgd.predict(X.iloc[test_index])
end = time.time()
lgbm_time_test = (end-start)
print('Время выполнения', lgbm_time_test, 'c')


# In[48]:


print('RMSE=', (rmse(y.iloc[test_index], test_pred_lgbm)))


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Тестовую выборку нужно оставить для финального тестирования. Для выбора лучшей модели и подбора гиперпараметров нужна ещё одна выборка - валидационная. Именно по ней оцениваются предсказания на "конкурс моделей". Или можно использовать кроссвалидацию, там валидационная выборка генерится автоматически.</s></font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'><s>Отлично, что используешь валидационную выборку для оценки моделей.</s></font>

# <font color='Purple'><b>Комментарий студента: </b></font> \
# <font color='red'> Отдельно специально прописал предикт и на валидационной и на тестовой выбороке. И так как ранне написал необходимо провести тестирование всех моделей на тестовых выборках не вижу смысла переносить отдельно именно этот код в другой раздел в итоговом разделе также указана работа моделей на тестовых выборках. По возможности проверьте поскорее остаю уже очень. Заранее спасибо.</font>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ❌\
# <font color='blue'>Всему своё время и место. Сначала нужно оценить модели на валидационных данных, затем выбрать Лучшую и только затем проводить тесты.\
# Можно, конечно, сразу посчитать предикты для тестовывх данных, но тогда надо чётко понимать, что оценка должна быть по Валидационному предикту, не по Тестовому.\
# У тебя ранжирование моделей проводится по тестам на тестовых данных - это ошибка!</font>

# ### CAtBoostRegressor

# In[49]:


train_dataset = cb.Pool(X.iloc[train_index],y.iloc[train_index], 
                        cat_features=cat_features)                                                      
valid_dataset = cb.Pool(X.iloc[valid_index],y.iloc[valid_index],           
                       cat_features=cat_features)


# In[50]:


model_cbr = cb.CatBoostRegressor(loss_function='RMSE',  
                              eval_metric='RMSE')


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Верно, CatBoost может сам кодировать категории из типа данных 'object' и 'category', но эта кодровка занимает дополнительное время. Если кодировать OE, то получатся быстрее.</font>

# In[51]:


grid = {'learning_rate': [0.03, 0.5],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5,],
        'iterations': [50, 100, 150]}


# In[52]:


start=time.time()
model_cbr.grid_search(grid,train_dataset)
end=time.time()
study_cb=end-start
print('Время выполнения', study_cb,'c')


# In[53]:


model_cbr.best_score_


# In[54]:


start=time.time()
pred_cbr = model_cbr.predict(X.iloc[test_index])
print('RMSE_Caboost =', rmse(y.iloc[test_index], pred_cbr))
end=time.time()
pred_cb=end-start
print('Время выполнения', pred_cb,'c')


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# ### RandomForestRegressor

# In[55]:


model_rnd = RandomForestRegressor(random_state = 42, n_jobs = -1)


# In[56]:


# зададим распрпделение параметров для optuna
param_distributions_randomforest_optuna = {
    'n_estimators': optuna.distributions.IntDistribution(10,1000,10),
    'max_depth': optuna.distributions.IntDistribution(2,40,1),
    'min_samples_split': optuna.distributions.IntDistribution(20,60,5),
    'min_samples_leaf': optuna.distributions.IntDistribution(1, 201, 10),
    'max_features': optuna.distributions.CategoricalDistribution(['sqrt','log2']),
    'bootstrap' : optuna.distributions.CategoricalDistribution([True, False])}


# In[57]:


opt_search_rnd = optuna.integration.OptunaSearchCV(estimator=model_rnd,
                                                       param_distributions=param_distributions_randomforest_optuna, 
                                                       cv=15 , 
                                                       n_trials=100, 
                                                       scoring=scorer, 
                                                       timeout=10, 
                                                       random_state=42)


# In[58]:



opt_search_rnd.fit(features_or_train, y_or_train)


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'> Деревянные модели не нуждаются в OHE кодировании категорий и могут работать с OE (OrdinalEncoding). OHE сильно повышает размерность признаков, что требует дополнительных вычислительных ресурсов (и времени).</font>

# <font color='Purple'><b>Комментарий студента: </b></font> \
# <font color='red'> Сделал.</font>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ✔️\
# <font color='green'> Отлично!  👍</font>

# In[59]:


opt_search_rnd_score = opt_search_rnd.best_score_


# In[60]:


opt_search_best_model_rnd = opt_search_rnd.best_params_


# In[61]:


print('Best model =', opt_search_best_model_rnd)
print('RMSE =', opt_search_rnd_score*-1)


# In[62]:


start=time.time()
best_model_rnd = RandomForestRegressor(n_estimators = 79, max_depth= 23, min_samples_split= 21, min_samples_leaf= 3, max_features= 'sqrt', bootstrap= True)
best_model_rnd.fit(features_or_train, y_or_train)
end=time.time()
study_rfr=end-start
print('Время выполнения', study_rfr,'c')


# In[63]:


pred_RFR_valid = best_model_rnd.predict(features_or_valid)
print('RMSE_RandomForrestRegressor =', rmse(y_or_valid, pred_RFR_valid))


# In[64]:


y_or_test


# In[65]:


features_or_test.info()


# In[66]:



start=time.time()
pred_RFR_test = best_model_rnd.predict(features_or_test)
print('RMSE_RandomForrestRegressor =', rmse(y_or_test, pred_RFR_test))
end=time.time()
pred_rfr=end-start
print('Время выполнения', pred_rfr,'c')


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# ### DecisionTreeRegressor

# In[67]:


model_dtr = DecisionTreeRegressor()


# In[68]:


grid_params={'max_depth':[3,5,10],
              'max_features':[1,3,5,7],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[2,3]
           }


# In[69]:


grid = GridSearchCV(model_dtr,param_grid=grid_params,cv=3,scoring=scorer)

model_grid = grid.fit(Xo.iloc[train_index],y.iloc[train_index])


# In[70]:


print('Best hyperparameters are: '+str(model_grid.best_params_))
print('Best score is: '+str(model_grid.best_score_*-1))


# In[71]:


start=time.time()
best_dtr = DecisionTreeRegressor(max_depth= 10, max_features= 5, min_samples_leaf= 1, min_samples_split= 2)
best_dtr.fit(Xo.iloc[train_index],y.iloc[train_index])
end=time.time()
study_DTR=end-start
print('Время выполнения', study_DTR,'c')


# In[72]:



start=time.time()
best_pred_dtr = best_dtr.predict(Xo.iloc[test_index])
print('RMSE_DecisionTreeRegressor =', rmse(y.iloc[test_index], best_pred_dtr))
end=time.time()
pred_DTR=end-start
print('Время выполнения', pred_DTR,'c')


# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>👍</font>

# ### LinearRegression

# In[73]:


model_lr = LinearRegression()


# In[74]:


start=time.time()
model_lr.fit(Xo.iloc[train_index],y.iloc[train_index])
end=time.time()
study_lin=end-start
print('Время выполнения', study_lin,'c')


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'> Кодирование признков соответсвует модели!</font>

# In[75]:


pred_lr_train = model_lr.predict(Xo.iloc[train_index])
print('RMSE_train =', rmse(y.iloc[train_index], pred_lr_train))


# In[76]:


start=time.time()
pred_lr_test = model_lr.predict(Xo.iloc[test_index])
print('RMSE_LR =', rmse(y.iloc[test_index], pred_lr_test))
end=time.time()
pred_lr=end-start
print('Время выполнения', pred_lr,'c')


# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Линейные модели нуждаются в OHE кодировке категорий.</s></font>

# <font color='purple'> Поправил.</font>

# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>👍</font>

# ## Анализ моделей

# Мы получили следующие результаты работы моделей в порядке убывания:

# In[79]:


print('RMSE_LGBM =', (rmse(y.iloc[test_index], test_pred_lgbm)))
print('RMSE_Catboost =', rmse(y.iloc[test_index], pred_cbr))
print('RMSE_RandomForrestRegressor =', rmse(y_or_test, pred_RFR_test))
print('RMSE_DecisionTreeRegressor =', rmse(y.iloc[test_index], best_pred_dtr))
print('RMSE_LinearRegression =', rmse(y.iloc[test_index], pred_lr_test))


# При этом время обучения моделей составляет следующие значения:

# In[80]:


print('LGBM:', study_lgbm)
print('CatBoost :',study_cb)
print('RandomForestRegressor :',study_rfr)
print('DecisionTreeRegressor :',study_DTR)
print('LinearRegressor :',study_lin)


# А время предсказания выглядит так:

# In[81]:


print('LGBM:', lgbm_time_test)
print('Catboost :', pred_cb)
print('RandomForestRegressor :', pred_rfr)
print('DecisionTreeRegressor :', pred_DTR)
print('LinearRegressor :', pred_lr)


# <font color='blue'><b>Комментарий ревьюера 2: </b></font> ✔️\
# <font color='green'>Классно, что время обучения и предсказания измеряется отдельно.</font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Заказчику важны:
# качество предсказания;
# скорость предсказания;
# время обучения.\
# Для оценки и сравнения моделей нам понадобится время обучения модели (без подбора параметров) и время предсказания. Их надо сохранить по отдельности.</s></font>

# <font color='blue'><b>Комментарий ревьюера: </b></font> ❌\
# <font color='red'><s> Нужно провести тестирование моделей на тестовых данных.</s></font>

# <font color='Purple'><b>Коментарий студента: </b></font> \
# <font color='red'> Оно проведено. Причем для каждой модели.</font>

# <font color='blue'><b>Комментарий ревьюера 3: </b></font> ❌\
# <font color='blue'> Тестирование проводится для подтверждения результата и контроля переобучения моделей при оптимизации на валидации. Здесь нечего подтверждать, тест не может подтвердить сам себя.</font>

# Хуже всего с задачей справился DecisionTreeRegressor с показателем RMSE=3389, лучше всего с задачей предсказания стоимости cправился ансамбль CatBoost с показателем RMSE=1359, но как и ожидалость на обучение его потребовалось больше всего времени t=6минут 4 секунды. Быстрее всех обучился DecisionTreeRegressor t=0.068 секунды. Но ее предсказания далеки от идеальных, RMSE=3497. Время предсказания на тестовых данных у всех моделей меньше секунды. Причем быстрее всех справился Catboost : t=0.068 с, а дольше всех предсказывал RandomForrestRegressor: t=0.767 с.

# <font color='blue'><b>Комментарий ревьюера: </b></font> ✔️\
# <font color='green'>Очень приятно видеть вывод в конце проекта! Приведены ответы на главные вопросы проекта.</font>

# <font color='blue'><b>Итоговый комментарий ревьюера 3</b></font>\
# <font color='green'>Сергей, проект принят! \
# Все этапы пройдены. \
# Надеюсь, тебе понравился процесс выполнения и результат.</font> \
# <font color='blue'><b>Спасибо, удачи в освоении профессии!</b></font>

# <font color='blue'><b>Итоговый комментарий ревьюера 2</b></font>\
# <font color='green'> Сергей, большое спасибо за доработку. Удалены неинформативные признаки, почищены выбросы, время обучения и работы моделей измерено отдельно.
# </font>
# 
# <font color='blue'>Что нужно исправить:</font>
# <ul><font color='red'>Попробуй обучить деревья в OE кодированием категорий. Скорость от этого должна улучшиться.</font></ul>
# <ul><font color='red'>Если используешь предикт для сравнения моделей выдели для этого отдельную выборку, или оценивай кроссвалидацией.</font></ul>
# 
# <font color='blue'>Что можно сделать лучше:</font>
# <font color='orange'>В работе я оставил несколько советов. Буду рад, если ты учтешь их.</font></ul>
# 
# <font color='blue'><b>Жду новую версию проекта :)</b></font>

# <font color='Purple'><b>Комментарий студента: </b></font> \
# <font color='red'> Для каждой модели представлены результаты именно на тестовой выборке.</font>

# <font color='blue'><b>Итоговый комментарий ревьюера</b></font>\
# <font color='green'> Сергей, хороший проект получился!
# Большое спасибо за проделанную работу. Видно, что приложено много усилий.
# </font>
# 
# <font color='blue'>Что нужно исправить:</font>
# <ul><font color='red'>Обработай нерельные значения в количественных признаках.</font></ul>
# <ul><font color='red'>Если среди признаков имеются неинформативные, удалите их.</font></ul>
# <ul><font color='red'>Измеряй отдельно время обучения и время предсказания моделей. </font></ul>
# <ul><font color='red'>Не обучай Энкодер на всех данных.</font></ul>
# <ul><font color='red'>Используй кодирование категорий адекватное моделям.</font></ul>
# <ul><font color='red'>Проведи тестирование на тестовых данных.</font></ul>
# 
# <font color='blue'>Что можно сделать лучше:</font>
# <font color='orange'>В работе я оставил несколько советов. Буду рад, если ты учтешь их.</font></ul>
# 
# <font color='blue'><b>Жду новую версию проекта :)</b></font>

# <font color='blue'><b>Комментарий ревьюера : </b></font> ✔️\
# <font color='green'> Мы, с коллегами, решили сделать небольшое дополнение к курсу по этому спринту.  https://colab.research.google.com/drive/1_gAMXcQKoCShB_l8FNtYEejMnosm9mvt?usp=sharing с примерами кодировки признаков разного уровня сложности. Посмотри, пожалуйста. Это может сильно помочь с предобработкой данных для моделей.</font>

# ## Чек-лист проверки

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Выполнена загрузка и подготовка данных
# - [x]  Выполнено обучение моделей
# - [x]  Есть анализ скорости работы и качества моделей
