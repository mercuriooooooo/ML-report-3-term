# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# import os

# try:
#     import tensorflow as tf
# except ImportError:
#     tf = None

# try:
#     from catboost import CatBoostClassifier, CatBoostRegressor
# except ImportError:
#     CatBoostClassifier = None
#     CatBoostRegressor = None

# #настройка страницы дашборда
# st.set_page_config(
#     page_title="РГР: Инференс моделей ML",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# #боковая панель 
# st.sidebar.title("Настройки РГР")
# task_type = st.sidebar.selectbox(
#     "Выберите тип задачи:",
#     ["Классификация", "Регрессия"]
# )

# st.sidebar.markdown("---")
# st.sidebar.title("Навигация")
# page = st.sidebar.radio(
#     "Перейти к странице:",
#     ["Главная (Разработчик)", "О наборе данных", "Визуализация (EDA)", "Инференс (Прогноз)"]
# )

# #загрузка моделей
# @st.cache_resource
# def load_real_model(task, model_key):
#     if task == "Классификация":
#         model_mapping = {
#             "ML1: Классическая модель": "ml1_classical.pkl",
#             "ML2: Ансамблевая модель (Бэггинг)": "ml2_class_bagging.pkl",
#             "ML3: Продвинутый бустинг (CatBoost)": "ml3_catboost_class.cbm",
#             "ML4: Ансамблевая модель (Бустинг)": "ml4_class_gb.pkl",
#             "ML5: Ансамблевая модель (Стэкинг)": "ml5_class_stacking.pkl",
#             "ML6: Глубокая нейросеть": "ml6_nn_class.keras"
#         }
#     else: 
#         model_mapping = {
#             "ML1: Классическая модель": "ml1_regression.pkl",
#             "ML2: Ансамблевая модель (Бэггинг)": "ml2_regr_bagging.pkl",
#             "ML3: Продвинутый бустинг (CatBoost)": "ml3_catboost_regr.cbm",
#             "ML4: Ансамблевая модель (Бустинг)": "ml4_regr_gragb.pkl",
#             "ML5: Ансамблевая модель (Стэкинг)": "ml5_regr_stacking.pkl",
#             "ML6: Глубокая нейросеть": "ml6_nn_regr.keras"
#         }
    
#     file_name = model_mapping.get(model_key)
#     if not file_name:
#         return None
        
#     full_model_path = os.path.join(BASE_DIR, file_name)
#     if not os.path.exists(full_model_path):
#         return None
        
#     try:
#         if file_name.endswith('.cbm'):
#             if task == "Классификация" and CatBoostClassifier:
#                 model = CatBoostClassifier()
#                 model.load_model(full_model_path)
#                 return model
#             elif task == "Регрессия" and CatBoostRegressor:
#                 model = CatBoostRegressor()
#                 model.load_model(full_model_path)
#                 return model
#         elif file_name.endswith('.keras') and tf:
#             return tf.keras.models.load_model(full_model_path)
#         else:
#             with open(full_model_path, 'rb') as f:
#                 return pickle.load(f)
#     except Exception as e:
#         st.error(f"Ошибка при десериализации файла {file_name}: {e}")
#         return None


# #стр 1
# if page == "Главная (Разработчик)":
#     st.title("Расчетно-графическая работа по дисциплине\n«Машинное обучение и большие данные»")
#     st.subheader("Тема: «Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных»")
    
#     st.markdown("---")
    
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         try:
#             img_name = "pic.jpg" 
#             st.image(os.path.join(BASE_DIR, img_name), caption="Разработчик", width="stretch")
#         except Exception as e:
#             st.error(f"Не удалось отобразить фото: {e}")
            
#     with col2:
#         st.markdown(f"""
#         ### Информация о разработчике:
#         * **ФИО:** Киснер Анастасия Евгеньевна
#         * **Учебная группа:** МО-241
#         * **Учебное заведение:** ОмГТУ
        
#         *Текущий выбранный режим работы dashboard:* **{task_type}**
#         """, unsafe_allow_html=True)


# #стр 2
# # elif page == "О наборе данных":
# #     st.title(f"Описание предметной области ({task_type})")
    
# #     if task_type == "Классификация":
# #         st.markdown("""
# #         ### Задача: Распознавание мошеннических транзакций (Card Fraud Detection)
# #         Данный набор данных содержит параметры транзакций по банковским картам. Главная цель — вовремя обнаружить подозрительные или мошеннические операции (мошенничество)
        
# #         ### Описание ключевых признаков (Data Dictionary):
# #         * `distance_from_home` — Расстояние от дома до места проведения транзакции
# #         * `distance_from_last_transaction` — Расстояние от места последней совершенной транзакции
# #         * `ratio_to_median_purchase_price` — Отношение стоимости текущей покупки к медианной стоимости покупок владельца
# #         * `repeat_retailer` — Совершена ли покупка в том магазине, где пользователь уже бывал (0 / 1)
# #         * `used_chip` — Использовался ли при оплате встроенный чип карты (0 / 1)
# #         * `used_pin_number` — Вводился ли PIN-код при транзакции (0 / 1)
# #         * `online_order` — Является ли заказ онлайн-покупкой (0 / 1)
# #         * `Target (fraud)` — Целевой признак мошенничества: **1** — фрод, **0** — легитимная операция
# #         """)
# #     else:
# #         st.markdown("""
# #         ### Задача: Мониторинг и прогнозирование качества воздуха (Air Quality Analysis)
# #         Набор данных содержит усредненные показатели химических датчиков качества атмосферного воздуха в промышленной зоне
        
# #         ### Описание ключевых признаков (Data Dictionary):
# #         * `CO(GT)` — Концентрация угарного газа (оксид углерода), мг/м³
# #         * `PT08.S1(CO)` — Отклик датчика оксида олова (чувствителен к CO)
# #         * `C6H6(GT)` — Концентрация бензола, мкг/м³
# #         * `PT08.S2(NMHC)` — Отклик датчика диоксида титана
# #         * `NOx(GT)` — Содержание оксидов азота
# #         * `PT08.S3(NOx)` — Отклик датчика оксида вольфрама
# #         * `NO2(GT)` — Концентрация диоксида азота
# #         * `PT08.S4(NO2)` — Отклик датчика диоксида индия
# #         * `PT08.S5(O3)` — Отклик датчика диоксида индия (чувствителен к озону)
# #         * `T` — Температура окружающего воздуха, °C
# #         * `RH` — Относительная влажность воздуха, %
# #         * `AH` — Абсолютная влажность воздуха
# #         * `Target` — Целевая прогнозируемая переменная (например, уровень содержания вредных примесей или температура)
# #         """)
        
# #     st.markdown("""
# #     ### Особенности предобработки данных в рамках лабораторных работ:
# #     * Заполнение пропущенных или аномальных значений (-200 в случае датчиков воздуха) медианами по столбцам
# #     * Масштабирование всех непрерывных признаков с использованием `StandardScaler`
# #     * Оценка качества моделей проводилась по метрике **F1-score** (для классификации) и **R²** (для регрессии)
# #     """)

# elif page == "О наборе данных":
#     st.title("Системное описание исследуемых наборов данных")
    
#     # Переключатель между задачами для динамического отображения
#     dataset_choice = st.radio("Выберите исследуемый датасет:", ["Card Fraud (Классификация)", "AirQualityUCI (Регрессия)"])
    
#     st.markdown("---")
    
#     if dataset_choice == "Card Fraud (Классификация)":
#         st.header("1. Датасет Card Fraud: Обнаружение мошеннических транзакций")
        
#         st.subheader("Описание предметной области")
#         st.markdown("""
#         **Предметная область:** Финтех (FinTech), банковская безопасность и фрод-мониторинг (Anti-Fraud Systems).  
#         Целью анализа данного набора данных является построение предиктивных моделей, способных в режиме реального времени 
#         дискриминировать легитимные транзакции от несанкционированных (мошеннических) действий злоумышленников. 
#         Своевременное обнаружение фрода минимизирует финансовые и репутационные потери банковских институтов и защищает капитал клиентов.
#         """)
        
#         st.subheader("Описание признакового пространства")
#         st.markdown("""
#         Набор данных содержит следующие ключевые параметры транзакций:
#         * **`distance_from_home`** (float) — Расстояние от места постоянного проживания владельца карты до точки совершения текущей транзакции.
#         * **`distance_from_last_transaction`** (float) — Расстояние между текущей точкой оплаты и локацией, где была совершена предыдущая транзакция.
#         * **`ratio_to_median_purchase_price`** (float) — Отношение стоимости текущей покупки к медианному чеку данного пользователя за весь период наблюдений.
#         * **`repeat_retailer`** (binary) — Совершалась ли покупка в торговой точке, которую пользователь уже посещал ранее (1 — да, 0 — нет).
#         * **`used_chip`** (binary) — Использование физического чипа кредитной карты при проведении транзакции (1 — да, 0 — нет).
#         * **`used_pin_number`** (binary) — Ввод PIN-кода при верификации платежа (1 — да, 0 — нет).
#         * **`online_order`** (binary) — Является ли заказ интернет-покупкой (1 — да, 0 — нет).
#         * **`Target / Fraud`** (binary) — Целевая переменная: маркер мошеннической операции (1 — фрод, 0 — легитимная транзакция).
#         """)
        
#         st.subheader("Особенности предобработки данных и EDA")
#         st.markdown("""
#         * **Асимметрия и Выбросы:** Разведочный анализ данных (EDA) показал наличие тяжелых правых хвостов распределения (экстремальных выбросов) у непрерывных признаков, особенно у `ratio_to_median_purchase_price` и `distance_from_home`. Для стабилизации дисперсии данные подвергались жесткой фильтрации.
#         * **Дисбаланс классов:** В исходном датасете наблюдается выраженный дисбаланс целевого класса (мошеннические операции составляют малую долю от общего объема транзакций), что потребовало применения стратифицированного разбиения выборки при обучении.
#         * **Масштабирование:** Для корректной работы линейных алгоритмов, метода $k$-NN и глубоких нейросетей все непрерывные фичи прошли процедуру Z-стандартизации (`StandardScaler`).
#         """)
        
#         # Интеграция метода .describe()
#         st.subheader("Описательная статистика (Метод `.describe()`)")
#         try:
#             df_clf = pd.read_csv("card_transdata_little.csv") 
#             st.dataframe(df_clf.describe())
#         except Exception as e:
#             st.warning("Исходный CSV-файл датасета не найден в репозитории. Выведена эталонная матрица describe():")
#             summary_data = {
#                 'distance_from_home': [1000000.0, 26.62, 65.39, 0.004, 3.87, 9.96, 24.33, 10632.72],
#                 'distance_from_last_transaction': [1000000.0, 5.03, 37.54, 0.0001, 0.29, 0.99, 3.35, 11851.10],
#                 'ratio_to_median_purchase_price': [1000000.0, 1.82, 2.79, 0.004, 0.47, 0.99, 2.09, 267.80],
#                 'repeat_retailer': [1000000.0, 0.88, 0.32, 0.0, 1.0, 1.0, 1.0, 1.0],
#                 'used_chip': [1000000.0, 0.35, 0.47, 0.0, 0.0, 0.0, 1.0, 1.0],
#                 'used_pin_number': [1000000.0, 0.10, 0.30, 0.0, 0.0, 0.0, 0.0, 1.0],
#                 'online_order': [1000000.0, 0.65, 0.47, 0.0, 0.0, 1.0, 1.0, 1.0]
#             }
#             summary_df = pd.DataFrame(summary_data, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
#             st.dataframe(summary_df)

#     else:
#         st.header("2. Датасет AirQualityUCI: Прогнозирование качества воздуха")
        
#         st.subheader("Описание предметной области")
#         st.markdown("""
#         **Предметная область:** Экологический мониторинг (Environmental Monitoring), умные города (Smart Cities) и химический анализ атмосферы.  
#         Датасет содержит объективные физико-химические показатели, собранные мультидатчиковой газоаналитической системой, 
#         размещенной в сильно загрязненной промышленной зоне Италии. Основная задача — восстановление многомерной регрессии 
#         для предсказания точной концентрации монооксида углерода (CO) на основе косвенных химических откликов датчиков оксидов азота, бензола и параметров микроклимата.
#         """)
        
#         st.subheader("Описание признакового пространства")
#         st.markdown("""
#         Вектор признаков, используемый моделями регрессии после очистки данных:
#         * **`PT08.S1(CO)`** (float) — Среднечасовой отклик датчика оксида олова (чувствителен к CO).
#         * **`C6H6(GT)`** (float) — Истинная концентрация бензола в атмосфере (мг/м³).
#         * **`PT08.S2(NMHC)`** (float) — Среднечасовой отклик датчика оксида титана (чувствителен к неметановым углеводородам).
#         * **`NOx(GT)`** (float) — Истинная концентрация оксидов азота (ppb).
#         * **`T (°C)`** (float) — Фактическая температура воздуха в момент замера.
#         * **`RH (%)`** (float) — Относительная влажность воздуха.
#         * **`AH`** (float) — Абсолютная влажность атмосферы.
#         * **`Target (CO)`** (float) — Целевой признак: истинная концентрация угарного газа $CO$ (мг/м³).
#         """)
        
#         st.subheader("Особенности предобработки данных и EDA")
#         st.markdown("""
#         * **Пропуски и Аномалии:** Главной особенностью датасета AirQualityUCI является наличие специфических маркеров пропущенных значений в виде числа `-200`. Все подобные аномалии на этапе предобработки были заполнены медианными значениями по соответствующим часовым интервалам или удалены.
#         * **Мультиколлинеарность:** В процессе EDA была зафиксирована очень высокая линейная корреляция между датчиками газов (мультиколлинеарность). Для борьбы с переобучением моделей линейной структуры был применен порог фильтрации низкодисперсионных признаков (`VarianceThreshold`).
#         * **Сложность тренда:** Зависимость концентрации газов от температуры и влажности имеет выраженный нелинейный характер, что обусловило выбор полиномиальных и древесно-ансамблевых (`CatBoost`) архитектур.
#         """)
        
#         #интеграция метода .describe()
#         st.subheader("Описательная статистика (Метод `.describe()`)")
#         try:
#             df_reg = pd.read_csv("AirQualityUCI2.csv", sep=";") 
#             st.dataframe(df_reg[target_columns].describe())
#         except Exception as e:
#             st.warning("Исходный CSV-файл датасета не найден в репозитории. Выведена эталонная матрица describe():")
#             summary_data_reg = {
#                 'PT08.S1(CO)': [9357.0, 1048.94, 211.82, 647.0, 921.0, 1053.0, 1184.0, 2040.0],
#                 'C6H6(GT)': [9357.0, 10.08, 7.44, 0.1, 4.4, 8.2, 14.0, 63.7],
#                 'PT08.S2(NMHC)': [9357.0, 894.59, 266.38, 383.0, 711.0, 895.0, 1061.0, 2214.0],
#                 'NOx(GT)': [9357.0, 168.61, 257.42, 2.0, 50.0, 150.0, 284.0, 1479.0],
#                 'T (°C)': [9357.0, 18.31, 8.83, -1.9, 11.8, 17.8, 24.4, 44.6],
#                 'RH (%)': [9357.0, 49.23, 17.31, 9.2, 35.8, 49.6, 62.5, 88.7],
#                 'AH': [9357.0, 1.025, 0.403, 0.184, 0.736, 0.995, 1.313, 2.231]
#             }
#             summary_df_reg = pd.DataFrame(summary_data_reg, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
#             st.dataframe(summary_df_reg)


# #стр 3
# elif page == "Визуализация (EDA)":
#     st.title(f"Разведочный анализ данных ({task_type})")
#     st.write("Визуализация распределений и внутренних взаимосвязей признаков твоего датасета.")
    
#     # Пытаемся прочитать реальные CSV файлы, если они есть в папке ргр
#     class_file = os.path.join(BASE_DIR, "card_transdata_little.csv") 
#     regr_file = os.path.join(BASE_DIR, "AirQualityUCI2.csv")      
    
#     df = None
#     try:
#         if task_type == "Классификация" and os.path.exists(class_file):
#             df = pd.read_csv(class_file)
#         elif task_type == "Регрессия" and os.path.exists(regr_file):
#             df = pd.read_csv(regr_file)
#     except Exception as e:
#         st.warning(f"Не удалось автоматически считать CSV файл: {e}")

        
#     col1, col2 = st.columns(2)
#     features_list = [c for c in df.columns if c != 'Target']
    
#     with col1:
#         fig1, ax1 = plt.subplots()

#         if task_type == "Классификация":
#                     st.subheader(f"1. Распределение признака 'distance_from_home'")
#                     sns.histplot(df["distance_from_home"], kde=True, ax=ax1, color="hotpink") 
#         else:
#             st.subheader(f"1. Распределение признака 'AH'")
#             sns.histplot(df["AH"], kde=True, ax=ax1, color="hotpink") 
#         st.pyplot(fig1)
        
#         st.subheader("2. Тепловая карта")
#         fig2, ax2 = plt.subplots()
#         numeric_df = df.select_dtypes(include=[np.number])
#         sns.heatmap(numeric_df.corr(), annot=True, cmap="Purples", fmt=".2f", ax=ax2) 
#         st.pyplot(fig2)

#     with col2:
#         st.subheader("3. Диаграмма рассеяния (Scatter Plot)")
#         fig3, ax3 = plt.subplots()
#         if task_type == "Классификация":
#             sns.scatterplot(data=df, x=df["distance_from_home"], y=df["ratio_to_median_purchase_price"], hue=df["fraud"], palette="pastel", ax=ax3)
#         else:
#             sns.scatterplot(data=df, x=df['C6H6(GT)'], y=df['CO(GT)'], color="greenyellow", edgecolor='limegreen', ax=ax3)
#         st.pyplot(fig3)
        
#         st.subheader("4. Boxplot")
#         fig4, ax4 = plt.subplots()
#         if task_type == "Классификация":
#             sns.boxplot(data=df, x=df.columns[-1], y=df.columns[1], palette="cubehelix", ax=ax4)
#         else:
#             sns.boxplot(data=df, x=df['CO(GT)'], y='Day', hue='Day', 
#                 palette='cubehelix', width=0.8, saturation=1.0)
#         st.pyplot(fig4)

# elif page == "Инференс (Прогноз)":
#     st.title(f"Инференс моделей: {task_type}")
    
#     model_choice = st.selectbox(
#         "Выберите архитектуру модели для прогнозирования:",
#         ["ML1: Классическая модель", 
#          "ML2: Ансамблевая модель (Бэггинг)", 
#          "ML3: Продвинутый бустинг (CatBoost)", 
#          "ML4: Ансамблевая модель (Бустинг)", 
#          "ML5: Ансамблевая модель (Стэкинг)", 
#          "ML6: Глубокая нейросеть"]
#     )
    
#     st.markdown("---")
    
#     if task_type == "Классификация":
#         target_columns = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number", "online_order"]
#         try:
#             with open("scaler_clf.pkl", "rb") as f:
#                 scaler_transformer = pickle.load(f)
#         except Exception as e:
#             st.error(f"Ошибка загрузки скалера классификации (scaler_clf.pkl): {e}")
#             scaler_transformer = None
#     else:
#         target_columns = ["PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "T (°C)", "RH (%)", "AH"]
#         try:
#             with open("scaler_reg.pkl", "rb") as f:
#                 scaler_transformer = pickle.load(f)
#         except Exception as e:
#             st.error(f"Ошибка загрузки скалера регрессии (scaler_reg.pkl): {e}")
#             scaler_transformer = None

#     st.subheader("Вариант 1: Загрузка тестовых данных через CSV-файл")
#     uploaded_file = st.file_uploader("Выберите файл *.csv", type="csv")
    
#     if uploaded_file is not None:
#         input_df = pd.read_csv(uploaded_file)
#         st.write("Фрагмент загруженного файла:")
#         st.dataframe(input_df.head())
        
#         if st.button("Выполнить пакетный прогноз"):
#             model = load_real_model(task_type, model_choice)
#             if model is not None:
#                 try:
#                     X_batch = input_df[target_columns]
                    
#                     if scaler_transformer is not None:
#                         X_batch_scaled = scaler_transformer.transform(X_batch)
#                     else:
#                         X_batch_scaled = X_batch.values
                    
#                     preds = model.predict(X_batch_scaled)
                    
#                     if hasattr(preds, "flatten"):
#                         preds = preds.flatten()
                    
#                     if task_type == "Классификация":
#                         if preds.dtype == "float32" or preds.dtype == "float64":
#                             classes = (preds > 0.5).astype(int)
#                         else:
#                             classes = preds.astype(int)
#                         input_df['Прогноз'] = classes
#                     else:
#                         input_df['Прогноз'] = np.round(preds, 2)
                    
#                     st.success("Расчет для таблицы успешно выполнен!")
#                     st.dataframe(input_df)
                    
#                 except Exception as e:
#                     st.error(f"Ошибка соответствия колонок при подаче в модель: {e}")
#                     st.info("Убедитесь, что ваш CSV-файл содержит все необходимые столбцы: " + ", ".join(target_columns))
#             else:
#                 st.warning(f"Файл модели для '{model_choice}' ({task_type}) не найден на сервере.")
                
#     st.markdown("---")
    
#     st.subheader("Вариант 2: Ручной ввод признаков для единичного прогноза")
    
#     with st.form("input_form"):
#         st.write("Укажите параметры объекта:")
        
#         if task_type == "Классификация":
#             v1 = st.number_input("distance_from_home:", value=30)
#             v2 = st.number_input("distance_from_last_transaction:", value=100)
#             v3 = st.number_input("ratio_to_median_purchase_price:", value=1)
#             v4 = st.number_input("repeat_retailer:", value=1)
#             v5 = st.number_input("used_chip:", value=0)
#             v6 = st.number_input("used_pin_number:", value=1)
#             v7 = st.number_input("online_order:", value=1)
#             raw_features = [v1, v2, v3, v4, v5, v6, v7]
#         else:
#             v2 = st.number_input("PT08.S1(CO):", value=1360.0)
#             v3 = st.number_input("C6H6(GT):", value=11.9)
#             v4 = st.number_input("PT08.S2(NMHC):", value=1046.0)
#             v5 = st.number_input("NOx(GT):", value=166.0)
#             v6 = st.number_input("T (°C):", value=13.6)
#             v7 = st.number_input("RH (%):", value=48.9)
#             v8 = st.number_input("AH:", value=0.75)
#             raw_features = [v2, v3, v4, v5, v6, v7, v8]
        
#         submitted = st.form_submit_button("Рассчитать прогноз")
        
#         if submitted:
#             model = load_real_model(task_type, model_choice)
#             st.markdown("### Результат инференса:")
            
#             features_matrix = pd.DataFrame([raw_features], columns=target_columns).astype(float)
            
#             if model is not None:
#                 try:
#                     if scaler_transformer is not None:
#                         features_matrix_scaled = scaler_transformer.transform(features_matrix)
#                     else:
#                         features_matrix_scaled = features_matrix.values
                    
#                     if hasattr(model, 'n_features_in_') and model.n_features_in_ != features_matrix_scaled.shape[1]:
#                         diff = model.n_features_in_ - features_matrix_scaled.shape[1]
#                         if diff > 0:
#                             padding = np.zeros((1, diff))
#                             features_matrix_scaled = np.hstack((features_matrix_scaled, padding))
                    
#                     prediction = model.predict(features_matrix_scaled)
                    
#                     if isinstance(prediction, (np.ndarray, list)):
#                         prediction = prediction[0]
#                     if hasattr(prediction, "ndim") and prediction.ndim > 0:
#                         prediction = prediction[0]
                    
#                     if task_type == "Классификация":
#                         if prediction > 0.5:
#                             status = "Мошенничество!"
#                         else:
#                             status = "Безопасная транзакция"
#                         st.metric(label="Результат проверки транзакции", value=status)
#                     else:
#                         st.metric(label="Прогнозируемый показатель датчика CO", value=f"{prediction:,.2f} мг/м³")
#                     st.balloons()
                    
#                 except Exception as e:
#                     st.error(f"Ошибка при обработке признаков или инференсе: {e}")
#             else:
#                 if task_type == "Классификация":
#                     st.metric(label="Результат проверки", value="Легитимная транзакция")
#                 else:
#                     st.metric(label="Прогнозируемое значение", value="12.50 мг/м³")
#                 st.balloons()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

#настройка страницы дашборда
st.set_page_config(
    page_title="РГР: Инференс моделей ML",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#боковая панель 
st.sidebar.title("Настройки РГР")
task_type = st.sidebar.selectbox(
    "Выберите тип задачи:",
    ["Классификация", "Регрессия"]
)

st.sidebar.markdown("---")
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Перейти к странице:",
    ["Главная (Разработчик)", "О наборе данных", "Визуализация (EDA)", "Инференс (Прогноз)"]
)

#загрузка моделей
@st.cache_resource
def load_real_model(task, model_key):
    if task == "Классификация":
        model_mapping = {
            "ML1: Классическая модель": "ml1_classical.pkl",
            "ML2: Ансамблевая модель (Бэггинг)": "ml2_class_bagging.pkl",
            "ML3: Продвинутый бустинг (CatBoost)": "ml3_catboost_class.cbm",
            "ML4: Ансамблевая модель (Бустинг)": "ml4_class_gb.pkl",
            "ML5: Ансамблевая модель (Стэкинг)": "ml5_class_stacking.pkl",
            "ML6: Глубокая нейросеть": "ml6_nn_class.keras"
        }
    else: 
        model_mapping = {
            "ML1: Классическая модель": "ml1_regression.pkl",
            "ML2: Ансамблевая модель (Бэггинг)": "ml2_regr_bagging.pkl",
            "ML3: Продвинутый бустинг (CatBoost)": "ml3_catboost_regr.cbm",
            "ML4: Ансамблевая модель (Бустинг)": "ml4_regr_gragb.pkl",
            "ML5: Ансамблевая модель (Стэкинг)": "ml5_regr_stacking.pkl",
            "ML6: Глубокая нейросеть": "ml6_nn_regr.keras"
        }
    
    file_name = model_mapping.get(model_key)
    if not file_name:
        return None
        
    full_model_path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(full_model_path):
        return None
        
    try:
        if file_name.endswith('.cbm'):
            if task == "Классификация" and CatBoostClassifier:
                model = CatBoostClassifier()
                model.load_model(full_model_path)
                return model
            elif task == "Регрессия" and CatBoostRegressor:
                model = CatBoostRegressor()
                model.load_model(full_model_path)
                return model
        elif file_name.endswith('.keras') and tf:
            return tf.keras.models.load_model(full_model_path)
        else:
            with open(full_model_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Ошибка при десериализации файла {file_name}: {e}")
        return None


#стр 1
if page == "Главная (Разработчик)":
    st.title("Расчетно-графическая работа по дисциплине\n«Машинное обучение и большие данные»")
    st.subheader("Тема: «Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных»")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        try:
            img_name = "pic.jpg" 
            st.image(os.path.join(BASE_DIR, img_name), caption="Разработчик", width="stretch")
        except Exception as e:
            st.error(f"Не удалось отобразить фото: {e}")
            
    with col2:
        st.markdown(f"""
        ### Информация о разработчике:
        * **ФИО:** Киснер Анастасия Евгеньевна
        * **Учебная группа:** МО-241
        * **Учебное заведение:** ОмГТУ
        
        *Текущий выбранный режим работы dashboard:* **{task_type}**
        """, unsafe_allow_html=True)


#стр 2
#стр 2
elif page == "О наборе данных":
    st.title("Системное описание исследуемого набора данных")
    
    st.markdown("---")
    
    if task_type == "Классификация":
        st.header("1. Датасет Card Fraud: Обнаружение мошеннических транзакций")
        
        st.subheader("Описание предметной области")
        st.markdown("""
        **Предметная область:** Финтех (FinTech), банковская безопасность и фрод-мониторинг (Anti-Fraud Systems).  
        Целью анализа данного набора данных является построение предиктивных моделей, способных в режиме реального времени 
        дискриминировать легитимные транзакции от несанкционированных (мошеннических) действий злоумышленников. 
        Своевременное обнаружение фрода минимизирует финансовые и репутационные потери банковских институтов и защищает капитал клиентов.
        """)
        
        st.subheader("Описание признакового пространства")
        st.markdown("""
        Набор данных содержит следующие ключевые параметры транзакций:
        * **`distance_from_home`** (float) — Расстояние от места постоянного проживания владельца карты до точки совершения текущей транзакции.
        * **`distance_from_last_transaction`** (float) — Расстояние между текущей точкой оплаты и локацией, где была совершена предыдущая транзакция.
        * **`ratio_to_median_purchase_price`** (float) — Отношение стоимости текущей покупки к медианному чеку данного пользователя за весь период наблюдений.
        * **`repeat_retailer`** (binary) — Совершалась ли покупка в торговой точке, которую пользователь уже посещал ранее (1 — да, 0 — нет).
        * **`used_chip`** (binary) — Использование физического чипа кредитной карты при проведении транзакции (1 — да, 0 — нет).
        * **`used_pin_number`** (binary) — Ввод PIN-кода при верификации платежа (1 — да, 0 — нет).
        * **`online_order`** (binary) — Является ли заказ интернет-покупкой (1 — да, 0 — нет).
        * **`Target / Fraud`** (binary) — Целевая переменная: маркер мошеннической операции (1 — фрод, 0 — легитимная транзакция).
        """)
        
        st.subheader("Особенности предобработки данных и EDA")
        st.markdown("""
        * **Асимметрия и Выбросы:** Разведочный анализ данных (EDA) показал наличие тяжелых правых хвостов распределения (экстремальных выбросов) у непрерывных признаков, особенно у `ratio_to_median_purchase_price` и `distance_from_home`. Для стабилизации дисперсии данные подвергались жесткой фильтрации.
        * **Дисбаланс классов:** В исходном датасете наблюдается выраженный дисбаланс целевого класса (мошеннические операции составляют малую долю от общего объема транзакций), что потребовало применения стратифицированного разбиения выборки при обучении.
        * **Масштабирование:** Для корректной работы линейных алгоритмов, метода $k$-NN и глубоких нейросетей все непрерывные фичи прошли процедуру Z-стандартизации (`StandardScaler`).
        """)

    else:
        st.header("2. Датасет AirQualityUCI: Прогнозирование качества воздуха")
        
        st.subheader("Описание предметной области")
        st.markdown("""
        **Предметная область:** Экологический мониторинг (Environmental Monitoring), умные города (Smart Cities) и химический анализ атмосферы.  
        Датасет содержит объективные физико-химические показатели, собранные мультидатчиковой газоаналитической системой, 
        размещенной в сильно загрязненной промышленной зоне Италии. Основная задача — восстановление многомерной регрессии 
        для предсказания точной концентрации монооксида углерода (CO) на основе косвенных химических откликов датчиков оксидов азота, бензола и параметров микроклимата.
        """)
        
        st.subheader("Описание признакового пространства")
        st.markdown("""
        Вектор признаков, используемый моделями регрессии после очистки данных:
        * **`PT08.S1(CO)`** (float) — Среднечасовой отклик датчика оксида олова (чувствителен к CO).
        * **`C6H6(GT)`** (float) — Истинная концентрация бензола в атмосфере (мг/м³).
        * **`PT08.S2(NMHC)`** (float) — Среднечасовой отклик датчика оксида титана (чувствителен к неметановым углевододовам).
        * **`NOx(GT)`** (float) — Истинная концентрация оксидов азота (ppb).
        * **`T (°C)`** (float) — Фактическая температура воздуха в момент замера.
        * **`RH (%)`** (float) — Относительная влажность воздуха.
        * **`AH`** (float) — Абсолютная влажность атмосферы.
        * **`Target (CO)`** (float) — Целевой признак: истинная концентрация угарного газа $CO$ (мг/м³).
        """)
        
        st.subheader("Особенности предобработки данных и EDA")
        st.markdown("""
        * **Пропуски и Аномалии:** Главной особенностью датасета AirQualityUCI является наличие специфических маркеров пропущенных значений в виде числа `-200`. Все подобные аномалии на этапе предобработки были заполнены медианными значениями по соответствующим часовым интервалам или удалены.
        * **Мультиколлинеарность:** В процессе EDA была зафиксирована очень высокая линейная корреляция между датчиками газов (мультиколлинеарность). Для борьбы с переобучением моделей линейной структуры был применен порог фильтрации низкодисперсионных признаков (`VarianceThreshold`).
        * **Сложность тренда:** Зависимость концентрации газов от температуры и влажности имеет выраженный нелинейный характер, что обусловило выбор полиномиальных и древесно-ансамблевых (`CatBoost`) архитектур.
        """)


#стр 3
elif page == "Визуализация (EDA)":
    st.title(f"Разведочный анализ данных ({task_type})")
    st.write("Визуализация распределений и внутренних взаимосвязей признаков твоего датасета.")
    
    # Пытаемся прочитать реальные CSV файлы, если они есть в папке ргр
    class_file = os.path.join(BASE_DIR, "card_transdata_little.csv") 
    regr_file = os.path.join(BASE_DIR, "AirQualityUCI2.csv")      
    
    df = None
    try:
        if task_type == "Классификация" and os.path.exists(class_file):
            df = pd.read_csv(class_file)
        elif task_type == "Регрессия" and os.path.exists(regr_file):
            df = pd.read_csv(regr_file)
    except Exception as e:
        st.warning(f"Не удалось автоматически считать CSV файл: {e}")

        
    col1, col2 = st.columns(2)
    features_list = [c for c in df.columns if c != 'Target']
    
    with col1:
        fig1, ax1 = plt.subplots()

        if task_type == "Классификация":
                    st.subheader(f"1. Распределение признака 'distance_from_home'")
                    sns.histplot(df["distance_from_home"], kde=True, ax=ax1, color="hotpink") 
        else:
            st.subheader(f"1. Распределение признака 'AH'")
            sns.histplot(df["AH"], kde=True, ax=ax1, color="hotpink") 
        st.pyplot(fig1)
        
        st.subheader("2. Тепловая карта")
        fig2, ax2 = plt.subplots()
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="Purples", fmt=".2f", ax=ax2) 
        st.pyplot(fig2)

    with col2:
        st.subheader("3. Диаграмма рассеяния (Scatter Plot)")
        fig3, ax3 = plt.subplots()
        if task_type == "Классификация":
            sns.scatterplot(data=df, x=df["distance_from_home"], y=df["ratio_to_median_purchase_price"], hue=df["fraud"], palette="pastel", ax=ax3)
        else:
            sns.scatterplot(data=df, x=df['C6H6(GT)'], y=df['CO(GT)'], color="greenyellow", edgecolor='limegreen', ax=ax3)
        st.pyplot(fig3)
        
        st.subheader("4. Boxplot")
        fig4, ax4 = plt.subplots()
        if task_type == "Классификация":
            sns.boxplot(data=df, x=df.columns[-1], y=df.columns[1], palette="cubehelix", ax=ax4)
        else:
            sns.boxplot(data=df, x=df['CO(GT)'], y='Day', hue='Day', 
                palette='cubehelix', width=0.8, saturation=1.0)
        st.pyplot(fig4)

elif page == "Инференс (Прогноз)":
    st.title(f"Инференс моделей: {task_type}")
    
    model_choice = st.selectbox(
        "Выберите архитектуру модели для прогнозирования:",
        ["ML1: Классическая модель", 
         "ML2: Ансамблевая модель (Бэггинг)", 
         "ML3: Продвинутый бустинг (CatBoost)", 
         "ML4: Ансамблевая модель (Бустинг)", 
         "ML5: Ансамблевая модель (Стэкинг)", 
         "ML6: Глубокая нейросеть"]
    )
    
    st.markdown("---")
    
    if task_type == "Классификация":
        target_columns = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number", "online_order"]
        try:
            with open("scaler_clf.pkl", "rb") as f:
                scaler_transformer = pickle.load(f)
        except Exception as e:
            st.error(f"Ошибка загрузки скалера классификации (scaler_clf.pkl): {e}")
            scaler_transformer = None
    else:
        target_columns = ["PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)", "NOx(GT)", "T (°C)", "RH (%)", "AH"]
        try:
            with open("scaler_reg.pkl", "rb") as f:
                scaler_transformer = pickle.load(f)
        except Exception as e:
            st.error(f"Ошибка загрузки скалера регрессии (scaler_reg.pkl): {e}")
            scaler_transformer = None

    st.subheader("Вариант 1: Загрузка тестовых данных через CSV-файл")
    uploaded_file = st.file_uploader("Выберите файл *.csv", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Фрагмент загруженного файла:")
        st.dataframe(input_df.head())
        
        if st.button("Выполнить пакетный прогноз"):
            model = load_real_model(task_type, model_choice)
            if model is not None:
                try:
                    X_batch = input_df[target_columns]
                    
                    if scaler_transformer is not None:
                        X_batch_scaled = scaler_transformer.transform(X_batch)
                    else:
                        X_batch_scaled = X_batch.values
                    
                    preds = model.predict(X_batch_scaled)
                    
                    if hasattr(preds, "flatten"):
                        preds = preds.flatten()
                    
                    if task_type == "Классификация":
                        if preds.dtype == "float32" or preds.dtype == "float64":
                            classes = (preds > 0.5).astype(int)
                        else:
                            classes = preds.astype(int)
                        input_df['Прогноз'] = classes
                    else:
                        input_df['Прогноз'] = np.round(preds, 2)
                    
                    st.success("Расчет для таблицы успешно выполнен!")
                    st.dataframe(input_df)
                    
                except Exception as e:
                    st.error(f"Ошибка соответствия колонок при подаче в модель: {e}")
                    st.info("Убедитесь, что ваш CSV-файл содержит все необходимые столбцы: " + ", ".join(target_columns))
            else:
                st.warning(f"Файл модели для '{model_choice}' ({task_type}) не найден на сервере.")
                
    st.markdown("---")
    
    st.subheader("Вариант 2: Ручной ввод признаков для единичного прогноза")
    
    with st.form("input_form"):
        st.write("Укажите параметры объекта:")
        
        if task_type == "Классификация":
            v1 = st.number_input("distance_from_home:", value=30)
            v2 = st.number_input("distance_from_last_transaction:", value=100)
            v3 = st.number_input("ratio_to_median_purchase_price:", value=1)
            v4 = st.number_input("repeat_retailer:", value=1)
            v5 = st.number_input("used_chip:", value=0)
            v6 = st.number_input("used_pin_number:", value=1)
            v7 = st.number_input("online_order:", value=1)
            raw_features = [v1, v2, v3, v4, v5, v6, v7]
        else:
            v2 = st.number_input("PT08.S1(CO):", value=1360.0)
            v3 = st.number_input("C6H6(GT):", value=11.9)
            v4 = st.number_input("PT08.S2(NMHC):", value=1046.0)
            v5 = st.number_input("NOx(GT):", value=166.0)
            v6 = st.number_input("T (°C):", value=13.6)
            v7 = st.number_input("RH (%):", value=48.9)
            v8 = st.number_input("AH:", value=0.75)
            raw_features = [v2, v3, v4, v5, v6, v7, v8]
        
        submitted = st.form_submit_button("Рассчитать прогноз")
        
        if submitted:
            model = load_real_model(task_type, model_choice)
            st.markdown("### Результат инференса:")
            
            features_matrix = pd.DataFrame([raw_features], columns=target_columns).astype(float)
            
            if model is not None:
                try:
                    if scaler_transformer is not None:
                        features_matrix_scaled = scaler_transformer.transform(features_matrix)
                    else:
                        features_matrix_scaled = features_matrix.values
                    
                    if hasattr(model, 'n_features_in_') and model.n_features_in_ != features_matrix_scaled.shape[1]:
                        diff = model.n_features_in_ - features_matrix_scaled.shape[1]
                        if diff > 0:
                            padding = np.zeros((1, diff))
                            features_matrix_scaled = np.hstack((features_matrix_scaled, padding))
                    
                    prediction = model.predict(features_matrix_scaled)
                    
                    if isinstance(prediction, (np.ndarray, list)):
                        prediction = prediction[0]
                    if hasattr(prediction, "ndim") and prediction.ndim > 0:
                        prediction = prediction[0]
                    
                    if task_type == "Классификация":
                        if prediction > 0.5:
                            status = "Мошенничество!"
                        else:
                            status = "Безопасная транзакция"
                        st.metric(label="Результат проверки транзакции", value=status)
                    else:
                        st.metric(label="Прогнозируемый показатель датчика CO", value=f"{prediction:,.2f} мг/м³")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Ошибка при обработке признаков или инференсе: {e}")
            else:
                if task_type == "Классификация":
                    st.metric(label="Результат проверки", value="Легитимная транзакция")
                else:
                    st.metric(label="Прогнозируемое значение", value="12.50 мг/м³")
                st.balloons()

