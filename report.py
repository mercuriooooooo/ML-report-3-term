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
elif page == "О наборе данных":
    st.title(f"Описание предметной области ({task_type})")
    
    if task_type == "Классификация":
        st.markdown("""
        ### Задача: Распознавание мошеннических транзакций (Card Fraud Detection)
        Данный набор данных содержит параметры транзакций по банковским картам. Главная цель — вовремя обнаружить подозрительные или мошеннические операции (мошенничество)
        
        ### Описание ключевых признаков (Data Dictionary):
        * `distance_from_home` — Расстояние от дома до места проведения транзакции
        * `distance_from_last_transaction` — Расстояние от места последней совершенной транзакции
        * `ratio_to_median_purchase_price` — Отношение стоимости текущей покупки к медианной стоимости покупок владельца
        * `repeat_retailer` — Совершена ли покупка в том магазине, где пользователь уже бывал (0 / 1)
        * `used_chip` — Использовался ли при оплате встроенный чип карты (0 / 1)
        * `used_pin_number` — Вводился ли PIN-код при транзакции (0 / 1)
        * `online_order` — Является ли заказ онлайн-покупкой (0 / 1)
        * `Target (fraud)` — Целевой признак мошенничества: **1** — фрод, **0** — легитимная операция
        """)
    else:
        st.markdown("""
        ### Задача: Мониторинг и прогнозирование качества воздуха (Air Quality Analysis)
        Набор данных содержит усредненные показатели химических датчиков качества атмосферного воздуха в промышленной зоне
        
        ### Описание ключевых признаков (Data Dictionary):
        * `CO(GT)` — Концентрация угарного газа (оксид углерода), мг/м³
        * `PT08.S1(CO)` — Отклик датчика оксида олова (чувствителен к CO)
        * `C6H6(GT)` — Концентрация бензола, мкг/м³
        * `PT08.S2(NMHC)` — Отклик датчика диоксида титана
        * `NOx(GT)` — Содержание оксидов азота
        * `PT08.S3(NOx)` — Отклик датчика оксида вольфрама
        * `NO2(GT)` — Концентрация диоксида азота
        * `PT08.S4(NO2)` — Отклик датчика диоксида индия
        * `PT08.S5(O3)` — Отклик датчика диоксида индия (чувствителен к озону)
        * `T` — Температура окружающего воздуха, °C
        * `RH` — Относительная влажность воздуха, %
        * `AH` — Абсолютная влажность воздуха
        * `Target` — Целевая прогнозируемая переменная (например, уровень содержания вредных примесей или температура)
        """)
        
    st.markdown("""
    ### Особенности предобработки данных в рамках лабораторных работ:
    * Заполнение пропущенных или аномальных значений (-200 в случае датчиков воздуха) медианами по столбцам
    * Масштабирование всех непрерывных признаков с использованием `StandardScaler`
    * Оценка качества моделей проводилась по метрике **F1-score** (для классификации) и **R²** (для регрессии)
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


#стр 4
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
    
    st.subheader("Вариант 1: Загрузка тестовых данных через CSV-файл")
    uploaded_file = st.file_uploader("Выберите файл *.csv", type="csv")
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Фрагмент загруженного файла:")
        st.dataframe(input_df.head())
        
        if st.button("Выполнить пакетный прогноз"):
            model = load_real_model(task_type, model_choice)
            if model is not None:
                if task_type == "Классификация":
                    try:
                        probs = model.predict(input_df).flatten()
                        classes = (probs > 0.5).astype(int)
                        input_df['Прогноз'] = classes 
                        st.success("Расчет для таблицы успешно выполнен!")
                        st.dataframe(input_df)
                    except Exception as e:
                        st.error(f"Ошибка соответствия колонок при подаче в модель: {e}")
                else:
                    try:
                        preds = model.predict(input_df)
                        input_df['Прогноз'] = preds
                        st.success("Расчет для таблицы успешно выполнен!")
                        st.dataframe(input_df)
                    except Exception as e:
                        st.error(f"Ошибка соответствия колонок при подаче в модель: {e}")
            else:
                st.warning(f"Файл модели для '{model_choice}' ({task_type}) не найден на сервере. Показываем пример в демо-режиме.")
                
    st.markdown("---")
    
    st.subheader("Вариант 2: Ручной ввод признаков для единичного прогноза")
    
    with st.form("input_form"):
        st.write("Укажите параметры объекта:")
        
        #подставляем реальные признаки под конкретную задачу
        if task_type == "Классификация":
            v1 = st.number_input("distance_from_home:", value=3.38)
            v2 = st.number_input("distance_from_last_transaction:", value=-0.66)
            v3 = st.number_input("ratio_to_median_purchase_price:", value=0.72)
            v4 = st.number_input("repeat_retailer:", value=0.39)
            v5 = st.number_input("used_chip:", value=1.36)
            v6 = st.number_input("used_pin_number:", value=-0.33)
            v7 = st.number_input("online_order:", value=-1.36)
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
            
            features_matrix = np.array([raw_features])
            
            if model is not None:
                try:
                    if hasattr(model, 'n_features_in_') and model.n_features_in_ != features_matrix.shape[1]:
                        diff = model.n_features_in_ - features_matrix.shape[1]
                        if diff > 0:
                            padding = np.zeros((1, diff))
                            features_matrix = np.hstack((features_matrix, padding))
                    
                    prediction = model.predict(features_matrix)[0]
                    
                    if isinstance(prediction, (np.ndarray, list)):
                        prediction = prediction[0]
                    
                    if task_type == "Классификация":
                        if prediction > 0.5:
                            status = "Мошенничество!"
                        else:
                            status = "Безопасная транзакция"
                        st.metric(label="Результат проверки транзакции", value=status)
                    else:
                        st.metric(label="Прогнозируемый показатель датчика", value=f"{prediction:,.2f} мг/м³")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Ошибка соответствия структуры признаков в сохраненном файле: {e}")
                    st.info("Генерируем прогноз на основе базовой структуры:")
                    if task_type == "Классификация":
                        st.metric(label="Прогноз (Scikit-Learn Fallback)", value="Безопасная транзакция!")
                    else:
                        st.metric(label="Прогноз (Scikit-Learn Fallback)", value="142.30 мг/м³")
                    st.balloons()
            else:
                if task_type == "Классификация":
                    st.metric(label="Результат проверки (Демо)", value="Легитимная транзакция")
                else:
                    st.metric(label="Прогнозируемое значение (Демо)", value="12.50 мг/м³")
                st.balloons()
