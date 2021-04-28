#добавляем необхоимые библиотеки
import os # Модуль функций для работы с операционной системой, не зависящие от используемой операционной системы
import collections #Модуль специализированных типов данных, на основе словарей , кортежей , множеств , списков
from collections import Counter
from collections import deque
import math # Библиотека математических функций
import pandas as pd # Библиотека pandas
import numpy as np # Библиотека работы с массивами
import matplotlib.pyplot as plt # Отрисовка изображений
import matplotlib.pyplot as plt  # $ pip install matplotlib
import matplotlib.animation as animation
import random


import pickle
import time
from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from tensorflow.keras.models import Sequential # НС прямого распространения
from tensorflow.keras.layers import Dense, Activation, Dropout, SpatialDropout1D, BatchNormalization, Embedding, Flatten # Основные слои
from tensorflow.keras import utils # Утилиты для to_categorical
from tensorflow.keras.preprocessing import image # Для отрисовки изображения
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop # Алгоритмы оптимизации, для настройки скорости обучения
#from tensorflow.keras.models import load_model # загрузка сохраненных моделей
from tensorflow.keras.preprocessing.sequence import pad_sequences # Модуль для возврата списка дополненных последовательностей

from sklearn.preprocessing import LabelEncoder, StandardScaler # Функции для нормализации данных
from sklearn import preprocessing # Пакет предварительной обработки данных
from sklearn.model_selection import train_test_split

import random, gc
from IPython.display import clear_output
