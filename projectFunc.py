import matplotlib.pyplot as plt

from projectLib import *

def write_file(path):
    loaded_data = []
    allLoadData = []
    classes = []


    for filename in os.listdir(path):

        loaded_data = np.load(path + '/' + filename, allow_pickle=True)
        classes.append(filename.replace(".npy", ""))
        allLoadData.append(loaded_data)
        #print('Файл прочитался: ', filename.replace(".npy", ""))

    #for i in tqdm(range(10)):
    #    time.sleep(0.1)


    return allLoadData, classes

def make_array(allLoadData, classes):
    sumGroupInClass = []
    sumIdInClass = []
    allIdForVoc = []
    sumGroup = 0
    sumId = 0
    sumAllId = 0
    lenSumID = []

    for i in range(len(allLoadData)):
        for num in allLoadData[i]:
            allId = []
            for sym in num:
                allId.append(sym)
                allIdForVoc.append(sym)
            sumId += len(allId)
            sumIdInClass.append(allId)

        sumGroupInClass.append(sumIdInClass)
        #print(f'В классе {classes[i]}: количество групп симптомов - {len(sumIdInClass)}, количество симптомов - {sumId}')

        sumAllId += sumId
        sumGroup += len(sumGroupInClass[i])
        lenSumID.append(sumId)

        sumIdInClass = []
        sumId = 0

    return allIdForVoc, lenSumID, sumGroup, sumGroupInClass


def make_id(allLoadData, classes):
    allIdInClass = []
    allIdForVoc = []
    allSum = 0

    for i in range(len(allLoadData)):
        allNum = []
        for num in allLoadData[i]:
            for sym in num:
                allNum.append(sym)
                allIdForVoc.append(sym)

        allIdInClass.append(allNum)

        # print(newAllLoadData[i])
        #print('Общее количество симптомов в классе ', classes[i], ' - ', len(allIdInClass[i]))
        allSum += len(allIdInClass[i])
    #print('Общее количество симптомов по всем классам -', allSum)

    return allIdInClass

def compute_tfidf(corpus):
    def compute_tf(text):
        tf_text = Counter(text)

        for i in tf_text:
            tf_text[i] = tf_text[i]/float(len(text))

        return tf_text

    def compute_idf(word, corpus):
        return math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))

    documents_list = []

    for text in corpus:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpus)
            documents_list.append(tf_idf_dictionary)

    return documents_list


def concept2Indexes(concepts, vocabulary):
    '''
      concept2Indexes - Функция создание индексов концептов
      вход:
        concepts - концепты
        vocabulary - словарь концептов
        maxConceptCount - максимальное количество всех концептов словаря
      выход:
        список индексов всх концептов

    '''
    conceptsIndexes = []

    for concept in concepts:
        if (concept in vocabulary):
            conceptsIndexes.append(vocabulary[concept])

    return conceptsIndexes

# Формирование обучающей выборки по листу индексов концептов (разделение на короткие векторы)
def getSetFromIndexes(conceptIndexes, xLen, step):
    xTrain = []
    conceptLen = len(conceptIndexes)
    index = 0
    while (index + xLen <= conceptLen):
        xTrain.append(conceptIndexes[index:index + xLen])
        index += step
    return xTrain


# Формирование обучающей и проверочной выборки выборки из 10 листов индексов от 10 классов
def createSetsMultiClasses(conceptIndexes, xLen, step):

    nClasses = len(conceptIndexes)
    classesXTrain = []
    for cI in conceptIndexes:  # Для каждого из 10 классов
        classesXTrain.append(getSetFromIndexes(cI, xLen, step))  # Создаём обучающую выборку из индексов

    xTrain = []  # Формируем один общий xTrain
    yTrain = []

    for t in range(nClasses):
        xT = classesXTrain[t]
        for i in range(len(xT)):
            xTrain.append(xT[i])

        currY = utils.to_categorical(t, nClasses)  # Формируем yTrain по номеру класса
        for i in range(len(xT)):
            yTrain.append(currY)

    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    return (xTrain, yTrain)

# Преобразование одного короткого вектора в вектор из 0 и 1 # По принципу words bag
def changeXTo01(trainVector, conceptsCount):
    out = np.zeros(conceptsCount)
    for x in trainVector:
        out[x] = 1
    return out

# Преобразование выборки (обучающей или проверочной) к виду 0 и 1 # По принципу words bag
def changeSetTo01(trainSet, conceptsCount):
    out = []
    for x in trainSet:
        out.append(changeXTo01(x, conceptsCount))
    return np.array(out)

# Преобразование одного короткого вектора в вектор из 0 и 1 # По принципу words bag с множественным вхождением
def changeXTo01Multi(trainVector, conceptsCount):
    out = []
    for x in trainVector:
        out[x] += 1
    return out

# Преобразование выборки (обучающей или проверочной) к виду 0 и 1 # По принципу words bag с множественным вхождением
def changeSetTo01Multi(trainSet, conceptsCount):
    out = []
    for x in trainSet:
        out.append(changeXTo01Multi(x, conceptsCount))
    return np.array(out)


def createTestsClasses(allIndexes, i, train_size):
    # Формируем общий xTrain и общий xTest
    X_train, X_test, y_train, y_test = np.array(train_test_split(allIndexes, np.ones(len(allIndexes), 'int') * (i + 1), train_size=train_size))

    return (X_train, y_train, X_test, y_test)

def plot_history(history):
  '''
  plot_history - функция отрисовки истории обучения сети
    вход:
      history - история обучения сети
    выход:
      график хода обучения по loss и lav_loss
  '''
  plt.figure(figsize=(20,10))
  plt.title('График ошибки нейронной сети')
  plt.plot(history.history['loss'], label='Ошибка на тренировочном наборе')
  plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
  plt.legend()
  plt.xlabel('Эпоха обучения')
  plt.ylabel('Значение ошибки')
  plt.show()

def plot_symptom(lenSumID, classes):
  plt.axis([0, 10, 0, 1500])
  plt.grid(True)
  color_rectangle = np.random.rand(10, 3)
  y_pos = np.arange(len(classes))


  plt.bar(y_pos, lenSumID, align='center', color=color_rectangle, alpha=0.75)
  plt.xticks(y_pos, classes, rotation=90)
  plt.xlabel('Количество классов')
  plt.ylabel('Количество симптомов')

  plt.show()