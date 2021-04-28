from projectLib import *
from projectFunc import *


path = 'base/base_id'
path2 = 'models'



allLoadData, classes = write_file(path)
nClasses = len(classes)
allIdForVoc, lenSumID, sumGroup, sumGroupInClass = make_array(allLoadData, classes)
allIdInClass = make_id(allLoadData, classes)

#print("=========================================================================================")
#print('Общее количество групп симптомов по всем классам -', sumGroup,
#      'Общее количество всех симптомов по всем классам -', len(allIdForVoc))

vocabulary = compute_tfidf(allIdInClass)



newVocabulary = {}
for i in range(len(vocabulary)):
    for keys, values in vocabulary[i].items():
        if(values != 0):
            newVocabulary[keys] = int(round(values, 5) * 100000)

conceptIndexes = []

for i in range(len(sumGroupInClass)):
    conceptIndexes.append(concept2Indexes(allIdInClass[i], newVocabulary))
#    print(i, classes[i], len(conceptIndexes[i]))

xTrainIndex = []
xTestIndex = []

for i in range(len(conceptIndexes)):

    (xTrain, yTrain, xTest, yTest) = createTestsClasses(conceptIndexes[i], i, 0.8)
    xTrainIndex.append(xTrain)
    xTestIndex.append(xTest)

#print('=== Общее количество ================')
#for i in range(len(conceptIndexes)):
#    print('Количество симптомов в классе - ', len(conceptIndexes[i]))

#print('=== Обучающая выборка =============')
#for i in range(len(xTrainIndex)):
#    print('Количество симптомов в классе - ', len(xTrainIndex[i]))

#print('=== Тестовая выборка ================')
#for i in range(len(xTestIndex)):
#    print('Количество симптомов в классе - ', len(xTestIndex[i]))

allIdcount = len(vocabulary)
xLen = 50
step = 1
libs = {'vocabulary': vocabulary, 'classes': classes, 'xLen': xLen}
with open(path2 + '/model_best220421.pickle', 'wb') as outfile:
      pickle.dump(libs, outfile)


(xTrain, yTrain) = createSetsMultiClasses(xTrainIndex, xLen, step)
xTrain01 = changeSetTo01(xTrain, allIdcount)

(xVal, yVal) = createSetsMultiClasses(xTestIndex, xLen, step)
xVal01 = changeSetTo01(xVal, allIdcount)

#print(xTrain01.shape)
#print(yTrain.shape)
#print(xVal01.shape)
#print(yVal.shape)


def build_model_bow():

      model = Sequential()
      model.add(Dense(800, input_dim=allIdcount, activation="relu"))

      model.add(Dense(800, activation="relu"))
      model.add(Dense(800, activation="relu"))
      model.add(Dropout(0.25))
      model.add(BatchNormalization())

      model.add(Dense(10, activation='softmax'))

      model.compile(optimizer=RMSprop(lr=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

      return model

model_BOW = build_model_bow()
#mcp_save = ModelCheckpoint('/content/drive/My Drive/Neuro/NLP/models/model_BOW_best220421.hdf5', save_best_only=True, monitor='val_loss', mode='min')
#model_BOW.fit(xTrain01, yTrain, epochs=40, batch_size=64, callbacks=[mcp_save], validation_data=(xVal01, yVal))
#model_BOW.summary()

history = model_BOW.fit(xTrain01, yTrain, epochs=20, batch_size=64, validation_data=(xVal01, yVal))
y = range(len(history.history['val_accuracy']))
xdat = []
ydat = []

def draw_fig(i):

    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    #plt.legend()
    #plt.ylim(0, 20)
    plt.title('NLP')
    plt.grid(True)

    xdat.append(history.history['val_accuracy'][i])
    ydat.append(y[i])
    plt.plot(ydat, xdat, color='black', linewidth=2)


for i in range(len(y)):
    draw_fig(i)
    plt.pause(.1)

plt.show()