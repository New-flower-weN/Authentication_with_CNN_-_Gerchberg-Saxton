import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import keras
import tensorflow_datasets as tfds

print(tf.__version__)

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = keras.utils.get_file(origin=dataset_url, extract=True) #Downloads a file from a URL if it not already in the cache.
data_dir = pathlib.Path(archive).with_suffix('')

print(archive) # prints -> C:\Users\Proku\.keras\datasets\flower_photos.tgz
print(data_dir) # prints -> C:\Users\Proku\.keras\datasets\flower_photos

'''     Пример для функции with_suffix
path = Path('/home/user/documents/report.txt')

# Удаление суффикса
new_path = path.with_suffix('')
print(new_path) # Output: /home/user/documents/report

# Изменение суффикса
new_path = path.with_suffix('.pdf')
print(new_path) # Output: /home/user/documents/report.pdf
'''

#print(data_dir) # returns path to the data folder

image_count = len(list(data_dir.glob('*/*.jpg'))) # glob -> Iterate over this subtree and yield all existing files (of any kind, including directories) matching the given relative pattern.
print(image_count)

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))

batch_size = 32
img_height = 180
img_width = 180

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

#print(train_ds) # prints <_PrefetchDataset element_spec=(TensorSpec(shape=(None, 180, 180, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names # returns all class names
print(class_names) 

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): #Метод `.take(1)` в TensorFlow (и других библиотеках, работающих с итераторами данных) извлекает один элемент (в данном случае, один батч) из набора данных `train_ds`
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

normalization_layer = keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
'''
1. `normalized_ds`: Это объект `tf.data.Dataset`, представляющий ваш нормализованный набор данных. `Dataset` в TensorFlow — это итерируемая последовательность элементов, где каждый элемент, как правило, является батчем данных (изображения и метки в вашем случае).

2. `iter(normalized_ds)`: Преобразует `Dataset` в итератор. Итератор — это объект, который позволяет получать элементы последовательности по одному за раз.

3. `next(...)`: Функция `next()` принимает итератор и возвращает следующий элемент из последовательности. В данном контексте, она вернет следующий батч из `normalized_ds`. Так как вы только что создали итератор, `next()` вернет самый первый батч данных.

Таким образом, `image_batch, labels_batch = next(iter(normalized_ds))` извлекает первый батч изображений (`image_batch`) и соответствующих меток (`labels_batch`) из вашего нормализованного набора данных `normalized_ds`. `image_batch` будет содержать тензор изображений, а `labels_batch` — тензор меток классов для этих изображений.
Если убрать `next()`, то `image_batch` и `labels_batch` будут содержать не батч данных, а сам итератор по набору данных.
'''

first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

'''
`tf.data.AUTOTUNE` — это специальное значение в TensorFlow, которое позволяет динамически настраивать параметры производительности операций ввода-вывода данных. 
Когда вы используете `AUTOTUNE`, TensorFlow автоматически определяет оптимальное количество ресурсов (например, количество параллельных потоков для предварительной обработки данных), которые следует использовать для достижения наилучшей производительности.

В контексте `tf.data`, `AUTOTUNE` чаще всего используется с методами `.map()`, `.prefetch()` и `.cache()`. Например:
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

############################# OR

def preprocess_image(image):
    # ... some preprocessing ...
    return image

train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

'''


train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

'''
`cache()`:

Эта функция кэширует (сохраняет в оперативной памяти) весь набор данных `train_ds` или `val_ds`. При первом проходе по набору данных, все данные загружаются в кэш. При последующих проходах (например, в процессе обучения на нескольких эпохах), данные берутся из кэша, а не заново считываются с диска. 
Это значительно ускоряет обработку данных, так как чтение с диска — относительно медленная операция.

Важно: Кэширование требует достаточного объема оперативной памяти. Если ваш набор данных слишком большой для доступной памяти, `cache()` может привести к ошибкам из-за нехватки памяти. 
В таком случае, лучше использовать другие методы оптимизации или разбить набор данных на меньшие части.


`prefetch(buffer_size=AUTOTUNE)`:

Эта функция выполняет предварительную загрузку (prefetch) следующего батча данных в фоновом режиме, пока текущий батч обрабатывается моделью. 
Параметр `buffer_size` определяет размер буфера для предварительной загрузки. `AUTOTUNE` — это оптимальный выбор, так как TensorFlow автоматически настраивает размер буфера, основываясь на доступных ресурсах и производительности системы.

В результате, когда модель закончит обработку текущего батча, следующий батч уже будет готов к обработке, сводя к минимуму время ожидания и повышая эффективность использования процессора и графического процессора. 
Без `prefetch()`, модель будет простаивать, ожидая загрузки следующего батча с диска или из памяти.
'''

num_classes = 5

model = keras.Sequential([
  keras.layers.Rescaling(1./255),
  keras.layers.Conv2D(32, 3, activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1)

"""                          FINER CONTROL                      """

print(str(data_dir/'*/*'))

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

'''
1. `str(data_dir/'*/*')`: Эта часть кода создает строку пути, которая указывает на все файлы во всех поддиректориях `data_dir`. 
Давайте разберем подробнее:

    * `data_dir`: Это объект `pathlib.Path`, представляющий путь к вашей директории с данными (в вашем случае, директория с распакованными изображениями цветов).

    * `/`: Это оператор объединения путей, используемый в `pathlib`.

    * `'*/*'`: Это шаблон пути (wildcard), который соответствует всем поддиректориям (`*`) и всем файлам внутри этих поддиректорий (`*`). Таким образом, `data_dir/'*/*'` собирает все файлы во всех подпапках `data_dir`.

    * `str(...)`: Преобразует объект `pathlib.Path` в строку, так как функция `tf.data.Dataset.list_files` ожидает строку в качестве аргумента.

2. `tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)`: Эта строка создает объект `tf.data.Dataset` из списка файлов, найденных по указанному пути. `shuffle=False` означает, что на этом этапе файлы *не* перемешиваются. 
Функция возвращает набор данных (`Dataset`), каждый элемент которого - это строка, представляющая путь к одному файлу изображения.

3. `list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)`: Эта строка перемешивает элементы набора данных (`list_ds`).

    * `image_count`: Это целое число, представляющее общее количество изображений в вашем наборе данных. Эта переменная используется как буфер для перемешивания, обеспечивая, что все файлы будут перемешаны.

    * `reshuffle_each_iteration=False`: Эта опция указывает, что перемешивание должно выполняться только один раз. Если бы она была `True`, то файлы перемешивались бы заново на каждой эпохе обучения. В данном случае, перемешивание выполняется только один раз, что может быть предпочтительнее, если вы хотите воспроизводимые результаты.
'''


for f in list_ds.take(5):
    print(f.numpy())

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print(class_names)

'''
* `data_dir`: Предполагается, что это объект `pathlib.Path`, представляющий собой путь к директории.

* `.glob('*')`: Это вызов метода `glob()`. `'*'` — это *шаблон* или *wildcard*, который означает "все". 
Поэтому `.glob('*')` возвращает итератор, который при итерации (проходе в цикле) будет последовательно отдавать каждый элемент (файл или поддиректория) в директории `data_dir`.


Пример:

Если `data_dir` указывает на директорию с файлами `file1.txt`, `file2.jpg`, и поддиректорией `subdir`, то `list(data_dir.glob('*'))` (преобразование итератора в список) вернет список, подобный этому:
[PosixPath('/путь/к/data_dir/file1.txt'), PosixPath('/путь/к/data_dir/file2.jpg'), PosixPath('/путь/к/data_dir/subdir')]

Важно: `glob()` возвращает итератор, а не список. Для того, чтобы увидеть содержимое, нужно либо перебрать итератор в цикле, либо преобразовать его в список (как в примере выше) с помощью `list()`. 
Однако, преобразование в список может быть неэффективно для очень больших директорий, так как весь список будет загружен в память сразу. 
Чаще всего итератор используется напрямую в цикле.
'''

val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

'''
Этот код разделяет набор данных `list_ds` на тренировочный (`train_ds`) и валидационный (`val_ds`) наборы. `skip()` и `take()` работают с итератором файлов, созданным ранее, определяя, какие файлы попадут в каждый набор.

* `val_size = int(image_count * 0.2)`: Вычисляется размер валидационного набора данных, составляющий 20% от общего числа изображений (`image_count`). 
Результат округляется до ближайшего целого числа.

* `train_ds = list_ds.skip(val_size)`: Метод `skip(val_size)` пропускает первые `val_size` элементов из набора данных `list_ds`. 
Это означает, что он исключает из `list_ds` первые 20% файлов, оставляя оставшиеся 80% для тренировочного набора. `train_ds` теперь содержит итератор, начинающийся с элемента, следующего после валидационного набора.

* `val_ds = list_ds.take(val_size)`: Метод `take(val_size)` берет первые `val_size` элементов из набора данных `list_ds`. 
Это формирует валидационный набор данных, содержащий первые 20% файлов из исходного списка.


Важно: Этот код предполагает, что файлы в `list_ds` были предварительно перемешаны (с помощью `shuffle()`), иначе разделение на тренировочный и валидационный наборы будет не случайным, а просто возьмёт первые 20% файлов и остальные 80%.
'''

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

'''
Этот код выводит количество элементов (файлов) в тренировочном (`train_ds`) и валидационном (`val_ds`) наборах данных.

* `tf.data.experimental.cardinality(train_ds)`: Эта функция определяет количество элементов в наборе данных `train_ds`. 
`tf.data.experimental` указывает на то, что эта функция находится в экспериментальном разделе TensorFlow и её API может измениться в будущем. Функция возвращает тензор, содержащий число элементов.

* `.numpy()`: Этот метод преобразует тензор, возвращенный `cardinality()`, в число Python (скалярное значение). Это необходимо для вывода на экран.
'''

def get_label(file_path):
    # Convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep) # os.path.sep -> 
    # print(file_path) # prints C:\Users\Proku\.keras\datasets\flower_photos\*\*
    # print(parts) # prints: 
                   # b'C:\\Users\\Proku\\.keras\\datasets\\flower_photos\\roses\\3873271620_1d9d314f01_n.jpg'
                   # b'C:\\Users\\Proku\\.keras\\datasets\\flower_photos\\tulips\\13513644515_a51470b899.jpg'
                   # b'C:\\Users\\Proku\\.keras\\datasets\\flower_photos\\sunflowers\\7581713708_8eae6794f2.jpg'
                   # b'C:\\Users\\Proku\\.keras\\datasets\\flower_photos\\roses\\14943194730_f48b4d4547_n.jpg'
                   # b'C:\\Users\\Proku\\.keras\\datasets\\flower_photos\\daisy\\16819071290_471d99e166_m.jpg'
    '''
    EXAMPLES FOR SPLIT():
    >>> tf.strings.split('hello world').numpy()
    array([b'hello', b'world'], dtype=object)
    >>> tf.strings.split(['hello world', 'a b c'])
    <tf.RaggedTensor [[b'hello', b'world'], [b'a', b'b', b'c']]>

    ##################################

    tf.string — это модуль в TensorFlow, который позволяет эффективно обрабатывать строковые данные внутри операций и моделей. 

    tf.strings.split(file_path, os.path.sep) конвертирует путь к файлу в список компонентов.
    Также эта функция позволяет разделить файл на пару (изображение, метка)
    '''
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names

    '''
    a = b = 1
    a = b == 1
    print(a)
    
    >>> True -> вывод
    '''

    # print(0.0, one_hot[0], 0.1, one_hot[1], 0.2, one_hot[2], 0.3, one_hot[3], 0.4, one_hot[4], sep='\n')
    # print(1, one_hot) 
    # print(2, class_names)
    ''' THESE PRINTS PRINT:

    0.0
    Tensor("strided_slice_1:0", shape=(), dtype=bool)
    0.1
    Tensor("strided_slice_2:0", shape=(), dtype=bool)
    0.2
    Tensor("strided_slice_3:0", shape=(), dtype=bool)
    0.3
    Tensor("strided_slice_4:0", shape=(), dtype=bool)
    0.4
    Tensor("strided_slice_5:0", shape=(), dtype=bool)
    1 Tensor("Equal:0", shape=(5,), dtype=bool)
    2 ['daisy' 'dandelion' 'roses' 'sunflowers' 'tulips']

    0.0
    Tensor("strided_slice_1:0", shape=(), dtype=bool)
    0.1
    Tensor("strided_slice_2:0", shape=(), dtype=bool)
    0.2
    Tensor("strided_slice_3:0", shape=(), dtype=bool)
    0.3
    Tensor("strided_slice_4:0", shape=(), dtype=bool)
    0.4
    Tensor("strided_slice_5:0", shape=(), dtype=bool)
    1 Tensor("Equal:0", shape=(5,), dtype=bool)
    2 ['daisy' 'dandelion' 'roses' 'sunflowers' 'tulips']
    '''

    # Integer encode the label
    # print(tf.argmax(one_hot)) # prints -> Tensor("ArgMax:0", shape=(), dtype=int64)
    '''
    a = [True, False, False]
    print(tf.argmax(a))
    >>> tf.Tensor(0, shape=(), dtype=int64)

    a = [True, True, False]
    print(tf.argmax(a))
    >>> tf.Tensor(0, shape=(), dtype=int64) -> возвращает первый номер первого True

    a = [False, True, False]
    print(tf.argmax(a))
    >>> tf.Tensor(1, shape=(), dtype=int64)
    '''

    return tf.argmax(one_hot) # Returns the index with the largest value across axes of a tensor

def decode_img(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    '''
    Args:
    contents -> A Tensor of type string. 0-D. The JPEG-encoded image.
    channels -> An optional int. Defaults to 0. Number of color channels for the decoded image.
    ratio -> An optional int. Defaults to 1. Downscaling ratio.
    fancy_upscaling -> An optional bool. Defaults to True. If true use a slower but nicer upscaling of the chroma planes (yuv420/422 only).
    try_recover_truncated -> An optional bool. Defaults to False. If true try to recover an image from truncated input.
    acceptable_fraction -> An optional float. Defaults to 1. The minimum required fraction of lines before a truncated input is accepted.
    dct_method -> An optional string. Defaults to "". string specifying a hint about the algorithm used for decompression. Defaults to "" which maps to a system-specific default. Currently valid values are ["INTEGER_FAST", "INTEGER_ACCURATE"]. The hint may be ignored (e.g., the internal jpeg library changes to a version that does not have that specific option.)
    name -> A name for the operation (optional).
    '''

    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
    label = get_label(file_path)
    print('first', label) # prints -> first Tensor("ArgMax:0", shape=(), dtype=int64)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path) # Reads the contents of file. This operation returns a tensor with the entire contents of the input filename. It does not do any parsing, it just returns the contents as they are. Usually, this is the first step in the input pipeline.
    print('second', img) # prints -> second Tensor("ReadFile:0", shape=(), dtype=string)
    img = decode_img(img)
    print('third', img) # prints -> third Tensor("resize/Squeeze:0", shape=(180, 180, 3), dtype=float32)
    return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

'''
Функция `configure_for_performance` оптимизирует набор данных `ds` для повышения производительности во время обучения нейронной сети. 
Она выполняет четыре ключевые операции:

1. `ds = ds.cache()`: 
Кэширует весь набор данных в памяти. При последующих итерациях (эпохах) обучения данные будут извлекаться из кэша, что значительно ускоряет процесс, так как чтение с диска занимает больше времени, чем чтение из памяти. 
Однако, это требует достаточного объема оперативной памяти. Если данных слишком много, может возникнуть ошибка `OutOfMemoryError`.

2. `ds = ds.shuffle(buffer_size=1000)`: 
Перемешивает данные в наборе. `buffer_size=1000` указывает, что для перемешивания используется буфер размером 1000 элементов. Это означает, что TensorFlow будет брать 1000 элементов, перемешивать их, и затем выводить их по одному. 
Это эффективнее, чем перемешивание всего набора данных сразу, особенно для больших наборов. Полное перемешивание происходит только один раз, в начале.

3. `ds = ds.batch(batch_size)`: 
Объединяет данные в батчи (группы) размером `batch_size`. Обработка данных в батчах повышает эффективность вычислений на GPU или TPU, поскольку операции выполняются над множеством образцов одновременно. 
`batch_size` - это гиперпараметр, который обычно устанавливается пользователем до вызова функции.

4. `ds = ds.prefetch(buffer_size=AUTOTUNE)`: 
Предварительно загружает (prefetch) следующий батч данных в фоновом режиме. `AUTOTUNE` позволяет TensorFlow автоматически оптимизировать размер буфера предварительной загрузки, что обеспечивает максимальную производительность в зависимости от аппаратных ресурсов. 
Это предотвращает простаивание модели в ожидании загрузки следующего батча.
'''

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis("off")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
    )

"""                      Using TensorFlow Datasets                          """

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
    )

num_classes = metadata.features['label'].num_classes
print(num_classes)

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

