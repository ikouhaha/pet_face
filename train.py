from Util import *

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

dir_path = os.path.dirname(os.path.realpath(__file__))
source_path = os.path.abspath(os.path.join(dir_path, os.pardir))
data_path = dir_path+"/dataset"

img_size = 224

mode = 'bbs' # [bbs, lmks]
if mode == 'bbs':
  output_size = 4
elif mode == 'lmks':
  output_size = 18



start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

dataList = []
imgs = []
modes = []

for path in os.listdir(data_path):
  if path.startswith('CAT_'):
    print(path)
    data = readNPY(data_path+"/"+path)
    getImg = data.item().get('imgs')
    getMode = data.item().get(mode)
    imgs.extend(getImg)
    modes.extend(getMode)
  
all_images = np.array(imgs)
all_modes = np.array(modes)

  
x_train, x_test, y_train, y_test = train_test_split(all_images, all_modes, test_size=.2, shuffle=False)
#test = np.concatenate((data_00.item().get('imgs'),data_01.item().get('imgs'),data_02.item().get('imgs'),data_03.item().get('imgs'),data_04.item().get('imgs'),data_05.item().get('imgs'),data_06.item().get('imgs')), axis=0)


print('Training', x_train.shape[0])
print('Validation', x_test.shape[0])
#print('Test', test_data.shape[0])


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (-1, img_size, img_size, 3))
x_test = np.reshape(x_test, (-1, img_size, img_size, 3))

y_train = np.reshape(y_train, (-1, output_size))
y_test = np.reshape(y_test, (-1, output_size))

inputs = Input(shape=(img_size, img_size, 3))

mobilenetv2_model = mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
net = Dense(64, activation='relu')(net)
net = Dense(output_size, activation='linear')(net)

model = Model(inputs=inputs, outputs=net)

model.summary()

# training
model.compile(optimizer=keras.optimizers.Adam(), loss='mse')


model.fit(x_train, y_train, epochs=50, batch_size=32, shuffle=True,
  validation_data=(x_test, y_test), verbose=1,
  callbacks=[
    TensorBoard(log_dir='logs/%s' % (start_time)),
    ModelCheckpoint('./models/%s.h5' % (start_time), monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto')
  ]
)
