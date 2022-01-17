from Util import *

img_size = 224

#source_path = "D:/IVE/source/cat"
if(len(sys.argv)>0):
  source_path = sys.argv[0]



for dirs in os.listdir(source_path):
  print(dirs)
  dirname = dirs
  base_path = source_path+"/"+dirname
  file_list = sorted(os.listdir(base_path))
  random.shuffle(file_list)
  
  dataset = {
  'filename':[],
  'imgs': [],
  'lmks': [],
  'bbs': []
  }

  for f in file_list:
    if '.cat' not in f:
      continue

    # if '00000001_000.jpg' not in f:
    #   continue

    # read landmarks
    pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
    landmarks = (pd_frame.values[0][1:-1]).reshape((-1, 2))

    # load image
    img_filename, ext = os.path.splitext(f)

    img = cv2.imread(os.path.join(base_path, img_filename))

    # resize image and relocate landmarks
    img, ratio, top, left = resize_img(img,img_size)
    landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)
    bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

    dataset['imgs'].append(img)
    dataset['lmks'].append(landmarks.flatten())
    dataset['bbs'].append(bb.flatten())
    dataset['filename'].append(img_filename)

    # for l in landmarks:
    #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    # plt.imshow(img)
    # plt.show()


  np.save('dataset/%s.npy' % dirname, np.array(dataset))








