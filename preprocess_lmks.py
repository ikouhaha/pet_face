from Util import *


img_size = 224

dataset = {
  'imgs': [],
  'lmks': [],
  'bbs': []
}

for dirs in os.listdir(source_path):
  print(dirs)
  dirname = dirs
  base_path = source_path+"/"+dirname
  file_list = sorted(os.listdir(base_path))
  random.shuffle(file_list)

  for f in file_list:
    if '.cat' not in f:
      continue

    # read landmarks
    pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
    landmarks = (pd_frame.values[0][1:-1]).reshape((-1, 2))
    bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)]).astype(np.int)
    center = np.mean(bb, axis=0)
    face_size = max(np.abs(np.max(landmarks, axis=0) - np.min(landmarks, axis=0)))
    new_bb = np.array([
      center - face_size * 0.6,
      center + face_size * 0.6
    ]).astype(np.int)
    new_bb = np.clip(new_bb, 0, 99999)
    new_landmarks = landmarks - new_bb[0]
    

    # load image
    img_filename, ext = os.path.splitext(f)

    img = cv2.imread(os.path.join(base_path, img_filename))

    new_img = img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]

    # resize image and relocate landmarks
    img, ratio, top, left = resize_img(new_img,img_size)
    new_landmarks = ((new_landmarks * ratio) + np.array([left, top])).astype(np.int)

    dataset['imgs'].append(img)
    dataset['lmks'].append(new_landmarks.flatten())
    dataset['bbs'].append(new_bb.flatten())

    # for l in new_landmarks:
    #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

    # cv2.imshow('img', img)
    # if cv2.waitKey(0) == ord('q'):
    #   sys.exit(1)

  np.save('dataset/lmks_%s.npy' % dirname, np.array(dataset))

