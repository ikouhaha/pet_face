from package import *

def genFolderTime():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def readNPY(path):
    return np.load(path,allow_pickle=True)


def load_keypoints(path):    
    """Load keypoints from .cat file
    
    The .cat file is a single-line text file in format: 'nb_keypoints x1, y1, x2, y2, ...'
    """
    with open(path, 'r') as f:
        line = f.read().split()  # [nb_keypoints, x1, y1, x2, y2, ...]
    keypoints_nb = int(line[0])  # int
    keypoints_1d = np.array(line[1:], dtype=int)  # np.ndarray, [x1, y1, x2, y2, ...]
    keypoints_xy = keypoints_1d.reshape((-1, 2))  # np.ndarray, [[x1, y1], [x2, y2], ...]
    assert keypoints_nb == len(keypoints_xy)
    assert keypoints_nb == 9                # always nine keypoints, eyes, nose, two ears todo change it dynamic
    return keypoints_xy       # np.ndarray, [[x1, y1], [x2, y2], ...]

def scale_img_kps(image, keypoints, target_size):
    width, height = image.size
    ratio_w = width / target_size
    ratio_h = height / target_size
    
    image_new = image.resize((target_size, target_size), resample=PIL.Image.LANCZOS)
    
    keypoints_new = np.zeros_like(keypoints)
    keypoints_new[range(len(keypoints_new)), 0] = keypoints[:,0] / ratio_w
    keypoints_new[range(len(keypoints_new)), 1] = keypoints[:,1] / ratio_h
    
    return image_new, keypoints_new

def undo_preprocess_images(images_batch):
    tmp = images_batch
    tmp = (tmp + tmp.min()) / (tmp.max() - tmp.min())
    tmp = (tmp * 255).astype(np.uint8)
    return tmp
    
def undo_preprocess_keypts(keypoints_batch, img_size):
    return (keypoints_batch * (img_size // 2)) + (img_size // 2)      

def resize_img(im,img_size):
  old_size = im.shape[:2] # old_size is in (height, width) format
  ratio = float(img_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  # new_size should be in (width, height) format
  im = cv2.resize(im, (new_size[1], new_size[0]))
  delta_w = img_size - new_size[1]
  delta_h = img_size - new_size[0]
  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)
  new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=[0, 0, 0])
  return new_im, ratio, top, left    

def genDataSetForbbs(source_path,img_size):
    imgs = []
    lmks = []
    bbs = []
    for dirs in os.listdir(source_path):
        print(dirs)
        dirname = dirs
        base_path = source_path+"/"+dirname
        file_list = sorted(os.listdir(base_path))
        random.shuffle(file_list)

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

            imgs.append(img)
            lmks.append(landmarks.flatten())
            bbs.append(bb.flatten())
            

    return (imgs,lmks,bbs)

def genDataSetForLmks(source_path,img_size):
    imgs = []
    lmks = []
    bbs = []
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

            imgs.append(img)
            lmks.append(new_landmarks.flatten())
            bbs.append(new_bb.flatten())

    return (imgs,lmks,bbs)

source_path = ""
dataset_path = ""
model_path = ""
result_path = ""
sample_path = ""
logs_path = "" 
lmk_logs_path = ""
images_path = "" 

if(platform.node()=="LAPTOP-MEFC1PDG"):
    source_path = "./source/cats"
    dataset_path = "./dataset"
    model_path = "./models"
    result_path = "./result"
    sample_path = "./samples"
    logs_path = "./logs" 
    lmk_logs_path = "./lmk_logs"
    images_path = "./images" 
else:
    source_path = "./source/cats"
    dataset_path = "./dataset"
    model_path = "/content/drive/MyDrive/PET_FACE/models"
    result_path = "/content/drive/MyDrive/PET_FACE/result"
    sample_path = "/content/drive/MyDrive/PET_FACE/samples"
    logs_path = "/content/drive/MyDrive/PET_FACE/logs"
    lmk_logs_path = "/content/drive/MyDrive/PET_FACE/lmk_logs" 
    images_path = "/content/drive/MyDrive/PET_FACE/images" 

shutil.rmtree(logs_path,ignore_errors=True)
shutil.rmtree(lmk_logs_path,ignore_errors=True)

Path(source_path).mkdir(parents=True, exist_ok=True)
Path(dataset_path).mkdir(parents=True, exist_ok=True)
Path(model_path).mkdir(parents=True, exist_ok=True)
Path(result_path).mkdir(parents=True, exist_ok=True)
Path(sample_path).mkdir(parents=True, exist_ok=True)
Path(logs_path).mkdir(parents=True, exist_ok=True)
Path(lmk_logs_path).mkdir(parents=True, exist_ok=True)
Path(images_path).mkdir(parents=True, exist_ok=True)
 