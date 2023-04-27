import dlib
import cv2
import os
from IPython import embed

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def get_bb(image,rect):
    for i, d in enumerate(rect):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        startX = d.rect.left()
        startY = d.rect.top()
        endX = d.rect.right()
        endY = d.rect.bottom()
        # ensure the bounding box coordinates fall within the spatial
        # dimensions of the image
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(endX, image.shape[1])
        endY = min(endY, image.shape[0])
        # compute the width and height of the bounding box
        w = endX - startX
        h = endY - startY
        # return our bounding box coordinates
        return (startX, startY, w, h)
        


def get_frames(input_path,output_path,file_name):
    cap = cv2.VideoCapture(input_path)
    frame_id = 0
    # a variable to set how many frames you want to skip
    frame_skip = 20
    # a variable to keep track of the frame to be saved
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id > frame_skip - 1:
            frame_count += 1
            boxes = get_face_landmarks(frame)
            if len(boxes) > 0:
                try:
                    cropped = get_cropped_image(frame, boxes)
                except:
                    continue
            output_file =  file_name +'_'+ str(frame_count) +'.jpg'
            if cropped is not None:
                try:
                    cv2.imwrite(os.path.join(output_path, output_file), cropped)
                except:
                    continue
            frame_id = 0
            print(output_file)
            continue
        frame_id += 1
        

    cap.release()
    cv2.destroyAllWindows()

def get_face_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    # detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    rects = detector(rgb)
    # boxes = [get_bb(image, r) for r in rects]
    boxes = [convert_and_trim_bb(image, r) for r in rects]
    return boxes

def get_cropped_image(image, boxes):
    height,  width = image.shape[0], image.shape[1]
    scale_factor = 1.5

    for (x, y, w, h) in boxes:
        new_h = h * scale_factor
        new_w = w * scale_factor
        delta_h = (new_h - h) / 2
        delta_w = (new_w - w) / 2
        new_y = abs(y - delta_h)
        new_x = abs(x - delta_w)
        cropped = image[int(new_y):int(new_y+new_h), int(new_x):int(new_x+new_w)]
    return cropped


fn_dir = "/NCSU Course work/Semester 2/Advanced ML/Phase 2/Celeb-DF-v2/"

(images, directories, id) = ([], {}, 0)

for (subdirs, dirs, files) in os.walk(fn_dir):
    print(subdirs)
    for subdir in dirs:
        directories[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        output_path = subjectpath + '/' + subdir+ '_images'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for filename in os.listdir(subjectpath):
            if filename.endswith('.mp4'):
                input_path = subjectpath + '/' + filename
                file_name = filename.split('.')[0]
                get_frames(input_path,output_path,file_name)
        id += 1


