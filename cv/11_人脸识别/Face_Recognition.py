
# # 使用Face Recognition

# 第一：pip install cmake
# 第二：pip install dlib(人脸特征点)
# 第三：pip install face_recognition


# 检测人脸面部特征
from PIL import Image, ImageDraw
import face_recognition

# 将图像文件加载到NumPy数组中
image = face_recognition.load_image_file("victor_test.jpeg")
# 查找图像中所有的面部特征
face_landmarks_list = face_recognition.face_landmarks(image)
print("找到{}张人脸。".format(len(face_landmarks_list)))

for face_landmarks in face_landmarks_list:

    # 图像中每个面部特征的key
    # 通过face_landmarks.keys()获取到的是一样的
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]

    # 将NumPy数组对象转换PIL.Image.Image对象
    pil_image = Image.fromarray(image)
    # 再将PIL.Image.Image对象转换成PIL.ImageDraw.ImageDraw，可供绘图
    new_img = ImageDraw.Draw(pil_image)

    # 根据找到的面部特征值
    for facial_feature in facial_features:
        # 来绘制线条，线条宽度是5
        new_img.line(face_landmarks[facial_feature], width=2)

    # 显示已经绘制过特征线条的图像
    pil_image.show()



# 实时人脸识别
import face_recognition
import cv2

# 获取网络摄像头webcam实例对象
video_capture = cv2.VideoCapture(0)

# 加载人脸图像函数
def load_person_image(person_filename):
    # 将图像文件加载到内存，以数组呈现
    person_image = face_recognition.load_image_file(person_filename)
    # 将图像数据编码为128维度的人脸矩阵数据，以NumPy数组返回
    person_face_encoding = face_recognition.face_encodings(person_image)[0]
    return person_face_encoding

# 加载已知人脸对象
victor_face_encoding = load_person_image("victor_test.jpeg")

# 创建已知人脸编码数组
known_face_encodings = [
    victor_face_encoding
]
# 创建已知人脸对象名字数组
known_face_names = [
    "Zhang Qiang"
]

# 初始化一些变量
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# 创建一个无限循环的环境，以便于让程序一直在webcam摄像头中读取图像内容
while True:
    # 获取摄像头的单帧frame图像
    ret, frame = video_capture.read()
    
    # 为了检测和识别人脸更快些，我们将摄像头拍摄到的图像frame重新定义大小，是原大小的4分之1
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # OpenCV使用BGR颜色通道
    # face_recognition使用RGB颜色通道
    # 这就将图像从BGR转换成RGB颜色通道
    rgb_small_frame = small_frame[:, :, ::-1]

    # 只处理每一帧视频，以节省时间
    if process_this_frame:
        # 在当前的视频帧中找出所有的人脸位置和人脸编码
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        # 遍历人脸图像编码，因为图像中可能会有多个人脸
        for face_encoding in face_encodings:
            # 将已知的人脸编码和检测到的人脸编码进行对比，就知道它们是否是同一个人了
            # 已知的人脸编码和检测到的人脸编码，可能在数值上有些差异，但是，只要在容错率范围内就可以，默认值是0.6
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # 如果人脸编码匹配上了，则将人脸对应的人名字取出来，反之，如果没有匹配上，就显示未知人脸
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            face_names.append(name)

    process_this_frame = not process_this_frame


    # 遍历人物名字和找到的人脸特征位置，并把它们绘制到图像上显示
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # 上面在做人脸检测和识别时，我们缩小了图像到4分之1大小，现在我们要把图像在还原回去
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # 在人脸周围绘制一个矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 在人脸矩形框的下面再绘制一个实心的矩形框，用来将人物的名字显示上去
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 将图像显示到视频窗口中
    cv2.imshow('Video', frame)

    # 在视频窗口，允许按下q键，退出程序
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 将创建的webcam实例对象手动释放掉
video_capture.release()
# 销毁所有的cv2窗口
cv2.destroyAllWindows()


