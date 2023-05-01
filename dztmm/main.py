import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from threading import Thread
import mediapipe
import pywavefront
import ObjLoader
import math,sys,numpy,random,ctypes
X_AXIS = 0.0
Y_AXIS = 0.0
Z_AXIS = 0.0

image=Image.open("data/textures/nb574.jpg")
image=image.transpose(Image.FLIP_TOP_BOTTOM)
image_data=image.convert("RGBA").tobytes()

vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;

uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

out vec2 v_texture;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330

in vec2 v_texture;

out vec4 out_color;

uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture);
}
"""

x_point = 0.0
y_point = 0.0

scene = pywavefront.Wavefront('data/nb574.obj', collect_faces=True)

scene_box = (scene.vertices[0], scene.vertices[0])
for vertex in scene.vertices:
    min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
    max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
    scene_box = (min_v, max_v)

scene_trans = [-(scene_box[1][i] + scene_box[0][i] ) / 2 for i in range(3)]

def Model():
    glPushMatrix()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

    glScalef(*scene_scale)

    for mesh in scene.mesh_list:
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_i in face:
                glVertex3f(*scene.vertices[vertex_i])
        glEnd()


    glPopMatrix()

scaled_size = 3

scene_size = [scene_box[1][i] - scene_box[0][i] for i in range(3)]
max_scene_size = max(scene_size)
scene_scale = [scaled_size / max_scene_size for i in range(3)]

texture_id = 0
thread_quit = 0
X_AXIS = 0.0
Y_AXIS = 0.0
Z_AXIS = 0.0
DIRECTION = 1
cap = cv2.VideoCapture(0)
new_frame = cap.read()[1]

Static_Image_Mode = False
Max_Number_Of_Objects = 2
Minimum_Detection_Confidence = 0.2
Min_Tracking_Confidence = 0.1
Model_Name = 'Shoe'

# Setup MediaPipe:
Draw = mediapipe.solutions.drawing_utils
My_Objectron = mediapipe.solutions.objectron.Objectron(Static_Image_Mode, Max_Number_Of_Objects,
                                                       Minimum_Detection_Confidence, Min_Tracking_Confidence,
                                                       Model_Name)


def init():
    video_thread = Thread(target=update, args=())
    video_thread.start()





def init_gl(width, height):
    global texture_id
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(35.0, float(width) / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_TEXTURE_2D)
    texture_id = glGenTextures(1)


def update():
    global new_frame
    global shoeExsists
    global scaled_size
    global x_point, y_point
    while (True):
        Frame = cap.read()[1]
        if thread_quit == 1:
            break

        Flipped_Frame = cv2.flip(Frame, 1)
        Flipped_Gray_Frame = cv2.cvtColor(Flipped_Frame, cv2.COLOR_BGR2GRAY)
        BGR_Flipped_Gray_Frame = cv2.cvtColor(Flipped_Gray_Frame, cv2.COLOR_GRAY2BGR)
        # Collect Objectron Results:
        Detection_Results = My_Objectron.process(Flipped_Frame)
        # If There Are Results:
        if Detection_Results.detected_objects:
            shoeExsists = True
            # For Each Result:
            for Object_Detected in Detection_Results.detected_objects:
                landmarks = Object_Detected.landmarks_2d
                x_point = landmarks.landmark[3].x
                y_point = landmarks.landmark[3].y


                Draw.draw_landmarks(Flipped_Frame, Object_Detected.landmarks_2d,
                                    mediapipe.solutions.objectron.BOX_CONNECTIONS)
                Draw.draw_axis(Flipped_Frame, Object_Detected.rotation, Object_Detected.translation)
        else:
            shoeExsists = False
        new_frame = Flipped_Frame

    cap.release()
    cv2.destroyAllWindows()


def draw_gl_scene():
    global cap
    global new_frame
    global X_AXIS, Y_AXIS, Z_AXIS
    global DIRECTION
    global texture_id
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    frame = new_frame
    # convert image to OpenGL texture format
    tx_image = cv2.flip(frame, 0)
    tx_image = Image.fromarray(tx_image)
    ix = tx_image.size[0]
    iy = tx_image.size[1]
    tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)
    # create texture
    # create texture
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glPushMatrix()
    glTranslatef(0.0, 0.0, -6.0)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(4.0, -3.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(4.0, 3.0, 0.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-4.0, 3.0, 0.0)
    glEnd()
    glPopMatrix()


    glPushMatrix()


    glTranslatef(x_point, y_point, -5.0)

    glRotatef(X_AXIS, 1.0, 0.0, 0.0)
    glRotatef(Y_AXIS, 0.0, 1.0, 0.0)
    glRotatef(Z_AXIS, 0.0, 0.0, 1.0)

    if shoeExsists:
        Model()
    glPopMatrix()



    glPushMatrix()

    X_AXIS = 90
    Z_AXIS = 180
    Y_AXIS = 180


    glBegin(GL_QUADS)


    glColor3f(6.0, 1.0, 1.0)
    glEnd()
    glPopMatrix()


    glutSwapBuffers()


def key_pressed(key, x, y):
    global thread_quit
    if key == chr(27) or key == "q":
        thread_quit = 1
        sys.exit()


def run():

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(1280, 720)
    glutInitWindowPosition(200, 200)
    window = glutCreateWindow('My and Cube')
    glutDisplayFunc(draw_gl_scene)
    glutIdleFunc(draw_gl_scene)
    glutKeyboardFunc(key_pressed)
    init_gl(1280, 720)
    glutMainLoop()


if __name__ == "__main__":
    init()
    run()
