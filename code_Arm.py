import RPi.GPIO as GPIO
import math
import time
from picamera import PiCamera
from time import sleep
from io import BytesIO
import numpy as np 
from PIL import Image

GPIO.setmode(GPIO.BCM)
BASE_PIN = 17
SHOULDER_PIN = 18
ELBOW_PIN = 27
WRIST_PIN = 22
GRIPPER_PIN = 23
SERVO_FREQUENCY = 50
GPIO.setup(BASE_PIN, GPIO.OUT)
GPIO.setup(SHOULDER_PIN, GPIO.OUT)
GPIO.setup(ELBOW_PIN, GPIO.OUT)
GPIO.setup(WRIST_PIN, GPIO.OUT)
GPIO.setup(GRIPPER_PIN, GPIO.OUT)

base_pwm = GPIO.PWM(BASE_PIN, SERVO_FREQUENCY)
shoulder_pwm = GPIO.PWM(SHOULDER_PIN, SERVO_FREQUENCY)
elbow_pwm = GPIO.PWM(ELBOW_PIN, SERVO_FREQUENCY)
wrist_pwm = GPIO.PWM(WRIST_PIN, SERVO_FREQUENCY)
gripper_pwm = GPIO.PWM(GRIPPER_PIN, SERVO_FREQUENCY)

base_pwm.start(0)
shoulder_pwm.start(0)
elbow_pwm.start(0)
wrist_pwm.start(0)
gripper_pwm.start(0)

longeur_base_shoulder = 10    
longeur_shoulder_elbow = 15   
longeur_elbow_wrist = 10      
longeur_wrist_gripper = 5     
board_taille = 25              
square_taille = board_taille / 8

def angle_to_duty_cycle(angle):
    return 2.5 + (angle / 180.0) * 10

def get_square_coordinates(row, col):
    x = (col - 3.5) * square_taille
    y = (3.5 - row) * square_taille
    z = 0 
    return x, y, z


def inverse_kinematics(x, y, z):
    base_angle = math.atan2(y, x)
    r = math.sqrt(x**2 + y**2)
    s = z
    D = (r**2 + s**2 - longeur_shoulder_elbow**2 - longeur_elbow_wrist**2) / (2 * longeur_shoulder_elbow * longeur_elbow_wrist)
    elbow_angle = math.acos(D)
    shoulder_angle = math.atan2(s, r) - math.atan2(longeur_elbow_wrist * math.sin(elbow_angle),
                                                   longeur_shoulder_elbow + longeur_elbow_wrist * math.cos(elbow_angle))
    wrist_angle = - (shoulder_angle + elbow_angle)

    return (math.degrees(base_angle), 
            math.degrees(shoulder_angle), 
            math.degrees(elbow_angle), 
            math.degrees(wrist_angle))


def move_servos(base_angle, shoulder_angle, elbow_angle, wrist_angle):
    base_pwm.ChangeDutyCycle(angle_to_duty_cycle(base_angle))
    shoulder_pwm.ChangeDutyCycle(angle_to_duty_cycle(shoulder_angle))
    elbow_pwm.ChangeDutyCycle(angle_to_duty_cycle(elbow_angle))
    wrist_pwm.ChangeDutyCycle(angle_to_duty_cycle(wrist_angle))
    time.sleep(1) 

def control_gripper(grip):
    if grip:  
        gripper_pwm.ChangeDutyCycle(angle_to_duty_cycle(45))  
    else:  
        gripper_pwm.ChangeDutyCycle(angle_to_duty_cycle(90))  
    time.sleep(0.5)


def move_piece(from_row, from_col, to_row, to_col):
    x1, y1, z1 = get_square_coordinates(from_row, from_col)
    base_angle1, shoulder_angle1, elbow_angle1, wrist_angle1 = inverse_kinematics(x1, y1, z1)
    move_servos(base_angle1, shoulder_angle1, elbow_angle1, wrist_angle1)
    control_gripper(True)
    move_servos(base_angle1, shoulder_angle1, elbow_angle1, wrist_angle1 + 20)

   
    x2, y2, z2 = get_square_coordinates(to_row, to_col)
    base_angle2, shoulder_angle2, elbow_angle2, wrist_angle2 = inverse_kinematics(x2, y2, z2)
    move_servos(base_angle2, shoulder_angle2, elbow_angle2, wrist_angle2)
    control_gripper(False)
    move_servos(base_angle2, shoulder_angle2, elbow_angle2, wrist_angle2 + 20)

def cleanup():
    base_pwm.stop()
    shoulder_pwm.stop()
    elbow_pwm.stop()
    wrist_pwm.stop()
    gripper_pwm.stop()
    GPIO.cleanup()


def get_pict():
    camera = PiCamera()
    camera.resolution = (1024, 768)

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.IN) 
    camera.start_preview(fullscreen=False, window=(50, 50, 640, 480))
    sleep(5) 

    try:
        while True:
            if GPIO.input(17):
                stream = BytesIO() 
                camera.capture(stream, format='jpeg')
                stream.seek(0)
                image = np.array(Image.open(stream))
                sleep(2)
                break

    except KeyboardInterrupt:
        print("Programme arrete.")
    finally:
        camera.stop_preview()  
        GPIO.cleanup()  
    return image    
