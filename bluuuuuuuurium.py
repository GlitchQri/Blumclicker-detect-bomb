import cv2
import numpy as np
import pyautogui
from pynput import keyboard
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import random

# Диапазон смещения курсора от центра блумов вниз
RAND_MIN = 5
RAND_MAX = 10

# Кнопка активации/деактивации кликера
ACTIVE_BTN = keyboard.Key.ctrl_r
# Кнопка завершения программы
EXIT_BTN = keyboard.Key.backspace

# Параметры захвата экрана
region = (900, 550, 370, 530)
element_lower = np.array([45, 75, 75])
element_upper = np.array([75, 255, 255])

# Диапазоны для темно-зеленого цвета
dark_green_lower = np.array([6, 22, 0])
dark_green_upper = np.array([85, 255, 125])

# Диапазоны для бомбы (примерные значения)
bomb_lower = np.array([0, 0, 100])
bomb_upper = np.array([10, 10, 255])

# Диапазоны для ледяного элемента (примерные значения)
ice_lower = np.array([100, 100, 255])
ice_upper = np.array([140, 255, 255])

# Минимальная и максимальная площадь контура для фильтрации
min_contour_area = 150
max_contour_area = 1000

# Радиус, в котором проверяется наличие бомбы
BOMB_RADIUS = 50

# Количество блумов для пропуска
MIN_BLOOMS_TO_SKIP = 5
MAX_BLOOMS_TO_SKIP = 17

clicking_enabled = False
program_running = True
executor = ThreadPoolExecutor(max_workers=10)

# Генерация случайного количества блумов для пропуска
blooms_to_skip = random.randint(MIN_BLOOMS_TO_SKIP, MAX_BLOOMS_TO_SKIP)
blooms_skipped = 0

def on_press(key):
    global clicking_enabled, program_running, blooms_to_skip, blooms_skipped
    try:
        if key == ACTIVE_BTN:
            clicking_enabled = not clicking_enabled
            blooms_to_skip = random.randint(MIN_BLOOMS_TO_SKIP, MAX_BLOOMS_TO_SKIP)
            blooms_skipped = 0
            print(f"Clicking enabled: {clicking_enabled}, Blooms to skip: {blooms_to_skip}")
        elif key == EXIT_BTN:
            program_running = False
            print("Exiting program...")
            return False  # Останавливает слушатель клавиш
    except AttributeError:
        pass

def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Создаем основную маску
    mask = cv2.inRange(hsv, element_lower, element_upper)   
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтруем контуры по площади и убираем вложенные контуры
    filtered_contours = []
    for i, cnt in enumerate(contours):
        if min_contour_area <= cv2.contourArea(cnt) <= max_contour_area and hierarchy[0][i][3] == -1:  # Только верхние контуры
            filtered_contours.append(cnt)
    return filtered_contours

def find_bombs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bomb_mask = cv2.inRange(hsv, bomb_lower, bomb_upper)
    bomb_contours, _ = cv2.findContours(bomb_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return bomb_contours

def click_on_position(screen_x, screen_y):
    if clicking_enabled:
        pyautogui.click(screen_x, screen_y + random.randint(RAND_MIN, RAND_MAX))

def click_element_contours(contours, bomb_contours):
    global blooms_skipped, blooms_to_skip
    for cnt in contours:
        if not clicking_enabled:
            break
        (x, y, w, h) = cv2.boundingRect(cnt)
        center_x = x + w // 2
        center_y = y + h // 2
        screen_x = region[0] + center_x
        screen_y = region[1] + center_y

        # Проверяем наличие бомбы в радиусе BOMB_RADIUS
        for bomb in bomb_contours:
            bomb_x, bomb_y, bomb_w, bomb_h = cv2.boundingRect(bomb)
            bomb_center_x = bomb_x + bomb_w // 2
            bomb_center_y = bomb_y + bomb_h // 2
            distance = np.sqrt((center_x - bomb_center_x)**2 + (center_y - bomb_center_y)**2)
            if distance < BOMB_RADIUS:
                print("Bomb detected near bloom, skipping click")
                break
        else:
            # Случайный пропуск блумов
            if blooms_skipped < blooms_to_skip:
                blooms_skipped += 1
                print(f"Bloom skipped: {blooms_skipped}/{blooms_to_skip}")
            else:
                executor.submit(click_on_position, screen_x, screen_y)

def capture_and_process():
    global program_running
    while program_running:
        screenshot = pyautogui.screenshot(region=region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        contours = process_frame(frame)
        bomb_contours = find_bombs(frame)

        # Удаляем темно-зеленый цвет
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dark_green_mask = cv2.inRange(hsv_frame, dark_green_lower, dark_green_upper)
        frame[dark_green_mask > 0] = (0, 0, 0)

        # Рисуем контуры на кадре
        cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
        cv2.drawContours(frame, bomb_contours, -1, (255, 0, 0), 2)  # Красный контур для бомб

        cv2.imshow("Captured Region", frame)
        cv2.waitKey(1)
        
        if clicking_enabled:
            click_element_contours(contours, bomb_contours)
        
        time.sleep(0.05)
        
    cv2.destroyAllWindows()
    print("Capture and processing thread terminated")

listener = keyboard.Listener(on_press=on_press)
listener.start()

capture_thread = threading.Thread(target=capture_and_process)
capture_thread.start()

try:
    listener.join()
    capture_thread.join()
except KeyboardInterrupt:
    cv2.destroyAllWindows()
    print("Program terminated")
