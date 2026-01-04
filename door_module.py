import lgpio
import time

# ================= CONFIGURATION =================
# Hardware Mappings
SERVO_GPIO = 18
LOCK_BUTTON_GPIO = 23    # Button inside
DOORBELL_BUTTON_GPIO = 24 # Button outside
BUZZER_GPIO = 25         # Orange wire of Buzzer

FREQ = 50

# Angles
ANGLE_OPEN = 10
ANGLE_CLOSE = 90

MIN_PW = 500
MAX_PW = 2500

h = None

def _us_to_duty(us):
    return (us / 1_000_000.0) * FREQ * 100.0

def _get_pulse_width(angle):
    return MIN_PW + (angle / 180.0) * (MAX_PW - MIN_PW)

def init():
    """Initializes the GPIO chip."""
    global h
    if h is None:
        try:
            h = lgpio.gpiochip_open(0)
            
            # Outputs
            lgpio.gpio_claim_output(h, SERVO_GPIO)
            lgpio.gpio_claim_output(h, BUZZER_GPIO)
            
            # Inputs (Pull-Up Resistors enabled)
            lgpio.gpio_claim_input(h, LOCK_BUTTON_GPIO, lgpio.SET_PULL_UP)
            lgpio.gpio_claim_input(h, DOORBELL_BUTTON_GPIO, lgpio.SET_PULL_UP)
            
        except Exception as e:
            print(f"GPIO Error: {e}")

def set_angle(angle):
    if h is None: init()
    pw = _get_pulse_width(angle)
    duty = _us_to_duty(pw)
    lgpio.tx_pwm(h, SERVO_GPIO, FREQ, duty)
    time.sleep(0.5)
    lgpio.tx_pwm(h, SERVO_GPIO, FREQ, 0)

def door_open():
    set_angle(ANGLE_OPEN)

def door_close():
    set_angle(ANGLE_CLOSE)

def is_lock_button_pressed():
    if h is None: init()
    return lgpio.gpio_read(h, LOCK_BUTTON_GPIO) == 0

def is_doorbell_button_pressed():
    if h is None: init()
    return lgpio.gpio_read(h, DOORBELL_BUTTON_GPIO) == 0

def play_buzzer_sequence():
    """
    Sends signal to Orange wire to trigger Active Buzzer.
    Pattern: Ding (Long) ... Dong (Long)
    """
    if h is None: init()
    
    # "Ding"
    lgpio.gpio_write(h, BUZZER_GPIO, 1) # Signal High (On)
    time.sleep(0.4)
    lgpio.gpio_write(h, BUZZER_GPIO, 0) # Signal Low (Off)
    
    time.sleep(0.2) # Short pause
    
    # "Dong"
    lgpio.gpio_write(h, BUZZER_GPIO, 1)
    time.sleep(0.6)
    lgpio.gpio_write(h, BUZZER_GPIO, 0)

def cleanup():
    global h
    if h is not None:
        lgpio.tx_pwm(h, SERVO_GPIO, FREQ, 0)
        lgpio.gpio_write(h, BUZZER_GPIO, 0)
        lgpio.gpiochip_close(h)
        h = None