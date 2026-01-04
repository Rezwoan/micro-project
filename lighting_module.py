import lgpio

# CONFIGURATION
LIGHT1_GPIO = 27
LIGHT2_GPIO = 22
BTN1_GPIO = 5
BTN2_GPIO = 6

h = None

def init():
    global h
    if h is None:
        try:
            h = lgpio.gpiochip_open(0)
            # Start High (OFF for most Relays)
            lgpio.gpio_claim_output(h, LIGHT1_GPIO, 1)
            lgpio.gpio_claim_output(h, LIGHT2_GPIO, 1)
            lgpio.gpio_claim_input(h, BTN1_GPIO, lgpio.SET_PULL_UP)
            lgpio.gpio_claim_input(h, BTN2_GPIO, lgpio.SET_PULL_UP)
        except Exception as e:
            print(f"Lighting GPIO Error: {e}")

def set_light(light_id, state):
    """state: 1=ON, 0=OFF (Logic Inverted for Relay)"""
    if h is None: init()
    pin = LIGHT1_GPIO if light_id == 1 else LIGHT2_GPIO
    # Relay: 0 is ON, 1 is OFF
    hw_val = 0 if state else 1
    lgpio.gpio_write(h, pin, hw_val)

def get_light_state(light_id):
    if h is None: init()
    pin = LIGHT1_GPIO if light_id == 1 else LIGHT2_GPIO
    # Read hardware state and invert back for software
    hw_val = lgpio.gpio_read(h, pin)
    return 1 if hw_val == 0 else 0

def is_btn_pressed(btn_id):
    if h is None: init()
    pin = BTN1_GPIO if btn_id == 1 else BTN2_GPIO
    return lgpio.gpio_read(h, pin) == 0