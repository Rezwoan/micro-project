import lgpio
import time

# ================= CONFIG =================
GPIO = 18
FREQ = 50

STOP_US = 1475        # calibrated stop
RANGE_US = 400        # speed strength
ROT_90_TIME = 0.18    # seconds for 90° (tuned)

# =========================================

def us_to_duty(us):
    return (us / 1_000_000.0) * FREQ * 100.0

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, GPIO)

def stop():
    lgpio.tx_pwm(h, GPIO, FREQ, us_to_duty(STOP_US))

def forward():
    lgpio.tx_pwm(h, GPIO, FREQ, us_to_duty(STOP_US + RANGE_US))

def backward():
    lgpio.tx_pwm(h, GPIO, FREQ, us_to_duty(STOP_US - RANGE_US))

# ================= STATE =================
# 0 = logical 0°
# 1 = logical 90°
state = 0

try:
    print("Servo TOGGLE control (0 ↔ 90)")
    print("Press ENTER to toggle")
    print("q = quit\n")

    stop()

    while True:
        cmd = input("Toggle (ENTER/q): ").strip().lower()

        if cmd == "q":
            break

        if state == 0:
            print("0 → 90")
            forward()
            time.sleep(ROT_90_TIME)
            stop()
            state = 1
        else:
            print("90 → 0")
            backward()
            time.sleep(ROT_90_TIME)
            stop()
            state = 0

finally:
    stop()
    time.sleep(0.3)
    lgpio.tx_pwm(h, GPIO, 0, 0)
    lgpio.gpiochip_close(h)
    print("Stopped cleanly.")
