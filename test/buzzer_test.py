from gpiozero import DigitalOutputDevice
from time import sleep

GPIO = 21  # Pin 40

# Most modules: active_high=True (HIGH = beep)
# Some modules: active_high=False (LOW = beep)
active_high = True

buzzer = DigitalOutputDevice(GPIO, active_high=active_high, initial_value=False)

print("Buzzer test starting on GPIO21 (Pin 40).")
print("If you hear nothing, edit active_high = False and run again.\n")

def beep(seconds=0.2):
    buzzer.on()
    sleep(seconds)
    buzzer.off()

# Test 1: single long beep
print("Test 1: Long beep (1s)")
beep(1.0)
sleep(0.5)

# Test 2: 5 short beeps
print("Test 2: 5 short beeps")
for _ in range(5):
    beep(0.15)
    sleep(0.15)

# Test 3: SOS-ish pattern
print("Test 3: Pattern")
for t in [0.1,0.1,0.1, 0.3,0.3,0.3, 0.1,0.1,0.1]:
    beep(t)
    sleep(0.12)

print("\nDone.")
