import lgpio
import time

class DoorController:
    """
    A class to control a continuous rotation servo as a door toggle.
    
    HOW TO USE FROM ANOTHER SCRIPT:
    -------------------------------
    from door_controller import DoorController
    
    door = DoorController(gpio_pin=18)
    door.open_door()
    door.close_door()
    door.cleanup()
    """

    def __init__(self, gpio_pin=18, freq=50):
        # Configuration Constants
        self.GPIO = gpio_pin
        self.FREQ = freq
        self.STOP_US = 1475     # Calibrated stop point
        self.RANGE_US = 400    # Speed/Strength
        self.ROT_TIME = 0.18   # Seconds to move 90 degrees
        
        # Initialize hardware
        self.h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.h, self.GPIO)
        self.is_open = False
        self.stop()

    def _us_to_duty(self, us):
        return (us / 1_000_000.0) * self.FREQ * 100.0

    def stop(self):
        """Stops the servo movement immediately."""
        lgpio.tx_pwm(self.h, self.GPIO, self.FREQ, self._us_to_duty(self.STOP_US))

    def open_door(self):
        """Rotates the servo forward to the open position."""
        if not self.is_open:
            print("Opening door...")
            lgpio.tx_pwm(self.h, self.GPIO, self.FREQ, self._us_to_duty(self.STOP_US + self.RANGE_US))
            time.sleep(self.ROT_TIME)
            self.stop()
            self.is_open = True
        else:
            print("Door is already open.")

    def close_door(self):
        """Rotates the servo backward to the closed position."""
        if self.is_open:
            print("Closing door...")
            lgpio.tx_pwm(self.h, self.GPIO, self.FREQ, self._us_to_duty(self.STOP_US - self.RANGE_US))
            time.sleep(self.ROT_TIME)
            self.stop()
            self.is_open = False
        else:
            print("Door is already closed.")

    def cleanup(self):
        """Safely shuts down the PWM and releases the GPIO chip."""
        self.stop()
        time.sleep(0.3)
        lgpio.tx_pwm(self.h, self.GPIO, 0, 0)
        lgpio.gpiochip_close(self.h)
        print("Hardware cleaned up.")
