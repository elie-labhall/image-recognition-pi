import RPi.GPIO as GPIO
import time

# Pin configuration
RELAY_PIN = 6  

# Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.LOW)

print("Starting relay loop. Press Ctrl+C to stop.")

try:
    while True:
        # Turn relay ON (motor powered)
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time.sleep(0.25)  # relay ON for 0.5 seconds

        # Turn relay OFF (motor off)
        GPIO.output(RELAY_PIN, GPIO.LOW)
        time.sleep(1.0)  # wait 2 seconds before next cycle

except KeyboardInterrupt:
    print("Stopping script...")

finally:
    # Clean up GPIO to reset pins safely
    GPIO.output(RELAY_PIN, GPIO.LOW)
    GPIO.cleanup()
