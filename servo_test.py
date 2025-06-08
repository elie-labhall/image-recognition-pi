#!/usr/bin/env python3
import time
import board
import busio
from adafruit_pca9685 import PCA9685

# ─── CONFIGURABLE PARAMETERS ───────────────────────────────
CHANNELS     = [0, 1]  # PCA9685 channels for your two servos
SWEEP_START  = 80       # degrees: start of sweep
SWEEP_END    = 100     # degrees: end of sweep
STEP         = 1       # degrees per step
DELAY        = 0.1   # seconds between steps (smaller = faster)
PAUSE        = 1       # seconds to wait at each end
# ────────────────────────────────────────────────────────────

# Setup I2C + PCA9685
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  # 50 Hz for servos

def set_servo_angle(channel: int, angle: float) -> None:
    """
    Move an SG90/180° servo on `channel` to `angle` (0–180°).
    Uses a safe pulse range of 500–2400 µs.
    """
    pulse_min = 500    # µs @ 0°
    pulse_max = 2400   # µs @ 180°
    pulse = int((angle / 180.0) * (pulse_max - pulse_min) + pulse_min)
    duty  = int(pulse * 65535 / 20000)  # scale to 16-bit for 20 ms frame
    pca.channels[channel].duty_cycle = duty

def sweep_both(start: int, end: int) -> None:
    """
    Sweep both servos from `start` to `end` in `STEP` increments.
    """
    direction = 1 if end >= start else -1
    for angle in range(start, end + direction, STEP * direction):
        for ch in CHANNELS:
            set_servo_angle(ch, angle)
        print(f"Angles → {angle}°")
        time.sleep(DELAY)

try:
    print(f"Sweeping channels {CHANNELS}: {SWEEP_START}° ↔ {SWEEP_END}° (Ctrl+C to stop)")
    while True:
        sweep_both(SWEEP_START, SWEEP_END)
        time.sleep(PAUSE)
        sweep_both(SWEEP_END, SWEEP_START)
        time.sleep(PAUSE)

except KeyboardInterrupt:
    print("\nUser interrupted — stopping servos...")

finally:
    # release both servos
    for ch in CHANNELS:
        pca.channels[ch].duty_cycle = 0
    pca.deinit()
    print("Servos released, exiting.")
