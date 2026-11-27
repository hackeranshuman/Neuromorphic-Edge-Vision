import serial
import time
import random

# ⚠️ Keep your specific COM port here
ESP32_PORT = 'COM8' 
BAUD_RATE = 115200

try:
    # Added a timeout so it doesn't get stuck waiting
    ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) 
    print("Connected to ESP32. Sending mock event streams...")

    while True:
        # 1. Send the mock data
        mock_event_stream = bytes([random.randint(0, 255) for _ in range(32)])
        ser.write(mock_event_stream)
        print("Sent 32 bytes...")
        
        # 2. Listen for the ESP32's response
        # Read all available lines from the ESP32
        while ser.in_waiting > 0:
            response = ser.readline().decode('utf-8', errors='ignore').strip()
            if response:
                print(f"🧠 ESP32 Output: {response}")
        
        time.sleep(0.5) 

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
