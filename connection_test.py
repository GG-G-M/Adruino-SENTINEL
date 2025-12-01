import serial
import time

# Change COM3 if needed
PORT = "COM3"
BAUD = 9600

try:
    print(f"Connecting to Arduino on {PORT}...")
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
    print("CONNECTED!\n")
except Exception as e:
    print("ERROR: Cannot connect to Arduino:", e)
    exit()

while True:
    print("\nSelect option:")
    print("1. Authorized")
    print("2. Not Authorized")
    print("3. Exit")

    choice = input("Enter option: ").strip()

    if choice == "1":
        arduino.write(b"AUTHORIZED\n")
        print("✔ Sent AUTHORIZED")

    elif choice == "2":
        arduino.write(b"NOT_AUTHORIZED\n")
        print("✖ Sent NOT_AUTHORIZED")

    elif choice == "3":
        print("Goodbye!")
        break

    else:
        print("Invalid option.")
