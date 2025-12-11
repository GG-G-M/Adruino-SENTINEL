import serial
import time
import sys

class ContinuousServo:
    def __init__(self):
        self.arduino = None

    def connect(self, port=None):
        import serial.tools.list_ports

        if not port:
            ports = list(serial.tools.list_ports.comports())
            if not ports:
                print("No COM ports found.")
                return False

            print("\nAvailable ports:")
            for i, p in enumerate(ports):
                print(f"[{i}] {p.device}")

            select = input("Choose port: ").strip()
            if select.isdigit():
                port = ports[int(select)].device
            else:
                port = select

        try:
            print(f"Connecting to {port}...")
            self.arduino = serial.Serial(port, 9600, timeout=1)
            time.sleep(2)
            self.arduino.reset_input_buffer()
            self.read_output()
            print("Connected.\n")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def read_output(self, timeout=1.0):
        start = time.time()
        while time.time() - start < timeout:
            if self.arduino.in_waiting:
                msg = self.arduino.readline().decode(errors="ignore").strip()
                if msg:
                    print("Arduino:", msg)

    def send(self, command):
        print(f"> Sending: {command}")
        self.arduino.write((command + "\n").encode())
        time.sleep(0.2)
        self.read_output(1.5)

    def menu(self):
        while True:
            print("\n=== Continuous Servo Control ===")
            print("1. Rotate Right")
            print("2. Rotate Left")
            print("3. Stop")
            print("4. Change Speed")
            print("5. Change Pin")
            print("0. Exit")
            choice = input("Select: ").strip()

            if choice == "1":
                self.send("right")
            elif choice == "2":
                self.send("left")
            elif choice == "3":
                self.send("stop")
            elif choice == "4":
                speed = input("Speed 0–100: ").strip()
                self.send(f"speed{speed}")
            elif choice == "5":
                pin = input("Enter pin (3,5,6,9,10,11): ").strip()
                self.send(f"pin{pin}")
            elif choice == "0":
                print("Goodbye.")
                break
            else:
                print("Invalid choice.")

def main():
    print("Continuous Servo Python Controller")
    app = ContinuousServo()

    if not app.connect():
        return

    try:
        app.menu()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
