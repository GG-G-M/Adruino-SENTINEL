"""
ARDUINO COMMUNICATION MODULE
Handles serial communication between Python and Arduino
"""

import serial
import serial.tools.list_ports
import time
import sys

# ============================================
# CONFIGURATION
# ============================================
# TinkerCAD uses virtual serial ports
# Real Arduino typically uses COM3, COM4 (Windows) or /dev/ttyUSB0, /dev/ttyACM0 (Linux/Mac)

DEFAULT_BAUD_RATE = 9600
TIMEOUT = 2  # seconds

# Command codes to send to Arduino
COMMANDS = {
    'UNLOCK': 'U',      # Unlock door
    'LOCK': 'L',        # Lock door
    'GRANTED': 'G',     # Access granted (LED green)
    'DENIED': 'D',      # Access denied (LED red)
    'SCANNING': 'S',    # Scanning in progress (LED yellow)
    'RESET': 'R'        # Reset system
}

# ============================================
# ARDUINO COMMUNICATION CLASS
# ============================================

class ArduinoConnection:
    """Manage serial communication with Arduino"""
    
    def __init__(self, port=None, baud_rate=DEFAULT_BAUD_RATE):
        self.port = port
        self.baud_rate = baud_rate
        self.connection = None
        self.is_connected = False
        
        if port:
            self.connect(port, baud_rate)
    
    def list_available_ports(self):
        """List all available serial ports"""
        ports = serial.tools.list_ports.comports()
        available = []
        
        print("\n📡 Available Serial Ports:")
        print("─"*60)
        
        if not ports:
            print("❌ No serial ports found")
            return available
        
        for i, port in enumerate(ports, 1):
            available.append(port.device)
            print(f"{i}. {port.device}")
            print(f"   Description: {port.description}")
            print(f"   Hardware ID: {port.hwid}")
            print()
        
        return available
    
    def auto_detect_arduino(self):
        """Automatically detect Arduino port"""
        ports = serial.tools.list_ports.comports()
        
        # Common Arduino identifiers
        arduino_keywords = ['Arduino', 'CH340', 'USB Serial', 'ttyUSB', 'ttyACM']
        
        for port in ports:
            port_info = f"{port.description} {port.hwid}".lower()
            for keyword in arduino_keywords:
                if keyword.lower() in port_info:
                    print(f"✅ Auto-detected Arduino on: {port.device}")
                    return port.device
        
        return None
    
    def connect(self, port=None, baud_rate=None):
        """Connect to Arduino"""
        if port is None:
            port = self.auto_detect_arduino()
            if port is None:
                print("❌ Could not auto-detect Arduino")
                return False
        
        if baud_rate is None:
            baud_rate = self.baud_rate
        
        try:
            print(f"🔌 Connecting to {port} at {baud_rate} baud...")
            
            self.connection = serial.Serial(
                port=port,
                baudrate=baud_rate,
                timeout=TIMEOUT,
                write_timeout=TIMEOUT
            )
            
            # Wait for Arduino to reset (important!)
            time.sleep(2)
            
            # Flush any startup data
            self.connection.reset_input_buffer()
            self.connection.reset_output_buffer()
            
            self.is_connected = True
            self.port = port
            print(f"✅ Connected to Arduino on {port}")
            
            return True
            
        except serial.SerialException as e:
            print(f"❌ Connection failed: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.is_connected = False
            print("🔌 Disconnected from Arduino")
    
    def send_command(self, command_key):
        """Send command to Arduino"""
        if not self.is_connected:
            print("❌ Not connected to Arduino")
            return False
        
        if command_key not in COMMANDS:
            print(f"❌ Invalid command: {command_key}")
            return False
        
        try:
            command = COMMANDS[command_key]
            self.connection.write(command.encode())
            print(f"📤 Sent to Arduino: {command_key} ({command})")
            
            # Wait for acknowledgment (optional)
            time.sleep(0.1)
            if self.connection.in_waiting > 0:
                response = self.connection.readline().decode().strip()
                print(f"📥 Arduino response: {response}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending command: {e}")
            return False
    
    def send_raw(self, data):
        """Send raw data to Arduino"""
        if not self.is_connected:
            print("❌ Not connected to Arduino")
            return False
        
        try:
            if isinstance(data, str):
                data = data.encode()
            
            self.connection.write(data)
            print(f"📤 Sent raw data: {data}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending data: {e}")
            return False
    
    def read_response(self, timeout=2):
        """Read response from Arduino"""
        if not self.is_connected:
            print("❌ Not connected to Arduino")
            return None
        
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.connection.in_waiting > 0:
                    response = self.connection.readline().decode().strip()
                    print(f"📥 Received: {response}")
                    return response
                time.sleep(0.1)
            
            print("⏱️ Timeout waiting for response")
            return None
            
        except Exception as e:
            print(f"❌ Error reading response: {e}")
            return None
    
    def unlock_door(self, duration=5):
        """Unlock door for specified duration"""
        print(f"🔓 Unlocking door for {duration} seconds...")
        
        if self.send_command('UNLOCK'):
            # Send duration (optional, depends on your Arduino code)
            self.send_raw(f"{duration}\n")
            return True
        return False
    
    def lock_door(self):
        """Lock door"""
        print("🔒 Locking door...")
        return self.send_command('LOCK')
    
    def signal_access_granted(self):
        """Signal that access was granted"""
        print("✅ Signaling: ACCESS GRANTED")
        return self.send_command('GRANTED')
    
    def signal_access_denied(self):
        """Signal that access was denied"""
        print("❌ Signaling: ACCESS DENIED")
        return self.send_command('DENIED')
    
    def signal_scanning(self):
        """Signal that scanning is in progress"""
        print("🔍 Signaling: SCANNING")
        return self.send_command('SCANNING')
    
    def test_connection(self):
        """Test Arduino connection with blink pattern"""
        if not self.is_connected:
            print("❌ Not connected")
            return False
        
        print("\n🧪 Testing connection...")
        print("Arduino should blink LED 3 times")
        
        for i in range(3):
            self.send_command('GRANTED')
            time.sleep(0.5)
            self.send_command('DENIED')
            time.sleep(0.5)
        
        print("✅ Test complete")
        return True

# ============================================
# INTERACTIVE SETUP FUNCTION
# ============================================

def interactive_setup():
    """Interactive Arduino setup"""
    print("\n" + "="*60)
    print("  ARDUINO CONNECTION SETUP")
    print("="*60)
    
    arduino = ArduinoConnection()
    
    # List available ports
    available_ports = arduino.list_available_ports()
    
    if not available_ports:
        print("\n⚠️  No serial ports detected!")
        print("\nFor TinkerCAD:")
        print("  1. Install 'Virtual Serial Port' software")
        print("  2. Create virtual COM ports (e.g., COM1, COM2)")
        print("\nFor Real Arduino:")
        print("  1. Connect Arduino via USB")
        print("  2. Install Arduino drivers if needed")
        return None
    
    print("\nOptions:")
    print("1. Auto-detect Arduino")
    print("2. Manual port selection")
    print("3. Exit")
    
    choice = input("\nChoose option: ").strip()
    
    if choice == "1":
        # Auto-detect
        port = arduino.auto_detect_arduino()
        if port and arduino.connect(port):
            return arduino
        else:
            print("❌ Auto-detection failed")
            return None
    
    elif choice == "2":
        # Manual selection
        port_num = input(f"\nEnter port number (1-{len(available_ports)}): ").strip()
        try:
            port = available_ports[int(port_num) - 1]
            if arduino.connect(port):
                return arduino
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return None
    
    return None

# ============================================
# TESTING FUNCTIONS
# ============================================

def test_arduino_connection(port=None):
    """Test Arduino connection"""
    print("\n" + "="*60)
    print("  ARDUINO CONNECTION TEST")
    print("="*60)
    
    arduino = ArduinoConnection()
    
    if port:
        success = arduino.connect(port)
    else:
        arduino_conn = interactive_setup()
        if arduino_conn:
            arduino = arduino_conn
            success = True
        else:
            success = False
    
    if not success:
        print("\n❌ Connection test failed")
        return None
    
    # Test commands
    print("\n📋 Testing commands...")
    print("─"*60)
    
    tests = [
        ('SCANNING', 'Scanning mode'),
        ('GRANTED', 'Access granted'),
        ('UNLOCK', 'Unlock door'),
        ('LOCK', 'Lock door'),
        ('DENIED', 'Access denied')
    ]
    
    for cmd, description in tests:
        print(f"\nTesting: {description}")
        arduino.send_command(cmd)
        time.sleep(1)
    
    print("\n✅ Test sequence complete")
    
    # Keep connection open for manual testing
    print("\n" + "─"*60)
    print("Manual Testing Mode")
    print("─"*60)
    print("Commands: U(nlock), L(ock), G(ranted), D(enied), S(canning), Q(uit)")
    
    while True:
        cmd = input("\nEnter command: ").strip().upper()
        
        if cmd == 'Q':
            break
        elif cmd == 'U':
            arduino.unlock_door()
        elif cmd == 'L':
            arduino.lock_door()
        elif cmd == 'G':
            arduino.signal_access_granted()
        elif cmd == 'D':
            arduino.signal_access_denied()
        elif cmd == 'S':
            arduino.signal_scanning()
        elif cmd == 'T':
            arduino.test_connection()
        else:
            print("Invalid command")
    
    arduino.disconnect()
    return arduino

# ============================================
# MAIN (FOR TESTING)
# ============================================

def main():
    """Main function for standalone testing"""
    print("\n" + "="*60)
    print("  ARDUINO COMMUNICATION MODULE")
    print("="*60)
    
    print("\nOptions:")
    print("1. Interactive setup and test")
    print("2. Quick connect (auto-detect)")
    print("3. List available ports")
    print("4. Exit")
    
    choice = input("\nChoose option: ").strip()
    
    if choice == "1":
        test_arduino_connection()
    
    elif choice == "2":
        arduino = ArduinoConnection()
        if arduino.connect():
            arduino.test_connection()
            arduino.disconnect()
    
    elif choice == "3":
        arduino = ArduinoConnection()
        arduino.list_available_ports()
    
    elif choice == "4":
        print("Goodbye!")
    
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()