import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import cv2
from PIL import Image, ImageTk
from main import SecuritySystem, SAMPLES_PER_PERSON

class SecurityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Authentication Security System")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e1e")
        
        # Initialize backend
        self.system = SecuritySystem()
        
        # Color scheme
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#0078d4',
            'success': '#28a745',
            'danger': '#dc3545',
            'card_bg': '#2d2d2d',
            'hover': '#3d3d3d'
        }
        
        # Configure styles
        self.setup_styles()
        
        # Show main menu
        self.show_main_menu()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Menu.TButton',
                       background=self.colors['card_bg'],
                       foreground=self.colors['fg'],
                       borderwidth=0,
                       focuscolor='none',
                       font=('Segoe UI', 11))
        style.map('Menu.TButton',
                  background=[('active', self.colors['hover'])])
        
        style.configure('Title.TLabel',
                       background=self.colors['bg'],
                       foreground=self.colors['fg'],
                       font=('Segoe UI', 20, 'bold'))
        
        style.configure('Info.TLabel',
                       background=self.colors['bg'],
                       foreground=self.colors['fg'],
                       font=('Segoe UI', 10))
    
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_main_menu(self):
        self.clear_window()
        
        # Title frame
        title_frame = tk.Frame(self.root, bg=self.colors['bg'])
        title_frame.pack(pady=30)
        
        title = ttk.Label(title_frame, text="DUAL AUTHENTICATION SYSTEM",
                         style='Title.TLabel')
        title.pack()
        
        subtitle = ttk.Label(title_frame, text="⚡ Thunder Robot 15n",
                           style='Info.TLabel')
        subtitle.pack()
        
        # Menu buttons frame
        menu_frame = tk.Frame(self.root, bg=self.colors['bg'])
        menu_frame.pack(pady=20)
        
        buttons = [
            ("📡 Start System (Sensor-Triggered Authentication)", self.start_system),
            ("📝 Register New Person", self.register_person),
            ("🔐 Authenticate", self.authenticate),
            ("👥 List Users", self.list_users),
            ("➕ Add Face/Gesture", self.add_face_gesture),
            ("🧪 Test Recognition", self.test_recognition),
            ("🔧 Test External Devices", self.test_external_devices),
            ("🗑️ Delete User", self.delete_user),
            ("🚪 Exit", self.exit_app)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = tk.Button(menu_frame, text=text, command=command,
                          bg=self.colors['card_bg'],
                          fg=self.colors['fg'],
                          font=('Segoe UI', 12),
                          width=35, height=2,
                          relief=tk.FLAT,
                          cursor='hand2')
            btn.pack(pady=5)
            
            # Hover effects
            btn.bind('<Enter>', lambda e, b=btn: b.config(bg=self.colors['hover']))
            btn.bind('<Leave>', lambda e, b=btn: b.config(bg=self.colors['card_bg']))
        
        # Status bar
        status_frame = tk.Frame(self.root, bg=self.colors['card_bg'])
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        users_count = len(self.system.face_recognition.known_encodings)
        status_label = ttk.Label(status_frame, 
                                text=f"Status: Ready | Users: {users_count} | Arduino: {'Connected' if self.system.arduino.connected else 'Simulation'}",
                                style='Info.TLabel')
        status_label.pack(pady=10)
    
    def start_system(self):
        """Start sensor-triggered authentication system"""
        # Check if users are registered
        if not self.system.face_recognition.known_encodings:
            messagebox.showwarning("Warning", "No registered users found!\nPlease register users first.")
            return
        
        # Start system in thread
        threading.Thread(target=self._start_system_thread, daemon=True).start()
    
    def _start_system_thread(self):
        """Start system thread"""
        try:
            result = self.system.start_system()
            if result:
                messagebox.showinfo("Success", "System authentication successful! Access granted.")
            else:
                messagebox.showerror("Failed", "System authentication failed. Access denied.")
        except Exception as e:
            messagebox.showerror("Error", f"System authentication failed: {e}")
    
    def register_person(self):
        """Register new person with GUI input"""
        name = simpledialog.askstring("Register Person", "Enter person name:")
        if not name:
            return
        
        # Check if already exists
        if name in self.system.face_recognition.known_encodings:
            messagebox.showerror("Error", f"User '{name}' already exists")
            return
        
        # Start registration in thread
        threading.Thread(target=self._register_thread, args=(name,), daemon=True).start()
    
    def _register_thread(self, name):
        """Registration thread"""
        try:
            # Note: This still uses OpenCV windows which is unavoidable
            # Face registration requires camera which opens in separate window
            self.system.face_recognition.register_face(name, SAMPLES_PER_PERSON)
            
            # Register gesture
            if messagebox.askyesno("Gesture Registration", f"Register gesture for {name}?"):
                self.system.gesture_recognition.register_gesture(name)
            
            messagebox.showinfo("Success", f"User '{name}' registered successfully!")
            self.show_main_menu()  # Refresh menu
        except Exception as e:
            messagebox.showerror("Error", f"Registration failed: {e}")
    
    def authenticate(self):
        """Start authentication"""
        threading.Thread(target=self._authenticate_thread, daemon=True).start()
    
    def _authenticate_thread(self):
        """Authentication thread"""
        try:
            result = self.system.authenticate_person()
            if result:
                messagebox.showinfo("Success", "Authentication successful! Access granted.")
            else:
                messagebox.showerror("Failed", "Authentication failed. Access denied.")
        except Exception as e:
            messagebox.showerror("Error", f"Authentication failed: {e}")
    
    def list_users(self):
        """Show list of users in dialog"""
        users = list(self.system.face_recognition.known_encodings.keys())
        
        if not users:
            messagebox.showinfo("Users", "No registered users")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Registered Users")
        dialog.geometry("400x300")
        dialog.configure(bg=self.colors['bg'])
        
        # Title
        title = tk.Label(dialog, text="👥 Registered Users", 
                        bg=self.colors['bg'], fg=self.colors['fg'],
                        font=('Segoe UI', 14, 'bold'))
        title.pack(pady=10)
        
        # Users list
        for user in users:
            has_gesture = user in self.system.gesture_recognition.registered_gestures
            samples = len(self.system.face_recognition.known_encodings[user])
            
            user_frame = tk.Frame(dialog, bg=self.colors['card_bg'])
            user_frame.pack(fill=tk.X, padx=10, pady=5)
            
            info = f"• {user} - Samples: {samples} | Gesture: {'✅' if has_gesture else '❌'}"
            label = tk.Label(user_frame, text=info, bg=self.colors['card_bg'],
                           fg=self.colors['fg'], font=('Segoe UI', 10))
            label.pack(pady=5)
    
    def add_face_gesture(self):
        """Add face/gesture submenu"""
        users = list(self.system.face_recognition.known_encodings.keys())
        if not users:
            messagebox.showwarning("Warning", "No registered users")
            return
        
        # Ask what to add
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Face or Gesture")
        dialog.geometry("400x200")
        dialog.configure(bg=self.colors['bg'])
        
        tk.Label(dialog, text="What would you like to add?", 
                bg=self.colors['bg'], fg=self.colors['fg'],
                font=('Segoe UI', 12)).pack(pady=20)
        
        def add_face():
            dialog.destroy()
            self._add_face()
        
        def add_gesture():
            dialog.destroy()
            self._add_gesture()
        
        tk.Button(dialog, text="📸 Add Face Samples", command=add_face,
                 bg=self.colors['accent'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=20, height=2).pack(pady=5)
        
        tk.Button(dialog, text="✋ Add/Update Gesture", command=add_gesture,
                 bg=self.colors['accent'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=20, height=2).pack(pady=5)
    
    def _add_face(self):
        """Add more face samples"""
        users = list(self.system.face_recognition.known_encodings.keys())
        name = self._select_user_dialog(users, "Select user to add face samples:")
        if not name:
            return
        
        num_str = simpledialog.askstring("Number of Samples", 
                                        f"How many samples? (default {SAMPLES_PER_PERSON}):")
        num = int(num_str) if num_str and num_str.isdigit() else SAMPLES_PER_PERSON
        
        threading.Thread(target=self._add_face_thread, args=(name, num), daemon=True).start()
    
    def _add_face_thread(self, name, num):
        try:
            self.system.face_recognition.register_face(name, num)
            messagebox.showinfo("Success", f"Added {num} samples for {name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
    
    def _add_gesture(self):
        """Add/update gesture"""
        users = list(self.system.face_recognition.known_encodings.keys())
        name = self._select_user_dialog(users, "Select user to add/update gesture:")
        if not name:
            return
        
        threading.Thread(target=self._add_gesture_thread, args=(name,), daemon=True).start()
    
    def _add_gesture_thread(self, name):
        try:
            self.system.gesture_recognition.register_gesture(name)
            messagebox.showinfo("Success", f"Gesture registered for {name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {e}")
    
    def test_recognition(self):
        """Test face/gesture submenu"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Test Recognition")
        dialog.geometry("400x200")
        dialog.configure(bg=self.colors['bg'])
        
        tk.Label(dialog, text="What would you like to test?", 
                bg=self.colors['bg'], fg=self.colors['fg'],
                font=('Segoe UI', 12)).pack(pady=20)
        
        def test_face():
            dialog.destroy()
            threading.Thread(target=self._test_face, daemon=True).start()
        
        def test_gesture():
            dialog.destroy()
            self._test_gesture()
        
        tk.Button(dialog, text="👁️ Test Face Recognition", command=test_face,
                 bg=self.colors['accent'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=20, height=2).pack(pady=5)
        
        tk.Button(dialog, text="✋ Test Gesture Recognition", command=test_gesture,
                 bg=self.colors['accent'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=20, height=2).pack(pady=5)
    
    def test_external_devices(self):
        """Test all external devices submenu"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Test External Devices")
        dialog.geometry("450x500")
        dialog.configure(bg=self.colors['bg'])
        
        tk.Label(dialog, text="🔧 External Devices Testing", 
                bg=self.colors['bg'], fg=self.colors['fg'],
                font=('Segoe UI', 14, 'bold')).pack(pady=20)
        
        def test_face():
            dialog.destroy()
            threading.Thread(target=self._test_face, daemon=True).start()
        
        def test_gesture():
            dialog.destroy()
            self._test_gesture()
        
        def test_arduino():
            dialog.destroy()
            self._test_arduino()
        
        def test_sonar():
            dialog.destroy()
            threading.Thread(target=self.system.arduino.test_sonar, daemon=True).start()
        
        def test_servo():
            dialog.destroy()
            threading.Thread(target=self.system.arduino.test_servo, daemon=True).start()
        
        def manual_servo():
            dialog.destroy()
            self._manual_servo_control()
        
        tk.Button(dialog, text="👁️ Test Face Recognition", command=test_face,
                 bg=self.colors['card_bg'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=25, height=2).pack(pady=5)
        
        tk.Button(dialog, text="✋ Test Gesture Recognition", command=test_gesture,
                 bg=self.colors['card_bg'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=25, height=2).pack(pady=5)
        
        tk.Button(dialog, text="🔧 Test Arduino", command=test_arduino,
                 bg=self.colors['card_bg'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=25, height=2).pack(pady=5)
        
        tk.Button(dialog, text="📡 Test Sonar Sensor", command=test_sonar,
                 bg=self.colors['card_bg'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=25, height=2).pack(pady=5)
        
        tk.Button(dialog, text="⚙️ Test Servo System", command=test_servo,
                 bg=self.colors['card_bg'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=25, height=2).pack(pady=5)
        
        tk.Button(dialog, text="🎮 Manual Servo Control", command=manual_servo,
                 bg=self.colors['card_bg'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=25, height=2).pack(pady=5)
    
    def _test_face(self):
        try:
            name, conf = self.system.face_recognition.verify_face_continuous(duration=None)
            if name:
                messagebox.showinfo("Result", f"Recognized: {name} ({conf:.0%})")
            else:
                messagebox.showwarning("Result", "Face not recognized")
        except Exception as e:
            messagebox.showerror("Error", f"Test failed: {e}")
    
    def _test_gesture(self):
        users = list(self.system.gesture_recognition.registered_gestures.keys())
        if not users:
            messagebox.showwarning("Warning", "No registered gestures")
            return
        
        name = self._select_user_dialog(users, "Select user to test gesture:")
        if not name:
            return
        
        threading.Thread(target=self._test_gesture_thread, args=(name,), daemon=True).start()
    
    def _test_gesture_thread(self, name):
        try:
            self.system.gesture_recognition.test_gesture(name)
        except Exception as e:
            messagebox.showerror("Error", f"Test failed: {e}")
    
    def _test_arduino(self):
        if self.system.arduino.connected:
            self.system.arduino.send_command("TEST")
            messagebox.showinfo("Arduino Test", "Test command sent")
        else:
            messagebox.showwarning("Arduino", "Arduino not connected - running in simulation mode")
    
    def _manual_servo_control(self):
        """Manual servo control dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manual Servo Control")
        dialog.geometry("400x300")
        dialog.configure(bg=self.colors['bg'])
        
        tk.Label(dialog, text="🎮 Manual Servo Control", 
                bg=self.colors['bg'], fg=self.colors['fg'],
                font=('Segoe UI', 14, 'bold')).pack(pady=20)
        
        def servo_right():
            self.system.arduino.servo_right()
        
        def servo_left():
            self.system.arduino.servo_left()
        
        def servo_stop():
            self.system.arduino.servo_stop()
        
        def set_speed():
            speed_str = simpledialog.askstring("Set Speed", "Enter speed (0-100):", initialvalue="60")
            if speed_str and speed_str.isdigit():
                speed_val = int(speed_str)
                if 0 <= speed_val <= 100:
                    self.system.arduino.servo_set_speed(speed_val)
                    messagebox.showinfo("Success", f"Speed set to {speed_val}%")
                else:
                    messagebox.showerror("Error", "Speed must be between 0-100")
            elif speed_str:
                messagebox.showerror("Error", "Invalid speed value")
        
        tk.Button(dialog, text="🔄 Rotate Right", command=servo_right,
                 bg=self.colors['success'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=15, height=2).pack(pady=5)
        
        tk.Button(dialog, text="🔄 Rotate Left", command=servo_left,
                 bg=self.colors['success'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=15, height=2).pack(pady=5)
        
        tk.Button(dialog, text="🛑 Stop", command=servo_stop,
                 bg=self.colors['danger'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=15, height=2).pack(pady=5)
        
        tk.Button(dialog, text="⚡ Set Speed", command=set_speed,
                 bg=self.colors['accent'], fg=self.colors['fg'],
                 font=('Segoe UI', 11), width=15, height=2).pack(pady=5)
    
    def delete_user(self):
        """Delete user"""
        users = list(self.system.face_recognition.known_encodings.keys())
        if not users:
            messagebox.showwarning("Warning", "No registered users")
            return
        
        name = self._select_user_dialog(users, "Select user to delete:")
        if not name:
            return
        
        if messagebox.askyesno("Confirm", f"Delete user '{name}'?"):
            try:
                self.system.delete_user_by_name(name)
                messagebox.showinfo("Success", f"User '{name}' deleted")
                self.show_main_menu()  # Refresh
            except Exception as e:
                messagebox.showerror("Error", f"Deletion failed: {e}")
    
    def _select_user_dialog(self, users, prompt):
        """Show dialog to select a user"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select User")
        dialog.geometry("300x400")
        dialog.configure(bg=self.colors['bg'])
        
        tk.Label(dialog, text=prompt, bg=self.colors['bg'], fg=self.colors['fg'],
                font=('Segoe UI', 11)).pack(pady=10)
        
        selected = tk.StringVar()
        
        for user in users:
            rb = tk.Radiobutton(dialog, text=user, variable=selected, value=user,
                               bg=self.colors['bg'], fg=self.colors['fg'],
                               selectcolor=self.colors['card_bg'],
                               font=('Segoe UI', 10))
            rb.pack(anchor=tk.W, padx=20, pady=5)
        
        result = [None]
        
        def confirm():
            result[0] = selected.get()
            dialog.destroy()
        
        tk.Button(dialog, text="Confirm", command=confirm,
                 bg=self.colors['accent'], fg=self.colors['fg'],
                 font=('Segoe UI', 10)).pack(pady=10)
        
        dialog.wait_window()
        return result[0]
    
    def exit_app(self):
        """Exit application"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.system.close()
            self.root.quit()

def main():
    root = tk.Tk()
    app = SecurityGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
