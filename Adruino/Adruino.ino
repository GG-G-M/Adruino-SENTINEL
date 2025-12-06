#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo gateServo;
#define SERVO_PIN 9
#define BUZZER_PIN 7

String currentCommand = "";

void setup() {
  Serial.begin(9600);
  
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("Security System");
  
  gateServo.attach(SERVO_PIN);
  gateServo.write(0);  // Closed position
  
  pinMode(BUZZER_PIN, OUTPUT);
  tone(BUZZER_PIN, 1500, 200);
  delay(1000);
  
  lcd.clear();
  lcd.print("Ready");
  lcd.setCursor(0, 1);
  lcd.print("Waiting...");
}

void loop() {
  if (Serial.available() > 0) {
    currentCommand = Serial.readStringUntil('\n');
    currentCommand.trim();
    
    if (currentCommand == "GRANTED") accessGranted();
    else if (currentCommand == "DENIED") accessDenied();
    else if (currentCommand == "READY") systemReady();
    else if (currentCommand == "FACE_VERIFIED") faceVerified();
    else if (currentCommand == "GESTURE_REQUIRED") gestureRequired();
    else if (currentCommand == "TEST") testComponents();
  }
}

void accessGranted() {
  lcd.clear();
  lcd.print("ACCESS GRANTED");
  tone(BUZZER_PIN, 2000, 800);
  
  // Open gate
  gateServo.write(90);
  delay(3000);
  
  // Close gate
  gateServo.write(0);
  delay(1000);
  
  lcd.clear();
  lcd.print("Ready");
}

void accessDenied() {
  lcd.clear();
  lcd.print("ACCESS DENIED");
  for(int i=0; i<3; i++) {
    tone(BUZZER_PIN, 4000, 500);
    delay(600);
  }
  delay(2000);
  lcd.clear();
  lcd.print("Ready");
}

void systemReady() {
  lcd.clear();
  lcd.print("System Ready");
  tone(BUZZER_PIN, 1500, 300);
}

void faceVerified() {
  lcd.clear();
  lcd.print("Face Verified");
  lcd.setCursor(0, 1);
  lcd.print("Show Gesture...");
  tone(BUZZER_PIN, 1500, 200);
  delay(300);
  tone(BUZZER_PIN, 1500, 200);
}

void gestureRequired() {
  lcd.clear();
  lcd.print("Gesture");
  lcd.setCursor(0, 1);
  lcd.print("Required");
  for(int i=0; i<2; i++) {
    tone(BUZZER_PIN, 1500, 100);
    delay(200);
  }
}

void testComponents() {
  lcd.clear();
  lcd.print("Testing...");
  
  // Test servo
  gateServo.write(90);
  delay(1000);
  gateServo.write(0);
  delay(1000);
  
  // Test buzzer
  tone(BUZZER_PIN, 2000, 500);
  delay(1000);
  
  lcd.clear();
  lcd.print("Test Complete");
  delay(2000);
  lcd.clear();
  lcd.print("Ready");
}