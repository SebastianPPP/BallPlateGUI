#include <Servo.h>

Servo servo1;
Servo servo2;
Servo servo3; 

// --- PARAMETRY KONFIGURACYJNE ---
// Bezpieczna wartość startowa w mikrosekundach (np. 1300, 1000, 1500 w zależności od montażu orczyka)
const int BASE_PULSE1 = 1790; 
const int BASE_PULSE2 = 1862; 
const int BASE_PULSE3 = 1336; 
// Bezpieczny limit wychylenia w stopniach chroniący przeguby
const float MAX_ANGLE = 20.0; 

// Przelicznik stopni na mikrosekundy (zazwyczaj od 10.0 do 12.0 w zależności od serwa)
const float US_PER_DEGREE = 12; 
// --------------------------------

const byte numChars = 64;
char receivedChars[numChars];
boolean newData = false;

float u1_out = 0.0;
float u2_out = 0.0;
float u3_out = 0.0;

void setup() {
    Serial.begin(115200);
    
    // Podłącz serwa pod piny z obsługą PWM
    servo1.attach(9);
    servo2.attach(10);
    servo3.attach(11);
    
    // Ustawienie w bezpiecznej pozycji początkowej zaraz po restarcie
    servo1.writeMicroseconds(1790);
    servo2.writeMicroseconds(1862);
    servo3.writeMicroseconds(1336);
}

void loop() {
    recvWithStartEndMarkers();
    if (newData == true) {
        parseData();
        updateServos();
        newData = false;
    }
}

// Funkcja nieblokująca, czytająca ramkę <u1, u2, u3>
void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            } else {
                receivedChars[ndx] = '\0'; // Zakończenie stringa
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        } else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

// Parsowanie danych tekstowych na zmiennoprzecinkowe (float)
void parseData() {
    char * strtokIndx;
    strtokIndx = strtok(receivedChars, ",");
    if (strtokIndx != NULL) u1_out = atof(strtokIndx); 
    
    strtokIndx = strtok(NULL, ",");
    if (strtokIndx != NULL) u2_out = atof(strtokIndx);
    
    strtokIndx = strtok(NULL, ",");
    if (strtokIndx != NULL) u3_out = atof(strtokIndx);
}

// Przeliczenie kątów na mikrosekundy i wysłanie do serw
void updateServos() {
    // Zabezpieczenie przed przekroczeniem bezpiecznego limitu
    u1_out = constrain(u1_out, -MAX_ANGLE, MAX_ANGLE);
    u2_out = constrain(u2_out, -MAX_ANGLE, MAX_ANGLE);
    u3_out = constrain(u3_out, -MAX_ANGLE, MAX_ANGLE);

    // Konwersja z uwzględnieniem bezpiecznej bazy
    int ms1 = BASE_PULSE1 + (int)(u1_out * US_PER_DEGREE);
    int ms2 = BASE_PULSE2 + (int)(u2_out * US_PER_DEGREE);
    int ms3 = BASE_PULSE3 + (int)(u3_out * US_PER_DEGREE);

    servo1.writeMicroseconds(ms1);
    servo2.writeMicroseconds(ms2);
    servo3.writeMicroseconds(ms3);
}
