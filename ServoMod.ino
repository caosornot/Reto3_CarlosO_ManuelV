#include <SPI.h>
#include <ServoCds55.h>
ServoCds55 myservo;

int servoNum = 1;
int ROT = 0;
int CTR = 0;

void setup () {
  Serial.begin (115200);
  myservo.begin ();
  myservo.Reset(servoNum);//Restore servo to factory Settings ( ID:1  Baud rate:1000000)
  myservo.rotate(servoNum, ROT); //ID:1  Pos:300  velocity:150
}

void loop () {
  if (Serial.available()!= 0){
    ROT = Serial.readStringUntil("\n").toInt();
    //ROT = Serial.parseInt();
    //Serial.println(ROT);
    Serial.flush();
      }
    CTR = map(ROT,-50,50,-300,300);
    CTR = CTR*(0.1);
    myservo.rotate(servoNum, CTR); //ID:1  Pos:300  velocity:150
   }


