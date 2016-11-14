/*  Luke Goodman (lng26)
  Link to YouTube video:
  https://www.youtube.com/watch?v=Rezz9YrnrYM */
#include <Stepper.h>
#include "pitches.h"

// pin_coil_{1,2,3,4}: the pin used for each coil (D0, D1, D2, D3)
const int pin_coil_1 = D0;
const int pin_coil_2 = D1;
const int pin_coil_3 = D2;
const int pin_coil_4 = D3;

// steps: number of steps per revolution (200)
const int steps = 200;

// knob_pin: analog pin used for potentiometer
const int knob_pin = A0;
// beep_pin: pin used for P
const int beep_pin = D4;

//Mario main theme melody
int melody[] = {
  NOTE_E7, NOTE_E7, 0, NOTE_E7,
  0, NOTE_C7, NOTE_E7, 0,
  NOTE_G7, 0, 0,  0,
  NOTE_G6, 0, 0, 0,

  NOTE_C7, 0, 0, NOTE_G6,
  0, 0, NOTE_E6, 0,
  0, NOTE_A6, 0, NOTE_B6,
  0, NOTE_AS6, NOTE_A6, 0,

  NOTE_G6, NOTE_E7, NOTE_G7,
  NOTE_A7, 0, NOTE_F7, NOTE_G7,
  0, NOTE_E7, 0, NOTE_C7,
  NOTE_D7, NOTE_B6, 0, 0,

  NOTE_C7, 0, 0, NOTE_G6,
  0, 0, NOTE_E6, 0,
  0, NOTE_A6, 0, NOTE_B6,
  0, NOTE_AS6, NOTE_A6, 0,

  NOTE_G6, NOTE_E7, NOTE_G7,
  NOTE_A7, 0, NOTE_F7, NOTE_G7,
  0, NOTE_E7, 0, NOTE_C7,
  NOTE_D7, NOTE_B6, 0, 0
};
//Mario main them tempo
int tempo[] = {
  12, 12, 12, 12,
  12, 12, 12, 12,
  12, 12, 12, 12,
  12, 12, 12, 12,

  12, 12, 12, 12,
  12, 12, 12, 12,
  12, 12, 12, 12,
  12, 12, 12, 12,

  9, 9, 9,
  12, 12, 12, 12,
  12, 12, 12, 12,
  12, 12, 12, 12,

  12, 12, 12, 12,
  12, 12, 12, 12,
  12, 12, 12, 12,
  12, 12, 12, 12,

  9, 9, 9,
  12, 12, 12, 12,
  12, 12, 12, 12,
  12, 12, 12, 12,
};

Stepper stepper(steps, pin_coil_1, pin_coil_2, pin_coil_3, pin_coil_4);

int knob_pos = 0;
int prev_pos = 0;
int pos = 0;
bool marker = false;

void setup() {
  stepper.setSpeed(40); // intialize stepper speed in RPM
  pinMode(beep_pin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  prev_pos = pos; // set previous position to position from last loop
  knob_pos = analogRead(knob_pin);  // read the knob pin

  pos = map (knob_pos, 0, 4095, 0, 200); // map the knob position to a value in the step range (0, 200)

  /*if (pos != prev_pos) {
    int step_dist = pos - prev_pos; // set the step distance to the difference between current and previous
    stepper.step(step_dist);  // turn stepper
  }*/
  if (pos < 190) {
    noTone(beep_pin);
    marker = false;
  } else if (!marker) {
    stepper.step(1000);
    sing();
    marker = true;
  }
}

void sing() {
  int size = sizeof(melody) / sizeof(int);
  for (int note = 0; note < size; note++) {

    // to calculate the note duration, take one second
    // divided by the note type.
    //e.g. quarter note = 1000 / 4, eighth note = 1000/8, etc.
    int noteDuration = 1000 / tempo[note];

    buzz(melody[note], noteDuration);

    // to distinguish the notes, set a minimum time between them.
    // the note's duration + 30% seems to work well:
    int pauseBetweenNotes = noteDuration * 1.30;
    delay(pauseBetweenNotes);

    // stop the tone playing:
    noTone(beep_pin);

  }
}

void buzz(long frequency, long length) {
  long delayValue = 1000000 / frequency / 2; // calculate the delay value between transitions
  //// 1 second's worth of microseconds, divided by the frequency, then split in half since
  //// there are two phases to each cycle
  long numCycles = frequency * length / 1000; // calculate the number of cycles for proper timing
  //// multiply frequency, which is really cycles per second, by the number of seconds to
  //// get the total number of cycles to produce
  for (long i = 0; i < numCycles; i++) { // for the calculated length of time...
    digitalWrite(beep_pin, HIGH); // write the buzzer pin high to push out the diaphram
    delayMicroseconds(delayValue); // wait for the calculated delay value
    digitalWrite(beep_pin, LOW); // write the buzzer pin low to pull back the diaphram
    delayMicroseconds(delayValue); // wait again or the calculated delay value
  }

}
