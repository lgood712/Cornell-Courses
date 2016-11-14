//Based on the code by Chris Ball - chrisballprojects.wordpress.com

#include "pitches.h"

const int notes_laser=7;          // set up number of notes

boolean notePlaying[notes_laser];  //note states array: keeps track of whether a MIDI note is "on" or "off"

int scale[notes_laser]={0,2,4,5,7,9,11};       //major scale

int beginNote=60;                                  //starting note value (default middle C)

int threshold=50;                                //value which analogRead() must cross for a midi note to be triggered

int pitchbend = 224;

int pitchbendVal = 8192;

int value = 0;
bool midiMode=true;                          //set this to "true" before uploading MIDI firmware (http://hunt.net.nz/users/darran/weblog/52882/Arduino_UNO_MIDI_USB_version_02.html) to 16u2

const int sensor_pin[notes_laser] = {A0,A1,A2,A3,A4,A5,A6};
//constants for lasers
const int num_lasers = 7;
const int laser_pin[7] = {D0,D1,D2,D3,D4,D5,D6};
const char notes[7] = {'C','D','E','F','G','A','B'};
int instruments[] ={1,9, 12, 26, 50, 100,128};  // change between 1 - 128

bool on_list[8] = {0,0,0,0,0,0,0,0}; //indicates status of a laser (0=off, 1=on)

char stop = 'S'; //to indicate end of song
// TUTORIAL SONGS:
// Twinkle Twinkle Little Star
char twinkle_notes[] = "CCGGAAG FFEEDDC GGFFEED GGFFEED CCGGAAG FFEEDDC S"; // a space represents a rest
int twinkle_beats[] = { 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2,
                        1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2,
                        1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1 };
int twinkle_tempo = 300;
// Harry Potter song
char potter_notes[] = "DFEDAGE DFECE S"; // a space represents a rest
int potter_beats[] = { 3, 1, 1, 2, 1, 3, 3,
                        3, 1, 1, 2, 3};
int potter_tempo = 300;

// One Love by Bob Marley
char one_love_notes[] = "EEDD FEDCDCEDC EEDD EFFFEDC CCDEDC FEDCDCEDC S"; // a space represents a rest
int one_love_beats[] = { 2, 3, 2, 3, 2,
                        1, 1, 1, 3, 1, 2, 1, 1, 2, 2,
                        2, 3, 2, 3, 2,
                        1, 1, 1, 3, 1, 2, 1,
                        1, 1, 1, 1, 2, 1, 1,
                        1, 1, 1, 3, 1, 2, 1, 1, 2, 2, 1};
int one_love_tempo = 300;

int intstatus = 11000000;
int noteOn = 10000000;
int noteOff = 10010000;
int MIDIControll = 10110000;

int buttonState = 0;  // variable for reading pushbutton status
int InstrButtonState = 0;
const int buttonPin = D7;
const int buttonInstrument = A7;
int instrument_index=0;

int harp_mode = 0; //records which mode the harp is in

void setup(){
  if(midiMode){
    Serial.begin(115200);           //midi USB baud rate = 115200
  }

  pinMode(buttonPin, INPUT); // initialize pushbutton as input
  pinMode(buttonInstrument, INPUT);
  for (int p=0; p<num_lasers; p++) {
    pinMode(laser_pin[p], OUTPUT);
    digitalWrite(laser_pin[p], HIGH);
  }
}


void loop() {
  checkButton(); //check if button was pressed
  if (harp_mode == 0) {
    //Serial.println(buttonState);
    /*Serial.println("==0");*/
    //Serial.println(harp_mode);
    freePlayMode();

  } else if (harp_mode > 0) {
    //Serial.println(buttonState);
    /*Serial.println("==1");*/
    //Serial.println(harp_mode);
    tutorialMode();
  }
  //freePlayMode();
  delay(500);
}

int checkButton() {
  // read the state of the pushbutton value:
  int prevState = buttonState;
  buttonState = digitalRead(buttonPin);
  // check if the pushbutton was pressed, switch to next mode if so
  if (buttonState == LOW && prevState == HIGH) {
    if (harp_mode < 3) {
      harp_mode++;
    } else {
      harp_mode = 0;
    }
  }
  return harp_mode;
}

void checkInstrButton() {
  // read the state of the pushbutton value:
  int prevState = InstrButtonState;
  InstrButtonState = digitalRead(buttonInstrument);
  Serial.print(InstrButtonState);
  // check if the pushbutton was pressed, switch to next mode if so
  if (InstrButtonState == 0 && prevState == 1) {
    if (instrument_index < 7) {
      Serial.println("index");
      Serial.println(instrument_index);
      changeInstrument(intstatus,instruments[instrument_index]);
      Serial.println(instrument_index);
      instrument_index ++;
    }
  }
}


/* Switches laser on or off based on note it represents ('C'-'G' + 'A'-'B')
  Returns the integer index of the laser */
int laserSwitch(byte laser, int on_off) {
  switch (laser) {
    case 'C':
      /*if (on_list[0]) {
        digitalWrite(laser_pin[0], LOW);
        on_list[0] = 0;
      } else {
        digitalWrite(laser_pin[0], HIGH);
        on_list[0] = 1;
      }*/
      digitalWrite(laser_pin[0], on_off);
      return 0;
      break;
    case 'D':
      /*if (on_list[1]) {
        digitalWrite(laser_pin[1], LOW);
        on_list[1] = 0;
      } else {
        digitalWrite(laser_pin[1], HIGH);
        on_list[1] = 1;
      }*/
      digitalWrite(laser_pin[1], on_off);
      return 1;
      break;
    case 'E':
      /*if (on_list[2]) {
        digitalWrite(laser_pin[2], LOW);
        on_list[2] = 0;
      } else {
        digitalWrite(laser_pin[2], HIGH);
        on_list[2] = 1;
      }*/
      digitalWrite(laser_pin[2], on_off);
      return 2;
      break;
    case 'F':
      /*if (on_list[3]) {
        digitalWrite(laser_pin[3], LOW);
        on_list[3] = 0;
      } else {
        digitalWrite(laser_pin[3], HIGH);
        on_list[3] = 1;
      }*/
      digitalWrite(laser_pin[3], on_off);
      return 3;
      break;
    case 'G':
      /*if (on_list[4]) {
        digitalWrite(laser_pin[4], LOW);
        on_list[4] = 0;
      } else {
        digitalWrite(laser_pin[4], HIGH);
        on_list[4] = 1;
      }*/
      digitalWrite(laser_pin[4], on_off);
      return 4;
      break;
    case 'A':
      /*if (on_list[5]) {
        digitalWrite(laser_pin[5], LOW);
        on_list[5] = 0;
      } else {
        digitalWrite(laser_pin[5], HIGH);
        on_list[5] = 1;
      }*/
      digitalWrite(laser_pin[5], on_off);
      return 5;
      break;
    case 'B':
      /*if (on_list[6]) {
        digitalWrite(laser_pin[6], LOW);
        on_list[6] = 0;
      } else {
        digitalWrite(laser_pin[6], HIGH);
        on_list[6] = 1;
      }*/
      digitalWrite(laser_pin[6], on_off);
      return 6;
      break;
  }
}

//change instrument in freePlayMode
void changeInstrument(int status, int inst){
  Serial.write(status);
  Serial.write(inst);
  delay(200);
}

void freePlayMode(){
  for (int p=0; p<num_lasers; p++) {
    digitalWrite(laser_pin[p], HIGH);
  }
  //checkInstrButton();

  for(int i=0;i<notes_laser;i++){
    value = analogRead(i);
    int distance = map(value, 0, 4095, 100, 15);
    if((distance<threshold)&&(notePlaying[i]==false)){        //note on & CC messages
      int sensorRead = analogRead(i);
      int sensorVal = map(sensorRead, 0,4095,7000, 0);
      //Serial.print(sensorRead);
      int note=beginNote+scale[i];
      /*if (instruments[i] != 0) {*/
        /*changeInstrument(intstatus,instruments[i]);*/
          MIDImessage(144, note, 100);
          notePlaying[i]=true;
        /*notePlaying[i]=true;*/
      /*}*/
      /*if (i==1){
        MIDImessage(10011000, note+1, 100);
        notePlaying[i]=true;
      }else if (i==2){
        MIDImessage(10110001, 0, 100);
        notePlaying[i]=true;
      }else if (i==3){
        MIDImessage(10110002, 0, 100);
        notePlaying[i]=true;
      }else{
        int note=beginNote+scale[i];
        MIDImessage(0x90, note, 100);
        delay(1);
        notePlaying[i]=true;
      }*/



    }

    if((distance>=threshold)&&(notePlaying[i]==true)){        //note off messages
      int note=beginNote+scale[i];
      pitchbendVal = 8192+40;
      //MIDImessage(pitchbend, (pitchbendVal&127), (pitchbendVal>>7));//send pitchbend message
      MIDImessage(128, note, 0);
      notePlaying[i]=false;
      }
  }
}

void MIDImessage(int command, int MIDInote, int MIDIvelocity) {
    Serial.write(command); //send note on or note off command
    Serial.write(MIDInote); //send pitch data
    Serial.write(MIDIvelocity); //send velocity data
}

/*Tutorial Mode teaches the user a specified song by turning on lasers
corresponding to the correct notes in sequence*/
void tutorialMode() {
  delay(500); // short delay

  // start teaching song
  if (harp_mode == 1) {
    for (int p=0; p<num_lasers; p++) {
      digitalWrite(laser_pin[p], LOW);
    }
    teachSong(twinkle_notes, twinkle_beats, twinkle_tempo, 1);
  }
  if (harp_mode == 2) {
    for (int p=0; p<num_lasers; p++) {
      digitalWrite(laser_pin[p], LOW);
    }
    teachSong(potter_notes, potter_beats, potter_tempo, 2);
  }
  if (harp_mode == 3) {
    for (int p=0; p<num_lasers; p++) {
      digitalWrite(laser_pin[p], LOW);
    }
    teachSong(one_love_notes, one_love_beats, one_love_tempo, 3);
  }
  harp_mode = 0; // switch to freePlayMode
}

/*Teaches the user a song given a char list of the song notes
and two int lists of the beats and tempo*/
void teachSong(char song[], int song_beats[], int tempo, int mode) {
  // loop through song notes
  for (int i = 0; song[i]!=stop; i++) {
    if (song[i] == ' ') {
      delay(song_beats[i] * tempo); // rest
    } else {
      int index = laserSwitch(song[i], HIGH); // switch on laser and get the index for corresponding sensor
      bool no_interrupt = true; // indicates whether the user has played the note yet (true=not yet played)
      while (no_interrupt) {        // wait for interruption of specified laser/sensor
        if (checkButton()  != mode) return; //check to see if they want to change modes midsong
        value = analogRead(sensor_pin[index]);
        int distance = map(value, 0, 4095, 100, 15);
        if(distance<threshold){        // interruption of string
          int note=beginNote+scale[index];
          // play note for amount of time specified by beats*tempo,
          // then turn off laser and break while loop
          if (index==1) {
            MIDImessage(144, note, 100);
            laserSwitch(song[i], LOW);
            delay(song_beats[i] * tempo);
            MIDImessage(128, note, 0);
            no_interrupt = false;
          } else if (index==2) {
            MIDImessage(144, note, 100);
            laserSwitch(song[i], LOW);
            delay(song_beats[i] * tempo);
            MIDImessage(128, 0, 0);
            no_interrupt = false;
          } else{
            int note=beginNote+scale[index];
            MIDImessage(144, note, 100);
            laserSwitch(song[i], LOW);
            delay(song_beats[i] * tempo);
            MIDImessage(128, note, 0);
            no_interrupt = false;
          }
      }
      /*while (!no_interrupt) {
        if (checkButton() != 2) return; //check to see if they want to change modes midsong
        value = analogRead(sensor_pin[index]);
        distance = map(value, 0, 4095, 100, 15);
        if(distance>=threshold) {
          no_interrupt = true;
        }
      }*/
    delay(200); // pause between notes
    }
  }
}
}


/*Concert Mode plays a song for the users while giving a laser show in sync with
  the music */
void concertMode() {
  // turn off all lasers to start
  for (int p=0; p<num_lasers; p++) {
    digitalWrite(laser_pin[p], LOW);
  }
  delay(500); // short delay

  // start teaching Twinkle Twinkle
  singSong(twinkle_notes, twinkle_beats, twinkle_tempo);
}

/*Plays song and flashes corresponding lasers*/
void singSong(char song[], int song_beats[], int tempo) {
    // loop through song notes
    for (int i = 0; song[i]!=stop; i++) {
      if (song[i] == ' ') {
        delay(song_beats[i] * tempo); // rest
      } else {
        int index = laserSwitch(song[i], HIGH); // switch on laser and get the index for corresponding sensor
        if (checkButton() != 1) return; //check to see if they want to change modes midsong
          int note=beginNote+scale[index];
          // play note for amount of time specified by beats*tempo,
          // then turn off laser and break while loop
          if (index==1) {
            MIDImessage(144, note, 100);
            delay(song_beats[i] * tempo);
            laserSwitch(song[i], LOW);
            MIDImessage(128, note, 0);
          } else if (index==2) {
            MIDImessage(144, note, 100);
            delay(song_beats[i] * tempo);
            laserSwitch(song[i], LOW);
            MIDImessage(128, 0, 0);
          } else{
            int note=beginNote+scale[index];
            MIDImessage(144, note, 100);
            delay(song_beats[i] * tempo);
            laserSwitch(song[i], LOW);
            MIDImessage(128, note, 0);
          }

        }

      delay(tempo / 2); // pause between notes
    }
  }

/*

int song = 0;

void concertMode() {
  //sing the tunes
  sing(1);
  sing(1);
  sing(2);

}*/
