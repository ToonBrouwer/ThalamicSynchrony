// =============================================================================
// Pulvinar_task2.ino
// =============================================================================
// Trial types (6 total, one dedicated TTL pin each):
//   AUDIO_2K4K  — 2kHz repeated (3-5x) then 4kHz oddball
//   AUDIO_4K2K  — 4kHz repeated (3-5x) then 2kHz oddball
//   AUDIO_6K8K  — 6kHz repeated (3-5x) then 8kHz oddball
//   AUDIO_8K6K  — 8kHz repeated (3-5x) then 6kHz oddball
//   VIS_SINGLE  — single 100ms flash
//   VIS_SEQ     — sequence of N flashes (N = 3, 4, or 5, drawn from repR)
//
// TTL STRATEGY:
//   trialPin     — fires HIGH for 10ms at prestim onset (marks trial start)
//   ttlAudio2k4k — fires HIGH for 10ms at onset of 4kHz oddball
//   ttlAudio4k2k — fires HIGH for 10ms at onset of 2kHz oddball
//   ttlAudio6k8k — fires HIGH for 10ms at onset of 8kHz oddball
//   ttlAudio8k6k — fires HIGH for 10ms at onset of 6kHz oddball
//   ttlVisSingle — fires HIGH for 10ms at flash onset
//   ttlVisSeq    — fires HIGH for 10ms at onset of each flash in sequence
//
// This gives you:
//   - Trial identity from which type TTL fires
//   - Oddball/flash onset timing with 10ms marker pulse
//   - Trial start timing from trialPin
//   - Flash count recoverable from number of ttlVisSeq pulses per trial
//
// HARDWARE (Arduino Mega):
//   Speaker      — pin 7
//   LEDpin1      — pin 49
//   LEDpin2      — pin 51
//   LEDpin3      — pin 53
//   trialPin     — pin 22  (trial start TTL → all recording systems)
//   ttlAudio2k4k — pin 23
//   ttlAudio4k2k — pin 25
//   ttlAudio6k8k — pin 27
//   ttlAudio8k6k — pin 29
//   ttlVisSingle — pin 31
//   ttlVisSeq    — pin 33
// =============================================================================

// ── USER PARAMETERS ──────────────────────────────────────────────────────────

const int maxTrialNum = 120;   // total trials (20 per type × 6 types)
const int prestimDur  = 1000;  // ms before stimulus
const int stimInterval = 1000; // ms between tones (includes tone duration)
const int toneDur     = 100;   // ms each tone
const int lightDur    = 100;   // ms each flash
const int flashITI    = 300;   // ms between flashes in sequence (gap only)
const int ttlDur      = 10;    // ms TTL pulse width

// Pseudorandom repeat counts (3–5) — same approach as original
// Used for: tone repeats before oddball AND flash count in VIS_SEQ
const int repR[] = {
  3,5,4,5,3,4,5,4,3,3,4,5,4,5,3,4,3,5,3,5,
  4,5,3,4,5,4,3,3,4,5,4,5,3,4,3,5,3,5,4,5,
  3,4,5,4,3,3,4,5,4,5,3,4,3,5,3,5,4,5,3,4,
  5,4,3,3,4,5,4,5,3,4,3,5,3,5,4,5,3,4,5,4,
  3,3,4,5,4,5,3,4,3,5,3,5,4,5,3,4,5,4,3,3,
  4,5,4,5,3,4,3,5,3,5,4,5,3,4,5,4,3,3,4,5
};

// ── PIN DEFINITIONS ───────────────────────────────────────────────────────────

const int speakerPin  = 7;
const int LEDpin1     = 49;
const int LEDpin2     = 51;
const int LEDpin3     = 53;

// One TTL pin per trial type + one for trial start
const int trialPin     = 22;   // fires at start of every trial
const int ttlAudio2k4k = 23;   // fires at oddball onset: 4kHz
const int ttlAudio4k2k = 25;   // fires at oddball onset: 2kHz
const int ttlAudio6k8k = 27;   // fires at oddball onset: 8kHz
const int ttlAudio8k6k = 29;   // fires at oddball onset: 6kHz
const int ttlVisSingle = 31;   // fires at single flash onset
const int ttlVisSeq    = 33;   // fires at each flash onset in sequence

// ── TONE FREQUENCIES ─────────────────────────────────────────────────────────

const int FREQ_2K = 2000;
const int FREQ_4K = 4000;
const int FREQ_6K = 6000;
const int FREQ_8K = 8000;

// ── TRIAL TYPES ───────────────────────────────────────────────────────────────

const int AUDIO_2K4K = 0;
const int AUDIO_4K2K = 1;
const int AUDIO_6K8K = 2;
const int AUDIO_8K6K = 3;
const int VIS_SINGLE = 4;
const int VIS_SEQ    = 5;
const int NUM_TYPES  = 6;

// Pre-shuffled trial sequence: 20 reps × 6 types = 120 trials
// Generated with seed 42: each block of 6 contains all types once,
// repeated 20 times and shuffled. Edit maxTrialNum if you change this.
// To regenerate: python3 -c "
//   import numpy as np; rng=np.random.default_rng(42)
//   t=np.tile(range(6),20); rng.shuffle(t); print(','.join(map(str,t)))"
const int trialSequence[120] = {
  4,2,0,5,3,1,2,4,1,3,0,5,1,3,5,0,4,2,3,0,
  5,2,4,1,0,3,1,4,2,5,5,0,3,1,2,4,2,5,4,0,
  3,1,1,2,0,4,5,3,4,3,2,5,1,0,0,1,3,4,2,5,
  3,5,0,2,4,1,5,4,1,0,3,2,0,2,5,3,1,4,1,0,
  4,5,2,3,2,1,3,0,5,4,4,3,5,2,0,1,0,4,2,1,
  5,3,3,2,4,1,5,0,1,5,0,3,4,2,5,1,2,0,3,4
};

// ── STATE ─────────────────────────────────────────────────────────────────────

int trialNum = 0;

// ── HELPERS ───────────────────────────────────────────────────────────────────

// All TTL pins
const int allTTL[] = {
  trialPin, ttlAudio2k4k, ttlAudio4k2k,
  ttlAudio6k8k, ttlAudio8k6k, ttlVisSingle, ttlVisSeq
};
const int nTTL = 7;

void pulseTTL(int pin) {
  digitalWrite(pin, HIGH);
  delay(ttlDur);
  digitalWrite(pin, LOW);
}

void flashLEDs(bool on) {
  digitalWrite(LEDpin1, on ? HIGH : LOW);
  digitalWrite(LEDpin2, on ? HIGH : LOW);
  digitalWrite(LEDpin3, on ? HIGH : LOW);
}

// ── SETUP ─────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(9600);

  pinMode(LEDpin1, OUTPUT); digitalWrite(LEDpin1, LOW);
  pinMode(LEDpin2, OUTPUT); digitalWrite(LEDpin2, LOW);
  pinMode(LEDpin3, OUTPUT); digitalWrite(LEDpin3, LOW);
  pinMode(speakerPin, OUTPUT);

  for (int i = 0; i < nTTL; i++) {
    pinMode(allTTL[i], OUTPUT);
    digitalWrite(allTTL[i], LOW);
  }

  trialNum = 0;
}

// ── MAIN LOOP ─────────────────────────────────────────────────────────────────

void loop() {
  if (trialNum >= maxTrialNum) return;

  int trialType = trialSequence[trialNum];
  int reps      = repR[trialNum];   // 3, 4, or 5

  // ── Prestim period — TTL marks trial start ─────────────────────────────
  pulseTTL(trialPin);
  delay(prestimDur - ttlDur);

  // ── Stimulus ───────────────────────────────────────────────────────────
  switch (trialType) {

    case AUDIO_2K4K:
      // Standard: 2kHz × reps, then 4kHz oddball
      for (int i = 0; i < reps; i++) {
        tone(speakerPin, FREQ_2K, toneDur);
        delay(stimInterval - toneDur);
      }
      pulseTTL(ttlAudio2k4k);               // TTL at oddball onset
      tone(speakerPin, FREQ_4K, toneDur);
      delay(stimInterval - toneDur - ttlDur);
      break;

    case AUDIO_4K2K:
      // Inverted: 4kHz × reps, then 2kHz oddball
      for (int i = 0; i < reps; i++) {
        tone(speakerPin, FREQ_4K, toneDur);
        delay(stimInterval - toneDur);
      }
      pulseTTL(ttlAudio4k2k);
      tone(speakerPin, FREQ_2K, toneDur);
      delay(stimInterval - toneDur - ttlDur);
      break;

    case AUDIO_6K8K:
      // Standard: 6kHz × reps, then 8kHz oddball
      for (int i = 0; i < reps; i++) {
        tone(speakerPin, FREQ_6K, toneDur);
        delay(stimInterval - toneDur);
      }
      pulseTTL(ttlAudio6k8k);
      tone(speakerPin, FREQ_8K, toneDur);
      delay(stimInterval - toneDur - ttlDur);
      break;

    case AUDIO_8K6K:
      // Inverted: 8kHz × reps, then 6kHz oddball
      for (int i = 0; i < reps; i++) {
        tone(speakerPin, FREQ_8K, toneDur);
        delay(stimInterval - toneDur);
      }
      pulseTTL(ttlAudio8k6k);
      tone(speakerPin, FREQ_6K, toneDur);
      delay(stimInterval - toneDur - ttlDur);
      break;

    case VIS_SINGLE:
      // Single flash
      pulseTTL(ttlVisSingle);               // TTL at flash onset
      flashLEDs(true);
      delay(lightDur - ttlDur);
      flashLEDs(false);
      delay(stimInterval - lightDur);
      break;

    case VIS_SEQ:
      // Sequence of reps flashes, TTL at each flash onset
      for (int i = 0; i < reps; i++) {
        pulseTTL(ttlVisSeq);                // TTL at each flash onset
        flashLEDs(true);
        delay(lightDur - ttlDur);
        flashLEDs(false);
        if (i < reps - 1) delay(flashITI); // gap between flashes
      }
      delay(stimInterval - lightDur);
      break;
  }

  // ── ITI: 3, 4, or 5 seconds ───────────────────────────────────────────
  delay(random(3, 6) * 1000);

  // ── Log to serial ──────────────────────────────────────────────────────
  Serial.print(trialNum + 1);
  Serial.print(",");
  Serial.print(trialType);
  Serial.print(",");
  Serial.println(reps);

  trialNum++;
}
