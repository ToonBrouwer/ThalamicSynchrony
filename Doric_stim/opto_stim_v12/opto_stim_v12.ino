// =============================================================================
// TRN Optogenetic Stimulation Paradigm — Arduino Due  v12
// =============================================================================
// Based on v11 (MCP4725 I2C DAC, runtime-benchmarked fs).
// Changes from v11:
//   - Removed `sinf()` from inner loops.  A 1024-entry uint16 sine-to-DAC
//     lookup table (SINE_DAC_LUT) is built once in setup().
//   - Removed `powf(k, ts)` from the chirp inner loop.  Chirps now use a
//     phase accumulator with a per-sample constant frequency multiplier
//     (`inst_f *= f_mult`).  One float multiply + one add per sample,
//     no transcendentals.
//   - Sine phase is a uint32 fixed-point accumulator; the top 10 bits
//     index the LUT directly, so sine playback is essentially free on
//     top of the I2C DAC cost.
//   - Result: chirp, sine, noise and ramp all run at the same measured
//     effective fs (no more 5 s / 6.75 s / 17 s spread), and commanded
//     frequencies match reality.
//
// IMPORTANT — noise spectrum:
//   NOISE_LP100_TBL / NOISE_LP500_TBL were generated for fs = 10 kHz.
//   Played out at the measured effective rate (likely ~8.5–8.7 kHz),
//   cutoffs scale by (real_fs / 10000).  Regenerate tables at the
//   measured rate for true 100 / 500 Hz cutoffs if that matters.
//
// WIRING (unchanged):
//   MCP4725 VCC -> 3.3V
//   MCP4725 GND -> GND
//   MCP4725 SDA -> SDA (pin 20)
//   MCP4725 SCL -> SCL (pin 21)
//   MCP4725 OUT -> BNC centre -> Doric LED driver BNC input
//   GND         -> BNC shield -> Doric driver GND
//   Pin 2       -> single TTL recording channel
// =============================================================================

#include <Wire.h>
#include <avr/pgmspace.h>
#include "noise_tables.h"
#include "trial_sequence.h"

// ── USER PARAMETERS ──────────────────────────────────────────────────────────
#define USE_TTL          true
#define TTL_PIN          2
#define MCP4725_ADDR     0x60

const float  STIM_DURATION  = 5.0f;     // seconds, wall-clock
const float  RAMP_RISE_MS   = 250.0f;
const int    DAC_TARGET     = 2047;
const int    DAC_BASELINE   = 25;
const float  ISI_BASE_SEC   = 3.0f;
const float  ISI_JITTER_PCT = 20.0f;

// TTL timing
const int    TTL_ONSET_MS   = 10;
const int    TTL_PRE_MS     = 200;

const float  NOMINAL_FS     = 10000.0f;

// ── INTERNALS (benchmark-filled) ─────────────────────────────────────────────
float SAMPLE_RATE_REAL   = NOMINAL_FS;
int   TOTAL_SAMPLES_REAL = (int)(STIM_DURATION * NOMINAL_FS);

// ── Sine-to-DAC lookup table ────────────────────────────────────────────────
// 1024 entries storing the final uint16 DAC value for each phase bin.
// Built once in setup() -> no sinf() inside any waveform loop.
static const int LUT_SIZE   = 1024;
static const int LUT_SHIFT  = 22;   // uint32 phase >> 22 gives index 0..1023
static uint16_t SINE_DAC_LUT[LUT_SIZE];

enum StimulusType {
  CHIRP_FWD = 0,
  CHIRP_REV,
  NOISE_LP100,
  NOISE_LP500,
  SINE_10HZ,
  SINE_25HZ,
  SINE_60HZ,
  RAMP,
  NUM_STIM_TYPES
};

const int TTL_TYPE_MS[NUM_STIM_TYPES] = {
  100, 150, 200, 250, 300, 350, 400, 450
};
const char* STIM_NAMES[] = {
  "ChirpFwd", "ChirpRev",
  "NoiseLp100", "NoiseLp500",
  "Sine10", "Sine25", "Sine60",
  "Ramp"
};
const float SINE_FREQS[] = { 10.0f, 25.0f, 60.0f };

// NOISE_TBL_LEN now comes from noise_tables.h (#define written by the
// notebook). Should be >= TOTAL_SAMPLES_REAL so the player never wraps.
#ifndef NOISE_TBL_LEN
#define NOISE_TBL_LEN 50000   // safe fallback if you forget to regenerate
#endif

// ── DAC HELPER (MCP4725 over I2C) ────────────────────────────────────────────
void dacWrite(uint16_t value) {
  value = constrain(value, 0, 4095);
  Wire.beginTransmission(MCP4725_ADDR);
  Wire.write(0x40);
  Wire.write((value >> 4) & 0xFF);
  Wire.write((value << 4) & 0xFF);
  Wire.endTransmission();
}

void dacOff() { dacWrite(DAC_BASELINE); }

void dacOffDelay(unsigned long ms) {
  unsigned long start = millis();
  while (millis() - start < ms) {
    dacOff();
    delay(100);
  }
}

// ── TTL HELPERS ──────────────────────────────────────────────────────────────
void ttlOnset() {
  digitalWrite(TTL_PIN, HIGH);
  delay(TTL_ONSET_MS);
  digitalWrite(TTL_PIN, LOW);
}
void ttlEncodeType(StimulusType stim) {
  delay(TTL_PRE_MS);
  digitalWrite(TTL_PIN, HIGH);
  delay(TTL_TYPE_MS[stim]);
  digitalWrite(TTL_PIN, LOW);
}

// ── LUT BUILD + CALIBRATION ──────────────────────────────────────────────────
void buildSineLUT() {
  for (int i = 0; i < LUT_SIZE; i++) {
    float s = 0.5f + 0.5f * sinf(2.0f * PI * (float)i / (float)LUT_SIZE);
    int v = (int)(s * DAC_TARGET + 0.5f);
    if (v < 0) v = 0; if (v > 4095) v = 4095;
    SINE_DAC_LUT[i] = (uint16_t)v;
  }
}

void benchmarkDac() {
  // Capture a slightly-representative workload: LUT lookup + phase add +
  // dacWrite, which is exactly what the sine player does per sample.
  const int N_BENCH = 2000;
  dacOff();
  uint32_t phase = 0;
  uint32_t inc   = 0x00400000;       // arbitrary, doesn't matter
  uint32_t t0    = micros();
  for (int i = 0; i < N_BENCH; i++) {
    dacWrite(SINE_DAC_LUT[phase >> LUT_SHIFT]);
    phase += inc;
  }
  float us_per_sample = (micros() - t0) / (float)N_BENCH;
  SAMPLE_RATE_REAL   = 1.0e6f / us_per_sample;
  TOTAL_SAMPLES_REAL = (int)(STIM_DURATION * SAMPLE_RATE_REAL);

  Serial.print("us/sample        = "); Serial.println(us_per_sample, 2);
  Serial.print("effective fs (Hz)= "); Serial.println(SAMPLE_RATE_REAL, 1);
  Serial.print("TOTAL_SAMPLES    = "); Serial.println(TOTAL_SAMPLES_REAL);
  Serial.print("ratio real/nom   = ");
  Serial.println(SAMPLE_RATE_REAL / NOMINAL_FS, 4);
}

// ── FORWARD DECLARATIONS ─────────────────────────────────────────────────────
void runStimulus(StimulusType stim);
void playChirpFwd();
void playChirpRev();
void playRamp();
void playNoise(const int16_t* tbl);
void playSine(float freq);

// =============================================================================
// SETUP
// =============================================================================
void setup() {
  Serial.begin(115200);

  Wire.begin();
  Wire.setClock(400000);

  #if USE_TTL
    pinMode(TTL_PIN, OUTPUT);
    digitalWrite(TTL_PIN, LOW);
  #endif

  buildSineLUT();
  dacOff();

  Serial.println("TTL decode: <50ms=onset  >=100ms=type");
  Serial.println("100=ChirpFwd 150=ChirpRev 200=NoiseLp100 250=NoiseLp500");
  Serial.println("300=Sine10 350=Sine25 400=Sine60 450=Ramp");

  if (TOTAL_TRIALS == 0) {
    Serial.println("ERROR: TOTAL_TRIALS=0 — regenerate trial_sequence.h");
    while (true) delay(1000);
  }

  benchmarkDac();

  Serial.print("TRIALS:"); Serial.println(TOTAL_TRIALS);
  Serial.print("ORDER:");
  for (int i = 0; i < TOTAL_TRIALS; i++) {
    Serial.print(pgm_read_byte(&TRIAL_SEQUENCE[i]));
    if (i < TOTAL_TRIALS - 1) Serial.print(",");
  }
  Serial.println();

  delay(3000);
}

// =============================================================================
// MAIN LOOP
// =============================================================================
void loop() {
  for (int t = 0; t < TOTAL_TRIALS; t++) {
    dacOff();
    float isi = ISI_BASE_SEC * (1.0f
                + (random(-100, 101) / 100.0f) * (ISI_JITTER_PCT / 100.0f));
    dacOffDelay((unsigned long)(isi * 1000.0f));
    dacOff();

    StimulusType stim = (StimulusType)pgm_read_byte(&TRIAL_SEQUENCE[t]);
    Serial.print(t + 1); Serial.print(","); Serial.println(STIM_NAMES[stim]);

    #if USE_TTL
      ttlOnset();
    #endif

    uint32_t t_start = micros();
    runStimulus(stim);
    uint32_t t_elapsed = micros() - t_start;

    dacOff();

    #if USE_TTL
      ttlEncodeType(stim);
    #endif

    Serial.print("  elapsed_us="); Serial.println(t_elapsed);
  }

  dacOff();
  Serial.println("DONE");
  while (true) { dacOffDelay(1000); }
}

// =============================================================================
// DISPATCHER
// =============================================================================
void runStimulus(StimulusType stim) {
  switch (stim) {
    case CHIRP_FWD:   playChirpFwd();             break;
    case CHIRP_REV:   playChirpRev();             break;
    case NOISE_LP100: playNoise(NOISE_LP100_TBL); break;
    case NOISE_LP500: playNoise(NOISE_LP500_TBL); break;
    case SINE_10HZ:   playSine(SINE_FREQS[0]);    break;
    case SINE_25HZ:   playSine(SINE_FREQS[1]);    break;
    case SINE_60HZ:   playSine(SINE_FREQS[2]);    break;
    case RAMP:        playRamp();                  break;
    default: break;
  }
}

// =============================================================================
// WAVEFORM PLAYERS — no sinf()/powf() in any inner loop
// =============================================================================

// Pure uint32 phase accumulator + LUT lookup. Essentially free on top of I2C.
void playSine(float freq) {
  uint32_t phase = 0;
  uint32_t inc   = (uint32_t)((double)freq * 4294967296.0
                              / (double)SAMPLE_RATE_REAL);
  for (int i = 0; i < TOTAL_SAMPLES_REAL; i++) {
    dacWrite(SINE_DAC_LUT[phase >> LUT_SHIFT]);
    phase += inc;
  }
  dacOff();
}

// Exponential chirp via phase accumulator + per-sample constant multiplier.
// inst_f grows (or shrinks) by f_mult each sample so that inst_f at sample
// TOTAL_SAMPLES_REAL equals F_END (fwd) or F_START (rev).
void playChirpFwd() {
  const float F_START = 0.5f, F_END = 200.0f;
  float inst_f = F_START;
  float ratio  = F_END / F_START;
  float f_mult = expf(logf(ratio) / (float)TOTAL_SAMPLES_REAL);   // > 1
  float two_pi_over_fs = 2.0f * PI / SAMPLE_RATE_REAL;
  float phase = 0.0f;
  const float phase_to_idx = (float)LUT_SIZE / (2.0f * PI);       // ~163

  for (int i = 0; i < TOTAL_SAMPLES_REAL; i++) {
    int idx = (int)(phase * phase_to_idx) & (LUT_SIZE - 1);
    dacWrite(SINE_DAC_LUT[idx]);
    phase += two_pi_over_fs * inst_f;
    if (phase >= 2.0f * PI) phase -= 2.0f * PI;
    inst_f *= f_mult;
  }
  dacOff();
}

void playChirpRev() {
  const float F_START = 0.5f, F_END = 200.0f;
  float inst_f = F_END;
  float ratio  = F_END / F_START;
  float f_mult = expf(-logf(ratio) / (float)TOTAL_SAMPLES_REAL);  // < 1
  float two_pi_over_fs = 2.0f * PI / SAMPLE_RATE_REAL;
  float phase = 0.0f;
  const float phase_to_idx = (float)LUT_SIZE / (2.0f * PI);

  for (int i = 0; i < TOTAL_SAMPLES_REAL; i++) {
    int idx = (int)(phase * phase_to_idx) & (LUT_SIZE - 1);
    dacWrite(SINE_DAC_LUT[idx]);
    phase += two_pi_over_fs * inst_f;
    if (phase >= 2.0f * PI) phase -= 2.0f * PI;
    inst_f *= f_mult;
  }
  dacOff();
}

void playRamp() {
  int rise = (int)(RAMP_RISE_MS / 1000.0f * SAMPLE_RATE_REAL);
  if (rise < 2) rise = 2;
  if (rise > TOTAL_SAMPLES_REAL) rise = TOTAL_SAMPLES_REAL;
  for (int i = 0; i < rise; i++) {
    dacWrite((int)((float)i / (rise - 1) * DAC_TARGET));
  }
  for (int i = rise; i < TOTAL_SAMPLES_REAL; i++) {
    dacWrite(DAC_TARGET);
  }
  dacOff();
}

void playNoise(const int16_t* tbl) {
  int rise = (int)(RAMP_RISE_MS / 1000.0f * SAMPLE_RATE_REAL);
  if (rise < 2) rise = 2;
  if (rise > TOTAL_SAMPLES_REAL) rise = TOTAL_SAMPLES_REAL;
  for (int i = 0; i < rise; i++) {
    dacWrite((int)((float)i / (rise - 1) * DAC_TARGET));
  }
  int noise_samples = TOTAL_SAMPLES_REAL - rise;
  // No more modulo: the noise table is generated at NOISE_TBL_LEN >= one
  // full trial, so wrap-around is impossible during a normal stim. We
  // still clamp the index defensively in case the effective fs is much
  // higher than the rate the table was generated for.
  for (int i = 0; i < noise_samples; i++) {
    int idx = (i < NOISE_TBL_LEN) ? i : (NOISE_TBL_LEN - 1);
    int16_t val = (int16_t)pgm_read_word(tbl + idx);
    dacWrite(constrain(DAC_TARGET + (int)val, 0, 4095));
  }
  dacOff();
}
