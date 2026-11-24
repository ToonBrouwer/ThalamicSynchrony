"""
Thin ctypes wrapper around DoricSystem.dll.

Imports the authoritative struct definitions from Doric's own SDK example
folder so we stay in sync with their ABI on updates.

Public API:
    DoricLightSource(dll_path, uid, channel=1)
        .init()             — init DLL + wait
        .list_devices()     — available_devices_infos() — prints to stdout
        .open()             — open_device_uid + wait
        .close()            — stop + close_device_uid
        .quit()             — final DLL shutdown
        .send_custom(samples_1000, period_ms, n_seq=1)
        .send_complex(complex_mods)
        .start() / .stop()  — channel-level

Use as a context manager:
    with DoricLightSource(...) as doric:
        doric.send_custom(samples, 2000)
        doric.start(); time.sleep(2); doric.stop()
"""
from __future__ import annotations

import sys
import time
from ctypes import POINTER, cdll, c_bool, c_char_p, c_int, c_uint16, pointer

from config import (
    DLL_PATH,
    EXAMPLES_PYTHON_DIR,
    SETTLE_MS_AFTER_SEND_SETTINGS,
    WAIT_AFTER_INIT_MS,
    WAIT_AFTER_OPEN_MS,
)

# Pull Doric's SDK-provided struct definitions onto the path BEFORE importing
sys.path.insert(0, EXAMPLES_PYTHON_DIR)

from lightsource_defs import (  # noqa: E402
    LightSourceSettings,
    LightSourceMode,
    LightSourceCurrentMode,
    LightSourceComplexModulation,  # noqa: F401  (re-exported for callers)
    LightSourceComplexMode,        # noqa: F401
)
from doric_system_defs import Channel, TriggerType, TriggerMode  # noqa: E402


class DoricLightSource:
    """Context-managed wrapper for a single channel on one Doric LED driver."""

    def __init__(self, dll_path: str = DLL_PATH, uid: bytes | None = None, channel: int = 1):
        self._dll_path = dll_path
        self.uid = uid if (uid is None or isinstance(uid, bytes)) else uid.encode()
        # Zero-based enum internally, user-facing 1-based
        self.channel = Channel(channel - 1)
        self._opened = False
        self._init_done = False
        self._last_settings: LightSourceSettings | None = None
        self.dll = None

    # --------------------------------------------------------------- lifecycle
    def _load_dll(self):
        self.dll = cdll.LoadLibrary(self._dll_path)
        self._setup_argtypes()

    def _setup_argtypes(self):
        d = self.dll
        d.init.argtypes = [c_bool];                                d.init.restype = None
        d.wait.argtypes = [c_int];                                 d.wait.restype = None
        d.quit.argtypes = [];                                      d.quit.restype = None
        d.available_devices_infos.argtypes = [];                   d.available_devices_infos.restype = None
        d.open_device_uid.argtypes  = [c_char_p];                  d.open_device_uid.restype  = None
        d.close_device_uid.argtypes = [c_char_p];                  d.close_device_uid.restype = None
        # Light source — UID variants only (we're not using port-number API)
        d.ls_send_settings_uid.argtypes  = [c_char_p, POINTER(LightSourceSettings)]
        d.ls_send_settings_uid.restype   = None
        d.ls_start_channel_uid.argtypes  = [c_char_p, c_int]
        d.ls_start_channel_uid.restype   = None
        d.ls_stop_channel_uid.argtypes   = [c_char_p, c_int]
        d.ls_stop_channel_uid.restype    = None
        d.ls_start_all_uid.argtypes      = [c_char_p]
        d.ls_start_all_uid.restype       = None
        d.ls_stop_all_uid.argtypes       = [c_char_p]
        d.ls_stop_all_uid.restype        = None
        d.ls_send_current_uid.argtypes   = [c_char_p, c_int, c_uint16]
        d.ls_send_current_uid.restype    = None

    def init(self, debug: bool = False):
        if self.dll is None:
            self._load_dll()
        self.dll.init(debug)
        self.dll.wait(WAIT_AFTER_INIT_MS)
        self._init_done = True

    def list_devices(self):
        """Prints available devices to stdout (SDK side-effect)."""
        if self.dll is None:
            self._load_dll()
        if not self._init_done:
            self.dll.init(False)
            self.dll.wait(WAIT_AFTER_INIT_MS)
            self._init_done = True
        self.dll.available_devices_infos()

    def open(self):
        if self.uid is None:
            raise RuntimeError(
                "DORIC_UID is None — run list_devices() first, then set "
                "DORIC_UID in config.py to the UID of your device."
            )
        self.dll.open_device_uid(self.uid)
        self.dll.wait(WAIT_AFTER_OPEN_MS)
        self._opened = True

    def close(self):
        if self._opened:
            try:
                self.stop()
            except Exception:
                pass
            self.dll.close_device_uid(self.uid)
            self.dll.wait(1000)
            self._opened = False

    def quit(self):
        if self.dll is not None and self._init_done:
            self.dll.quit()
            self._init_done = False

    def __enter__(self):
        self.init()
        if self.uid is not None:
            self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        finally:
            self.quit()

    # --------------------------------------------------------- settings upload
    def _new_settings(self) -> LightSourceSettings:
        s = LightSourceSettings()
        s.channelIdx     = self.channel
        s.isTTLOutput    = True                       # sync pulse → OneBox
        s.triggerType    = TriggerType.Manual         # software-fired
        s.triggerMode    = TriggerMode.Uninterrupted
        s.currentMode    = LightSourceCurrentMode.Normal
        s.isTriggerRepeatable = False
        return s

    def send_custom(
        self,
        samples_1000,
        period_ms: float,
        n_seq: int = 1,
        starting_delay_ms: int = 0,
        delay_between_seq_ms: int = 0,
    ):
        """Upload a 1000-point arbitrary waveform.

        samples_1000 : iterable of 1000 ints in [0, MAX_CURRENT_MA]
        period_ms    : duration over which the 1000 points are played
        n_seq        : number of times to repeat the block
        """
        if len(samples_1000) != 1000:
            raise ValueError(f"samples_1000 must be length 1000, got {len(samples_1000)}")

        s = self._new_settings()
        s.mode = LightSourceMode.Custom
        s.ttlModulation.periodMs           = float(period_ms)
        s.ttlModulation.startingDelayMs    = int(starting_delay_ms)
        s.ttlModulation.delayBetweenSeqMs  = int(delay_between_seq_ms)
        s.ttlModulation.nbOfSeq            = int(n_seq)
        s.ttlModulation.nbOfPulsesPerSeq   = 0
        s.ttlModulation.current            = 0        # unused for Custom
        for i, v in enumerate(samples_1000):
            s.customDataPoint[i] = int(v)

        self.dll.ls_send_settings_uid(self.uid, pointer(s))
        self._last_settings = s                       # keep alive
        self.dll.wait(SETTLE_MS_AFTER_SEND_SETTINGS)

    def send_complex(self, complex_mods):
        """Upload a list of ComplexModulation primitives (up to 32)."""
        s = self._new_settings()
        s.mode = LightSourceMode.Complex
        s.clearAllComplexModulations()
        for m in complex_mods:
            s.addComplexModulation(m)

        self.dll.ls_send_settings_uid(self.uid, pointer(s))
        self._last_settings = s
        self.dll.wait(SETTLE_MS_AFTER_SEND_SETTINGS)

    # --------------------------------------------------------------- start/stop
    def start(self):
        self.dll.ls_start_channel_uid(self.uid, int(self.channel))

    def stop(self):
        self.dll.ls_stop_channel_uid(self.uid, int(self.channel))

    # -------------------------------------------------- direct current control
    def set_current(self, current_ma: int):
        """Manually set the channel current (useful for debugging)."""
        self.dll.ls_send_current_uid(self.uid, int(self.channel), int(current_ma))
