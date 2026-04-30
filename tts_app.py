"""
OfflineNarrator — ElevenLabs-style narration studio.
Per-segment voice control · replayable audio · full playback bar
"""
import sys, os, re, json, uuid, shutil, tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QFrame, QLabel, QPushButton, QSlider, QComboBox,
    QLineEdit, QTextEdit, QSplitter, QSizePolicy, QFileDialog,
    QStatusBar, QMenu, QToolButton, QDialog, QDialogButtonBox,
    QCheckBox, QColorDialog, QProgressBar, QInputDialog, QMessageBox
)
from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QTimer, QUrl, QSize,
                           QMimeData, QPoint, QEvent)
from PyQt6.QtGui import (QFont, QColor, QCursor, QPainter, QPen,
                          QIcon, QPixmap, QDrag, QKeySequence, QShortcut)
from PyQt6.QtWidgets import QGraphicsOpacityEffect

try:
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
    HAS_MEDIA = True
except ImportError:
    HAS_MEDIA = False

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent

VOICE_COLORS = [
    "#89b4fa", "#a6e3a1", "#f38ba8", "#fab387",
    "#cba6f7", "#89dceb", "#f9e2af", "#94e2d5",
]

ORPHEUS_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]

KOKORO_VOICES = {
    "Heart (AF)":    ("a", "af_heart"),
    "Bella (AF)":    ("a", "af_bella"),
    "Nicole (AF)":   ("a", "af_nicole"),
    "Sky (AF)":      ("a", "af_sky"),
    "Adam (AM)":     ("a", "am_adam"),
    "Michael (AM)":  ("a", "am_michael"),
    "Emma (BF)":     ("b", "bf_emma"),
    "Isabella (BF)": ("b", "bf_isabella"),
    "George (BM)":   ("b", "bm_george"),
    "Lewis (BM)":    ("b", "bm_lewis"),
}

EMOTION_TAGS  = ["<laugh>", "<chuckle>", "<sigh>", "<cough>",
                 "<sniffle>", "<groan>", "<yawn>", "<gasp>"]
SPEEDS        = ["0.5×", "0.75×", "1.0×", "1.25×", "1.5×", "2.0×"]
SPEED_VALUES  = [0.5,    0.75,    1.0,    1.25,    1.5,    2.0   ]

STATUS_STYLE = {
    "pending":    ("○", "#6c7086"),
    "generating": ("◌", "#fab387"),
    "done":       ("●", "#a6e3a1"),
    "error":      ("●", "#f38ba8"),
}

STYLE = """
QMainWindow, QWidget          { background: #1e1e2e; color: #cdd6f4; }
QScrollArea                   { background: #1e1e2e; border: none; }
QScrollBar:vertical           { background: #1e1e2e; width: 8px; }
QScrollBar::handle:vertical   { background: #45475a; border-radius: 4px; min-height: 20px; }
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal         { background: #1e1e2e; height: 8px; }
QScrollBar::handle:horizontal { background: #45475a; border-radius: 4px; min-width: 20px; }
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal { width: 0; }
QComboBox {
    background: #313244; border: 1px solid #45475a;
    border-radius: 5px; padding: 3px 8px; color: #cdd6f4; min-height: 26px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #313244; selection-background-color: #89b4fa;
    selection-color: #1e1e2e; border: 1px solid #45475a;
}
QComboBox:disabled { color: #6c7086; }
QPushButton {
    background: #313244; border: 1px solid #45475a;
    border-radius: 6px; padding: 6px 14px; color: #cdd6f4; font-size: 13px;
}
QPushButton:hover    { background: #45475a; border-color: #89b4fa; }
QPushButton:pressed  { background: #181825; }
QPushButton:disabled { background: #1e1e2e; color: #6c7086; border-color: #313244; }
QPushButton#primary  { background: #89b4fa; color: #ffffff; font-weight: bold; border: none; }
QPushButton#primary:hover     { background: #b4befe; }
QPushButton#primary:disabled  { background: #45475a; color: #6c7086; }
QPushButton#danger   { background: #f38ba8; color: #ffffff; font-weight: bold; border: none; }
QPushButton#danger:hover      { background: #eba0ac; }
QSlider::groove:horizontal    { height: 4px; background: #45475a; border-radius: 2px; }
QSlider::handle:horizontal    {
    width: 14px; height: 14px; background: #89b4fa;
    border-radius: 7px; margin: -5px 0;
}
QSlider::sub-page:horizontal  { background: #89b4fa; border-radius: 2px; }
QSlider::handle:horizontal:disabled { background: #45475a; }
QLineEdit {
    background: #181825; border: 1px solid #45475a;
    border-radius: 5px; padding: 4px 8px; color: #cdd6f4;
}
QLineEdit:focus { border-color: #89b4fa; }
QTextEdit {
    background: #181825; border: 1px solid #45475a;
    border-radius: 6px; padding: 8px; color: #cdd6f4; font-size: 13px;
}
QTextEdit:focus { border-color: #89b4fa; }
QLabel          { color: #a6adc8; }
QStatusBar      { background: #181825; color: #6c7086; font-size: 12px; }
QToolButton {
    background: transparent; border: none; color: #a6adc8;
    border-radius: 4px; padding: 3px 5px; font-size: 14px;
}
QToolButton:hover { background: #313244; color: #cdd6f4; }
QMenu { background: #313244; border: 1px solid #45475a; color: #cdd6f4; }
QMenu::item:selected { background: #89b4fa; color: #1e1e2e; }
QMenu::separator { background: #45475a; height: 1px; margin: 3px 0; }
QProgressBar {
    background: #313244; border: none; border-radius: 3px;
    height: 6px; text-align: center;
}
QProgressBar::chunk { background: #89b4fa; border-radius: 3px; }
"""

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VoiceConfig:
    id:                str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:              str   = "Voice 1"
    model:             str   = "orpheus"
    voice_id:          str   = "tara"
    color:             str   = "#89b4fa"
    temperature:       float = 0.6
    top_p:             float = 0.8
    repetition_penalty:float = 1.3
    speed:             float = 1.0
    exaggeration:      float = 0.5
    voice_ref:         str   = ""
    post_gap_ms:       int   = 0

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Segment:
    id:          str            = field(default_factory=lambda: str(uuid.uuid4()))
    text:        str            = ""
    voice_id:    str            = ""
    audio:       Optional[object] = field(default=None, repr=False)
    sample_rate: int            = 24000
    status:      str            = "pending"
    post_gap_ms: int            = 0        # per-segment silence after audio

    def duration_ms(self) -> int:
        if self.audio is None or self.audio.size == 0:
            return 0
        return int(len(self.audio) / self.sample_rate * 1000)

    def to_dict(self):
        import base64
        d = {"id": self.id, "text": self.text, "voice_id": self.voice_id,
             "status": self.status, "post_gap_ms": self.post_gap_ms}
        if self.audio is not None and self.audio.size > 0:
            pcm = (np.clip(self.audio, -1.0, 1.0) * 32767).astype(np.int16)
            d["audio_b64"]   = base64.b64encode(pcm.tobytes()).decode()
            d["sample_rate"] = self.sample_rate
        return d

    @classmethod
    def from_dict(cls, d):
        import base64
        seg = cls(id=d["id"], text=d["text"], voice_id=d.get("voice_id", ""),
                  status=d.get("status", "pending"),
                  post_gap_ms=d.get("post_gap_ms", 0))
        if "audio_b64" in d:
            raw       = base64.b64decode(d["audio_b64"])
            seg.audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            seg.sample_rate = d.get("sample_rate", 24000)
            seg.status = "done"
        return seg

# ── Helpers ───────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list:
    paragraphs = re.split(r'\n\s*\n', text.strip())
    out = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = re.split(r'(?<=[.!?])\s+', para)
        out.extend(p.strip() for p in parts if p.strip())
    return out

def ms_to_time(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"

def _make_icon() -> QIcon:
    px = QPixmap(64, 64)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#1e1e2e"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(0, 0, 64, 64, 12, 12)
    p.setBrush(QColor("#89b4fa"))
    bars = [14, 28, 46, 36, 54, 32, 20]
    bw, gap = 6, 3
    x = (64 - len(bars) * (bw + gap) + gap) // 2
    for h in bars:
        p.drawRoundedRect(x, (64 - h) // 2, bw, h, 3, 3)
        x += bw + gap
    p.end()
    return QIcon(px)

# ── Model cache + worker ──────────────────────────────────────────────────────

_MODELS: dict = {}
_MODEL_LOCK = __import__("threading").Lock()   # prevents concurrent model loads

def _get_orpheus():
    with _MODEL_LOCK:
        if "orpheus" not in _MODELS:
            import torch
            from orpheus_local import OrpheusLocal
            _MODELS["orpheus"] = OrpheusLocal(
                "cuda" if torch.cuda.is_available() else "cpu")
        return _MODELS["orpheus"]

def _get_kokoro(lang_code: str):
    with _MODEL_LOCK:
        key = f"kokoro_{lang_code}"
        if key not in _MODELS:
            import torch
            from kokoro import KPipeline
            _MODELS[key] = KPipeline(lang_code=lang_code,
                                     device="cuda" if torch.cuda.is_available() else "cpu")
        return _MODELS[key]

def _get_chatterbox():
    with _MODEL_LOCK:
        if "chatterbox" not in _MODELS:
            import torch
            from chatterbox.tts import ChatterboxTTS
            _MODELS["chatterbox"] = ChatterboxTTS.from_pretrained(
                device="cuda" if torch.cuda.is_available() else "cpu")
        return _MODELS["chatterbox"]


class GenerationWorker(QThread):
    segment_started = pyqtSignal(str)
    segment_done    = pyqtSignal(str, object, int)
    segment_error   = pyqtSignal(str, str)
    all_done        = pyqtSignal()
    status_msg      = pyqtSignal(str)
    progress        = pyqtSignal(int, int)   # done, total

    def __init__(self, segments: list, voices: dict):
        super().__init__()
        self._segments  = segments
        self._voices    = voices
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        total = len(self._segments)
        for i, seg in enumerate(self._segments):
            if self._cancelled:
                self.status_msg.emit("Cancelled.")
                return
            voice = self._voices.get(seg.voice_id)
            if voice is None:
                self.segment_error.emit(seg.id, "No voice assigned")
                self.progress.emit(i + 1, total)
                continue
            self.segment_started.emit(seg.id)
            self.status_msg.emit(
                f"Generating {i+1}/{total}  [{voice.name} · {seg.text[:40]}…]")
            try:
                audio, sr = self._generate(seg, voice)
                self.segment_done.emit(seg.id, audio, sr)
            except Exception as e:
                self.segment_error.emit(seg.id, str(e))
            self.progress.emit(i + 1, total)
        self.all_done.emit()

    def _generate(self, seg, voice):
        if voice.model == "orpheus":
            model     = _get_orpheus()
            sentences = split_sentences(seg.text) or [seg.text]
            chunks    = []
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                if len(sentences) > 1:
                    self.status_msg.emit(
                        f"[{voice.name}] sentence {i+1}/{len(sentences)}: {sentence[:50]}…")
                audio = model.generate_speech(
                    sentence, voice=voice.voice_id,
                    temperature=voice.temperature,
                    top_p=voice.top_p,
                    repetition_penalty=voice.repetition_penalty,
                )
                if audio.size > 0:
                    chunks.append(audio)
            return (np.concatenate(chunks) if chunks
                    else np.zeros(0, dtype=np.float32)), 24000

        elif voice.model == "kokoro":
            lang, vid = KOKORO_VOICES[voice.voice_id]
            pipeline  = _get_kokoro(lang)
            chunks = []
            for _, _, a in pipeline(seg.text, voice=vid, speed=voice.speed):
                chunks.append(a.numpy() if hasattr(a, "numpy") else a)
            return (np.concatenate(chunks) if chunks
                    else np.zeros(0, dtype=np.float32)), 24000

        elif voice.model == "chatterbox":
            model  = _get_chatterbox()
            kwargs = dict(exaggeration=voice.exaggeration, cfg_weight=0.5)
            if voice.voice_ref and os.path.exists(voice.voice_ref):
                kwargs["audio_prompt_path"] = voice.voice_ref
            wav = model.generate(seg.text, **kwargs)
            return wav.squeeze(0).cpu().numpy(), model.sr

        raise ValueError(f"Unknown model: {voice.model}")

# ── Plain-text editor ─────────────────────────────────────────────────────────

_EDITOR_FONT  = QFont("Segoe UI", 12)
_EDITOR_COLOR = "#cdd6f4"

class PlainTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_format()

    def _apply_format(self):
        from PyQt6.QtGui import QTextCharFormat
        fmt = QTextCharFormat()
        fmt.setFont(_EDITOR_FONT)
        fmt.setForeground(QColor(_EDITOR_COLOR))
        self.setCurrentCharFormat(fmt)
        self.document().setDefaultFont(_EDITOR_FONT)

    def insertFromMimeData(self, source):
        self.insertPlainText(source.text())
        self._apply_format()

# ── Text-import dialog ────────────────────────────────────────────────────────

class TextImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Text")
        self.setMinimumSize(640, 420)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        lbl = QLabel("Paste your script below. It will be split into sentences automatically.")
        lbl.setWordWrap(True)
        layout.addWidget(lbl)
        self.edit = PlainTextEdit()
        self.edit.setPlaceholderText(
            "Paste narration, story, or script here…\n\n"
            "Tip: separate paragraphs with a blank line to group them.")
        layout.addWidget(self.edit)
        file_btn = QPushButton("📂  Import from .txt file…")
        file_btn.setFixedWidth(200)
        file_btn.clicked.connect(self._from_file)
        layout.addWidget(file_btn)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _from_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open text file", "", "Text files (*.txt);;All files (*)")
        if path:
            with open(path, encoding="utf-8") as f:
                self.edit.setPlainText(f.read())

    def get_text(self):
        return self.edit.toPlainText()

# ── Drag handle ───────────────────────────────────────────────────────────────

class _DragHandle(QLabel):
    drag_initiated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("⠿", parent)
        self.setFixedWidth(18)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("color: #45475a; font-size: 16px; padding: 0;")
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        self._press_pos = None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press_pos = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self._press_pos and e.buttons() & Qt.MouseButton.LeftButton:
            dist = (e.position().toPoint() - self._press_pos).manhattanLength()
            if dist >= QApplication.startDragDistance():
                self._press_pos = None
                self.drag_initiated.emit()

    def mouseReleaseEvent(self, e):
        self._press_pos = None

# ── Segments drop container ───────────────────────────────────────────────────

class SegmentsContainer(QWidget):
    reorder_requested = pyqtSignal(str, int)   # seg_id, new_index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._drop_y = -1
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addStretch()

    def dragEnterEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasText():
            e.acceptProposedAction()
            self._drop_y = int(e.position().y())
            self.update()

    def dragLeaveEvent(self, e):
        self._drop_y = -1
        self.update()

    def dropEvent(self, e):
        if e.mimeData().hasText():
            seg_id = e.mimeData().text()
            drop_y = int(e.position().y())
            self._drop_y = -1
            self.update()
            self.reorder_requested.emit(seg_id, self._index_at(drop_y))
            e.acceptProposedAction()

    def _index_at(self, y: int) -> int:
        layout = self.layout()
        count  = layout.count() - 1   # exclude stretch
        for i in range(count):
            item = layout.itemAt(i)
            if item and item.widget():
                if y < item.widget().geometry().center().y():
                    return i
        return count

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._drop_y >= 0:
            p = QPainter(self)
            pen = QPen(QColor("#89b4fa"), 2)
            p.setPen(pen)
            p.drawLine(8, self._drop_y, self.width() - 8, self._drop_y)

# ── SegmentWidget ─────────────────────────────────────────────────────────────

class SegmentWidget(QFrame):
    regenerate_requested = pyqtSignal(str)
    voice_change_requested = pyqtSignal(str, str)
    text_edited            = pyqtSignal(str, str)
    delete_requested       = pyqtSignal(str)
    add_above_requested    = pyqtSignal(str)
    selection_toggled      = pyqtSignal(str, bool)
    split_requested        = pyqtSignal(str, int)   # seg_id, cursor_pos (-1 = auto)
    merge_requested        = pyqtSignal(str)         # merge with next
    duplicate_requested    = pyqtSignal(str)
    silence_requested      = pyqtSignal(str)
    play_from_requested    = pyqtSignal(str)

    def __init__(self, segment: Segment, voices: dict, parent=None):
        super().__init__(parent)
        self._seg      = segment
        self._voices   = voices
        self._selected = False
        self._playing  = False
        self._editing  = False
        self._build()
        self.refresh()

    def _build(self):
        self.setMinimumHeight(52)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 6, 8, 6)
        row.setSpacing(0)
        row.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        # Drag handle
        self._handle = _DragHandle(self)
        self._handle.drag_initiated.connect(self._start_drag)
        row.addWidget(self._handle)

        # Segment number
        self._num_lbl = QLabel("#")
        self._num_lbl.setFixedWidth(32)
        self._num_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._num_lbl.setStyleSheet("color: #45475a; font-size: 10px;")
        row.addWidget(self._num_lbl)

        # Checkbox
        self._check = QCheckBox()
        self._check.setFixedWidth(24)
        self._check.setToolTip("Check to include in Generate Selected")
        self._check.toggled.connect(self._on_check_toggled)
        self._check.setStyleSheet("""
            QCheckBox::indicator { width: 15px; height: 15px; border-radius: 3px;
                border: 1px solid #45475a; background: #313244; }
            QCheckBox::indicator:checked { background: #89b4fa; border-color: #89b4fa; }
        """)
        row.addWidget(self._check)

        # Color strip
        self._strip = QFrame()
        self._strip.setFixedWidth(5)
        self._strip.setMinimumHeight(36)
        row.addWidget(self._strip)
        row.addSpacing(4)

        # Voice badge
        self._voice_btn = QPushButton()
        self._voice_btn.setFixedWidth(76)
        self._voice_btn.setFlat(True)
        self._voice_btn.clicked.connect(self._pick_voice)
        row.addWidget(self._voice_btn)
        row.addSpacing(4)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #313244;")
        row.addWidget(sep)
        row.addSpacing(10)

        # Text display
        self._lbl = QTextEdit()
        self._lbl.setReadOnly(True)
        self._lbl.setFrameShape(QFrame.Shape.NoFrame)
        self._lbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._lbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._lbl.document().setDocumentMargin(0)
        self._lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._lbl.setStyleSheet(
            "QTextEdit { background: transparent; border: none; "
            "color: #cdd6f4; font-size: 14px; padding: 2px 0; }")
        self._lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row.addWidget(self._lbl, 1)

        # Inline editor
        self._edit = PlainTextEdit()
        self._edit.setMinimumHeight(52)
        self._edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._edit.setStyleSheet("""
            QTextEdit { background: #313244; border: 1px solid #89b4fa;
                        border-radius: 4px; padding: 4px; color: #cdd6f4; font-size: 14px; }
        """)
        self._edit.hide()
        row.addWidget(self._edit, 1)

        row.addSpacing(8)

        # Status dot
        self._dot = QLabel("○")
        self._dot.setFixedWidth(16)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._dot)
        row.addSpacing(4)

        # Regen button
        self._regen = QToolButton()
        self._regen.setText("↺")
        self._regen.setToolTip("Regenerate this segment")
        self._regen.setFixedSize(26, 26)
        self._regen.clicked.connect(lambda: self.regenerate_requested.emit(self._seg.id))
        row.addWidget(self._regen)

    # ── refresh ──

    def refresh(self):
        voice = self._voices.get(self._seg.voice_id)
        color = voice.color if voice else "#6c7086"
        name  = voice.name  if voice else "—"

        self._strip.setStyleSheet(f"background: {color}; border-radius: 0 2px 2px 0;")
        self._voice_btn.setText(name)
        self._voice_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
                text-align: left; font-size: 11px; font-weight: bold;
                padding: 2px 4px; color: {color};
            }}
            QPushButton:hover {{ background: #313244; border-radius: 4px; }}
        """)

        self._lbl.setPlainText(self._seg.text)
        self._update_height()

        sym, col = STATUS_STYLE.get(self._seg.status, ("○", "#6c7086"))
        self._dot.setText(sym)
        self._dot.setStyleSheet(f"font-size: 12px; color: {col};")

        if self._playing:
            border, bg = "2px solid #a6e3a1", "#152118"
        elif self._selected:
            border, bg = "2px solid #89b4fa", "#13182a"
        else:
            border, bg = "1px solid #313244", "#181825"

        self.setStyleSheet(f"""
            SegmentWidget {{
                background: {bg}; border: {border}; border-radius: 6px;
            }}
        """)

    def set_number(self, n: int, total: int):
        self._num_lbl.setText(f"{n}/{total}")

    def set_selected(self, v: bool):
        self._selected = v
        self._check.blockSignals(True)
        self._check.setChecked(v)
        self._check.blockSignals(False)
        self.refresh()

    def set_playing(self, v: bool):
        self._playing = v
        self.refresh()

    def set_regen_enabled(self, v: bool):
        self._regen.setEnabled(v)

    def _update_height(self):
        if not hasattr(self, '_lbl') or self._editing:
            return
        vp_w  = self._lbl.viewport().width()
        doc   = self._lbl.document()
        doc.setTextWidth(vp_w if vp_w > 10 else max(self.width() - 200, 100))
        doc_h = int(doc.size().height())
        needed = max(40, doc_h + 4)
        self._lbl.setFixedHeight(needed)
        new_min = max(52, needed + 18)
        if self.minimumHeight() != new_min:
            self.setMinimumHeight(new_min)
            self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._update_height)

    # ── drag ──

    def _start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(self._seg.id)
        drag.setMimeData(mime)
        px = self.grab().scaled(
            min(self.width(), 500), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        drag.setPixmap(px)
        drag.setHotSpot(QPoint(px.width() // 2, px.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)

    # ── selection & check ──

    def _on_check_toggled(self, checked: bool):
        self._selected = checked
        self.refresh()
        self.selection_toggled.emit(self._seg.id, checked)

    # ── edit ──

    def mouseDoubleClickEvent(self, e):
        self._start_edit()
        super().mouseDoubleClickEvent(e)

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        menu.addAction("▶  Play from here", lambda: self.play_from_requested.emit(self._seg.id))
        menu.addSeparator()
        menu.addAction("✏️  Edit",       self._start_edit)
        menu.addAction("↺  Regenerate", lambda: self.regenerate_requested.emit(self._seg.id))
        menu.addSeparator()

        if self._editing:
            menu.addAction("✂️  Split at cursor",
                           lambda: self.split_requested.emit(
                               self._seg.id,
                               self._edit.textCursor().position()))
        menu.addAction("✂️  Auto-split by sentences",
                       lambda: self.split_requested.emit(self._seg.id, -1))
        menu.addAction("⊕  Merge with next segment",
                       lambda: self.merge_requested.emit(self._seg.id))
        menu.addAction("⧉  Duplicate",
                       lambda: self.duplicate_requested.emit(self._seg.id))
        menu.addAction("⏱  Insert silence after…",
                       lambda: self.silence_requested.emit(self._seg.id))
        menu.addSeparator()

        voice_sub = menu.addMenu("🎤  Change voice")
        for vid, vc in self._voices.items():
            act = voice_sub.addAction(vc.name)
            act.setCheckable(True)
            act.setChecked(vid == self._seg.voice_id)
            act.setData(vid)

        menu.addSeparator()
        menu.addAction("➕  Add segment above",
                       lambda: self.add_above_requested.emit(self._seg.id))
        menu.addAction("🗑  Delete",
                       lambda: self.delete_requested.emit(self._seg.id))

        chosen = menu.exec(e.globalPos())
        if chosen and chosen.data():
            self._seg.voice_id = chosen.data()
            self.refresh()
            self.voice_change_requested.emit(self._seg.id, chosen.data())

    def _pick_voice(self):
        menu = QMenu(self)
        for vid, vc in self._voices.items():
            act = menu.addAction(vc.name)
            act.setCheckable(True)
            act.setChecked(vid == self._seg.voice_id)
            act.setData(vid)
        chosen = menu.exec(
            self._voice_btn.mapToGlobal(self._voice_btn.rect().bottomLeft()))
        if chosen:
            self._seg.voice_id = chosen.data()
            self.refresh()
            self.voice_change_requested.emit(self._seg.id, chosen.data())

    def _start_edit(self):
        self._editing = True
        self._edit.setMinimumHeight(max(52, self._lbl.height()))
        self._edit.setPlainText(self._seg.text)
        self._lbl.hide()
        self._edit.show()
        self._edit.setFocus()
        self._edit.selectAll()
        self._edit.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self._edit and event.type() == QEvent.Type.KeyPress:
            if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                    and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self._finish_edit()
                return True
        if obj is self._edit and event.type() == QEvent.Type.FocusOut:
            self._finish_edit()
        return super().eventFilter(obj, event)

    def _finish_edit(self):
        if not self._editing:
            return
        self._editing = False
        self._edit.setMinimumHeight(0)
        self._lbl.show()
        self._edit.hide()
        new = self._edit.toPlainText().strip()
        if new and new != self._seg.text:
            self._seg.text   = new
            self._seg.status = "pending"
            self.refresh()
            self.text_edited.emit(self._seg.id, new)
        else:
            self._update_height()

# ── VoiceCard ─────────────────────────────────────────────────────────────────

class VoiceCard(QFrame):
    changed = pyqtSignal(str)
    deleted = pyqtSignal(str)

    def __init__(self, voice: VoiceConfig, parent=None):
        super().__init__(parent)
        self._voice = voice
        self._build()

    def _build(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "VoiceCard { background: #181825; border: 1px solid #313244; border-radius: 8px; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(22, 22)
        self._color_btn.setToolTip("Click to change colour")
        self._color_btn.clicked.connect(self._pick_color)
        self._color_btn.setStyleSheet(
            f"background: {self._voice.color}; border-radius: 4px; border: none;")
        hdr.addWidget(self._color_btn)
        self._name_edit = QLineEdit(self._voice.name)
        self._name_edit.setFixedHeight(26)
        self._name_edit.textChanged.connect(self._on_name)
        hdr.addWidget(self._name_edit, 1)
        del_btn = QToolButton()
        del_btn.setText("✕")
        del_btn.setFixedSize(22, 22)
        del_btn.clicked.connect(lambda: self.deleted.emit(self._voice.id))
        hdr.addWidget(del_btn)
        root.addLayout(hdr)

        # Model
        mr = QHBoxLayout()
        mr.setSpacing(6)
        mr.addWidget(QLabel("Model"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["Orpheus", "Kokoro", "Chatterbox"])
        self._model_combo.setCurrentText(self._voice.model.capitalize())
        self._model_combo.currentTextChanged.connect(self._on_model)
        mr.addWidget(self._model_combo, 1)
        root.addLayout(mr)

        # Voice
        vr = QHBoxLayout()
        vr.setSpacing(6)
        vr.addWidget(QLabel("Voice"))
        self._voice_combo = QComboBox()
        self._voice_combo.currentTextChanged.connect(self._on_voice_id)
        vr.addWidget(self._voice_combo, 1)
        root.addLayout(vr)

        # ── Orpheus settings ──
        self._orpheus_box = QWidget()
        ob = QVBoxLayout(self._orpheus_box)
        ob.setContentsMargins(0, 0, 0, 0)
        ob.setSpacing(4)
        self._temp_sl = self._slider_row(ob, "Temperature", 1, 20,
                                          int(self._voice.temperature * 10), self._on_orpheus)
        self._topp_sl = self._slider_row(ob, "Top-P",       1, 10,
                                          int(self._voice.top_p * 10),        self._on_orpheus)
        self._rep_sl  = self._slider_row(ob, "Rep. Penalty",10, 20,
                                          int(self._voice.repetition_penalty * 10), self._on_orpheus)

        # Emotion tag buttons (Orpheus only)
        tag_lbl = QLabel("Emotion tags — copy to clipboard:")
        tag_lbl.setStyleSheet("color: #6c7086; font-size: 10px;")
        ob.addWidget(tag_lbl)
        tag_row1 = QHBoxLayout()
        tag_row2 = QHBoxLayout()
        for i, tag in enumerate(EMOTION_TAGS):
            btn = QPushButton(tag)
            btn.setFixedHeight(22)
            btn.setStyleSheet(
                "QPushButton { font-size: 10px; padding: 1px 4px; "
                "background: #313244; border: 1px solid #45475a; border-radius: 4px; }"
                "QPushButton:hover { background: #45475a; }")
            btn.setToolTip(f"Copy {tag} to clipboard")
            btn.clicked.connect(lambda _, t=tag: (
                QApplication.clipboard().setText(t),
                self._flash_status(f"Copied {t}")))
            (tag_row1 if i < 4 else tag_row2).addWidget(btn)
        ob.addLayout(tag_row1)
        ob.addLayout(tag_row2)
        root.addWidget(self._orpheus_box)

        # ── Kokoro settings ──
        self._kokoro_box = QWidget()
        kb = QVBoxLayout(self._kokoro_box)
        kb.setContentsMargins(0, 0, 0, 0)
        self._speed_sl = self._slider_row(kb, "Speed", 5, 20,
                                           int(self._voice.speed * 10), self._on_kokoro)
        root.addWidget(self._kokoro_box)

        # ── Chatterbox settings ──
        self._cb_box = QWidget()
        cbb = QVBoxLayout(self._cb_box)
        cbb.setContentsMargins(0, 0, 0, 0)
        cbb.setSpacing(4)
        self._exag_sl = self._slider_row(cbb, "Expressiveness", 0, 10,
                                          int(self._voice.exaggeration * 10), self._on_cb)
        ref_row = QHBoxLayout()
        self._ref_edit = QLineEdit(self._voice.voice_ref)
        self._ref_edit.setPlaceholderText("Reference audio (optional)")
        self._ref_edit.setReadOnly(True)
        self._ref_edit.setFixedHeight(26)
        ref_browse = QPushButton("…")
        ref_browse.setFixedSize(26, 26)
        ref_browse.clicked.connect(self._browse_ref)
        ref_row.addWidget(self._ref_edit, 1)
        ref_row.addWidget(ref_browse)
        cbb.addLayout(ref_row)
        root.addWidget(self._cb_box)

        # ── Gap after (all models) ──
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        root.addWidget(sep)
        self._gap_sl = self._slider_row(root, "Gap after (ms)", 0, 200,
                                         self._voice.post_gap_ms // 10, self._on_gap)
        self._gap_sl._val_lbl_ref.setText(f"{self._voice.post_gap_ms}")
        # Override label to show ms not 1/10 units
        self._gap_sl.valueChanged.connect(
            lambda v, l=self._gap_sl._val_lbl_ref: l.setText(f"{v*10}"))

        self._refresh_model()

    def _flash_status(self, msg: str):
        """Show brief status in the name field."""
        orig = self._name_edit.text()
        self._name_edit.setText(f"✓ {msg}")
        QTimer.singleShot(1200, lambda: self._name_edit.setText(orig))

    def _slider_row(self, layout, label: str, lo, hi, val, slot) -> QSlider:
        row  = QHBoxLayout()
        lbl  = QLabel(label)
        lbl.setFixedWidth(108)
        sl   = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(val)
        vlbl = QLabel(f"{val/10:.1f}")
        vlbl.setFixedWidth(34)
        sl._val_lbl_ref = vlbl
        sl.valueChanged.connect(lambda v, l=vlbl: l.setText(f"{v/10:.1f}"))
        sl.valueChanged.connect(slot)
        row.addWidget(lbl)
        row.addWidget(sl, 1)
        row.addWidget(vlbl)
        layout.addLayout(row)
        return sl

    def _on_name(self, text):
        self._voice.name = text or "Voice"
        self.changed.emit(self._voice.id)

    def _on_model(self, text):
        self._voice.model = text.lower()
        self._refresh_model()
        self.changed.emit(self._voice.id)

    def _refresh_model(self):
        model = self._voice.model
        self._voice_combo.blockSignals(True)
        self._voice_combo.clear()
        if model == "orpheus":
            self._voice_combo.addItems(ORPHEUS_VOICES)
            cur = self._voice.voice_id if self._voice.voice_id in ORPHEUS_VOICES else ORPHEUS_VOICES[0]
        elif model == "kokoro":
            keys = list(KOKORO_VOICES.keys())
            self._voice_combo.addItems(keys)
            cur  = self._voice.voice_id if self._voice.voice_id in keys else keys[0]
        else:
            self._voice_combo.addItem("default")
            cur = "default"
        self._voice_combo.setCurrentText(cur)
        self._voice.voice_id = cur
        self._voice_combo.blockSignals(False)
        self._orpheus_box.setVisible(model == "orpheus")
        self._kokoro_box.setVisible(model  == "kokoro")
        self._cb_box.setVisible(model      == "chatterbox")

    def _on_voice_id(self, text):
        self._voice.voice_id = text
        self.changed.emit(self._voice.id)

    def _on_orpheus(self):
        self._voice.temperature        = self._temp_sl.value() / 10
        self._voice.top_p              = self._topp_sl.value() / 10
        self._voice.repetition_penalty = self._rep_sl.value()  / 10
        self.changed.emit(self._voice.id)

    def _on_kokoro(self):
        self._voice.speed = self._speed_sl.value() / 10
        self.changed.emit(self._voice.id)

    def _on_cb(self):
        self._voice.exaggeration = self._exag_sl.value() / 10
        self.changed.emit(self._voice.id)

    def _on_gap(self):
        self._voice.post_gap_ms = self._gap_sl.value() * 10
        self.changed.emit(self._voice.id)

    def _browse_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference voice clip", "",
            "Audio files (*.wav *.mp3 *.flac *.ogg)")
        if path:
            self._voice.voice_ref = path
            self._ref_edit.setText(path)
            self.changed.emit(self._voice.id)

    def _pick_color(self):
        chosen = QColorDialog.getColor(QColor(self._voice.color), self, "Pick voice colour")
        if chosen.isValid():
            self._voice.color = chosen.name()
            self._color_btn.setStyleSheet(
                f"background: {self._voice.color}; border-radius: 4px; border: none;")
            self.changed.emit(self._voice.id)

# ── VoicePanel ────────────────────────────────────────────────────────────────

class VoicePanel(QWidget):
    voice_updated = pyqtSignal(str)
    voice_deleted = pyqtSignal(str)
    voice_added   = pyqtSignal(str)

    def __init__(self, voices: dict, parent=None):
        super().__init__(parent)
        self._voices = voices
        self._cards: dict = {}
        self.setFixedWidth(280)
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header with collapse toggle
        hdr = QWidget()
        hdr.setFixedHeight(36)
        hdr.setStyleSheet("background: #181825;")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 8, 0)
        title = QLabel("Voices")
        title.setStyleSheet(
            "color: #cdd6f4; font-weight: bold; font-size: 14px;")
        hl.addWidget(title, 1)
        self._collapse_btn = QToolButton()
        self._collapse_btn.setText("◀")
        self._collapse_btn.setToolTip("Collapse voice panel")
        self._collapse_btn.setFixedSize(24, 24)
        self._collapse_btn.clicked.connect(self._toggle_collapse)
        hl.addWidget(self._collapse_btn)
        root.addWidget(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244;")
        root.addWidget(sep)

        # Collapsible body
        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        self._list = QVBoxLayout(inner)
        self._list.setContentsMargins(8, 8, 8, 8)
        self._list.setSpacing(8)
        self._list.addStretch()
        scroll.setWidget(inner)
        body_layout.addWidget(scroll, 1)

        add_btn = QPushButton("+ Add Voice")
        add_btn.setObjectName("primary")
        add_btn.setFixedHeight(34)
        add_btn.clicked.connect(self._add_voice)
        wrapper = QWidget()
        wl = QHBoxLayout(wrapper)
        wl.setContentsMargins(8, 6, 8, 8)
        wl.addWidget(add_btn)
        body_layout.addWidget(wrapper)

        root.addWidget(self._body, 1)

        for vc in list(self._voices.values()):
            self._add_card(vc)

    def _toggle_collapse(self):
        collapsed = not self._body.isVisible()
        self._body.setVisible(collapsed)
        if collapsed:
            self.setFixedWidth(280)
            self._collapse_btn.setText("◀")
            self._collapse_btn.setToolTip("Collapse voice panel")
        else:
            self.setFixedWidth(36)
            self._collapse_btn.setText("▶")
            self._collapse_btn.setToolTip("Expand voice panel")

    def _add_card(self, vc: VoiceConfig):
        card = VoiceCard(vc, self)
        card.changed.connect(self.voice_updated)
        card.deleted.connect(self._on_delete)
        self._list.insertWidget(self._list.count() - 1, card)
        self._cards[vc.id] = card

    def _add_voice(self):
        idx = len(self._voices) % len(VOICE_COLORS)
        n   = len(self._voices) + 1
        vc  = VoiceConfig(name=f"Voice {n}", color=VOICE_COLORS[idx])
        self._voices[vc.id] = vc
        self._add_card(vc)
        self.voice_added.emit(vc.id)

    def _on_delete(self, vid: str):
        card = self._cards.pop(vid, None)
        if card:
            self._list.removeWidget(card)
            card.deleteLater()
        self._voices.pop(vid, None)
        self.voice_deleted.emit(vid)

# ── Segment timeline ──────────────────────────────────────────────────────────

class SegmentTimeline(QWidget):
    seek_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._segments: list = []
        self._total_ms  = 0
        self._pos_ms    = 0
        self._dragging  = False
        self.setFixedHeight(26)
        self.setMinimumWidth(100)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setMouseTracking(True)

    def load(self, segments: list):
        self._segments = segments
        self._total_ms = segments[-1][1] if segments else 0
        self.update()

    def set_position(self, ms: int):
        self._pos_ms = ms
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#313244"))
        if self._total_ms <= 0:
            p.end()
            return
        for start, end, color, _ in self._segments:
            x1 = int(start / self._total_ms * w)
            x2 = int(end   / self._total_ms * w)
            p.fillRect(x1, 3, max(1, x2 - x1 - 1), h - 6, QColor(color))
        cx = int(self._pos_ms / self._total_ms * w)
        p.fillRect(0, 0, cx, h, QColor(0, 0, 0, 60))
        pen = QPen(QColor("#ffffff"))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawLine(cx, 0, cx, h)
        p.end()

    def _ms_at(self, x: int) -> int:
        if self._total_ms <= 0 or self.width() <= 0:
            return 0
        return max(0, min(self._total_ms, int(x / self.width() * self._total_ms)))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.seek_requested.emit(self._ms_at(int(e.position().x())))

    def mouseMoveEvent(self, e):
        x = int(e.position().x())
        if self._dragging:
            self.seek_requested.emit(self._ms_at(x))
        if self._total_ms > 0:
            ms = self._ms_at(x)
            for start, end, _, name in self._segments:
                if start <= ms < end:
                    self.setToolTip(f"{name}  —  {ms_to_time(ms)}")
                    return
        self.setToolTip(ms_to_time(self._ms_at(x)) if self._total_ms > 0 else "")

    def mouseReleaseEvent(self, e):
        self._dragging = False

# ── PlaybackBar ───────────────────────────────────────────────────────────────

class PlaybackBar(QWidget):
    position_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(68)
        self.setStyleSheet("background: #11111b; border-top: 1px solid #313244;")
        self._player = None
        self._output = None
        self._build()
        self._setup_player()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 4, 16, 4)
        root.setSpacing(3)

        self.timeline = SegmentTimeline(self)
        root.addWidget(self.timeline)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(32, 32)
        self.btn_play.setEnabled(False)
        row.addWidget(self.btn_play)

        self.btn_back = QPushButton("◀ 10")
        self.btn_back.setFixedHeight(28)
        self.btn_back.setEnabled(False)
        row.addWidget(self.btn_back)

        self.btn_fwd = QPushButton("10 ▶")
        self.btn_fwd.setFixedHeight(28)
        self.btn_fwd.setEnabled(False)
        row.addWidget(self.btn_fwd)

        row.addStretch(1)

        self.lbl_time = QLabel("0:00 / 0:00")
        self.lbl_time.setFixedWidth(90)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_time.setStyleSheet("color: #6c7086; font-size: 12px;")
        row.addWidget(self.lbl_time)

        row.addWidget(QLabel("⚡"))

        self.speed_combo = QComboBox()
        for s in SPEEDS:
            self.speed_combo.addItem(s)
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.setFixedWidth(68)
        row.addWidget(self.speed_combo)

        row.addWidget(QLabel("🔊"))

        self.vol = QSlider(Qt.Orientation.Horizontal)
        self.vol.setRange(0, 200)
        self.vol.setValue(80)
        self.vol.setFixedWidth(80)
        row.addWidget(self.vol)

        root.addLayout(row)

    def _setup_player(self):
        if not HAS_MEDIA:
            return
        self._player = QMediaPlayer()
        self._output = QAudioOutput()
        self._player.setAudioOutput(self._output)
        self._output.setVolume(0.8)
        self._player.positionChanged.connect(self._on_pos)
        self._player.durationChanged.connect(self._on_dur)
        self._player.playbackStateChanged.connect(self._on_state)
        self.btn_play.clicked.connect(self._toggle)
        self.btn_back.clicked.connect(lambda: self._seek_rel(-10000))
        self.btn_fwd.clicked.connect(lambda:  self._seek_rel( 10000))
        self.timeline.seek_requested.connect(self._on_timeline_seek)
        self.speed_combo.currentIndexChanged.connect(self._on_speed)
        self.vol.valueChanged.connect(self._on_vol)

    def load(self, wav_path: str):
        if not self._player:
            return
        self._player.setSource(QUrl.fromLocalFile(wav_path))
        for b in (self.btn_play, self.btn_back, self.btn_fwd):
            b.setEnabled(True)

    def load_timeline(self, segments: list):
        self.timeline.load(segments)

    def stop(self):
        if self._player:
            self._player.stop()

    def toggle_play(self):
        self._toggle()

    def _toggle(self):
        if not self._player:
            return
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _seek_rel(self, delta_ms: int):
        if self._player:
            self._player.setPosition(max(0, self._player.position() + delta_ms))

    def _on_timeline_seek(self, ms: int):
        if self._player:
            self._player.setPosition(ms)

    def seek_and_play(self, ms: int):
        if self._player:
            self._player.setPosition(ms)
            self._player.play()

    def is_playing(self) -> bool:
        return (self._player is not None and
                self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState)

    def _on_pos(self, ms: int):
        dur = self._player.duration()
        self.timeline.set_position(ms)
        self.lbl_time.setText(f"{ms_to_time(ms)} / {ms_to_time(dur)}")
        self.position_changed.emit(ms)

    def _on_dur(self, dur: int):
        self.lbl_time.setText(f"{ms_to_time(0)} / {ms_to_time(dur)}")

    def _on_state(self, state):
        self.btn_play.setText(
            "⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶")

    def _on_speed(self, idx: int):
        if self._player:
            self._player.setPlaybackRate(SPEED_VALUES[idx])

    def _on_vol(self, v: int):
        if self._output:
            self._output.setVolume(v / 100.0)

# ── Main window ───────────────────────────────────────────────────────────────

class StudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(_make_icon())
        self.setMinimumSize(1100, 700)
        self.resize(1340, 840)

        self._voices: dict       = {}
        self._segments: list     = []
        self._widgets: dict      = {}
        self._selected: set      = set()
        self._worker             = None
        self._queued_segs: list  = []     # segments waiting for the next free worker
        self._temp_wav: str      = ""
        self._playback_map: list = []
        self._last_playing_id    = None
        self._dirty: bool        = False
        self._project_path: str  = ""
        self._seg_scroll         = None   # set in _build_segments_panel

        vc = VoiceConfig(id="default", name="Narrator", color=VOICE_COLORS[0])
        self._voices[vc.id] = vc

        self.setStyleSheet(STYLE)
        self._build()
        self._update_title()

        # Space = play/pause
        QShortcut(QKeySequence(Qt.Key.Key_Space), self,
                  activated=self._pb.toggle_play)

    # ── Title ──────────────────────────────────────────────────────────────────

    def _update_title(self):
        name = Path(self._project_path).stem if self._project_path else "Untitled"
        dirty = " •" if self._dirty else ""
        self.setWindowTitle(f"OfflineNarrator — {name}{dirty}")

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self._update_title()

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #313244; max-height: 1px;")
        root.addWidget(sep)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(1)
        self._splitter.setStyleSheet("QSplitter::handle { background: #313244; }")
        self._splitter.addWidget(self._build_segments_panel())

        self._voice_panel = VoicePanel(self._voices, self)
        self._voice_panel.voice_updated.connect(self._on_voice_updated)
        self._voice_panel.voice_deleted.connect(self._on_voice_deleted)
        self._voice_panel.voice_added.connect(self._on_voice_added)
        self._splitter.addWidget(self._voice_panel)
        self._splitter.setSizes([1060, 280])

        root.addWidget(self._splitter, 1)

        self._pb = PlaybackBar(self)
        self._pb.position_changed.connect(self._on_playback_pos)
        root.addWidget(self._pb)

        self.setStatusBar(QStatusBar())
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                _hw = f"GPU: {gpu}"
            else:
                _hw = "CPU mode (no CUDA)"
        except Exception:
            _hw = "Ready"
        self.statusBar().showMessage(f"Ready  ·  {_hw}")

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet("background: #181825;")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        title = QLabel("OfflineNarrator")
        title.setStyleSheet("color: #cba6f7; font-size: 16px; font-weight: bold;")
        row.addWidget(title)
        row.addSpacing(12)

        self.btn_gen_all = QPushButton("⚡  Generate All")
        self.btn_gen_all.setObjectName("primary")
        self.btn_gen_all.setFixedHeight(36)
        self.btn_gen_all.clicked.connect(self._on_generate_all)
        row.addWidget(self.btn_gen_all)

        self.btn_gen_missing = QPushButton("⚡  Generate Missing")
        self.btn_gen_missing.setFixedHeight(36)
        self.btn_gen_missing.setToolTip("Generate only segments without audio yet")
        self.btn_gen_missing.clicked.connect(self._on_generate_missing)
        row.addWidget(self.btn_gen_missing)

        self.btn_gen_sel = QPushButton("✦  Generate Selected")
        self.btn_gen_sel.setFixedHeight(36)
        self.btn_gen_sel.setEnabled(False)
        self.btn_gen_sel.setToolTip(
            "Tick the checkbox on each segment you want,\nthen click this button.")
        self.btn_gen_sel.clicked.connect(self._on_generate_selected)
        row.addWidget(self.btn_gen_sel)

        # Progress bar (hidden when idle)
        self._progress = QProgressBar()
        self._progress.setFixedSize(120, 8)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.hide()
        row.addWidget(self._progress)

        row.addStretch()

        for label, slot in [
            ("📂 Import",      self._on_import),
            ("💾 Save",        self._on_save_project),
            ("📁 Load",        self._on_load_project),
            ("⬇ Export WAV",  self._on_export),
            ("⬇ Export Parts", self._on_export_parts),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(34)
            btn.clicked.connect(slot)
            row.addWidget(btn)

        return bar

    def _build_segments_panel(self) -> QWidget:
        outer = QWidget()
        vl    = QVBoxLayout(outer)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # Search bar
        search_bar = QWidget()
        search_bar.setStyleSheet("background: #11111b;")
        search_bar.setFixedHeight(36)
        sl = QHBoxLayout(search_bar)
        sl.setContentsMargins(8, 4, 8, 4)
        sl.setSpacing(6)
        sl.addWidget(QLabel("🔍"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search segments…")
        self._search_edit.setFixedHeight(26)
        self._search_edit.textChanged.connect(self._on_search)
        self._search_clear = QToolButton()
        self._search_clear.setText("✕")
        self._search_clear.setFixedSize(20, 20)
        self._search_clear.clicked.connect(lambda: self._search_edit.clear())
        sl.addWidget(self._search_edit, 1)
        sl.addWidget(self._search_clear)
        vl.addWidget(search_bar)

        # Column header
        hdr = QWidget()
        hdr.setFixedHeight(26)
        hdr.setStyleSheet("background: #11111b;")
        hl  = QHBoxLayout(hdr)
        hl.setContentsMargins(60, 0, 12, 0)
        for txt, w in [("  #", 36), ("Voice", 84), ("Segment text", -1), ("Status", 60)]:
            lbl = QLabel(txt)
            lbl.setStyleSheet("color: #6c7086; font-size: 10px;")
            if w > 0:
                lbl.setFixedWidth(w)
            else:
                hl.addWidget(lbl, 1)
                continue
            hl.addWidget(lbl)
        vl.addWidget(hdr)

        # Scrollable segment list
        self._seg_scroll = QScrollArea()
        self._seg_scroll.setWidgetResizable(True)
        self._seg_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._seg_container = SegmentsContainer(self)
        self._seg_container.reorder_requested.connect(self._on_reorder)
        self._seg_layout = self._seg_container.layout()

        self._empty_lbl = QLabel(
            "No segments yet.\n\nClick  📂 Import  to paste a script,\n"
            "or right-click here to add a blank segment.")
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_lbl.setStyleSheet("color: #45475a; font-size: 14px;")
        self._seg_layout.insertWidget(0, self._empty_lbl)

        self._seg_scroll.setWidget(self._seg_container)
        vl.addWidget(self._seg_scroll, 1)

        self._seg_container.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self._seg_container.customContextMenuRequested.connect(
            self._panel_context_menu)

        return outer

    # ── Segment management ────────────────────────────────────────────────────

    def _add_segment(self, seg: Segment, index: int = -1):
        self._empty_lbl.hide()
        w = SegmentWidget(seg, self._voices, self._seg_container)
        w.regenerate_requested.connect(self._on_regen_one)
        w.delete_requested.connect(self._delete_segment)
        w.add_above_requested.connect(self._add_blank_above)
        w.play_from_requested.connect(self._on_play_from)
        w.selection_toggled.connect(self._on_selection_toggled)
        w.split_requested.connect(self._split_segment)
        w.merge_requested.connect(self._merge_with_next)
        w.duplicate_requested.connect(self._duplicate_segment)
        w.silence_requested.connect(self._insert_silence)
        w.text_edited.connect(lambda sid, t: self._mark_dirty())
        w.voice_change_requested.connect(lambda sid, vid: self._mark_dirty())

        if index < 0 or index >= len(self._segments):
            self._segments.append(seg)
            self._seg_layout.insertWidget(self._seg_layout.count() - 1, w)
        else:
            self._segments.insert(index, seg)
            # +1 to skip the _empty_lbl at layout position 0
            self._seg_layout.insertWidget(index + 1, w)

        self._widgets[seg.id] = w
        self._renumber()
        self._mark_dirty()

    def _delete_segment(self, seg_id: str):
        seg = next((s for s in self._segments if s.id == seg_id), None)
        if not seg:
            return
        self._segments.remove(seg)
        w = self._widgets.pop(seg_id, None)
        if w:
            self._seg_layout.removeWidget(w)
            w.deleteLater()
        self._selected.discard(seg_id)
        self._update_gen_sel_btn()
        self._renumber()
        if not self._segments:
            self._empty_lbl.show()
        self._mark_dirty()

    def _add_blank_above(self, ref_id: str):
        idx = next((i for i, s in enumerate(self._segments) if s.id == ref_id), -1)
        self._add_segment(Segment(voice_id=self._first_voice_id()), index=idx)

    def _add_blank_at_end(self):
        self._add_segment(Segment(voice_id=self._first_voice_id()))

    def _split_segment(self, seg_id: str, cursor_pos: int):
        seg = next((s for s in self._segments if s.id == seg_id), None)
        if not seg:
            return
        idx = next(i for i, s in enumerate(self._segments) if s.id == seg_id)

        if cursor_pos >= 0:
            # Split at exact cursor position in text
            left  = seg.text[:cursor_pos].strip()
            right = seg.text[cursor_pos:].strip()
            parts = [p for p in [left, right] if p]
        else:
            # Auto-split by sentence boundaries
            parts = split_sentences(seg.text)

        if len(parts) < 2:
            self.statusBar().showMessage("Nothing to split — need at least 2 sentences.")
            return

        vid = seg.voice_id
        self._delete_segment(seg_id)
        for i, part in enumerate(reversed(parts)):
            new_seg = Segment(text=part, voice_id=vid)
            self._add_segment(new_seg, index=idx)
        self.statusBar().showMessage(f"Split into {len(parts)} segments.")

    def _merge_with_next(self, seg_id: str):
        idx = next((i for i, s in enumerate(self._segments) if s.id == seg_id), -1)
        if idx < 0 or idx + 1 >= len(self._segments):
            self.statusBar().showMessage("No next segment to merge with.")
            return
        curr = self._segments[idx]
        nxt  = self._segments[idx + 1]
        merged_text = f"{curr.text} {nxt.text}".strip()
        self._delete_segment(nxt.id)
        curr.text   = merged_text
        curr.status = "pending"
        curr.audio  = None
        self._widgets[curr.id].refresh()
        self.statusBar().showMessage("Segments merged.")

    def _duplicate_segment(self, seg_id: str):
        seg = next((s for s in self._segments if s.id == seg_id), None)
        if not seg:
            return
        idx = next(i for i, s in enumerate(self._segments) if s.id == seg_id)
        dup = Segment(text=seg.text, voice_id=seg.voice_id)
        self._add_segment(dup, index=idx + 1)
        self.statusBar().showMessage("Segment duplicated.")

    def _insert_silence(self, seg_id: str):
        seg = next((s for s in self._segments if s.id == seg_id), None)
        if not seg:
            return
        ms, ok = QInputDialog.getInt(
            self, "Insert silence", "Silence duration (ms):",
            value=500, min=50, max=10000, step=50)
        if ok:
            seg.post_gap_ms = ms
            self.statusBar().showMessage(f"Gap of {ms}ms set after segment.")
            self._mark_dirty()

    def _on_reorder(self, seg_id: str, new_index: int):
        # new_index is a layout index (_empty_lbl sits at layout pos 0,
        # so layout pos N == _segments index N-1)
        old_idx = next((i for i, s in enumerate(self._segments) if s.id == seg_id), -1)
        old_layout_pos = old_idx + 1   # +1 to account for _empty_lbl
        if old_idx < 0 or new_index == old_layout_pos:
            return
        seg = self._segments.pop(old_idx)
        w   = self._widgets[seg_id]
        self._seg_layout.removeWidget(w)
        # After removal, indices past the old position shift down by 1
        if new_index > old_layout_pos:
            layout_target = new_index - 1
        else:
            layout_target = new_index
        seg_target = max(0, min(layout_target - 1, len(self._segments)))
        self._segments.insert(seg_target, seg)
        self._seg_layout.insertWidget(layout_target, w)
        self._renumber()
        self._rebuild_playback()
        self._mark_dirty()

    def _renumber(self):
        total = len(self._segments)
        for i, seg in enumerate(self._segments):
            if seg.id in self._widgets:
                self._widgets[seg.id].set_number(i + 1, total)

    def _first_voice_id(self) -> str:
        return next(iter(self._voices), "")

    def _panel_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("➕  Add blank segment", self._add_blank_at_end)
        menu.exec(self._seg_container.mapToGlobal(pos))

    # ── Search ────────────────────────────────────────────────────────────────

    def _on_search(self, query: str):
        q = query.strip().lower()
        first_match = None
        for seg in self._segments:
            w = self._widgets.get(seg.id)
            if not w:
                continue
            if q and q not in seg.text.lower():
                eff = QGraphicsOpacityEffect()
                eff.setOpacity(0.3)
                w.setGraphicsEffect(eff)
            else:
                w.setGraphicsEffect(None)
                if q and first_match is None:
                    first_match = w
        if first_match and self._seg_scroll:
            self._seg_scroll.ensureWidgetVisible(first_match)

    # ── Voice events ──────────────────────────────────────────────────────────

    def _on_voice_updated(self, vid: str):
        for w in self._widgets.values():
            w.refresh()
        self._mark_dirty()

    def _on_voice_deleted(self, vid: str):
        fallback = self._first_voice_id()
        for seg in self._segments:
            if seg.voice_id == vid:
                seg.voice_id = fallback
        for w in self._widgets.values():
            w.refresh()
        self._mark_dirty()

    def _on_voice_added(self, vid: str):
        for seg in self._segments:
            if not seg.voice_id or seg.voice_id not in self._voices:
                seg.voice_id = vid
        for w in self._widgets.values():
            w.refresh()
        self._mark_dirty()

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_selection_toggled(self, seg_id: str, selected: bool):
        if selected:
            self._selected.add(seg_id)
        else:
            self._selected.discard(seg_id)
        self._update_gen_sel_btn()

    def _update_gen_sel_btn(self):
        n = len(self._selected)
        self.btn_gen_sel.setEnabled(bool(n))
        self.btn_gen_sel.setText(
            f"✦  Generate Selected ({n})" if n else "✦  Generate Selected")

    # ── Generation ────────────────────────────────────────────────────────────

    def _on_generate_all(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._reset_gen_ui()
            return
        self._start_generation(self._segments)

    def _on_generate_missing(self):
        segs = [s for s in self._segments if s.status != "done"]
        if segs:
            self._start_generation(segs)
        else:
            self.statusBar().showMessage("All segments already have audio.")

    def _on_generate_selected(self):
        segs = [s for s in self._segments if s.id in self._selected]
        if segs:
            self._start_generation(segs)

    def _on_regen_one(self, seg_id: str):
        seg = next((s for s in self._segments if s.id == seg_id), None)
        if seg:
            self._start_generation([seg])

    def _start_generation(self, segs: list):
        if not segs:
            return

        # If a worker is already running, queue the segments instead
        if self._worker and self._worker.isRunning():
            existing_ids = {s.id for s in self._queued_segs}
            for s in segs:
                if s.id not in existing_ids:
                    self._queued_segs.append(s)
            n = len(self._queued_segs)
            self.statusBar().showMessage(
                f"Queued {len(segs)} segment(s) — {n} waiting after current run.")
            return

        self._queued_segs.clear()
        self._pb.stop()
        self._worker = GenerationWorker(segs, self._voices)
        self._worker.segment_started.connect(self._on_seg_started)
        self._worker.segment_done.connect(self._on_seg_done)
        self._worker.segment_error.connect(self._on_seg_error)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.status_msg.connect(self.statusBar().showMessage)
        self._worker.progress.connect(self._on_progress)

        total = len(segs)
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.show()

        self.btn_gen_all.setText("⬛  Stop")
        self.btn_gen_all.setObjectName("danger")
        self.btn_gen_all.setStyleSheet(
            "QPushButton#danger { background: #f38ba8; color: #ffffff; "
            "font-weight: bold; border: none; border-radius: 6px; padding: 6px 14px; }")
        self.btn_gen_sel.setEnabled(False)
        self.btn_gen_missing.setEnabled(False)

        # Disable individual regen buttons while a run is active
        for w in self._widgets.values():
            w.set_regen_enabled(False)

        self._worker.start()

    def _on_progress(self, done: int, total: int):
        self._progress.setValue(done)

    def _on_seg_started(self, seg_id: str):
        seg = next((s for s in self._segments if s.id == seg_id), None)
        if seg:
            seg.status = "generating"
            if seg_id in self._widgets:
                self._widgets[seg_id].refresh()

    def _on_seg_done(self, seg_id: str, audio, sr: int):
        seg = next((s for s in self._segments if s.id == seg_id), None)
        if seg:
            seg.audio = audio;  seg.sample_rate = sr;  seg.status = "done"
            if seg_id in self._widgets:
                self._widgets[seg_id].refresh()

    def _on_seg_error(self, seg_id: str, msg: str):
        seg = next((s for s in self._segments if s.id == seg_id), None)
        if seg:
            seg.status = "error"
            if seg_id in self._widgets:
                self._widgets[seg_id].refresh()
        self.statusBar().showMessage(f"Error: {msg}")

    def _on_all_done(self):
        self._reset_gen_ui()
        self._progress.hide()
        self._rebuild_playback()
        self._mark_dirty()

        # Process any queued segments
        if self._queued_segs:
            queued = self._queued_segs[:]
            self._queued_segs.clear()
            self.statusBar().showMessage(
                f"Starting queued generation ({len(queued)} segment(s))…")
            self._start_generation(queued)
        else:
            self.statusBar().showMessage("Generation complete.")

    def _reset_gen_ui(self):
        self.btn_gen_all.setText("⚡  Generate All")
        self.btn_gen_all.setObjectName("primary")
        self.btn_gen_all.setStyleSheet("")
        self.btn_gen_missing.setEnabled(True)
        self._update_gen_sel_btn()
        self._progress.hide()
        for w in self._widgets.values():
            w.set_regen_enabled(True)

    # ── Playback ──────────────────────────────────────────────────────────────

    def _on_play_from(self, seg_id: str):
        for start_ms, _end_ms, sid in self._playback_map:
            if sid == seg_id:
                self._pb.seek_and_play(start_ms)
                return

    def _rebuild_playback(self):
        generated = [(s, s.audio) for s in self._segments
                     if s.audio is not None and s.audio.size > 0]
        if not generated:
            return

        arrays, self._playback_map, timeline_map = [], [], []
        cursor = 0
        for seg, audio in generated:
            dur   = int(len(audio) / seg.sample_rate * 1000)
            voice = self._voices.get(seg.voice_id)
            color = voice.color if voice else "#6c7086"
            name  = voice.name  if voice else "—"
            self._playback_map.append((cursor, cursor + dur, seg.id))
            timeline_map.append((cursor, cursor + dur, color, name))
            arrays.append(audio.astype(np.float32))
            cursor += dur

            # Voice-level gap
            gap_ms = (voice.post_gap_ms if voice else 0) + seg.post_gap_ms
            if gap_ms > 0:
                silence = np.zeros(int(24000 * gap_ms / 1000), dtype=np.float32)
                arrays.append(silence)
                cursor += gap_ms

        combined = np.concatenate(arrays)
        if self._temp_wav and os.path.exists(self._temp_wav):
            try:
                os.unlink(self._temp_wav)
            except Exception:
                pass
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, combined, 24000)
        tmp.close()
        self._temp_wav = tmp.name
        self._pb.load(self._temp_wav)
        self._pb.load_timeline(timeline_map)

    def _on_playback_pos(self, ms: int):
        playing_id = None
        for start, end, sid in self._playback_map:
            if start <= ms < end:
                playing_id = sid
                break
        if playing_id == self._last_playing_id:
            return
        if self._last_playing_id and self._last_playing_id in self._widgets:
            self._widgets[self._last_playing_id].set_playing(False)
        self._last_playing_id = playing_id
        if playing_id and playing_id in self._widgets:
            w = self._widgets[playing_id]
            w.set_playing(True)
            if self._seg_scroll and self._pb.is_playing():
                self._seg_scroll.ensureWidgetVisible(w)

    # ── Import ────────────────────────────────────────────────────────────────

    def _on_import(self):
        dlg = TextImportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        text = dlg.get_text().strip()
        if not text:
            return

        if self._segments:
            ans = QMessageBox.question(
                self, "Replace segments?",
                "Replace all existing segments with imported text?\n"
                "Choose No to append instead.",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No  |
                QMessageBox.StandardButton.Cancel)
            if ans == QMessageBox.StandardButton.Cancel:
                return
            if ans == QMessageBox.StandardButton.Yes:
                self._clear_all_segments()

        sentences = split_sentences(text)
        fid = self._first_voice_id()
        for sent in sentences:
            self._add_segment(Segment(text=sent, voice_id=fid))
        self.statusBar().showMessage(f"Imported {len(sentences)} segments.")

    def _clear_all_segments(self):
        for w in list(self._widgets.values()):
            self._seg_layout.removeWidget(w)
            w.deleteLater()
        self._widgets.clear()
        self._segments.clear()
        self._selected.clear()
        self._empty_lbl.show()
        self._update_gen_sel_btn()

    # ── Save / Load / Export ──────────────────────────────────────────────────

    def _on_save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project",
            self._project_path or "project.ttsproj",
            "TTS project (*.ttsproj);;JSON (*.json)")
        if not path:
            return
        data = {
            "voices":   [v.to_dict() for v in self._voices.values()],
            "segments": [s.to_dict() for s in self._segments],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self._project_path = path
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"Saved → {path}")

    def _on_load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project", "",
            "TTS project (*.ttsproj);;JSON (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._clear_all_segments()
        self._voices.clear()
        for card in list(self._voice_panel._cards.values()):
            self._voice_panel._list.removeWidget(card)
            card.deleteLater()
        self._voice_panel._cards.clear()

        for vd in data.get("voices", []):
            vc = VoiceConfig.from_dict(vd)
            self._voices[vc.id] = vc
            self._voice_panel._add_card(vc)

        for sd in data.get("segments", []):
            seg = Segment.from_dict(sd)
            self._add_segment(seg)

        self._project_path = path
        self._dirty = False
        self._update_title()
        self._rebuild_playback()
        self.statusBar().showMessage(f"Loaded {len(self._segments)} segments.")

    def _on_export(self):
        generated = [s for s in self._segments
                     if s.audio is not None and s.audio.size > 0]
        if not generated:
            self.statusBar().showMessage("Nothing to export — generate audio first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audio", "output.wav", "WAV files (*.wav)")
        if not path:
            return
        arrays = [s.audio.astype(np.float32) for s in generated]
        sf.write(path, np.concatenate(arrays), 24000)
        self.statusBar().showMessage(f"Exported → {path}")

    def _on_export_parts(self):
        generated = [(i, s) for i, s in enumerate(self._segments, 1)
                     if s.audio is not None and s.audio.size > 0]
        if not generated:
            self.statusBar().showMessage("Nothing to export — generate audio first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export parts to folder")
        if not folder:
            return
        for n, seg in generated:
            voice = self._voices.get(seg.voice_id)
            vname = (voice.name if voice else "Unknown").replace(" ", "_")
            fname = os.path.join(folder, f"{n:03d}_{vname}.wav")
            sf.write(fname, seg.audio.astype(np.float32), seg.sample_rate)
        self.statusBar().showMessage(
            f"Exported {len(generated)} files to {folder}")

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, e):
        if self._dirty:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Save   |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel)
            if ans == QMessageBox.StandardButton.Cancel:
                e.ignore()
                return
            if ans == QMessageBox.StandardButton.Save:
                self._on_save_project()

        self._pb.stop()
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        if self._temp_wav and os.path.exists(self._temp_wav):
            try:
                os.unlink(self._temp_wav)
            except Exception:
                pass
        e.accept()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = StudioWindow()
    win.show()
    sys.exit(app.exec())
