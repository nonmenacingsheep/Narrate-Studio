# OfflineNarrator

A local, offline narration studio for Windows. Import a script, assign voices to each sentence, generate studio-quality audio, and export — no internet required after the first model download.

Built with [Orpheus TTS](https://github.com/canopylabs/orpheus-tts), [Kokoro](https://github.com/hexgrad/kokoro), and [Chatterbox](https://github.com/resemble-ai/chatterbox).

I made this for myself using Claude, but I figured someone else might be able to use it as well. I hope you find it useful!

![OfflineNarrator](assets/UI.PNG)

---

## Requirements

- **Windows 10/11**
- **Python 3.10, 3.11, 3.12, or 3.13** — [python.org](https://www.python.org/downloads/) *(tick "Add Python to PATH")*
  > ⚠️ Python 3.14+ is not supported yet — PyTorch does not have wheels for it.
  > 
  > Easiest way to get the right version on Windows — run this in a terminal:
  > ```
  > winget install Python.Python.3.13
  > ```
- **NVIDIA GPU strongly recommended** — an RTX card will generate in seconds; CPU works but is very slow
- **~10 GB free disk space** for model downloads (cached locally after first run)
- A free **HuggingFace account** for the Orpheus model (see setup)

---

## Installation

1. **Clone or download** this repository into a folder of your choice.

2. **Double-click `setup.bat`** and follow the prompts. It will:
   - Create an isolated Python environment (`.venv/`)
   - Install PyTorch with CUDA support
   - Install all dependencies
   - Walk you through getting a HuggingFace token for Orpheus

3. **Double-click `run.bat`** to launch the app.

> On first launch, each model you use will download automatically and be cached for all future runs. Orpheus is ~6 GB, so the first generation will take a while — subsequent ones are fast.

---

## HuggingFace Token (Orpheus only)

Orpheus is a gated model that requires a free HuggingFace account and one-click access approval.

1. Sign up at [huggingface.co](https://huggingface.co)
2. Visit the model page and click **Request access**: [canopylabs/orpheus-tts-0.1-finetune-prod](https://huggingface.co/canopylabs/orpheus-tts-0.1-finetune-prod)
3. Once approved, go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a **Read** token
4. Paste the token into `hf_token.txt` (one line, no spaces)

Kokoro and Chatterbox do not require an account.

---

## Usage

### Importing a script

Click **📂 Import** in the toolbar and paste your script, or load a `.txt` file. The text is automatically split into sentences — each sentence becomes a **segment** that can be individually voiced and regenerated.

### Voices

The **Voices panel** on the right lets you create and configure as many voices as you need. Each voice has:

- A **name** and **colour** (the colour shows on the segment list and playback timeline)
- A **model** (Orpheus, Kokoro, or Chatterbox)
- Model-specific settings (temperature, expressiveness, speed, etc.)
- A **gap-after** slider to add silence between segments

Click the **voice badge** on any segment (or right-click → Change voice) to assign a different voice to that segment.

### Generating audio

| Button | What it does |
|---|---|
| ⚡ **Generate All** | Generates every segment from scratch |
| ⚡ **Generate Missing** | Only generates segments that don't have audio yet |
| ✦ **Generate Selected** | Generates only the segments you've ticked |
| ↺ *(on each segment)* | Regenerates just that one segment |

Click **⚡ Generate All** again while generating to stop early.

### Editing segments

- **Double-click** a segment to edit its text inline. Press **Ctrl+Enter** or click away to save.
- **Right-click** a segment for the full context menu:
  - Split at cursor / auto-split by sentences
  - Merge with next segment
  - Duplicate
  - Insert silence after
  - Change voice
  - Add segment below / Delete

- **Drag the ⠿ handle** on the left to reorder segments.
- Use the **🔍 search bar** at the top to filter segments — non-matching ones fade out.

### Playback

The **colour-coded timeline** at the bottom shows which voice is speaking at every point. Click or drag it to seek. Hover for the voice name and timestamp.

| Control | Action |
|---|---|
| **Space** | Play / Pause |
| **◀ 10 / 10 ▶** | Skip back / forward 10 seconds |
| Speed combo | 0.5× – 2.0× playback speed |
| Volume slider | Output volume |

### Exporting

| Button | Output |
|---|---|
| **⬇ Export WAV** | Single combined WAV file |
| **⬇ Export Parts** | One WAV file per segment, named `001_VoiceName.wav` etc. |

### Saving & loading projects

Use **💾 Save** and **📁 Load** to save your work as a `.ttsproj` file. This saves everything — segments, text, voice assignments, settings, and all generated audio. When you reload a project, audio that was already generated plays immediately without re-generating.

> **Note:** Generated audio is embedded in the project file, so `.ttsproj` files can get large for long projects (~500 KB per minute of audio per segment).

---

## Models

| Model | Voice quality | Notes |
|---|---|---|
| **Orpheus** | ⭐⭐⭐⭐⭐ | Best expressiveness. Supports emotion tags. ~6 GB. Requires HF token. |
| **Kokoro** | ⭐⭐⭐⭐ | Fast, high quality, many voices. ~400 MB. No account needed. |
| **Chatterbox** | ⭐⭐⭐⭐ | Voice cloning from a reference clip. ~1 GB. No account needed. |

### Orpheus emotion tags

Insert these directly into segment text to add expression:

`<laugh>` `<chuckle>` `<sigh>` `<cough>` `<sniffle>` `<groan>` `<yawn>` `<gasp>`

Example: `She looked at the mess and <sigh> cleaned it up anyway.`

Copy buttons for all tags are in the Orpheus voice card.

---

## Tips

- **Long paragraphs** are fine — Orpheus automatically splits them into sentences internally and joins the audio seamlessly.
- **Voice panel** can be collapsed with the **◀** button to give more room to the segment list.
- A **•** in the title bar means you have unsaved changes.
- The segment list shows `x/total` numbering and a coloured status dot: ○ pending · ◌ generating · ● done · ● error.

---

## Credits

- [Orpheus TTS](https://github.com/canopylabs/orpheus-tts) by Canopy Labs
- [Kokoro](https://github.com/hexgrad/kokoro) by hexgrad
- [Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI
- [SNAC codec](https://github.com/hubertsiuzdak/snac) by Hubert Siuzdak

Built with [Claude](https://claude.ai/claude-code).

---

## License

MIT — do whatever you want, just keep the model credits.
