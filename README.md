# Goal # 
Preserve and make searchable Dharma CD teachings by ripping CDs to audio, transcribing with Whisper, and cleaning/extracting metadata using LLMs — using a pure functional, data‑oriented design.  

## Setup ##
This project relies on a dedicated Whisper transcription environment. Environment reproducibility is critical because ML dependencies (Torch, FFmpeg, NumPy) are platform-sensitive.  

To install the conda env of whisper_env, do one of the following:  

Windows user, run: ```conda env create -f whisper_env_win.yml```  
macOS user, run: ```conda env create -f whisper_env_win.yml```  

After creation:  
```conda activate whisper_env```  