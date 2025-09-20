# Profile-AI

Profile-AI is a project for running and emulating an AI-powered profile system. The project includes a server and optional AI model integration.

## Quick Start

1. **Clone the repository:**

   ```bash
   git clone <repo_url>
   cd Profile-AI
   ```

2. **Create virtual environment**

   ```bash
    python -m venv venv
    source venv/bin/activate
   ```

3. **Install dependencies:**

   - For running the server (emulation mode, without AI):
     `bash
	 pip install -r server/requirements.txt
	 `
   - For running with AI model support (optional, only if you want to use the neural network):
     `bash
	 pip install -r requirements.txt
	 `

4. **(Optional) Download the AI model:**

   - Due to GitHub file size restrictions, the model files are not included in this repository.
   - If you want to run the AI model, download it from the following link and place it in the `models/yoda/` directory:
     [Google Drive - Model Download](https://drive.google.com/drive/folders/1UZhohAOM_INeFmB6rw6rZjn47Q9m4fZT?usp=sharing)
   - This step is only required if you want to use the neural network.

5. **Run the server:**
   ```bash
   fastapi dev server/main.py
   ```

## Notes

- Ai model is tested on python version 3.10.18

- There are two requirements files:

  - `server/requirements.txt` — required for running the server (including emulation mode).
  - `requirements.txt` — only needed if you want to run the neural network (AI model). Not required for emulation mode.

- The AI models for different characters are located in `models/`.

## API Documentation

You can find interactive API documentation at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Project Structure

- `server/` — main server code
- `models/` — AI model files (optional)
- `ai_requirements.txt` — requirements for AI model (optional)
- `server/requirements.txt` — requirements for server
