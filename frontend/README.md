# AI Prompt Optimizer

A smart system that takes vague/short user prompts and transforms them into well-structured, detailed prompts using **NLP** and **Machine Learning** techniques. Built with **FastAPI** (backend), **React** (frontend), and deployable on **Google Colab** with **ngrok**.

## 🚀 Features

- **Slang Normalization** — Handles abbreviations (`abt` → `about`, `enna` → `what`) and Tamil/Hindi loanwords
- **Intent Classification** — Logistic Regression + TF-IDF classifies prompts as `informative`, `creative`, `analytical`, or `conversational`
- **Keyword Extraction** — TF-IDF based keyword extraction from a curated corpus
- **Named Entity Recognition** — spaCy extracts entities (organizations, locations, etc.)
- **Prompt Quality Scoring** — Scores prompts as Poor/Average/Excellent
- **Suggestion Engine** — Analyzes prompts and suggests improvements
- **Prompt Optimization** — Generates optimized prompts with domain, tone, audience, and length context
- **Structured Form (Skolemization)** — Converts prompts into logical forms like `Explain(Artificial Intelligence)`
- **Modern React UI** — Dark theme, responsive design, real-time feedback

## 📁 Project Structure

```
nl-final/
├── colab_backend.py          # FastAPI server for Google Colab + ngrok
├── finalprompt (2).ipynb     # Original Jupyter notebook
├── frontend/                 # React application
│   ├── src/
│   │   ├── App.tsx           # Main component with all UI logic
│   │   ├── App.css           # Dark theme styling
│   │   └── ...
│   ├── package.json
│   └── ...
└── README.md
```

Run the project
1. Start the backend
Open a PowerShell terminal and run:

2. Start the frontend
Open a second PowerShell terminal and run:

3. Open the UI
Frontend: http://localhost:3000
Backend health: http://localhost:8000
Backend docs: http://localhost:8000/docs

## 🛠️ Setup Instructions

### Part 1: Frontend (React App)

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```
   The app will open at `http://localhost:3000`

### Part 2: Backend (Google Colab + ngrok)

1. **Go to [Google Colab](https://colab.research.google.com/)**

2. **Upload `colab_backend.py`** to a new notebook (or paste its contents into a code cell)

3. **Get your ngrok Auth Token:**
   - Sign up at [ngrok.com](https://dashboard.ngrok.com/signup)
   - Go to [Your Authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
   - Copy your token

4. **In the Colab cell, uncomment and set your ngrok token:**
   ```python
   ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")  # Replace with your token
   ```

5. **Run the entire Colab cell.** You'll see output like:
   ```
   🚀 Backend API is live at: https://xxxx-xx-xx-xx-xx.ngrok-free.app
   📡 API Docs: https://xxxx-xx-xx-xx-xx.ngrok-free.app/docs
   ```

6. **Copy the ngrok URL** (e.g., `https://xxxx.ngrok-free.app`)

### Part 3: Connect Frontend to Backend

1. **In the React app** (running at `http://localhost:3000`), click the **⚙️ API Config** button in the top-right corner

2. **Paste your ngrok URL** (e.g., `https://xxxx.ngrok-free.app`) into the "Backend API URL" field

3. **Click Close** and start optimizing prompts!

## 🎯 How to Use

1. **Enter a prompt** (e.g., `hi tell abt machine learning`)
2. **Select options:**
   - **Domain:** Academic / Marketing / Coding / Chatbot
   - **Tone:** Formal / Simple / Persuasive
   - **Audience:** Beginner / Student / Expert
   - **Length:** Short / Medium / Long
3. **Click "✨ Optimize Prompt"**
4. **View results:**
   - 📊 Prompt Score (quality rating)
   - 🎯 Detected Intent (with confidence & language)
   - 💡 Suggestions for improvement
   - 🔧 Structured Form (logical representation)
   - 🔑 Extracted Keywords
   - 🏷️ Named Entities (via spaCy)
   - ✨ Optimized Prompt (copy to clipboard)
   - 🤖 Final AI Answer

## 🧠 NLP Techniques Used

| Technique | Purpose |
|-----------|---------|
| **TF-IDF Vectorization** | Feature extraction for keyword scoring & intent classification |
| **Logistic Regression** | Classifies prompt intent (informative/creative/analytical/conversational) |
| **spaCy NER** | Extracts named entities (organizations, locations, persons) |
| **POS Tagging** | Identifies parts of speech for better lemmatization |
| **Lemmatization** | Reduces words to base forms (better than stemming) |
| **Stopword Removal** | Filters out common words (the, is, at, etc.) |
| **Slang Normalization** | Maps informal abbreviations to formal words |
| **Prompt Scoring** | Heuristic-based quality scoring system |

## 📊 Intent Classification Examples

| User Prompt | Detected Intent |
|-------------|----------------|
| "what is machine learning" | Informative |
| "write a story about robots" | Creative |
| "compare python and java" | Analytical |
| "chat about technology" | Conversational |

## 🔌 API Endpoints

When the backend is running, you can access:

- **`GET /`** — Health check
- **`GET /docs`** — Interactive Swagger API documentation
- **`POST /optimize`** — Main endpoint to optimize prompts

### Request Body (POST /optimize)
```json
{
  "prompt": "tell abt AI",
  "domain": "academic",
  "tone": "formal",
  "audience": "beginner",
  "length": "short"
}
```

### Response
```json
{
  "score": "3 (Average)",
  "suggestions": "Add more detail...\nSpecify your audience...",
  "intent": "informative (92%) | Language: English",
  "structured": "Explain(Artificial Intelligence)",
  "optimized": "Explain Artificial Intelligence for a beginner...",
  "answer": "Artificial Intelligence is a concept...",
  "keywords": "artificial intelligence",
  "entities": [["AI", "ORG"]]
}
```

## 🚀 Deployment Options

### Option 1: Colab + ngrok (Free, Recommended for Demo)
- **Backend:** Google Colab (free GPU/TPU)
- **Frontend:** React on local machine or deploy to Vercel/Netlify
- **Connection:** ngrok tunnel

### Option 2: Local Development
- **Backend:** Run `colab_backend.py` locally (remove ngrok lines, just run uvicorn)
- **Frontend:** `npm start` on `localhost:3000`
- **Connection:** `http://localhost:8000`

### Option 3: Production Deployment
- **Backend:** Deploy FastAPI to Railway, Render, or AWS
- **Frontend:** Deploy React to Vercel or Netlify
- **Connection:** Update `apiUrl` in frontend to production backend URL

## 📦 Dependencies

### Backend (Python)
- `fastapi` — Web framework
- `uvicorn` — ASGI server
- `pyngrok` — ngrok integration
- `spacy` — NLP (Named Entity Recognition)
- `nltk` — Tokenization, POS tagging, lemmatization
- `scikit-learn` — TF-IDF, Logistic Regression

### Frontend (Node.js)
- `react` — UI library
- `typescript` — Type safety
- Standard Create React App dependencies

## 🐛 Troubleshooting

### "Failed to fetch" error in frontend
- Make sure the backend (Colab) is running
- Verify the ngrok URL in API Config is correct
- Check that CORS is enabled (it is by default in `colab_backend.py`)

### ngrok session expired
- ngrok free tier sessions last ~2 hours
- Restart the Colab cell to get a new URL
- Update the URL in the frontend API Config

### Colab runtime disconnected
- Reconnect and re-run the cell
- The model will reload (takes ~30 seconds)

### Port already in use (local development)
- Change the port in `colab_backend.py` from `8000` to another port
- Update `API_BASE_URL` in `App.tsx` accordingly

## 📝 License

This project is for educational purposes by SOMESH S.

## 👨‍💻 Author

Built as part of the **Computational NLP by SOMESH S** course project.

---

**Tech Stack:** `FastAPI` • `React` • `TypeScript` • `TF-IDF` • `Logistic Regression` • `spaCy` • `NLTK` • `scikit-learn` • `ngrok` • `Google Colab`
