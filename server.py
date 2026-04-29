import re
import nltk
import spacy
import nest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

# =============================================================================
# MODULE 1 - SLANG NORMALIZER
# =============================================================================
SLANG_MAP = {
    "u": "you", "abt": "about", "plz": "please",
    "pls": "please", "hw": "how", "wht": "what",
    "r": "are", "ur": "your", "cn": "can",
    "con": "can", "tell": "explain", "say": "explain",
    "giv": "give", "diff": "difference", "b/w": "between",
    "enna": "what", "epdi": "how", "sollu": "explain",
    "nw": "now", "bcz": "because", "coz": "because",
    "info": "information", "imp": "important",
}

def normalize_slang(text):
    text = text.lower().strip()
    words = text.split()
    words = [SLANG_MAP.get(w, w) for w in words]
    text = " ".join(words)
    text = re.sub(r'\b(hi|hello|please|can you|tell me)\b', '', text)
    return text.strip()

# =============================================================================
# MODULE 2 - NLP PREPROCESSING PIPELINE
# =============================================================================
def preprocess_pipeline(text):
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())
    words = word_tokenize(text)
    tagged = pos_tag(words)
    pos_map = {'N': 'n', 'V': 'v', 'R': 'r', 'J': 'a'}
    lemmas = [
        lemmatizer.lemmatize(w, pos_map.get(p[0], 'n'))
        for w, p in tagged
        if w not in STOPWORDS and w.isalpha() and len(w) > 1
    ]
    return " ".join(lemmas)

# =============================================================================
# MODULE 3 - KEYWORD EXTRACTION (TF-IDF based)
# =============================================================================
_CORPUS = [
    "explain artificial intelligence machine learning",
    "describe deep learning neural networks",
    "compare python java programming languages",
    "write story creative poem",
    "analyze climate change environment",
    "explain blockchain cryptocurrency",
    "describe data science statistics",
    "explain internet networking protocols",
    "analyze pros cons renewable energy",
    "explain quantum computing physics",
    "describe cybersecurity hacking prevention",
    "explain diabetes health disease symptoms",
    "compare supervised unsupervised learning algorithms",
]

_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
_tfidf.fit(_CORPUS)

def extract_keywords(text, top_n=4):
    cleaned = preprocess_pipeline(text)
    if not cleaned.strip():
        return text
    try:
        vec = _tfidf.transform([cleaned])
        scores = zip(_tfidf.get_feature_names_out(), vec.toarray()[0])
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        keywords = [w for w, s in ranked if s > 0][:top_n]
    except Exception:
        keywords = []
    if not keywords:
        tagged = pos_tag(word_tokenize(cleaned))
        keywords = [w for w, p in tagged if p.startswith(('NN', 'JJ'))][:top_n]
    if not keywords:
        keywords = cleaned.split()[:top_n]
    keywords = [w for w in keywords if w not in ["explain", "tell", "describe", "about"]]
    keywords = list(dict.fromkeys(keywords))
    final_keywords = []
    for word in keywords:
        if " " in word:
            final_keywords.append(word)
    for word in keywords:
        if " " not in word:
            if not any(word in phrase for phrase in final_keywords):
                final_keywords.append(word)
    if not final_keywords:
        return cleaned
    topic = " ".join(final_keywords[:min(3, len(final_keywords))])
    return topic

# =============================================================================
# MODULE 4 - NAMED ENTITY RECOGNITION (spaCy)
# =============================================================================
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# =============================================================================
# MODULE 5 - INTENT CLASSIFIER (Logistic Regression + TF-IDF)
# =============================================================================
_INTENT_CORPUS = [
    ("what is machine learning", "informative"),
    ("explain artificial intelligence", "informative"),
    ("how does blockchain work", "informative"),
    ("describe neural networks", "informative"),
    ("tell me about climate change", "informative"),
    ("define deep learning", "informative"),
    ("what are neural networks", "informative"),
    ("explain quantum computing", "informative"),
    ("how does internet work", "informative"),
    ("what is data science", "informative"),
    ("write a story about robots", "creative"),
    ("create a poem on nature", "creative"),
    ("generate a short story", "creative"),
    ("compose a blog post about ai", "creative"),
    ("write an essay on climate", "creative"),
    ("draft a speech about technology", "creative"),
    ("compare python and java", "analytical"),
    ("analyze pros and cons of ai", "analytical"),
    ("difference between ml and dl", "analytical"),
    ("evaluate renewable energy sources", "analytical"),
    ("assess the impact of social media", "analytical"),
    ("contrast supervised unsupervised learning", "analytical"),
    ("tell me something interesting", "conversational"),
    ("chat about technology", "conversational"),
    ("discuss movies", "conversational"),
    ("talk to me about life", "conversational"),
    ("share your thoughts on music", "conversational"),
]

_ix, _iy = zip(*_INTENT_CORPUS)
_intent_vec = TfidfVectorizer(ngram_range=(1, 2))
_Xintent = _intent_vec.fit_transform(_ix)
_intent_model = LogisticRegression(max_iter=500)
_intent_model.fit(_Xintent, _iy)

def detect_intent(text):
    vec = _intent_vec.transform([text.lower()])
    pred = _intent_model.predict(vec)[0]
    prob = _intent_model.predict_proba(vec).max()
    return pred, round(float(prob), 2)

# =============================================================================
# MODULE 6 - STRUCTURED FORM (Skolemization)
# =============================================================================
def skolemize(text, intent):
    entities = extract_entities(text)
    keywords = extract_keywords(text, top_n=3)
    if entities:
        topic = ", ".join(f"{e[0]}({e[1]})" for e in entities[:2])
    else:
        topic = keywords.title() if keywords else "General Topic"
    action_map = {
        "informative": "Explain", "creative": "Create",
        "analytical": "Analyze", "conversational": "Discuss",
    }
    action = action_map.get(intent, "Explain")
    return f"{action}({topic})"

# =============================================================================
# MODULE 7 - PROMPT QUALITY SCORER
# =============================================================================
def score_prompt(prompt):
    score = 0
    p = prompt.lower()
    word_count = len(prompt.split())
    if word_count > 8:
        score += 3
    elif word_count > 4:
        score += 1
    if any(w in p for w in ["explain", "write", "compare", "analyze", "describe", "define", "create"]):
        score += 2
    if any(w in p for w in ["beginner", "student", "expert", "children", "professional", "for"]):
        score += 2
    if any(w in p for w in ["formal", "simple", "persuasive", "friendly", "tone", "style"]):
        score += 2
    if any(w in p for w in ["words", "short", "brief", "paragraph", "long", "detailed"]):
        score += 1
    if score <= 3:
        level = "Poor"
    elif score <= 6:
        level = "Average"
    else:
        level = "Excellent"
    return score, level

# =============================================================================
# MODULE 8 - SUGGESTIONS ENGINE
# =============================================================================
def suggest_improvements(prompt):
    suggestions = []
    p = prompt.lower()
    if len(prompt.split()) < 5:
        suggestions.append("Add more detail - your prompt is too short")
    if not any(w in p for w in ["explain", "write", "compare", "describe", "define", "analyze", "create"]):
        suggestions.append("Add a clear purpose: explain / compare / write / analyze")
    if not any(w in p for w in ["beginner", "student", "expert", "child", "professional", "for"]):
        suggestions.append("Specify your audience (e.g., 'for beginners')")
    if not any(w in p for w in ["formal", "simple", "friendly", "persuasive", "tone"]):
        suggestions.append("Add a tone (formal / simple / persuasive)")
    if not any(w in p for w in ["words", "short", "brief", "long", "detailed", "paragraph"]):
        suggestions.append("Add a length constraint (e.g., 'in 100 words')")
    return "\n".join(suggestions) if suggestions else "Your prompt is well-structured!"

# =============================================================================
# MODULE 9 - PROMPT OPTIMIZER ENGINE
# =============================================================================
def optimize_prompt(keywords, domain, tone, audience, length, intent):
    topic = keywords.title().strip()
    action_map = {
        "informative": "Explain", "creative": "Write about",
        "analytical": "Analyze", "conversational": "Discuss"
    }
    action = action_map.get(intent, "Explain")
    domain_context = {
        "academic": "Use clear, structured academic explanation",
        "marketing": "Focus on persuasive and engaging content",
        "coding": "Include technical explanation with examples",
        "chatbot": "Keep it conversational and easy to understand"
    }
    domain_text = domain_context.get(domain, "")
    audience_map = {
        "beginner": "a beginner with no prior knowledge",
        "student": "a student with basic understanding",
        "expert": "an experienced professional"
    }
    audience_str = audience_map.get(audience, audience)
    length_map = {
        "short": "Keep it under 150 words.",
        "medium": "Explain in 250-400 words.",
        "long": "Give a detailed explanation with examples."
    }
    length_str = length_map.get(length, "")
    prompt = (
        f"{action} {topic} for {audience_str} in a {tone} tone. "
        f"{domain_text}. "
        f"Start with a simple definition, then explain step-by-step. "
        f"Include real-world examples and practical applications. "
        f"Ensure clarity and good structure. "
        f"{length_str}"
    )
    return prompt.strip()

def detect_language(text):
    if any(w in text.lower() for w in ["enna", "epdi", "sollu"]):
        return "Tamil"
    elif any(w in text.lower() for w in ["kya", "kaise", "hai"]):
        return "Hindi"
    return "English"

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(title="AI Prompt Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    domain: str = "academic"
    tone: str = "formal"
    audience: str = "beginner"
    length: str = "short"

class PromptResponse(BaseModel):
    score: str
    suggestions: str
    intent: str
    structured: str
    optimized: str
    answer: str
    keywords: str
    entities: list

@app.get("/")
def root():
    return {"message": "AI Prompt Optimizer API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/optimize", response_model=PromptResponse)
def optimize(req: PromptRequest):
    if not req.prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"detail": "Please enter a prompt."}
        )

    normalized = normalize_slang(req.prompt)
    language = detect_language(req.prompt)
    intent, confidence = detect_intent(normalized)
    keywords = extract_keywords(normalized, top_n=4)
    entities = extract_entities(normalized)
    structured = skolemize(normalized, intent)
    score, level = score_prompt(req.prompt)
    suggestions = suggest_improvements(req.prompt)
    optimized = optimize_prompt(keywords, req.domain, req.tone, req.audience, req.length, intent)

    answer = (
        f"{keywords.title()} is a concept in computer science where systems learn from data "
        f"and improve over time without being explicitly programmed. "
        f"It works by identifying patterns and making predictions or decisions. "
        f"For example, recommendation systems like Netflix or Amazon use it. "
        f"Overall, it is widely used in real-world applications like healthcare, finance, and automation."
    )

    intent_display = f"{intent} ({confidence:.0%}) | Language: {language}"

    return PromptResponse(
        score=f"{score} ({level})",
        suggestions=suggestions,
        intent=intent_display,
        structured=structured,
        optimized=optimized,
        answer=answer,
        keywords=keywords,
        entities=entities
    )

# =============================================================================
# RUN SERVER (Local - no ngrok needed)
# =============================================================================
print(f"\n{'='*60}")
print(f"  Backend API is live at: http://localhost:8000")
print(f"  API Docs: http://localhost:8000/docs")
print(f"  Frontend should use: http://localhost:8000")
print(f"{'='*60}\n")

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)
