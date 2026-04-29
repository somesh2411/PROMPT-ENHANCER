import { useState } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000'; // Change this to your ngrok URL when using Colab

interface ResultData {
  score: string;
  suggestions: string;
  intent: string;
  structured: string;
  optimized: string;
  answer: string;
  keywords: string;
  entities: [string, string][];
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [domain, setDomain] = useState('academic');
  const [tone, setTone] = useState('formal');
  const [audience, setAudience] = useState('beginner');
  const [length, setLength] = useState('short');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ResultData | null>(null);
  const [error, setError] = useState('');
  const [apiUrl, setApiUrl] = useState(API_BASE_URL);
  const [showApiConfig, setShowApiConfig] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      setError('Please enter a prompt.');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${apiUrl}/optimize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt,
          domain,
          tone,
          audience,
          length,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Something went wrong');
      }

      const data: ResultData = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setPrompt('');
    setDomain('academic');
    setTone('formal');
    setAudience('beginner');
    setLength('short');
    setResult(null);
    setError('');
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>🤖 AI Prompt Optimizer</h1>
          <p>Smart system to enhance prompts using real NLP + ML</p>
          <div className="tech-badges">
            <span>TF-IDF</span>
            <span>Logistic Regression</span>
            <span>spaCy NER</span>
            <span>POS Tagging</span>
            <span>Lemmatization</span>
          </div>
        </div>
        <button className="api-config-btn" onClick={() => setShowApiConfig(!showApiConfig)}>
          ⚙️ API Config
        </button>
      </header>

      {/* API Config Modal */}
      {showApiConfig && (
        <div className="api-config-modal">
          <div className="modal-content">
            <h3>API Configuration</h3>
            <label>
              Backend API URL:
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                placeholder="http://localhost:8000 or your ngrok URL"
              />
            </label>
            <p className="hint">
              For Colab: Replace with your ngrok public URL (e.g., https://xxxx.ngrok-free.app)
            </p>
            <button onClick={() => setShowApiConfig(false)}>Close</button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="main">
        <div className="container">
          {/* Input Section */}
          <form onSubmit={handleSubmit} className="input-section">
            <div className="form-group">
              <label htmlFor="prompt">Enter Prompt</label>
              <textarea
                id="prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="e.g. hi tell abt machine learning / compare python and java"
                rows={4}
              />
            </div>

            <div className="dropdowns">
              <div className="form-group">
                <label htmlFor="domain">Domain</label>
                <select id="domain" value={domain} onChange={(e) => setDomain(e.target.value)}>
                  <option value="academic">Academic</option>
                  <option value="marketing">Marketing</option>
                  <option value="coding">Coding</option>
                  <option value="chatbot">Chatbot</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="tone">Tone</label>
                <select id="tone" value={tone} onChange={(e) => setTone(e.target.value)}>
                  <option value="formal">Formal</option>
                  <option value="simple">Simple</option>
                  <option value="persuasive">Persuasive</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="audience">Audience</label>
                <select id="audience" value={audience} onChange={(e) => setAudience(e.target.value)}>
                  <option value="beginner">Beginner</option>
                  <option value="student">Student</option>
                  <option value="expert">Expert</option>
                </select>
              </div>

              <div className="form-group">
                <label htmlFor="length">Length</label>
                <select id="length" value={length} onChange={(e) => setLength(e.target.value)}>
                  <option value="short">Short</option>
                  <option value="medium">Medium</option>
                  <option value="long">Long</option>
                </select>
              </div>
            </div>

            <div className="buttons">
              <button type="button" className="btn-clear" onClick={handleClear}>
                Clear
              </button>
              <button type="submit" className="btn-submit" disabled={loading}>
                {loading ? 'Optimizing...' : '✨ Optimize Prompt'}
              </button>
            </div>
          </form>

          {/* Error Message */}
          {error && <div className="error-message">⚠️ {error}</div>}

          {/* Loading Spinner */}
          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <p>Processing your prompt with NLP...</p>
            </div>
          )}

          {/* Results Section */}
          {result && !loading && (
            <div className="results">
              <div className="result-grid">
                {/* Prompt Score */}
                <div className="result-card score-card">
                  <h3>📊 Prompt Score</h3>
                  <div className="score-value">
                    <span className="score-number">{result.score.split(' ')[0]}</span>
                    <span className={`score-level ${result.score.toLowerCase().includes('excellent') ? 'excellent' : result.score.toLowerCase().includes('average') ? 'average' : 'poor'}`}>
                      {result.score.split(' ')[1]?.replace(/[()]/g, '')}
                    </span>
                  </div>
                </div>

                {/* Detected Intent */}
                <div className="result-card intent-card">
                  <h3>🎯 Detected Intent</h3>
                  <p className="intent-text">{result.intent.split(' | ')[0]}</p>
                  <p className="language-text">{result.intent.split(' | ')[1]}</p>
                </div>
              </div>

              {/* Suggestions */}
              <div className="result-card suggestions-card">
                <h3>💡 Suggestions</h3>
                <div className="suggestions-list">
                  {result.suggestions.includes('\n')
                    ? result.suggestions.split('\n').map((s, i) => (
                        <p key={i}>• {s}</p>
                      ))
                    : <p>• {result.suggestions}</p>}
                </div>
              </div>

              {/* Structured Form */}
              <div className="result-card structured-card">
                <h3>🔧 Structured Form (spaCy NER + Keywords)</h3>
                <code>{result.structured}</code>
              </div>

              {/* Keywords & Entities */}
              <div className="result-grid">
                <div className="result-card keywords-card">
                  <h3>🔑 Extracted Keywords</h3>
                  <div className="keywords">
                    {result.keywords.split(' ').map((kw, i) => (
                      <span key={i} className="keyword-tag">{kw}</span>
                    ))}
                  </div>
                </div>

                <div className="result-card entities-card">
                  <h3>🏷️ Named Entities</h3>
                  {result.entities.length > 0 ? (
                    <div className="entities">
                      {result.entities.map((ent, i) => (
                        <span key={i} className="entity-tag">
                          {ent[0]} <small>({ent[1]})</small>
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="no-entities">No named entities detected</p>
                  )}
                </div>
              </div>

              {/* Optimized Prompt */}
              <div className="result-card optimized-card">
                <h3>✨ Optimized Prompt</h3>
                <div className="optimized-text">{result.optimized}</div>
                <button
                  className="copy-btn"
                  onClick={() => navigator.clipboard.writeText(result.optimized)}
                >
                  📋 Copy to Clipboard
                </button>
              </div>

              {/* AI Answer */}
              <div className="result-card answer-card">
                <h3>🤖 Final AI Answer</h3>
                <p>{result.answer}</p>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>AI Prompt Optimizer — Built with FastAPI + React + NLP</p>
      </footer>
    </div>
  );
}

export default App;
