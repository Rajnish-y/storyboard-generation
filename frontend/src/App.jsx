import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import HeroAtmosphere from './HeroAtmosphere.jsx';
import {
  Activity, AlertCircle, ArrowDownToLine, ArrowUpRight, Check, CheckCircle2,
  ChevronDown, CircleHelp, Clapperboard, Cloud, Cpu, FileImage, Film, Gauge,
  ImagePlus, Layers3, LoaderCircle, LockKeyhole, Play, RefreshCw, Settings2,
  ShieldCheck, Sparkles, Upload, WandSparkles, Zap, Users,
} from 'lucide-react';

const DEFAULT_API = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const DEFAULT_PROMPT = 'a young man sprinting at full speed through a rain-soaked narrow alley, mid-stride, one leg extended forward and the other behind, arms swinging naturally, dynamic running pose';
const STORY_PROMPTS = [
  'photorealistic wide shot, same character standing calmly in a quiet library, soft afternoon light, full scene visible',
  'photorealistic wide shot, same character walking through a crowded city market, visible stalls and pedestrians, daylight',
  'photorealistic wide shot, same character sprinting through a rain-soaked narrow alley, mid-stride, arms swinging, wet pavement, night lighting',
  'photorealistic wide shot, same character jumping over a fallen barrier while escaping, full body airborne, dynamic pose',
];
const MODES = [
  { id: 'fixed', name: 'Fixed baseline', scale: '0.50', note: 'Constant reference strength', icon: Layers3 },
  { id: 'adaptive_only', name: 'Adaptive-only', scale: 'SACS', note: 'Scene-aware initial scale', icon: Gauge },
  { id: 'proposed', name: 'Proposed method', scale: 'SACS + DINO', note: 'Adaptive feedback + retries', icon: WandSparkles },
];


const PAGE_META = {
  studio: {
    kicker: 'THE CREATIVE WORKSPACE',
    title: 'Direct the next frame.',
    accent: 'Make every scene count.',
    description: 'Anchor a character, describe the action, and compare fixed, adaptive, and feedback-driven generation.',
    cta: 'Open generation controls',
    icon: 'studio',
  },
  storyboard: {
    kicker: 'SEQUENCE DESIGN',
    title: 'A story, frame by frame.',
    accent: 'One reference. Many scenes.',
    description: 'Build a multi-scene sequence, review each generated frame, and export the assembled storyboard.',
    cta: 'Go to scene sequence',
    icon: 'storyboard',
  },
  research: {
    kicker: 'METHOD & EVIDENCE',
    title: 'Consistency, measured.',
    accent: 'Trade-offs made visible.',
    description: 'Explore SACS, DINO-guided correction, the three-arm ablation, and the measured identity-versus-alignment trade-off.',
    cta: 'Try the proposed method',
    icon: 'research',
  },
  gallery: {
    kicker: 'THE SCENE ARCHIVE',
    title: 'Same character.',
    accent: 'Different worlds.',
    description: 'Browse original project scene studies, then open any visual direction in the Studio to continue creating.',
    cta: 'Create a new scene',
    icon: 'gallery',
  },
};

function cx(...values) { return values.filter(Boolean).join(' '); }
function metric(value) { return typeof value === 'number' ? value.toFixed(4) : '—'; }
function prettyMode(mode) { return MODES.find((item) => item.id === mode)?.name || mode; }

async function parseResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === 'string' ? payload.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload;
}

function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const currentPage = ['studio', 'storyboard', 'research', 'gallery'].includes(location.pathname.replace(/^\/+|\/+$/g, ''))
    ? location.pathname.replace(/^\/+|\/+$/g, '')
    : 'home';
  const pageMeta = PAGE_META[currentPage];
  const [apiBase, setApiBase] = useState(() => localStorage.getItem('storyframe-api') || DEFAULT_API);
  const [apiToken, setApiToken] = useState('');
  const [health, setHealth] = useState({ status: 'checking', model_loaded: false });
  const [connectionFeedback, setConnectionFeedback] = useState({ kind: 'idle', message: 'Use Test connection to verify the configured API address.' });
  const [view, setView] = useState('single');
  const [mode, setMode] = useState('proposed');
  const [referenceFile, setReferenceFile] = useState(null);
  const [referencePreview, setReferencePreview] = useState('');
  const [characterAFile, setCharacterAFile] = useState(null);
  const [characterAPreview, setCharacterAPreview] = useState('');
  const [characterBFile, setCharacterBFile] = useState(null);
  const [characterBPreview, setCharacterBPreview] = useState('');
  const [multiPrompt, setMultiPrompt] = useState('Wide shot. Two friends at a village station at sunset, standing side by side, equal size.');
  const [multiScale, setMultiScale] = useState(0.75);
  const [multiResult, setMultiResult] = useState(null);
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [scenesText, setScenesText] = useState(STORY_PROMPTS.join('\n'));
  const [premise, setPremise] = useState('A young traveler explores a city, escapes through a storm, and finds refuge at the coast.');
  const [sceneCount, setSceneCount] = useState(4);
  const [threshold, setThreshold] = useState(0.54);
  const [maxRetries, setMaxRetries] = useState(2);
  const [steps, setSteps] = useState(30);
  const [guidance, setGuidance] = useState(7.5);
  const [seed, setSeed] = useState(42);
  const [showSettings, setShowSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState('Ready when your GPU backend is connected.');
  const [error, setError] = useState('');
  const [singleResult, setSingleResult] = useState(null);
  const [storyResults, setStoryResults] = useState([]);
  const [storySheet, setStorySheet] = useState(null);
  const [storyPdf, setStoryPdf] = useState(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [lastDuration, setLastDuration] = useState(0);
  const fileInputRef = useRef(null);
  const characterAInputRef = useRef(null);
  const characterBInputRef = useRef(null);
  const apiBaseNormalized = useMemo(() => apiBase.trim().replace(/\/$/, ''), [apiBase]);
  const headerObject = useMemo(() => apiToken.trim() ? { 'X-Demo-Token': apiToken.trim() } : {}, [apiToken]);

  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  };

  const chooseGalleryPrompt = (nextPrompt) => {
    setPrompt(nextPrompt);
    setView('single');
    navigate('/studio');
  };

  const refreshHealth = useCallback(async () => {
    if (!apiBaseNormalized) {
      const message = 'Enter an API base URL before testing the connection.';
      setHealth({ status: 'offline', model_loaded: false });
      setConnectionFeedback({ kind: 'error', message });
      setStatusText(message);
      return;
    }

    setHealth((old) => ({ ...old, status: 'checking' }));
    setConnectionFeedback({ kind: 'checking', message: 'Testing the API health endpoint…' });

    try {
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), 25000);
      let response;
      try {
        response = await fetch(`${apiBaseNormalized}/api/health`, {
          headers: headerObject,
          cache: 'no-store',
          signal: controller.signal,
        });
      } finally {
        window.clearTimeout(timeoutId);
      }

      const result = await parseResponse(response);
      setHealth(result);
      localStorage.setItem('storyframe-api', apiBaseNormalized);

      if (result.model_loaded) {
        const message = `Connection successful — ${result.gpu_name || result.device || 'GPU'} backend is ready.`;
        setConnectionFeedback({ kind: 'success', message });
        setStatusText(message);
      } else {
        const message = result.model_error || 'The API is reachable, but the generation model is not ready yet.';
        setConnectionFeedback({ kind: 'warning', message });
        setStatusText(message);
      }
    } catch (err) {
      const detail = err?.name === 'AbortError'
        ? 'The health check timed out after 25 seconds.'
        : (err?.message || 'Unknown network error.');
      const message = `Connection failed. Check the API URL, active Colab tunnel, and backend CORS settings. ${detail}`;
      setHealth({ status: 'offline', model_loaded: false });
      setConnectionFeedback({ kind: 'error', message });
      setStatusText(message);
    }
  }, [apiBaseNormalized, headerObject]);

  useEffect(() => { refreshHealth(); }, [refreshHealth]);
  useEffect(() => {
    if (currentPage === 'storyboard') setView('storyboard');
    if (currentPage === 'studio') setView('single');
    if (currentPage !== 'home') {
      window.scrollTo({
        top: 0,
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
      });
    }
  }, [currentPage]);
  useEffect(() => {
    const title = currentPage === 'home' ? 'StoryFrame Lab — Character-Consistent Visual Generation' : `StoryFrame Lab — ${currentPage.charAt(0).toUpperCase()}${currentPage.slice(1)}`;
    document.title = title;
  }, [currentPage]);

  useEffect(() => () => { if (referencePreview) URL.revokeObjectURL(referencePreview); }, [referencePreview]);
  useEffect(() => () => { if (characterAPreview) URL.revokeObjectURL(characterAPreview); }, [characterAPreview]);
  useEffect(() => () => { if (characterBPreview) URL.revokeObjectURL(characterBPreview); }, [characterBPreview]);

  const authFetch = async (url, options = {}) => fetch(`${apiBaseNormalized}${url}`, {
    ...options,
    headers: { ...(options.headers || {}), ...headerObject },
  });

  const handleReference = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Choose a PNG, JPG, or WebP image for the character reference.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Reference image must be 10 MB or smaller.');
      return;
    }
    if (referencePreview) URL.revokeObjectURL(referencePreview);
    setReferenceFile(file);
    setReferencePreview(URL.createObjectURL(file));
    setError('');
  };

  const handleMultiReference = (file, character) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Choose a PNG, JPG, or WebP image for each character reference.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Each character reference must be 10 MB or smaller.');
      return;
    }
    const isA = character === 'a';
    const oldPreview = isA ? characterAPreview : characterBPreview;
    if (oldPreview) URL.revokeObjectURL(oldPreview);
    const nextPreview = URL.createObjectURL(file);
    if (isA) {
      setCharacterAFile(file);
      setCharacterAPreview(nextPreview);
    } else {
      setCharacterBFile(file);
      setCharacterBPreview(nextPreview);
    }
    setError('');
    setMultiResult(null);
  };

  const appendScenePrompts = (prompts) => {
    const clean = prompts.map((item) => String(item || '').trim()).filter(Boolean);
    if (clean.length) setScenesText(clean.join('\n'));
    setView('storyboard');
    navigate('/storyboard');
  };

  const handleGenerate = async (event) => {
    event?.preventDefault();
    setError('');
    setSingleResult(null);
    if (!referenceFile) { setError('Upload a character reference image first.'); return; }
    if (!prompt.trim()) { setError('Enter a scene prompt first.'); return; }
    setBusy(true);
    setLastDuration(0);
    const started = performance.now();
    setStatusText(mode === 'proposed' ? 'Generating, scoring, and correcting if needed…' : 'Generating scene and calculating metrics…');
    try {
      const form = new FormData();
      form.append('reference_image', referenceFile);
      form.append('prompt', prompt.trim());
      form.append('mode', mode);
      form.append('threshold', String(threshold));
      form.append('max_retries', String(maxRetries));
      form.append('steps', String(steps));
      form.append('guidance', String(guidance));
      form.append('seed', String(seed));
      const response = await authFetch('/api/generate', { method: 'POST', body: form });
      const result = await parseResponse(response);
      setSingleResult(result);
      setLastDuration((performance.now() - started) / 1000);
      setStatusText('Scene generated. Review the image, metrics, and attempt trace.');
    } catch (err) {
      setError(err.message || 'Generation failed.');
      setStatusText('Generation could not be completed.');
    } finally {
      setBusy(false);
    }
  };

  const handleGenerateMulti = async (event) => {
    event?.preventDefault();
    setError('');
    setMultiResult(null);
    if (!characterAFile || !characterBFile) {
      setError('Upload both Character A and Character B reference images first.');
      return;
    }
    if (!multiPrompt.trim()) {
      setError('Enter a short scene prompt.');
      return;
    }
    if (multiPrompt.trim().length > 1200) {
      setError('Prompt must be 1,200 characters or fewer.');
      return;
    }
    if (characterAFile.name === characterBFile.name && characterAFile.size === characterBFile.size && characterAFile.lastModified === characterBFile.lastModified) {
      setError('Choose two distinct character reference images.');
      return;
    }

    setBusy(true);
    setLastDuration(0);
    setStatusText('Generating a two-character scene with masked reference conditioning…');
    const started = performance.now();
    try {
      const form = new FormData();
      form.append('character_a', characterAFile);
      form.append('character_b', characterBFile);
      form.append('prompt', multiPrompt.trim());
      form.append('scale', String(multiScale));
      form.append('steps', String(steps));
      form.append('guidance', String(guidance));
      form.append('seed', String(seed));

      const response = await authFetch('/api/generate-multi', { method: 'POST', body: form });
      const result = await parseResponse(response);
      setMultiResult(result);
      setLastDuration((performance.now() - started) / 1000);
      setStatusText('Multi-character scene generated. Review the composition and both character appearances.');
    } catch (err) {
      setError(err.message || 'Multi-character generation failed.');
      setStatusText('Multi-character generation could not be completed.');
    } finally {
      setBusy(false);
    }
  };

  const handleStoryboard = async (event) => {
    event?.preventDefault();
    setError('');
    setStoryResults([]);
    setStorySheet(null);
    setStoryPdf(null);
    if (!referenceFile) { setError('Upload a character reference image first.'); return; }
    const lines = scenesText.split('\n').map((line) => line.trim()).filter(Boolean);
    if (!lines.length) { setError('Enter at least one scene prompt.'); return; }
    if (lines.length > 10) { setError('The demo supports up to 10 scenes per generation request.'); return; }
    setBusy(true);
    const started = performance.now();
    setStatusText(`Generating ${lines.length} storyboard scenes. This can take several minutes on a T4…`);
    try {
      const form = new FormData();
      form.append('reference_image', referenceFile);
      form.append('scenes_text', lines.join('\n'));
      form.append('mode', mode);
      form.append('threshold', String(threshold));
      form.append('max_retries', String(maxRetries));
      form.append('steps', String(steps));
      form.append('guidance', String(guidance));
      form.append('seed', String(seed));
      const response = await authFetch('/api/storyboard', { method: 'POST', body: form });
      const result = await parseResponse(response);
      setStoryResults(result.results || []);
      setStorySheet(result.storyboard_sheet_url || null);
      setStoryPdf(result.storyboard_pdf_url || null);
      setLastDuration((performance.now() - started) / 1000);
      setStatusText(`Completed ${result.generated_scenes} of ${result.requested_scenes} scenes.`);
    } catch (err) {
      setError(err.message || 'Storyboard generation failed.');
      setStatusText('Storyboard generation could not be completed.');
    } finally {
      setBusy(false);
    }
  };

  const handlePlan = async () => {
    setError('');
    setPlanLoading(true);
    try {
      const form = new FormData();
      form.append('premise', premise);
      form.append('num_scenes', String(sceneCount));
      const response = await authFetch('/api/plan', { method: 'POST', body: form });
      const result = await parseResponse(response);
      setScenesText((result.scenes || []).map((scene) => scene.prompt).join('\n'));
      setView('storyboard');
      navigate('/storyboard');
      setStatusText(`Scene plan created: ${(result.scenes || []).length} scenes. Review the prompts before generation.`);
    } catch (err) {
      setError(err.message || 'Scene planning failed. You can enter prompts manually.');
    } finally {
      setPlanLoading(false);
    }
  };

  const downloadImage = async (imageUrl, filename) => {
    try {
      const response = await authFetch(imageUrl);
      if (!response.ok) throw new Error('Download failed. The backend runtime may have reset.');
      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = objectUrl;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      setError(err.message || 'Could not download image.');
    }
  };

  const bestResult = singleResult;
  const selectedMode = MODES.find((item) => item.id === mode);

  return (
    <div className={cx('app-shell', `page-${currentPage}`)}>
      <header className="topbar">
        <Link to="/" className="brand-lockup brand-button" aria-label="StoryFrame home">
          <div className="brand-mark"><Clapperboard size={21} strokeWidth={1.8} /></div>
          <div>
            <div className="brand-name">STORYFRAME <span>LAB</span></div>
            <div className="brand-subtitle">Visual stories. Consistent characters.</div>
          </div>
        </Link>
        <nav className="top-nav" aria-label="Main navigation">
          <Link to="/" className={cx('nav-link', currentPage === 'home' && 'active')}>Home</Link>
          <Link to="/studio" className={cx('nav-link', currentPage === 'studio' && 'active')}>Studio</Link>
          <Link to="/storyboard" className={cx('nav-link', currentPage === 'storyboard' && 'active')}>Storyboard</Link>
          <Link to="/research" className={cx('nav-link', currentPage === 'research' && 'active')}>Research</Link>
          <Link to="/gallery" className={cx('nav-link', currentPage === 'gallery' && 'active')}>Gallery</Link>
        </nav>
        <div className="topbar-right">
          <div className={cx('connection-pill', health.model_loaded ? 'connected' : health.status === 'checking' ? 'checking' : 'disconnected')}>
            <span className="status-dot" />
            {health.model_loaded ? `GPU ready · ${health.gpu_name || health.device}` : health.status === 'checking' ? 'Checking GPU…' : 'GPU disconnected'}
          </div>
          <button type="button" className="icon-button" title="Backend settings" onClick={() => setShowSettings((value) => !value)}><Settings2 size={18} /></button>
          <button type="button" className="top-launch" onClick={() => navigate('/studio')}>Launch studio <ArrowUpRight size={15} /></button>
        </div>
      </header>

      {showSettings && <section className="settings-panel surface-card">
        <div className="settings-title"><Settings2 size={16} /> Backend connection <span className="muted-small">For local or Colab GPU API</span></div>
        <div className="settings-grid">
          <label>API base URL<input value={apiBase} onChange={(event) => setApiBase(event.target.value)} placeholder="https://your-colab-tunnel.trycloudflare.com" /></label>
          <label>Demo token (optional)<input value={apiToken} onChange={(event) => setApiToken(event.target.value)} placeholder="Enter token if backend authentication is enabled" type="password" /></label>
        </div>
        <div className="settings-footer"><span>Never put model/API secrets in the frontend source. Token is kept only in this tab's state.</span><button type="button" className="button button-secondary button-small" onClick={refreshHealth} disabled={connectionFeedback.kind === 'checking'}>{connectionFeedback.kind === 'checking' ? <LoaderCircle className="spin" size={14} /> : <RefreshCw size={14} />}{connectionFeedback.kind === 'checking' ? ' Testing connection…' : ' Test connection'}</button></div>
        <div
          role="status"
          aria-live="polite"
          className={`connection-feedback connection-feedback-${connectionFeedback.kind}`}
          style={{
            display: 'flex', alignItems: 'flex-start', gap: '9px', marginTop: '12px',
            padding: '11px 13px', borderRadius: '10px', fontSize: '13px', lineHeight: 1.45,
            color: connectionFeedback.kind === 'success' ? '#83e2bc' : connectionFeedback.kind === 'error' ? '#ffaaaa' : connectionFeedback.kind === 'warning' ? '#ffd18b' : '#a8c6f2',
            background: connectionFeedback.kind === 'success' ? 'rgba(16,185,129,0.08)' : connectionFeedback.kind === 'error' ? 'rgba(244,63,94,0.08)' : connectionFeedback.kind === 'warning' ? 'rgba(245,158,11,0.08)' : 'rgba(59,130,246,0.07)',
            border: `1px solid ${connectionFeedback.kind === 'success' ? 'rgba(16,185,129,0.28)' : connectionFeedback.kind === 'error' ? 'rgba(244,63,94,0.28)' : connectionFeedback.kind === 'warning' ? 'rgba(245,158,11,0.25)' : 'rgba(59,130,246,0.22)'}`,
          }}
        >
          {connectionFeedback.kind === 'success' ? <CheckCircle2 size={16} /> : connectionFeedback.kind === 'error' ? <AlertCircle size={16} /> : connectionFeedback.kind === 'checking' ? <LoaderCircle className="spin" size={16} /> : <CircleHelp size={16} />}
          <span>{connectionFeedback.message}</span>
        </div>
      </section>}

      <main className={cx('main-content', `main-${currentPage}`)}>
        {currentPage !== 'home' && pageMeta && (
          <section className={cx('route-hero', `route-hero-${currentPage}`)}>
            <div className="route-hero-backdrop" />
            <div className="route-hero-art" aria-hidden="true"><HeroAtmosphere /></div>
            <div className="route-hero-content">
              <div className="hero-kicker"><span className="live-kicker-dot" /> STORYFRAME LAB <span className="kicker-divider">/</span> {pageMeta.kicker}</div>
              <h1>{pageMeta.title}<br /><span>{pageMeta.accent}</span></h1>
              <p className="hero-copy">{pageMeta.description}</p>
              <div className="route-hero-actions">
                <button type="button" className="hero-cta" onClick={() => {
                  if (currentPage === 'storyboard') scrollToSection('workspace-tabs');
                  else if (currentPage === 'research') navigate('/studio');
                  else if (currentPage === 'gallery') navigate('/studio');
                  else scrollToSection('workspace');
                }}>{pageMeta.cta} <ArrowUpRight size={16} /></button>
                {currentPage !== 'research' && <span className="route-hero-note">SDXL × IP-Adapter · SACS · DINO feedback</span>}
              </div>
            </div>
            <div className="route-hero-index"><span>SCENE-ADAPTIVE CONTROL</span><i /><span>VISUAL STORY SYSTEM</span></div>
          </section>
        )}
        <section className="cinema-hero" id="home">
          <div className="hero-backdrop" />
          {currentPage === 'home' && <HeroAtmosphere />}
          <div className="hero-noise" />
          <div className="hero-artwork" aria-hidden="true">
            <div className="hero-main-image"><img src="/hero-character.jpg" alt="" /></div>
            <div className="hero-art-vignette" />
            <div className="hero-frame hero-frame-top"><img src="/frame-escape.jpg" alt="" /><div><span>FRAME 04</span><strong>Motion study</strong></div></div>
            <div className="hero-frame hero-frame-bottom"><img src="/frame-campfire.jpg" alt="" /><div><span>FRAME 05</span><strong>Quiet aftermath</strong></div></div>
            <div className="hero-art-index"><span>CHARACTER 01</span><i /> <span>10 SCENE STUDIES</span></div>
          </div>
          <div className="hero-content">
            <div className="hero-kicker"><span className="live-kicker-dot" /> AI STORYBOARD RESEARCH STUDIO <span className="kicker-divider">/</span> SDXL × IP-ADAPTER</div>
            <h1>One character.<br /><span>Infinite stories.</span></h1>
            <p className="hero-copy">A cinematic workspace for exploring character-consistent visual generation, scene-adaptive conditioning, and DINO-guided self-correction.</p>
            <div className="hero-actions">
              <button type="button" className="hero-cta" onClick={() => navigate('/studio')}>Enter the studio <ArrowUpRight size={17} /></button>
              <button type="button" className="hero-secondary" onClick={() => navigate('/research')}><Play size={14} fill="currentColor" /> Explore the method</button>
            </div>
            <div className="hero-proof-row">
              <div><span className="proof-label">DINO LIFT</span><strong>+15.2%</strong><small>vs fixed baseline</small></div>
              <div className="proof-divider" />
              <div><span className="proof-label">EVALUATION</span><strong>10 scenes</strong><small>3 controlled methods</small></div>
              <div className="proof-divider" />
              <div><span className="proof-label">CONTROL LOOP</span><strong>Up to 2</strong><small>additional retries</small></div>
            </div>
          </div>
          <div className="hero-bottom-note"><span className="hero-bottom-line" /> <span>RESEARCH PROTOTYPE</span><span className="hero-bottom-small">NOT A GUARANTEE OF IDENTITY</span></div>
        </section>

        {currentPage === 'home' && (
          <section className="home-highlights" aria-label="StoryFrame features">
            <button type="button" className="home-feature" onClick={() => navigate('/studio')}><span className="home-feature-index">01</span><strong>Create a scene</strong><small>Reference-conditioned generation</small><ArrowUpRight size={16} /></button>
            <button type="button" className="home-feature" onClick={() => navigate('/storyboard')}><span className="home-feature-index">02</span><strong>Build a storyboard</strong><small>Turn prompts into a visual sequence</small><ArrowUpRight size={16} /></button>
            <button type="button" className="home-feature" onClick={() => navigate('/research')}><span className="home-feature-index">03</span><strong>Explore the method</strong><small>SACS, DINO feedback, and results</small><ArrowUpRight size={16} /></button>
          </section>
        )}
        <section className="scene-gallery" id="gallery">
          <div className="gallery-heading">
            <div><div className="eyebrow"><span className="eyebrow-line" /> THE SCENE ARCHIVE</div><h2>Same character. <em>Different worlds.</em></h2><p>Explore the visual range of the reference-conditioned pipeline.</p></div>
            <button type="button" className="gallery-all-link" onClick={() => { setView('storyboard'); navigate('/storyboard'); }}>Open storyboard workspace <ArrowUpRight size={15} /></button>
          </div>
          <div className="poster-rail">
            <button type="button" className="poster-card poster-library" onClick={() => chooseGalleryPrompt('photorealistic wide shot, same character standing calmly in a quiet library, soft afternoon light, full scene visible')}><img src="/frame-library.jpg" alt="Character in a quiet library" /><span className="poster-shade" /><span className="poster-number">01 / STATIC</span><span className="poster-title">The Archive</span><span className="poster-hint">Stillness & identity <ArrowUpRight size={13} /></span></button>
            <button type="button" className="poster-card poster-rain" onClick={() => chooseGalleryPrompt(DEFAULT_PROMPT)}><img src="/frame-rain-alley.jpg" alt="Character in a rainy alley" /><span className="poster-shade" /><span className="poster-number">03 / DYNAMIC</span><span className="poster-title">After the Rain</span><span className="poster-hint">Movement & environment <ArrowUpRight size={13} /></span></button>
            <button type="button" className="poster-card poster-fire" onClick={() => chooseGalleryPrompt('photorealistic cinematic scene, same character sitting beside a campfire at night, reflective expression, warm firelight, wide shot')}><img src="/frame-campfire.jpg" alt="Character beside a campfire" /><span className="poster-shade" /><span className="poster-number">05 / MOOD</span><span className="poster-title">The Quiet After</span><span className="poster-hint">Emotion & atmosphere <ArrowUpRight size={13} /></span></button>
            <button type="button" className="poster-card poster-coast" onClick={() => chooseGalleryPrompt('cinematic wide shot, same character walking alone along a beach at sunset, wind moving jacket, reflective mood')}><img src="/frame-coast.jpg" alt="Character at the coast at sunset" /><span className="poster-shade" /><span className="poster-number">08 / ENVIRONMENT</span><span className="poster-title">Open Water</span><span className="poster-hint">A different world <ArrowUpRight size={13} /></span></button>
          </div>
        </section>

        <section className="studio-intro" id="workspace">
          <div><div className="eyebrow"><span className="eyebrow-line" /> INTERACTIVE STUDIO</div><h2>Direct the next frame.</h2><p>Anchor a character, shape a scene, then compare the conditioning strategy that brings it to life.</p></div>
          <div className="studio-step-key"><span><i>01</i> Character</span><b /> <span><i>02</i> Scene</span><b /> <span><i>03</i> Method</span><b /> <span><i>04</i> Generate</span></div>
        </section>

        <div className="workspace-tabs" id="workspace-tabs">
          <button className={cx('workspace-tab', view === 'single' && 'active')} onClick={() => { setView('single'); setError(''); }}><Sparkles size={16} /> Single scene</button>
          <button className={cx('workspace-tab', view === 'storyboard' && 'active')} onClick={() => { setView('storyboard'); setError(''); }}><Film size={16} /> Multi-scene storyboard</button>
          <button className={cx('workspace-tab', view === 'multi' && 'active')} onClick={() => { setView('multi'); setError(''); }}><Users size={16} /> Multi-character scene</button>
          <div className="tab-spacer" />
          <div className="runtime-note"><Cpu size={14} /> Generation runs on the connected GPU backend</div>
        </div>

        <div className="workspace-grid">
          <section className="control-column">
            <div className="surface-card input-card">
              <div className="section-heading"><span className="step-number">01</span><div><h2>{view === 'multi' ? 'Character references' : 'Character reference'}</h2><p>{view === 'multi' ? 'Upload one reference per character; the model conditions them into a shared scene.' : 'Upload one image to anchor appearance across scenes.'}</p></div></div>
              {view === 'multi' ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '14px' }}>
                  <div>
                    <div className="field-label" style={{ marginBottom: '8px' }}>Character A · left</div>
                    <input ref={characterAInputRef} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => handleMultiReference(event.target.files?.[0], 'a')} />
                    <button type="button" aria-label="Upload Character A reference" className={cx('upload-zone', characterAPreview && 'has-image')} onClick={() => characterAInputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); handleMultiReference(event.dataTransfer.files?.[0], 'a'); }} style={{ minHeight: '220px', width: '100%' }}>
                      {characterAPreview ? <img className="reference-preview" src={characterAPreview} alt="Character A reference" /> : <><div className="upload-symbol"><ImagePlus size={24} /></div><strong>Upload Character A</strong><span>PNG, JPG, WebP · max 10 MB</span></>}
                      {characterAPreview && <span className="replace-image"><Upload size={13} /> Replace image</span>}
                    </button>
                    {characterAFile && <div className="file-meta"><CheckCircle2 size={14} /><span>{characterAFile.name}</span><span>{(characterAFile.size / 1024 / 1024).toFixed(2)} MB</span></div>}
                  </div>
                  <div>
                    <div className="field-label" style={{ marginBottom: '8px' }}>Character B · right</div>
                    <input ref={characterBInputRef} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => handleMultiReference(event.target.files?.[0], 'b')} />
                    <button type="button" aria-label="Upload Character B reference" className={cx('upload-zone', characterBPreview && 'has-image')} onClick={() => characterBInputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); handleMultiReference(event.dataTransfer.files?.[0], 'b'); }} style={{ minHeight: '220px', width: '100%' }}>
                      {characterBPreview ? <img className="reference-preview" src={characterBPreview} alt="Character B reference" /> : <><div className="upload-symbol"><ImagePlus size={24} /></div><strong>Upload Character B</strong><span>PNG, JPG, WebP · max 10 MB</span></>}
                      {characterBPreview && <span className="replace-image"><Upload size={13} /> Replace image</span>}
                    </button>
                    {characterBFile && <div className="file-meta"><CheckCircle2 size={14} /><span>{characterBFile.name}</span><span>{(characterBFile.size / 1024 / 1024).toFixed(2)} MB</span></div>}
                  </div>
                </div>
              ) : <>
                <input ref={fileInputRef} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={(event) => handleReference(event.target.files?.[0])} />
                <button type="button" className={cx('upload-zone', referencePreview && 'has-image')} onClick={() => fileInputRef.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); handleReference(event.dataTransfer.files?.[0]); }}>
                  {referencePreview ? <img className="reference-preview" src={referencePreview} alt="Uploaded character reference" /> : <><div className="upload-symbol"><ImagePlus size={24} /></div><strong>Drop reference image here</strong><span>or click to browse · PNG, JPG, WebP · max 10 MB</span></>}
                  {referencePreview && <span className="replace-image"><Upload size={13} /> Replace image</span>}
                </button>
                {referenceFile && <div className="file-meta"><CheckCircle2 size={14} /><span>{referenceFile.name}</span><span>{(referenceFile.size / 1024 / 1024).toFixed(2)} MB</span></div>}
              </>}
            </div>

            <div className="surface-card input-card">
              <div className="section-heading"><span className="step-number">02</span><div><h2>{view === 'single' ? 'Scene prompt' : view === 'multi' ? 'Shared scene prompt' : 'Scene sequence'}</h2><p>{view === 'single' ? 'Describe the action, environment, pose and mood.' : view === 'multi' ? 'Describe one shared scene. Keep it concise so SDXL receives the full prompt.' : 'Use one prompt per line. Up to 10 scenes per request.'}</p></div></div>
              {view === 'single' ? <>
                <label className="field-label" htmlFor="scene-prompt">Prompt</label>
                <textarea id="scene-prompt" className="prompt-input" rows={5} value={prompt} onChange={(event) => setPrompt(event.target.value)} maxLength={1200} placeholder="Describe your scene…" />
                <div className="field-footer"><span>Be explicit about pose and action for better scene compliance.</span><span>{prompt.length}/1200</span></div>
                <div className="prompt-presets"><button type="button" onClick={() => setPrompt('photorealistic wide shot, same character standing calmly in a quiet library, soft afternoon light, full scene visible')}>Static scene</button><button type="button" onClick={() => setPrompt(DEFAULT_PROMPT)}>Dynamic scene</button><button type="button" onClick={() => setPrompt('photorealistic wide shot, same character kneeling beside a wounded friend, visible grief and concern, dramatic side lighting, full body visible')}>Emotional scene</button></div>
              </> : view === 'multi' ? <>
                <label className="field-label" htmlFor="multi-scene-prompt">Prompt</label>
                <textarea id="multi-scene-prompt" className="prompt-input" rows={4} value={multiPrompt} onChange={(event) => setMultiPrompt(event.target.value)} maxLength={1200} placeholder="Two friends at a village station at sunset, standing side by side, equal size." />
                <div className="field-footer"><span>Short prompt recommended; SDXL has a 77-token text encoder limit.</span><span>{multiPrompt.length}/1200</span></div>
                <div className="prompt-presets"><button type="button" onClick={() => setMultiPrompt('Wide shot. Two friends at a village station at sunset, standing side by side, equal size.')}>Balanced wide shot</button><button type="button" onClick={() => setMultiPrompt('Two friends talking in a quiet library, side by side, both faces visible.')}>Conversation</button><button type="button" onClick={() => setMultiPrompt('Two friends walking through a rainy city street, side by side.')}>Rainy street</button></div>
              </> : <>
                <label className="field-label" htmlFor="story-prompts">Prompts (one per line)</label>
                <textarea id="story-prompts" className="prompt-input storyboard-input" rows={8} value={scenesText} onChange={(event) => setScenesText(event.target.value)} placeholder="Enter one scene prompt per line…" />
                <div className="field-footer"><span>{scenesText.split('\n').map((line) => line.trim()).filter(Boolean).length} scene(s) queued</span><button type="button" className="text-button" onClick={() => setScenesText(STORY_PROMPTS.join('\n'))}>Load demo sequence</button></div>
                <div className="premise-box">
                  <label className="field-label" htmlFor="premise">Optional: create scenes from a story premise</label>
                  <textarea id="premise" rows={2} value={premise} onChange={(event) => setPremise(event.target.value)} placeholder="A one-sentence story premise…" />
                  <div className="premise-actions"><label>Scenes <select value={sceneCount} onChange={(event) => setSceneCount(Number(event.target.value))}>{[2,3,4,5,6,8,10].map((n) => <option key={n} value={n}>{n}</option>)}</select></label><button type="button" className="button button-secondary button-small" onClick={handlePlan} disabled={planLoading || busy}>{planLoading ? <LoaderCircle className="spin" size={14} /> : <WandSparkles size={14} />} Draft with LLM</button></div>
                  <p className="optional-note">Requires ANTHROPIC_API_KEY on the backend. Manual prompts work without it.</p>
                </div>
              </>}
            </div>

            <div className="surface-card input-card method-card">
              <div className="section-heading"><span className="step-number">03</span><div><h2>Experiment method</h2><p>Choose the conditioning strategy to run.</p></div></div>
              {view !== 'multi' ? <div className="method-list">
                {MODES.map((item) => {
                  const Icon = item.icon;
                  return <button type="button" key={item.id} onClick={() => setMode(item.id)} className={cx('method-option', mode === item.id && 'selected')}>
                    <span className="method-icon"><Icon size={18} /></span><span className="method-copy"><strong>{item.name}</strong><small>{item.note}</small></span><span className="method-scale">{item.scale}</span>{mode === item.id && <Check size={16} className="method-check" />}
                  </button>;
                })}
              </div> : <div className="surface-card" style={{ padding: '14px', marginBottom: '14px' }}>
                <div className="section-heading" style={{ marginBottom: '12px' }}><span className="step-number">M</span><div><h2>Multi-reference conditioning</h2><p>Masked IP-Adapter conditioning guides each character to one side of the frame.</p></div></div>
                <label className="field-label" htmlFor="multi-scale">Reference strength <strong>{multiScale.toFixed(2)}</strong></label>
                <input id="multi-scale" type="range" min="0.40" max="1.00" step="0.05" value={multiScale} onChange={(event) => setMultiScale(Number(event.target.value))} style={{ width: '100%' }} />
                <p className="optional-note" style={{ marginTop: '8px' }}>This experimental mode doesn't report CLIP-I/DINO as per-character identity scores. Review character regions manually.</p>
              </div>}
              {view !== 'multi' && <details className="advanced-settings">
                <summary><Settings2 size={14} /> Advanced inference settings <ChevronDown size={14} /></summary>
                <div className="advanced-grid">
                  <label>DINO threshold <div className="range-value">{threshold.toFixed(2)}</div><input type="range" min="0.30" max="0.80" step="0.01" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /><small>Calibration value used by the proposed method.</small></label>
                  <label>Max retries <select value={maxRetries} onChange={(event) => setMaxRetries(Number(event.target.value))}><option value={0}>0</option><option value={1}>1</option><option value={2}>2</option></select><small>Additional generations after the first attempt.</small></label>
                  <label>Inference steps <select value={steps} onChange={(event) => setSteps(Number(event.target.value))}>{[20,25,30,35,40].map((n) => <option key={n} value={n}>{n}</option>)}</select></label>
                  <label>Guidance scale <select value={guidance} onChange={(event) => setGuidance(Number(event.target.value))}>{[5,6,7,7.5,8,9].map((n) => <option key={n} value={n}>{n}</option>)}</select></label>
                  <label>Seed <input type="number" min="0" max="2147483647" value={seed} onChange={(event) => setSeed(Math.max(0, Number(event.target.value) || 0))} /></label>
                </div>
              </details>}
              {view !== 'multi' && <div className="method-summary"><ShieldCheck size={15} /><span>{mode === 'fixed' ? 'Control condition: one fixed 0.50 scale, no retry.' : mode === 'adaptive_only' ? 'SACS selects an initial scale; generation occurs once.' : 'SACS selects the initial scale; DINO can trigger up to two retries.'}</span></div>}
              {error && <div className="error-banner"><AlertCircle size={16} /><span>{error}</span><button type="button" onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
              <button type="button" className="button button-primary generate-button" disabled={busy || (view === 'single' ? !prompt.trim() : view === 'multi' ? (!characterAFile || !characterBFile || !multiPrompt.trim()) : !scenesText.trim())} onClick={view === 'single' ? handleGenerate : view === 'multi' ? handleGenerateMulti : handleStoryboard}>
                {busy ? <LoaderCircle className="spin" size={18} /> : view === 'single' ? <Sparkles size={18} /> : view === 'multi' ? <Users size={18} /> : <Clapperboard size={18} />}
                {busy ? 'Generating… keep this tab open' : view === 'single' ? 'Generate scene' : view === 'multi' ? 'Generate multi-character scene' : 'Generate storyboard'}
                {!busy && <ArrowUpRight size={17} className="button-arrow" />}
              </button>
              <p className="runtime-disclaimer"><LockKeyhole size={12} /> Images are generated by your configured backend. A reset Colab runtime clears its temporary outputs.</p>
            </div>
          </section>

          <section className="result-column">
            <div className="result-header"><div><div className="eyebrow"><span className="eyebrow-line" /> OUTPUT CANVAS</div><h2>{view === 'single' ? 'Generation result' : view === 'multi' ? 'Multi-character result' : 'Storyboard preview'}</h2></div><div className="result-tag"><span className={cx('status-dot', (singleResult || multiResult || storyResults.length) && 'live-dot')} /> {busy ? 'Running' : view === 'single' ? (singleResult ? 'Complete' : 'Waiting for input') : view === 'multi' ? (multiResult ? 'Complete' : 'Waiting for two characters') : (storyResults.length ? `${storyResults.filter((r) => r.image_url).length} frames` : 'Waiting for scenes')}</div></div>

            {view === 'single' ? bestResult ? <>
              <div className="result-image-frame">
                <img src={`${apiBaseNormalized}${bestResult.image_url}`} alt="Generated scene" />
                <div className="image-overlay-chip"><Sparkles size={13} /> {prettyMode(bestResult.mode)}</div>
                <button type="button" className="download-image-button" onClick={() => downloadImage(bestResult.image_url, `${bestResult.scene_id || 'storyframe'}.png`)}><ArrowDownToLine size={15} /> Download PNG</button>
              </div>
              <div className="result-caption"><div><span className="caption-label">SCENE PROMPT</span><p>{bestResult.prompt}</p></div><div className="result-scene-tag">{bestResult.scene_type}</div></div>
              <div className="metrics-grid">
                <MetricCard name="CLIP-I" value={bestResult.metrics.clip_i} label="Image similarity" accent="blue" />
                <MetricCard name="DINO" value={bestResult.metrics.dino} label="Visual feature similarity" accent="violet" />
                <MetricCard name="CLIP-T" value={bestResult.metrics.clip_t} label="Prompt alignment" accent="mint" />
              </div>
              <div className="detail-grid">
                <div className="detail-card"><span>Selected scale</span><strong>{Number(bestResult.metrics.selected_scale ?? bestResult.metrics.initial_scale).toFixed(2)}</strong><small>{bestResult.metrics.initial_scale === bestResult.metrics.selected_scale ? 'Initial setting retained' : 'Best attempt selected'}</small></div>
                <div className="detail-card"><span>Retries performed</span><strong>{bestResult.metrics.retries}</strong><small>{bestResult.metrics.attempt_count} generation attempt(s)</small></div>
                <div className="detail-card"><span>Elapsed time</span><strong>{bestResult.metrics.generation_seconds}s</strong><small>Generation + metric evaluation</small></div>
                <div className="detail-card"><span>Threshold status</span><strong className={bestResult.metrics.threshold_met === false ? 'warning-text' : 'success-text'}>{bestResult.metrics.threshold_met == null ? 'N/A' : bestResult.metrics.threshold_met ? 'Reached' : 'Not reached'}</strong><small>{bestResult.metrics.threshold_met == null ? 'No feedback threshold in this mode' : `DINO threshold ${Number(bestResult.metrics.threshold).toFixed(2)}`}</small></div>
              </div>
              {bestResult.attempts?.length > 0 && <div className="attempt-card surface-card">
                <div className="attempt-title"><div><h3>Self-correction trace</h3><p>Scores are evaluated against the same reference.</p></div><span>{bestResult.attempts.length} attempt(s)</span></div>
                <div className="attempt-list">{bestResult.attempts.map((attempt) => <div className="attempt-row" key={attempt.attempt}><div className="attempt-index">{String(attempt.attempt).padStart(2, '0')}</div><div className="attempt-main"><div className="attempt-line"><strong>Scale {Number(attempt.scale).toFixed(2)}</strong><span>DINO {Number(attempt.dino).toFixed(4)}</span></div><div className="score-track"><div className="score-fill" style={{ width: `${Math.max(0, Math.min(100, Number(attempt.dino) * 100))}%` }} /></div></div><span className={cx('attempt-status', attempt.threshold_met === true ? 'pass' : attempt.threshold_met === false ? 'low' : 'neutral')}>{attempt.threshold_met === true ? 'Threshold met' : attempt.threshold_met === false ? 'Below threshold' : 'Measured'}</span></div>)}</div>
                {bestResult.metrics.threshold_met === false && <div className="notice-line"><CircleHelp size={15} /><span>Threshold was not reached. The application displays the best-DINO image among the attempts rather than claiming success.</span></div>}
              </div>}
            </> : <EmptyResult busy={busy} statusText={statusText} onLoadPrompt={() => { setPrompt(DEFAULT_PROMPT); setView('single'); }} /> : view === 'multi' ? (multiResult ? <>
              <div className="result-image-frame">
                <img src={`${apiBaseNormalized}${multiResult.image_url}`} alt="Generated scene containing Character A and Character B" />
                <div className="image-overlay-chip"><Users size={13} /> Two-character prototype</div>
                <button type="button" className="download-image-button" onClick={() => downloadImage(multiResult.image_url, 'storyframe_two_character_scene.png')}><ArrowDownToLine size={15} /> Download PNG</button>
              </div>
              <div className="result-caption"><div><span className="caption-label">SCENE PROMPT</span><p>{multiPrompt}</p></div><div className="result-scene-tag">MULTI-CHARACTER</div></div>
              <div className="detail-grid">
                <div className="detail-card"><span>References</span><strong>{multiResult.reference_count || 2}</strong><small>Character A left · Character B right</small></div>
                <div className="detail-card"><span>Reference strength</span><strong>{Number(multiResult.conditioning_scale ?? multiScale).toFixed(2)}</strong><small>Masked IP-Adapter conditioning</small></div>
                <div className="detail-card"><span>Elapsed time</span><strong>{Number(multiResult.generation_seconds || lastDuration).toFixed(2)}s</strong><small>Single generation pass</small></div>
                <div className="detail-card"><span>Identity metrics</span><strong>Manual review</strong><small>No per-character CLIP-I/DINO score reported</small></div>
              </div>
              <div className="notice-line"><CircleHelp size={15} /><span>{multiResult.note || 'Review both character regions manually. This mode does not claim per-character identity scores.'}</span></div>
            </> : <EmptyResult busy={busy} statusText={statusText} onLoadPrompt={() => { setMultiPrompt('Wide shot. Two friends at a village station at sunset, standing side by side, equal size.'); setView('multi'); }} />) : <>
              {storyResults.length ? <>
                {storySheet && <div className="contact-sheet-card"><div className="contact-sheet-heading"><div><strong>Composed storyboard sheet</strong><span>Generated from the successful frames below.</span></div><div className="contact-sheet-actions"><button className="button button-secondary button-small" onClick={() => downloadImage(storySheet, 'storyframe_storyboard.png')}><ArrowDownToLine size={14} /> PNG</button>{storyPdf && <button className="button button-secondary button-small" onClick={async () => { try { const response = await authFetch(storyPdf); if (!response.ok) throw new Error('PDF download failed.'); const blob = await response.blob(); const objectUrl = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = objectUrl; anchor.download = 'storyframe_storyboard.pdf'; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(objectUrl); } catch (err) { setError(err.message || 'Could not download PDF.'); } }}><ArrowDownToLine size={14} /> PDF</button>}</div></div><img src={`${apiBaseNormalized}${storySheet}`} alt="Composed multi-scene storyboard sheet" /></div>}
                <div className="storyboard-topline"><div><strong>{storyResults.filter((r) => r.image_url).length} / {storyResults.length} frames generated</strong><span>Method: {prettyMode(mode)}</span></div><button className="button button-secondary button-small" onClick={() => { const prompts = storyResults.filter((item) => item.image_url).map((item) => item.prompt); appendScenePrompts(prompts); }}><RefreshCw size={14} /> Reuse prompts</button></div>
                <div className="story-grid">{storyResults.map((item, index) => item.image_url ? <article className="story-frame-card" key={item.scene_id || index}><div className="story-frame-image"><img src={`${apiBaseNormalized}${item.image_url}`} alt={item.prompt} /><span>#{String(index + 1).padStart(2, '0')}</span><button onClick={() => downloadImage(item.image_url, `${item.scene_id}.png`)} title="Download frame"><ArrowDownToLine size={14} /></button></div><div className="story-frame-content"><p>{item.prompt}</p><div className="story-frame-metrics"><span>DINO <b>{metric(item.metrics?.dino)}</b></span><span>CLIP-T <b>{metric(item.metrics?.clip_t)}</b></span><span>{item.metrics?.retries || 0} retry</span></div></div></article> : <article className="story-frame-card scene-error" key={item.scene_id || index}><AlertCircle size={18} /><strong>{item.scene_id}</strong><p>{item.error || 'Generation failed'}</p></article>)}</div>
                <div className="story-export-note"><CheckCircle2 size={16} /> Frames are saved on the backend. Download individual PNGs to assemble a permanent storyboard sheet with <code>storyboard_composer.py</code>.</div>
              </> : <EmptyResult busy={busy} statusText={statusText} onLoadPrompt={() => setScenesText(STORY_PROMPTS.join('\n'))} />}
            </>}

            <div className="bottom-status"><span className={cx('status-dot', health.model_loaded && 'live-dot')} /><span>{statusText}</span><button type="button" onClick={refreshHealth} aria-label="Refresh backend status"><RefreshCw size={14} /></button></div>
          </section>
        </div>

        <section className="research-section" id="research">
          <div className="research-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> THE RESEARCH SNAPSHOT</div><h2>Better identity, <em>measured.</em></h2><p>Recorded averages from the project's 10-scene, three-arm evaluation. Similarity metrics are proxies—not identity guarantees.</p></div><div className="research-badge"><Activity size={15} /> 10-scene evaluation</div></div>
          <div className="research-grid">
            <article className="research-lead-card"><div className="research-card-top"><span>PROPOSED METHOD</span><span className="research-live-dot" /></div><div className="research-lead-number">0.6163</div><p>Mean DINO similarity</p><div className="research-bar-track"><span style={{ width: '61.63%' }} /></div><div className="research-lead-foot"><strong>+15.2%</strong><span>over fixed baseline (0.5351)</span></div></article>
            <article className="research-compare-card"><div className="compare-card-head"><span>METHOD COMPARISON</span><span>DINO ↑</span></div><div className="compare-row"><div><span className="compare-swatch fixed" />Fixed</div><div className="compare-bar"><i style={{ width: '53.51%' }} /></div><strong>0.5351</strong></div><div className="compare-row"><div><span className="compare-swatch adaptive" />Adaptive-only</div><div className="compare-bar"><i style={{ width: '54.39%' }} /></div><strong>0.5439</strong></div><div className="compare-row proposed-row"><div><span className="compare-swatch proposed" />Proposed</div><div className="compare-bar"><i style={{ width: '61.63%' }} /></div><strong>0.6163</strong></div><div className="compare-footnote">CLIP-T declined slightly and generation time increased; both trade-offs are reported in the evaluation.</div></article>
            <article className="research-method-card"><span className="research-card-label">THE CONTROL LOOP</span><div className="loop-step"><span>01</span><div><strong>Scene-adaptive scale</strong><small>SACS selects an initial strength.</small></div></div><div className="loop-connector" /><div className="loop-step"><span>02</span><div><strong>Score the output</strong><small>DINO compares visual features.</small></div></div><div className="loop-connector" /><div className="loop-step"><span>03</span><div><strong>Retry if needed</strong><small>Increase conditioning; retain best score.</small></div></div></article>
          </div>
        </section>
      </main>
      <footer className="app-footer"><div><span className="footer-brand">STORYFRAME LAB</span><span>Research prototype · Not a guarantee of identity</span></div><div><span>SDXL</span><i /> <span>IP-Adapter</span><i /> <span>SACS</span><i /> <span>DINO feedback</span></div></footer>
    </div>
  );
}

function MetricCard({ name, value, label, accent }) {
  return <div className={`metric-card metric-${accent}`}><div className="metric-top"><span>{name}</span><span className="metric-symbol" /></div><strong>{metric(value)}</strong><small>{label}</small></div>;
}

function EmptyResult({ busy, statusText, onLoadPrompt }) {
  return <div className={cx('empty-result', busy && 'is-busy')}>
    <div className="empty-orbit orbit-one" /><div className="empty-orbit orbit-two" />
    <div className="empty-icon">{busy ? <LoaderCircle className="spin" size={26} /> : <FileImage size={26} />}</div>
    <h3>{busy ? 'Creating your scene' : 'Your canvas is ready'}</h3>
    <p>{busy ? statusText : 'Upload a character reference, describe a scene, and generate a result. Metrics and correction traces will appear here.'}</p>
    {!busy && <button type="button" className="text-button" onClick={onLoadPrompt}>Load a demo prompt <ArrowUpRight size={14} /></button>}
  </div>;
}

export default App;
