import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, ArrowRight, BedDouble, BrainCircuit, Building2, CalendarDays,
  ChevronRight, CircleAlert, Clock3, HeartPulse, Hospital, Layers3,
  LogIn, LogOut, Menu, Moon, PackageCheck, RefreshCw, Search, ShieldCheck,
  SlidersHorizontal, Sparkles, Stethoscope, Sun, Users, X, Zap, Check,
  LockKeyhole, UserRound, Wifi, Bell, BarChart3
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from 'recharts';
import './styles.css';
import api, { setToken } from './api/client.js';

// ---------------------------------------------------------------- simulation chart data
// Reads the live what-if simulation response and turns it into charts.
const SIM_COLORS = {
  capacity: '#9bb3c9',
  demand: '#2468a6',
  allocated: '#0d8d8a',
  shortfall: '#c0392b',
};

function SimulationCapacityChart({ result, height = 240 }) {
  const dem = result?.demand_after || {};
  const st = result?.state_after || {};

  const rows = [
    { name: 'ICU beds', capacity: st.icu_beds_total ?? 0, demand: dem.icu ?? 0 },
    { name: 'General beds', capacity: st.general_beds_total ?? 0, demand: dem.general ?? 0 },
    { name: 'Emergency', capacity: st.emergency_capacity ?? 0, demand: dem.emergency ?? 0 },
    { name: 'Nursing staff', capacity: st.staff_total ?? 0, demand: dem.min_staff_required ?? 0 },
    { name: 'OR slots', capacity: st.or_capacity ?? 0, demand: dem.or_cases_planned ?? 0 },
  ].map(r => ({
    ...r,
    capacity: Math.max(0, r.capacity),
    demand: Math.max(0, r.demand),
    over: Math.max(0, r.demand - r.capacity),
  }));

  const stackedMax = Math.max(10, ...rows.map(r => r.capacity + r.over));
  const independentMax = Math.max(10, ...rows.map(r => Math.max(r.capacity, r.demand)));

  return (
    <div className="chart-panel sim-capacity-chart">
      <div className="panel-top">
        <div><b>Projected demand vs current capacity</b><small>Stacked view: current capacity plus any demand that exceeds it</small></div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} layout="vertical" margin={{ left: 110, right: 24, top: 10, bottom: 10 }}>
          <CartesianGrid horizontal={false} stroke="#e8eef2"/>
          <XAxis type="number" domain={[0, stackedMax]} hide />
          <YAxis dataKey="name" type="category" width={104} type="band" tick={{ fontSize: 12, fontWeight: 600 }} axisLine={false} tickLine={false}/>
          <Tooltip contentStyle={{ border: '1px solid #dce6ec', borderRadius: 12, boxShadow: '0 12px 28px rgba(15,47,74,.08)' }} />
          {/* Current capacity */}
          <Bar dataKey="capacity" stackId="a" fill={SIM_COLORS.capacity} name="Current capacity" radius={[0, 0, 0, 0]} />
          {/* Over-capacity demand shown as a separate stacked segment, taller = more overflow */}
          <Bar dataKey="over" stackId="a" fill={SIM_COLORS.demand} name="Projected demand" radius={[0, 0, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <div className="sim-chart-legend">
        <span><i style={{ background: SIM_COLORS.capacity }}/> Current capacity</span>
        <span><i style={{ background: SIM_COLORS.demand }}/> Projected demand</span>
      </div>
    </div>
  );
}

function SimulationAllocationChart({ result, height = 240 }) {
  const opt = result?.optimization || {};
  const alloc = opt.allocation || {};
  const cap = opt.capacity || {};
  const dem = opt.demand || {};
  const short = opt.shortfall_detail || {};
  const res = opt.reserved_capacity || {};

  // Recommended allocation as a simple vertical bar chart: one bar per resource.
  // If there is a shortfall, draw a thin red marker at the point where capacity ends
  // and the shortfall begins, so the gap reads clearly.
  const rows = [
    { name: 'ICU beds', allocated: alloc.icu ?? 0, capacity: cap.icu_total ?? 0, used: cap.icu_occupied ?? 0, reserve: res.icu ?? 0, shortfall: short.icu ?? 0 },
    { name: 'General beds', allocated: alloc.general_beds ?? 0, capacity: cap.general_total ?? 0, used: cap.general_occupied ?? 0, reserve: 0, shortfall: short.general ?? 0 },
    { name: 'Emergency', allocated: alloc.emergency ?? 0, capacity: cap.emergency_capacity ?? 0, used: 0, reserve: 0, shortfall: short.emergency ?? 0 },
    { name: 'Nursing staff', allocated: alloc.staff ?? 0, capacity: cap.staff_total ?? 0, used: 0, reserve: 0, shortfall: (opt.shortages || []).includes('Staffing') ? Math.max(0, dem.min_staff_required - cap.staff_available) : 0 },
    { name: 'OR slots', allocated: alloc.or ?? 0, capacity: cap.or_capacity ?? 0, used: 0, reserve: 0, shortfall: short.or ?? 0 },
  ].map(r => ({
    ...r,
    allocated: Math.max(0, r.allocated),
    capacity: Math.max(0, r.capacity),
    used: Math.max(0, r.used),
    reserve: Math.max(0, r.reserve),
    shortfall: Math.max(0, r.shortfall),
  }));

  const maxX = Math.max(10, ...rows.map(r => r.capacity + r.shortfall));

  return (
    <div className="chart-panel sim-alloc-chart">
      <div className="panel-top">
        <div><b>Optimizer recommended allocation</b><small>Recommended vs total capacity, with shortfall shown where present</small></div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={rows} layout="vertical" margin={{ left: 110, right: 24, top: 10, bottom: 10 }}>
          <CartesianGrid horizontal={false} stroke="#e8eef2"/>
          <XAxis type="number" domain={[0, maxX]} hide />
          <YAxis dataKey="name" type="category" width={104} type="band" tick={{ fontSize: 12, fontWeight: 600 }} axisLine={false} tickLine={false}/>
          <Tooltip contentStyle={{ border: '1px solid #dce6ec', borderRadius: 12, boxShadow: '0 12px 28px rgba(15,47,74,.08)' }} />
          {/* Already used units */}
          <Bar dataKey="used" stackId="a" fill={SIM_COLORS.capacity} name="Already used" radius={[0, 0, 0, 0]} />
          {/* Recommended allocation from free capacity */}
          <Bar dataKey="allocated" stackId="a" fill={SIM_COLORS.allocated} name="Recommended allocation" radius={[0, 0, 0, 0]} />
          {/* ICU safety reserve, only meaningful for ICU */}
          <Bar dataKey="reserve" stackId="a" fill={SIM_COLORS.demand} name="ICU reserve" radius={[0, 0, 0, 0]} />
          {/* Shortfall shown beyond capacity */}
          <Bar dataKey="shortfall" stackId="b" fill={SIM_COLORS.shortfall} name="Projected shortfall" radius={[0, 0, 4, 4]} />
        </BarChart>
      </ResponsiveContainer>
      <div className="sim-chart-legend">
        <span><i style={{ background: SIM_COLORS.capacity }}/> Already used</span>
        <span><i style={{ background: SIM_COLORS.allocated }}/> Recommended allocation</span>
        <span><i style={{ background: SIM_COLORS.demand }}/> ICU reserve</span>
        <span><i style={{ background: SIM_COLORS.shortfall }}/> Projected shortfall</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- simulation summary text rows
// Small readable numbers under the charts so the result is readable at a glance.
function simSummaryRows(result) {
  const pressure = result?.projected_pressure || {};
  const shortages = result?.shortages || [];
  const opt = result?.optimization || {};
  const dem = opt?.demand || {};
  const cap = opt?.capacity || {};
  const alloc = opt?.allocation || {};
  const short = opt?.shortfall_detail || {};

  const staffingShort = shortages.includes('Staffing');
  const staffingNeed = dem.min_staff_required ?? 0;
  const staffingHave = cap.staff_available ?? 0;
  const staffingAlloc = alloc.staff ?? 0;

  return [
    {
      key: 'icu',
      label: 'ICU',
      value: (pressure.icu ?? 0) + '% projected',
      note: `need ${dem.icu ?? 0}, alloc ${alloc.icu ?? 0}${short.icu ? ', shortfall ' + short.icu : ''}`,
    },
    {
      key: 'beds',
      label: 'General beds',
      value: (pressure.general_beds ?? 0) + '% projected',
      note: `need ${dem.general ?? 0}, alloc ${alloc.general_beds ?? 0}${short.general ? ', shortfall ' + short.general : ''}`,
    },
    {
      key: 'emergency',
      label: 'Emergency',
      value: (pressure.emergency ?? 0) + '% projected',
      note: `arrivals ${dem.emergency ?? 0}, capacity ${cap.emergency_capacity ?? 0}${short.emergency ? ', shortfall ' + short.emergency : ''}`,
    },
    {
      key: 'staffing',
      label: 'Nursing staffing',
      value: staffingShort ? 'below safe minimum' : 'within safe minimum',
      note: `need ${staffingNeed}, have ${staffingHave}, alloc ${staffingAlloc}`,
    },
    {
      key: 'or',
      label: 'OR / theatre',
      value: (pressure.or ?? 0) + '% projected',
      note: `cases ${dem.or_cases_planned ?? 0}, alloc ${alloc.or ?? 0}${short.or ? ', shortfall ' + short.or : ''}`,
    },
  ];
}

// ---------------------------------------------------------------- data hook
// Loads a resource from the FastAPI backend with loading/error/retry states.
// On failure the UI shows the error + Retry — never silently fake numbers.
function useApi(loader, deps = []) {
  const [state, setState] = useState({ loading: true, error: null, data: null });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    setState(s => ({ ...s, loading: true, error: null }));
    Promise.resolve(loader(ctrl.signal))
      .then(data => { if (alive) setState({ loading: false, error: null, data }); })
      .catch(err => {
        if (!alive || err.name === 'AbortError') return;
        setState({ loading: false, error: err.message || 'Request failed', data: null });
      });
    return () => { alive = false; ctrl.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);
  return { ...state, retry: () => setNonce(n => n + 1) };
}

function LoadError({ loading, error, retry, label = 'live hospital data' }) {
  if (loading) {
    return <div className="api-state"><RefreshCw size={15} className="spin"/> Loading {label}…</div>;
  }
  if (error) {
    return <div className="api-state error"><CircleAlert size={15}/> Unable to load {label}. <span>{error}</span>
      <button onClick={retry}>Retry</button></div>;
  }
  return null;
}

const images = {
  hero: 'https://images.unsplash.com/photo-1586773860418-d37222d8fce3?auto=format&fit=crop&w=1800&q=85',
  doctor: 'https://images.unsplash.com/photo-1551076805-e1869033e561?auto=format&fit=crop&w=1200&q=85',
  team: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1200&q=85',
  hospital: 'https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?auto=format&fit=crop&w=1400&q=85',
  nurse: 'https://images.unsplash.com/photo-1584982751601-97dcc096659c?auto=format&fit=crop&w=1000&q=85',
};

const bedTypes = ['All beds', 'General', 'ICU'];
// Resource name -> lucide icon for the pressure grid.
const RESOURCE_ICONS = {
  general_beds: BedDouble,
  icu_beds: HeartPulse,
  emergency_capacity: Zap,
  staff: Users,
  or_theatres: Activity,
};
const RESOURCE_LABELS = {
  general_beds: 'General beds',
  icu_beds: 'ICU capacity',
  emergency_capacity: 'Emergency capacity',
  staff: 'Nursing coverage',
  or_theatres: 'OR utilization',
};

function App() {
  const [authenticated, setAuthenticated] = useState(false);
  const [page, setPage] = useState('overview');
  const [menuOpen, setMenuOpen] = useState(false);
  const [scenario, setScenario] = useState('base');
  const [selectedId, setSelectedId] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshTick, setRefreshTick] = useState(0);
  const [dark, setDark] = useState(false);

  // Auto-refresh: bumps a tick every 60s; pages that care re-fetch cheaply.
  useEffect(() => {
    if (!autoRefresh || !authenticated) return undefined;
    const id = setInterval(() => setRefreshTick(t => t + 1), 60000);
    return () => clearInterval(id);
  }, [autoRefresh, authenticated]);

  if (!authenticated) return <Login onLogin={() => setAuthenticated(true)} dark={dark} setDark={setDark} />;
  const nav = [
    ['overview', 'Command Center', Hospital],
    ['risk', 'Patient Risk', ShieldCheck],
    ['forecast', 'Demand Forecast', Activity],
    ['simulation', 'What-If Simulation', SlidersHorizontal],
    ['optimization', 'Resource Optimization', PackageCheck],
  ];
  const go = (next) => { setPage(next); setMenuOpen(false); window.scrollTo({ top: 0, behavior: 'smooth' }); };

  return (
    <div className={dark ? 'app-shell dark' : 'app-shell'}>
      <header className="topbar">
        <div className="brand" onClick={() => go('overview')}>
          <div className="brand-mark"><HeartPulse size={21}/></div>
          <div><strong>MediCare AI</strong><span>Hospital Intelligence</span></div>
        </div>
        <nav className={menuOpen ? 'main-nav open' : 'main-nav'}>
          {nav.map(([key, label, Icon]) => <button key={key} className={page === key ? 'active' : ''} onClick={() => go(key)}><Icon size={17}/>{label}</button>)}
        </nav>
        <div className="top-actions">
          <button className="theme-btn" onClick={() => setDark(v => !v)} title="Toggle theme">{dark ? <Sun size={17}/> : <Moon size={17}/>}<span>{dark ? 'Light' : 'Dark'}</span></button>
          <button className="icon-btn" aria-label="Search"><Search size={18}/></button>
          <div className="status-pill"><span className="pulse-dot"/> Live data</div>
          <button className="profile-chip"><span className="profile-avatar">AD</span><span>Admin</span></button>
          <button className="menu-btn" onClick={() => setMenuOpen(v => !v)}>{menuOpen ? <X/> : <Menu/>}</button>
        </div>
      </header>

      <div className="announcement"><div><Wifi size={14}/> Connected to hospital operations feed</div><span>Last sync 2 min ago</span><button><Bell size={14}/></button></div>
      <main>
        <AnimatePresence mode="wait">
          {page === 'overview' && <Overview key={`ov-${refreshTick}`} go={go} autoRefresh={autoRefresh} setAutoRefresh={setAutoRefresh} />}
          {page === 'risk' && <RiskPage selectedId={selectedId} setSelectedId={setSelectedId} />}
          {page === 'forecast' && <ForecastPage />}
          {page === 'simulation' && <SimulationPage scenario={scenario} setScenario={setScenario} />}
          {page === 'optimization' && <OptimizationPage />}
        </AnimatePresence>
      </main>
      <footer className="footer"><span>MEDICARE AI</span><span>Decision support for hospital operations</span><span>24 / 48 / 72 hour intelligence</span><button onClick={() => setAuthenticated(false)}><LogOut size={13}/> Sign out</button></footer>
    </div>
  );
}

function Login({ onLogin, dark, setDark }) {
  const [email, setEmail] = useState('admin@medicare.ai');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true); setError(null);
    try {
      const res = await api.login(email, password);
      setToken(res.access_token);
      onLogin(res.user);
    } catch (e) {
      setError(e.message || 'Sign-in failed');
    } finally {
      setBusy(false);
    }
  };
  return <div className={dark ? 'login-shell dark' : 'login-shell'}>
    <div className="login-image"><img src={images.hero} alt="Hospital operations"/><div className="login-image-overlay"><div className="login-brand"><div className="brand-mark"><HeartPulse size={22}/></div><strong>MediCare AI</strong></div><h1>Operate ahead of the pressure.</h1><p>Forecast demand, test disruptions and optimize constrained hospital resources from one command center.</p><div className="login-proof"><span><Check size={14}/> 24/48/72h forecasting</span><span><Check size={14}/> What-if simulation</span><span><Check size={14}/> Constraint-aware optimization</span></div></div></div>
    <div className="login-panel"><button className="login-theme" onClick={() => setDark(v => !v)}>{dark ? <Sun size={16}/> : <Moon size={16}/>} {dark ? 'Light mode' : 'Dark mode'}</button><div className="login-card"><div className="login-icon"><LockKeyhole size={20}/></div><div className="eyebrow">Admin access</div><h2>Welcome back</h2><p>Sign in to the hospital operations command center.</p><label>Email<input value={email} onChange={e => setEmail(e.target.value)} /></label><label>Password<input type="password" value={password} onChange={e => setPassword(e.target.value)} onKeyDown={e => e.key === 'Enter' && submit()} /></label>{error && <div className="login-error"><CircleAlert size={14}/> {error}</div>}<button className="login-btn" disabled={busy} onClick={submit}>{busy ? <RefreshCw size={17} className="spin"/> : <LogIn size={17}/>} {busy ? 'Signing in…' : 'Sign in to command center'}</button><div className="demo-login"><span>Backend-authenticated</span><b>admin@medicare.ai</b><small>Credentials are validated by the FastAPI backend (ADMIN_EMAIL / ADMIN_PASSWORD in .env)</small></div></div><div className="login-footer">Protected operations workspace · Prototype environment</div></div>
  </div>;
}

function Page({ eyebrow, title, description, children, image }) {
  return <motion.section className="page" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: .35 }}>
    <div className="page-head"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{image && <div className="head-image"><img src={image} alt="Hospital operations"/><div className="image-caption"><Activity size={15}/> Live operations view</div></div>}</div>{children}
  </motion.section>
}

function Overview({ go, autoRefresh, setAutoRefresh }) {
  const dash = useApi(signal => api.dashboard(signal), []);
  const res = useApi(signal => api.resources(signal), []);
  const bedData = useApi(signal => api.beds(signal), []);
  const d = dash.data;
  const forecastRows = d ? d.forecast.timestamps.map((t, i) => ({
    time: t, icu: d.forecast.icu[i], beds: d.forecast.beds[i], ed: d.forecast.ed[i],
  })) : [];
  const f = d && d.forecast;
  const last = f ? f.icu.length - 1 : 0;
  return <Page eyebrow="Hospital command center" title={<>Know what the hospital will need <em>before</em> it needs it.</>} description="MediCare AI turns operational signals into forecasts, simulations and feasible resource plans for the next 24, 48 and 72 hours." image={images.hero}>
    <div className="hero-actions"><button className="primary-btn" onClick={() => go('simulation')}>Run a what-if scenario <ArrowRight size={17}/></button><button className="secondary-btn" onClick={() => go('optimization')}>View recommended allocation</button><label className="refresh-toggle"><input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)}/><span>{autoRefresh ? 'Auto-refresh on' : 'Auto-refresh off'}</span></label></div>
    <LoadError loading={dash.loading} error={dash.error} retry={dash.retry}/>
    {d && <>
    <div className="metric-strip"><Metric label="Beds occupied" value={`${Math.round(d.occupancy_pct)}%`} delta={`${d.beds_occupied} / ${d.beds_total} general beds`} tone={d.occupancy_pct > 85 ? 'warn' : 'good'} icon={BedDouble}/><Metric label="ICU capacity" value={`${Math.round(d.icu_occupancy_pct)}%`} delta={`${d.icu_available} beds available`} tone={d.icu_available <= 3 ? 'critical' : 'good'} icon={HeartPulse}/><Metric label="ED arrivals / day" value={d.ed_arrivals_now} delta={`capacity ${d.ed_capacity}/day`} tone={d.ed_arrivals_now / d.ed_capacity > .85 ? 'warn' : 'good'} icon={Zap}/><Metric label="High-risk patients" value={d.high_risk_patients} delta={`${d.critical_risk_patients} critical · ${d.total_patients_scored} scored`} tone="critical" icon={ShieldCheck}/></div>
    <div className="section-title"><div><span>Predict</span><h2>Next 72 hours</h2></div><button className="text-btn" onClick={() => go('forecast')}>Open full forecast <ChevronRight size={16}/></button></div>
    <div className="command-grid"><div className="chart-panel large"><div className="panel-top"><div><b>Resource pressure</b><small>Predicted utilization across core hospital resources</small></div><div className="legend"><span><i className="dot blue"/> ICU</span><span><i className="dot teal"/> Beds</span><span><i className="dot dark"/> ED</span></div></div><ForecastChart compact rows={forecastRows}/></div><div className="story-card"><img src={images.team} alt="Clinical team"/><div className="story-copy"><span>Decision intelligence</span><h3>From signal to action</h3><p>Forecast the pressure, test disruption scenarios, then allocate constrained resources with an explainable plan.</p><button onClick={() => go('optimization')}>See the plan <ArrowRight size={16}/></button></div></div></div>
    <div className="section-title"><div><span>Operate</span><h2>Bed occupancy</h2></div><div className="live-label"><span className="pulse-dot"/> {d.beds_occupied} / {d.beds_total} occupied</div></div>
    <BedOccupancy bedData={bedData}/>
    <div className="section-title"><div><span>Operate</span><h2>Resource pressure at a glance</h2></div></div><div className="resource-grid">{res.data ? res.data.resources.map((r, i) => <ResourceBar key={r.name} label={RESOURCE_LABELS[r.name] || r.name} value={r.value} total={r.total} icon={RESOURCE_ICONS[r.name] || Activity} delay={i * .08}/>) : null}</div>
    </>}
    <div className="flow-banner"><FlowStep n="01" title="Predict" text="Demand + patient risk"/><ArrowRight/><FlowStep n="02" title="Simulate" text="Test disruptions"/><ArrowRight/><FlowStep n="03" title="Optimize" text="Respect constraints"/><ArrowRight/><FlowStep n="04" title="Explain" text="Show why the plan works"/></div>
  </Page>
}

function BedOccupancy({ bedData }) {
  const [filter, setFilter] = useState('All beds');
  const all = bedData.data?.beds || [];
  const counts = bedData.data?.counts || {};
  const beds = useMemo(() => all.filter(b => filter === 'All beds' || b.type === filter), [all, filter]);
  return <div className="bed-board">
    <div className="bed-board-top"><div className="bed-tabs">{bedTypes.map(t => <button key={t} className={filter === t ? 'active' : ''} onClick={() => setFilter(t)}>{t}</button>)}</div><div className="bed-legend"><span><i className="bed-dot occupied"/> Occupied</span><span><i className="bed-dot reserved"/> Reserved</span><span><i className="bed-dot available"/> Available</span><span><i className="bed-dot cleaning"/> Cleaning</span></div></div>
    <LoadError loading={bedData.loading} error={bedData.error} retry={bedData.retry} label="bed status"/>
    <div className="bed-grid">{beds.map((bed, i) => <motion.button key={bed.id} className={`bed-tile ${bed.status}`} initial={{opacity:0, scale:.88}} animate={{opacity:1, scale:1}} transition={{delay:i*.025, type:'spring', stiffness:220}} whileHover={{y:-4, scale:1.035}}><div className="bed-shape"><BedDouble size={22}/></div><b>{bed.id}</b><small>{bed.type}</small><span>{bed.status}</span></motion.button>)}</div>
    <div className="bed-summary"><div><b>{bedData.data ? `${Math.round(bedData.data.occupancy_pct)}%` : '—'}</b><span>Overall occupancy</span></div><div><b>{counts.available ?? '—'}</b><span>Available</span></div><div><b>{counts.reserved ?? '—'}</b><span>Reserved</span></div><div><b>{counts.cleaning ?? '—'}</b><span>Cleaning</span></div><div className="bed-action"><button>Open bed management <ArrowRight size={15}/></button></div></div>
  </div>;
}

function Metric({ label, value, delta, tone, icon: Icon }) { return <div className="metric"><div className="metric-icon"><Icon size={18}/></div><div><span>{label}</span><strong>{value}</strong><small className={tone}>{delta}</small></div></div> }
function ResourceBar({ label, value, total, icon: Icon, delay }) { const pct = Math.round(value / total * 100); return <motion.div className="resource-item" initial={{ opacity:0, y:12 }} animate={{ opacity:1, y:0 }} transition={{ delay }}><div className="resource-head"><span><Icon size={17}/>{label}</span><b>{pct}%</b></div><div className="bar"><motion.div initial={{ width:0 }} animate={{ width:`${pct}%` }} transition={{ duration: .8, delay }}/></div><small>{total-value} units of buffer</small></motion.div> }
function FlowStep({ n, title, text }) { return <div className="flow-step"><span>{n}</span><div><b>{title}</b><small>{text}</small></div></div> }

function RiskPage({ selectedId, setSelectedId }) {
  const list = useApi(signal => api.patients(25, null, signal), []);
  const patients = list.data?.patients || [];
  const activeId = selectedId ?? patients[0]?.patient_id ?? null;
  const detail = useApi(signal => activeId != null ? api.patientRisk(activeId, signal) : Promise.resolve(null), [activeId]);
  const p = detail.data;
  const dotClass = l => ({ Critical: 'critical', High: 'high', Watch: 'watch', Low: 'watch' }[l] || 'watch');
  return <Page eyebrow="Patient risk intelligence" title="See acuity before it becomes a capacity problem." description="A trained deterioration model (synthetic clinical data) scores every admission on the risk of deterioration within 12 hours, helping the optimizer protect scarce capacity."><div className="risk-layout"><div className="risk-table"><div className="table-head"><span>Patient</span><span>Admission</span><span>Risk</span><span>Level</span></div>
    <LoadError loading={list.loading} error={list.error} retry={list.retry} label="patient risk list"/>
    {patients.map(r => <button key={r.patient_id} className={activeId === r.patient_id ? 'risk-row selected' : 'risk-row'} onClick={() => setSelectedId(r.patient_id)}><span><b>Patient {r.patient_id}</b><small>{r.age}{r.gender} · comorbidity {r.comorbidity_index}</small></span><span>{r.admission_type}</span><span><i className={`risk-dot ${dotClass(r.risk_label)}`}/>{r.risk_pct}%</span><span>{r.risk_label}</span></button>)}</div><div className="detail-panel"><span className="eyebrow">Selected patient</span>
      <LoadError loading={detail.loading} error={detail.error} retry={detail.retry} label="patient risk details"/>
      {p && <>
      <div className="patient-top"><div className="avatar">P{String(p.patient_id).slice(-2)}</div><div><h2>Patient {p.patient_id}</h2><span>{p.age} years · {p.gender} · {p.admission_type} admission · comorbidity {p.comorbidity_index}</span></div></div>
      <div className="risk-score"><div><small>Predicted risk (12h window)</small><strong>{p.risk_pct}<sup>/100</sup></strong></div><div className="score-ring"><span>{p.risk_label}</span></div></div>
      <div className="reason"><CircleAlert size={18}/><div><b>Why the model flagged this patient</b><p>{p.factors.join(' · ')}</p></div></div>
      <div className="detail-actions"><button className="secondary-btn"><CalendarDays size={15}/> Review care pathway</button><button className="secondary-btn"><BedDouble size={15}/> Reserve likely resource</button></div>
      <div className="insight-band"><BrainCircuit size={20}/><div><b>Feeds optimization</b><span>Risk scores are passed into the resource planning layer to protect high-acuity capacity. {p.synthetic_data_note}</span></div></div></>}
    </div></div></Page>
}

function ForecastPage() {
  const [horizon, setHorizon] = useState(72);
  const fc = useApi(signal => api.forecast(horizon, signal), [horizon]);
  const f = fc.data;
  const rows = f ? f.timestamps.map((t, i) => ({
    time: t, icu: f.icu[i], beds: f.general[i], ed: f.emergency[i],
  })) : [];
  const peak = f ? f.timestamps[f.icu.reduce((best, v, i, a) => v + f.general[i] > a[best] + f.general[best] ? i : best, 0)] : null;
  return <Page eyebrow="Demand intelligence" title="A 72-hour view of hospital pressure." description="Forecast emergency arrivals, admissions, ICU demand and bed occupancy so administrators can prepare before the bottleneck appears."><div className="forecast-toolbar"><div className="time-pills">{[24, 48, 72].map(h => <button key={h} className={horizon === h ? 'active' : ''} onClick={() => setHorizon(h)}>{h}h</button>)}</div><div className="forecast-note"><Clock3 size={16}/> {f ? `Model origin ${f.origin_date.slice(0, 10)} · Random-Forest per-horizon models` : 'Loading model output…'}</div></div>
    <LoadError loading={fc.loading} error={fc.error} retry={fc.retry} label="demand forecast"/>
    {f && <>
    <div className="forecast-grid"><div className="chart-panel"><div className="panel-top"><div><b>ICU demand</b><small>Model forecast · dataset demand units</small></div><strong>{f.icu[f.icu.length - 1]}</strong></div><ForecastChart metric="icu" rows={rows}/></div><div className="chart-panel"><div className="panel-top"><div><b>Bed occupancy</b><small>General bed demand forecast</small></div><strong>{f.general[f.general.length - 1]}</strong></div><ForecastChart metric="beds" rows={rows}/></div><div className="chart-panel"><div className="panel-top"><div><b>Emergency arrivals</b><small>Forecast demand index</small></div><strong>{f.emergency[f.emergency.length - 1]}</strong></div><ForecastChart metric="ed" rows={rows}/></div><div className="chart-panel image-panel"><img src={images.hospital} alt="Hospital building"/><div><span>Operational signal</span><h3>Peak pressure is expected around {peak}.</h3><p>Trained on {f.origin_date.slice(0, 10)}-anchored operations history. Test MAE by horizon is available in the model API; data is synthetic/simulated.</p></div></div></div>
    </>}</Page>
}

function ForecastChart({ metric, compact, rows }) { const key = metric || 'icu'; const stroke = key === 'beds' ? '#0d8f8a' : key === 'ed' ? '#233f57' : '#2468a6'; const data = rows || []; return <div className={compact ? 'chart compact' : 'chart'}><ResponsiveContainer width="100%" height="100%"><AreaChart data={data}><CartesianGrid vertical={false} stroke="#e8eef2"/><XAxis dataKey="time" tickLine={false} axisLine={false} fontSize={11}/><YAxis hide domain={['auto', 'auto']}/><Tooltip contentStyle={{border:'1px solid #dce6ec', borderRadius:12, boxShadow:'0 12px 28px rgba(15,47,74,.08)'}}/><Area type="monotone" dataKey={key} stroke={stroke} fill="transparent" strokeWidth={3}/></AreaChart></ResponsiveContainer></div> }

function SimulationPage({ scenario, setScenario }) {
  const presets = useApi(signal => api.simulationPresets(signal), []);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async (presetId) => {
    setBusy(true); setError(null);
    try {
      const res = await api.runSimulation({ preset: presetId, horizon_hours: 48 });
      setResult(res);
    } catch (e) {
      setError(e.message || 'Simulation failed');
    } finally { setBusy(false); }
  };

  // Run the base plan on first load and whenever the selection changes.
  useEffect(() => { run(scenario); /* eslint-disable-next-line */ }, [scenario]);

  const buttons = (presets.data?.presets || []).map(p => [p.id, p.label]);
  const label = (presets.data?.presets || []).find(p => p.id === scenario)?.label || 'Custom scenario';
  const pressure = result?.projected_pressure || {};
  const peak = Math.max(pressure.icu || 0, pressure.general_beds || 0, pressure.staff || 0, pressure.emergency || 0);
  return <Page eyebrow="Scenario lab" title="Test the disruption before it tests you." description="Change one operational assumption and watch projected pressure move. The simulation recalculates demand from the forecasting models and re-runs the OR-Tools optimizer on the new state."><div className="scenario-layout"><div className="scenario-controls"><div className="control-head"><SlidersHorizontal size={18}/><div><b>Scenario controls</b><small>Select a disruption</small></div></div>{buttons.map(([id, lbl]) => <button className={scenario === id ? 'scenario-btn active' : 'scenario-btn'} key={id} onClick={() => { setScenario(id); }}><span>{lbl}</span><ChevronRight size={16}/></button>)}<div className="scenario-photo"><img src={images.hero} alt="Hospital"/><div><span>Simulation mode</span><b>Safe to experiment</b></div></div></div>
    <motion.div className="scenario-result" key={scenario} initial={{opacity:.4, x:12}} animate={{opacity:1,x:0}} transition={{duration:.3}}>
      <LoadError loading={busy && !result} error={error} retry={() => run(scenario)} label="simulation result"/>
      {result && <>
      <div className="result-head"><div><span>Projected outcome</span><h2>{result.shortages.length ? `${result.shortages.join(' and ')} under pressure` : 'Stable — no projected shortages'}</h2></div><div className="scenario-tag"><Sparkles size={15}/> {label}</div></div>
      <div className="pressure-grid"><Pressure label="ICU" value={pressure.icu}/><Pressure label="Beds" value={pressure.general_beds}/><Pressure label="Staff" value={pressure.staff}/><Pressure label="ED" value={pressure.emergency}/></div>
      <div className="compare"><div><span>Current plan</span><strong>{result.scenario && Object.keys(result.scenario).length ? 'Baseline' : `${peak}%`}</strong><small>Peak projected pressure</small></div><ArrowRight/><div className="after"><span>Selected scenario</span><strong>{peak}%</strong><small>Peak projected pressure · {result.horizon_hours}h</small></div></div>
      <div className="recommendation"><div className="rec-icon"><Zap size={19}/></div><div><span>Optimizer response ({result.optimization.status})</span><b>{result.recommendations[0]}</b><small>{result.recommendations.slice(1).join(' ') || 'No further actions required.'}</small></div></div>
        <div className="sim-chart-grid"><SimulationCapacityChart result={result} /><SimulationAllocationChart result={result} /></div>
      </>}
    </motion.div></div></Page>
}
function Pressure({ label, value }) { const v = value ?? 0; return <div className="pressure"><div><span>{label}</span><b>{v}%</b></div><div className="bar"><motion.div animate={{width:`${Math.min(v,100)}%`}} transition={{duration:.5}} /></div><small>{v > 90 ? 'Critical' : v > 80 ? 'High pressure' : 'Within plan'}</small></div> }

function OptimizationPage() {
  const opt = useApi(signal => api.latestOptimization(signal), []);
  const [rerunning, setRerunning] = useState(false);
  const [error, setError] = useState(null);
  const data = opt.data;
  const rerun = async () => {
    setRerunning(true); setError(null);
    try {
      await api.runOptimization({ horizon_hours: 48 });
      opt.retry();
    } catch (e) { setError(e.message || 'Optimization failed'); }
    finally { setRerunning(false); }
  };
  const dem = data?.demand || {};
  const rows = data?.allocation ? [
    ['ICU beds', `${data.capacity?.icu_available ?? '—'} free`, `${data.allocation.icu} (+${data.reserved_capacity?.icu ?? 0} reserve)`,
      (data.shortfall_detail?.icu > 0 ? `Projected need ${dem.icu} vs ${data.capacity?.icu_occupied} occupied — shortfall ${data.shortfall_detail.icu}` : `Projected need ${dem.icu} covered; reserve kept for surge`)],
    ['General beds', `${data.capacity?.general_available ?? '—'} free`, `${data.allocation.general_beds}`,
      (data.shortfall_detail?.general > 0 ? `Shortfall ${data.shortfall_detail.general} vs projected need ${dem.general}` : `Projected need ${dem.general} within free capacity`)],
    ['Emergency', `${data.capacity?.emergency_capacity ?? '—'} capacity`, `${data.allocation.emergency}`,
      (data.shortfall_detail?.emergency > 0 ? `Arrivals ${dem.emergency} exceed capacity by ${data.shortfall_detail.emergency}` : `Capacity covers ${dem.emergency} arrivals`)],
    ['Nursing', `${data.capacity?.staff_available ?? '—'} on duty`, `${data.allocation.staff}`,
      (data.shortages?.includes('Staffing') ? `Safe-staffing need ${dem.min_staff_required} exceeds staff on duty` : `Meets safe-staffing ratio (${dem.min_staff_required} required)`)],
    ['OR capacity', `${data.capacity?.or_capacity ?? '—'} slots`, `${data.allocation.or}`,
      `${dem.or_cases_planned} scheduled cases`],
  ] : [];
  return <Page eyebrow="Constraint-aware planning" title="A feasible allocation, not just a forecast." description="MediCare AI passes predicted demand and current constraints into the OR-Tools CP-SAT solver to recommend how limited resources should be allocated."><div className="optimization-hero"><div className="optimization-flow"><div className="opt-node"><BrainCircuit/><span>Predicted demand</span><b>24 / 48 / 72h</b></div><ArrowRight/><div className="opt-node"><Layers3/><span>Hospital constraints</span><b>Beds · ICU · staff · OR</b></div><ArrowRight/><div className="opt-node highlight"><Zap/><span>OR-Tools optimizer</span><b>Find feasible plan</b></div><ArrowRight/><div className="opt-node"><PackageCheck/><span>Recommended allocation</span><b>Explainable actions</b></div></div></div>
    <LoadError loading={opt.loading} error={opt.error} retry={opt.retry} label="optimization plan"/>
    {error && <div className="api-state error"><CircleAlert size={15}/> {error}</div>}
    {data && <div className="allocation-grid"><div className="allocation-table"><div className="section-title"><div><span>Recommended plan · {data.status}</span><h2>Where the capacity moves</h2></div><button className="secondary-btn" onClick={rerun} disabled={rerunning}>{rerunning ? <RefreshCw size={15} className="spin"/> : <RefreshCw size={15}/>} Re-run</button></div><div className="alloc-head"><span>Resource</span><span>Current</span><span>Recommended</span><span>Reason</span></div>{rows.map(r => <div className="alloc-row" key={r[0]}><b>{r[0]}</b><span>{r[1]}</span><strong>{r[2]}</strong><span>{r[3]}</span></div>)}</div>    <div className="explain-card"><div className="explain-image"><img src={images.doctor} alt="Clinical operations"/><div className="explain-badge"><ShieldCheck size={16}/> Constraint-aware</div></div><div className="explain-copy"><span>Explain</span><h3>Why this plan?</h3>{(data.explanation || []).filter(b => b.resource !== 'Objective' && b.resource !== 'Scenario').map(b => <p key={b.resource}><b>{b.resource}:</b> {b.lines.join(' ')}</p>)}<div className="mini-proof"><div><b>{data.shortages?.length ?? 0}</b><span>shortages flagged</span></div><div><b>{data.recommendations?.length ?? 0}</b><span>actions recommended</span></div><div><b>0</b><span>hard constraints violated</span></div></div></div></div></div>}
  </Page>
}
createRoot(document.getElementById('root')).render(<App />);
