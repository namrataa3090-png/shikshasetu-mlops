

import { useEffect, useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8001'

const demoStudents = [
  { id: 'STU-1001', name: 'Aarav Sharma', study: 'Good', attendance: 92, risk: 'Low' },
  { id: 'STU-1002', name: 'Ananya Patel', study: 'Needs support', attendance: 68, risk: 'High' },
  { id: 'STU-1003', name: 'Rohan Verma', study: 'Average', attendance: 81, risk: 'Medium' },
  { id: 'STU-1004', name: 'Diya Singh', study: 'Good', attendance: 95, risk: 'Low' },
  { id: 'STU-1005', name: 'Vivaan Gupta', study: 'Needs support', attendance: 73, risk: 'High' },
]

const riskData = [
  { label: 'Low risk', count: 68, color: '#16a34a' },
  { label: 'Medium risk', count: 42, color: '#f59e0b' },
  { label: 'High risk', count: 20, color: '#ef4444' },
]

const featureFields = [
  { name: 'school', label: 'School', options: ['GP', 'MS'] },
  { name: 'sex', label: 'Sex', options: ['F', 'M'] },
  { name: 'age', label: 'Age', type: 'number', min: 15, max: 22, defaultValue: 17 },
  { name: 'address', label: 'Home address type', options: ['U', 'R'] },
  { name: 'famsize', label: 'Family size', options: ['GT3', 'LE3'] },
  { name: 'Pstatus', label: 'Parent status', options: ['T', 'A'] },
  { name: 'Medu', label: "Mother's education", options: [0, 1, 2, 3, 4], numeric: true },
  { name: 'Fedu', label: "Father's education", options: [0, 1, 2, 3, 4], numeric: true },
  { name: 'Mjob', label: "Mother's job", options: ['teacher', 'health', 'services', 'at_home', 'other'] },
  { name: 'Fjob', label: "Father's job", options: ['teacher', 'health', 'services', 'at_home', 'other'] },
  { name: 'reason', label: 'School choice reason', options: ['home', 'reputation', 'course', 'other'] },
  { name: 'guardian', label: 'Guardian', options: ['mother', 'father', 'other'] },
  { name: 'traveltime', label: 'Travel time category', options: [1, 2, 3, 4], numeric: true },
  { name: 'studytime', label: 'Weekly study time category', options: [1, 2, 3, 4], numeric: true, defaultValue: 2 },
  { name: 'failures', label: 'Past class failures', options: [0, 1, 2, 3], numeric: true },
  { name: 'schoolsup', label: 'Extra school support', options: ['yes', 'no'] },
  { name: 'famsup', label: 'Family educational support', options: ['yes', 'no'] },
  { name: 'paid', label: 'Extra paid classes', options: ['yes', 'no'] },
  { name: 'activities', label: 'Extracurricular activities', options: ['yes', 'no'] },
  { name: 'nursery', label: 'Attended nursery school', options: ['yes', 'no'] },
  { name: 'higher', label: 'Wants higher education', options: ['yes', 'no'] },
  { name: 'internet', label: 'Internet access at home', options: ['yes', 'no'] },
  { name: 'romantic', label: 'In a romantic relationship', options: ['yes', 'no'] },
  { name: 'famrel', label: 'Family relationship quality', options: [1, 2, 3, 4, 5], numeric: true },
  { name: 'freetime', label: 'Free time after school', options: [1, 2, 3, 4, 5], numeric: true },
  { name: 'goout', label: 'Going out with friends', options: [1, 2, 3, 4, 5], numeric: true },
  { name: 'Dalc', label: 'Weekday alcohol consumption category', options: [1, 2, 3, 4, 5], numeric: true },
  { name: 'Walc', label: 'Weekend alcohol consumption category', options: [1, 2, 3, 4, 5], numeric: true },
  { name: 'health', label: 'Self-reported health', options: [1, 2, 3, 4, 5], numeric: true },
  { name: 'absences', label: 'School absences', type: 'number', min: 0, max: 100, defaultValue: 0 },
]

const initialFeatures = Object.fromEntries(
  featureFields.map((field) => [
    field.name,
    field.defaultValue ?? field.options?.[0] ?? '',
  ])
)

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [search, setSearch] = useState('')
  const [riskFilter, setRiskFilter] = useState('All')
  const [studentName, setStudentName] = useState('')
  const [assessmentMessage, setAssessmentMessage] = useState('')
  const [features, setFeatures] = useState(initialFeatures)
  const [predictionResult, setPredictionResult] = useState(null)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionError, setPredictionError] = useState('')
  const [modelMetrics, setModelMetrics] = useState(null)
  const [metricsLoading, setMetricsLoading] = useState(true)
  const [metricsError, setMetricsError] = useState('')

  useEffect(() => {
  let cancelled = false

  async function loadMetrics() {
    try {
      setMetricsLoading(true)
      setMetricsError('')

      const response = await fetch(`${API_BASE}/metrics`)

      if (!response.ok) {
        throw new Error(`Metrics API returned HTTP ${response.status}`)
      }

      const data = await response.json()

      if (!data.logistic_regression) {
        throw new Error('Logistic regression metrics are missing from the response')
      }

      if (!cancelled) {
        setModelMetrics(data)
      }
    } catch (error) {
      if (!cancelled) {
        setMetricsError(error.message || 'Could not load model metrics')
      }
    } finally {
      if (!cancelled) {
        setMetricsLoading(false)
      }
    }
  }

  loadMetrics()

  return () => {
    cancelled = true
  }
}, [])

  const lrMetrics = modelMetrics?.logistic_regression

  const filteredStudents = demoStudents.filter((student) => {
    const matchesSearch = `${student.name} ${student.id}`
      .toLowerCase()
      .includes(search.toLowerCase())
    const matchesRisk = riskFilter === 'All' || student.risk === riskFilter
    return matchesSearch && matchesRisk
  })

  
async function handleAssessment(event) {
  event.preventDefault()

  setPredictionLoading(true)
  setPredictionError('')
  setPredictionResult(null)
  setAssessmentMessage('')

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ features }),
    })

    const data = await response.json()

    if (!response.ok) {
      const detail = data.detail
      throw new Error(
        typeof detail === 'string'
          ? detail
          : JSON.stringify(detail ?? data)
      )
    }

    setPredictionResult(data)
    setAssessmentMessage(
      `Assessment completed for ${studentName}.`
    )
  } catch (error) {
    setPredictionError(
      error.message || 'Could not connect to the prediction API.'
    )
  } finally {
    setPredictionLoading(false)
  }
}


  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <h1>ShikshaSetu</h1>
            <span>STUDENT SUCCESS PLATFORM</span>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>
        <nav className="navigation">
          {[
            ['Overview', '▦'],
            ['Students', '♙'],
            ['Risk Assessment', '◎'],
            ['Model Performance', '⌁'],
            ['Pipeline Health', '↗'],
          ].map(([page, icon]) => (
            <button
              className={`nav-item ${activePage === page ? 'active' : ''}`}
              key={page}
              onClick={() => {
                setActivePage(page)
                setAssessmentMessage('')
                setPredictionError('')
              }}
            >
              <span className="nav-icon">{icon}</span>
              {page}
              {page === 'Risk Assessment' && <span className="nav-dot" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="support-card">
            <span className="support-icon">✦</span>
            <strong>Support every learner</strong>
            <p>Use insights to guide timely, positive interventions.</p>
          </div>
          <div className="profile">
            <div className="avatar">SS</div>
            <div>
              <strong>School Administrator</strong>
              <span>Demo workspace</span>
            </div>
            <span className="profile-menu">···</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> {activePage}</div>
          <div className="topbar-right">
            <span className="environment"><span /> DEMO ENVIRONMENT</span>
            <button className="icon-button" aria-label="Notifications">♧</button>
            <div className="top-avatar">SA</div>
          </div>
        </header>

        {activePage === 'Overview' && (
          <>
            <section className="page-heading">
              <div>
                <p className="eyebrow">STUDENT SUCCESS OVERVIEW</p>
                <h2>Good afternoon 👋</h2>
                <p className="subheading">
                  Understand student needs and help educators take timely action.
                </p>
              </div>
              <span className="updated-label"><span /> Sample data · Not live</span>
            </section>

            <section className="metric-grid">
              <article className="metric-card">
                <div className="metric-top">
                  <span>Total students</span>
                  <span className="metric-icon blue">♙</span>
                </div>
                <strong className="metric-number">130</strong>
                <p className="metric-note">Illustrative dashboard cohort</p>
              </article>
              <article className="metric-card">
                <div className="metric-top">
                  <span>High-risk flags</span>
                  <span className="metric-icon red">!</span>
                </div>
                <strong className="metric-number">20</strong>
                <p className="metric-note"><span className="red-text">15.4%</span> of demo cohort</p>
              </article>
              <article className="metric-card">
                <div className="metric-top">
                  <span>Model recall</span>
                  <span className="metric-icon purple">◎</span>
                </div>
                <strong className="metric-number">{lrMetrics ? `${(lrMetrics.recall * 100).toFixed(1)}%` : (metricsLoading ? 'Loading…' : '—')}</strong>
                <p className="metric-note">Live API metric · prototype test set</p>
              </article>
              <article className="metric-card">
                <div className="metric-top">
                  <span>ROC-AUC</span>
                  <span className="metric-icon green">⌁</span>
                </div>
                <strong className="metric-number">{lrMetrics ? lrMetrics.roc_auc.toFixed(3) : (metricsLoading ? 'Loading…' : '—')}</strong>
                <p className="metric-note">Live API metric · prototype test set</p>
              </article>
            </section>

            <section className="content-grid">
              <article className="panel risk-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Student risk distribution</h3>
                    <p>Illustrative categories for the dashboard</p>
                  </div>
                  <button className="subtle-button" onClick={() => setActivePage('Students')}>
                    View students ↗
                  </button>
                </div>

                <div className="risk-chart">
                  <div className="donut">
                    <div className="donut-center">
                      <strong>130</strong>
                      <span>Students</span>
                    </div>
                  </div>
                  <div className="legend">
                    {riskData.map((item) => (
                      <div className="legend-item" key={item.label}>
                        <span className="legend-dot" style={{ background: item.color }} />
                        <span>{item.label}</span>
                        <strong>{item.count}</strong>
                        <span className="legend-percent">
                          {((item.count / 130) * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
                <p className="chart-disclaimer">
                  Risk categories above are demo values, not generated by the trained model.
                </p>
              </article>

              <article className="panel model-panel">
                <div className="panel-heading">
                  <div>
                    <h3>Model performance</h3>
                    <p>Logistic regression · Test set</p>
                  </div>
                  <span className="model-badge">Baseline model</span>
                </div>
                {[
  [
    'Precision',
    lrMetrics ? lrMetrics.precision * 100 : 0,
    lrMetrics ? `${(lrMetrics.precision * 100).toFixed(1)}%` : '—',
  ],
  [
    'Recall',
    lrMetrics ? lrMetrics.recall * 100 : 0,
    lrMetrics ? `${(lrMetrics.recall * 100).toFixed(1)}%` : '—',
  ],
  [
    'F1 score',
    lrMetrics ? lrMetrics.f1 * 100 : 0,
    lrMetrics ? `${(lrMetrics.f1 * 100).toFixed(1)}%` : '—',
  ],
  [
    'ROC-AUC',
    lrMetrics ? lrMetrics.roc_auc * 100 : 0,
    lrMetrics ? lrMetrics.roc_auc.toFixed(3) : '—',
  ],
].map(([label, value, display]) => (
                  <div className="performance-row" key={label}>
                    <div className="performance-label">
                      <span>{label}</span>
                      <strong>{display}</strong>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill" style={{ width: `${value}%` }} />
                    </div>
                  </div>
                ))}
                <p className="chart-disclaimer">{metricsLoading ? 'Loading metrics from FastAPI...': metricsError
                  ? `Metrics unavailable: ${metricsError}`
                  : 'Metrics loaded from the FastAPI backend.'}
                </p>
                <p className="chart-disclaimer">
                  Metrics are from the current prototype evaluation; they are not production validation.
                </p>
              </article>
            </section>

            <section className="panel students-panel">
              <div className="panel-heading">
                <div>
                  <h3>Student support overview</h3>
                  <p>Example records to demonstrate the interface</p>
                </div>
                <button className="subtle-button" onClick={() => setActivePage('Students')}>
                  View all students ↗
                </button>
              </div>
              <StudentTable
                students={demoStudents.slice(0, 4)}
                search={search}
                setSearch={setSearch}
                riskFilter={riskFilter}
                setRiskFilter={setRiskFilter}
                compact
              />
            </section>
          </>
        )}

        {activePage === 'Students' && (
          <section className="page-section">
            <div className="page-heading">
              <div>
                <p className="eyebrow">STUDENT DIRECTORY</p>
                <h2>Students</h2>
                <p className="subheading">Explore sample records and filter by demo category.</p>
              </div>
            </div>
            <section className="panel students-panel">
              <StudentTable
                students={filteredStudents}
                search={search}
                setSearch={setSearch}
                riskFilter={riskFilter}
                setRiskFilter={setRiskFilter}
              />
            </section>
          </section>
        )}

        {activePage === 'Risk Assessment' && (
          <section className="page-section">
            <div className="page-heading">
              <div>
                <p className="eyebrow">SUPPORTIVE EARLY INTERVENTION</p>
                <h2>Student risk assessment</h2>
                <p className="subheading">Enter student information to request a prototype prediction from the ML API.</p>
              </div>
            </div>
            <div className="assessment-layout">
              <form className="panel assessment-form" onSubmit={handleAssessment}>
                <h3>Student information</h3>
                <p className="form-description">This prototype sends the 30 trained features to the FastAPI model. Use illustrative data only.</p>
                <label>
                  Student name or demo ID
                  <input
                    value={studentName}
                    onChange={(event) => setStudentName(event.target.value)}
                    placeholder="e.g. STU-1006"
                    required
                  />
                </label>
                
<div className="feature-grid">
  {featureFields.map((field) => (
    <label key={field.name}>
      {field.label}

      {field.options ? (
        <select
          value={String(features[field.name])}
          onChange={(event) => {
            const rawValue = event.target.value
            setFeatures((previous) => ({
              ...previous,
              [field.name]: field.numeric
                ? Number(rawValue)
                : rawValue,
            }))
          }}
          required
        >
          {field.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="number"
          min={field.min}
          max={field.max}
          value={features[field.name]}
          onChange={(event) => {
            const rawValue = event.target.value
            setFeatures((previous) => ({
              ...previous,
              [field.name]:
                rawValue === '' ? '' : Number(rawValue),
            }))
          }}
          required
        />
      )}
    </label>
  ))}
</div>

<button
  className="primary-button"
  type="submit"
  disabled={predictionLoading}
>
  {predictionLoading
    ? 'Running prediction...'
    : 'Run ML assessment →'}
</button>

{assessmentMessage && (
  <p className="form-message">{assessmentMessage}</p>
)}

{predictionError && (
  <p className="form-message" role="alert">
    Prediction failed: {predictionError}
  </p>
)}

{predictionResult && (
  <div className="prediction-result" aria-live="polite">
    <h3>Model result for {studentName}</h3>

    <p>
      <strong>Result:</strong>{' '}
      {predictionResult.interpretation}
    </p>

    {predictionResult.positive_class_probability != null && (
      <p>
        <strong>Positive-class probability:</strong>{' '}
        {(predictionResult.positive_class_probability * 100).toFixed(1)}%
      </p>
    )}

    <p>{predictionResult.disclaimer}</p>
  </div>
)}

              </form>
              <aside className="panel guidance-panel">
                <span className="guidance-icon">✦</span>
                <h3>Designed to support, not label</h3>
                <p>
                  A risk score should prompt a supportive conversation, not determine a
                  student's future or replace educator judgment.
                </p>
                <ul>
                  <li>Review attendance and learning context.</li>
                  <li>Discuss barriers with the student.</li>
                  <li>Agree on a practical support plan.</li>
                </ul>
              </aside>
            </div>
          </section>
        )}

        {activePage === 'Model Performance' && (
          <section className="page-section">
            <div className="page-heading">
              <div>
                <p className="eyebrow">MACHINE LEARNING</p>
                <h2>Model performance</h2>
                <p className="subheading">Current saved evaluation metrics from the prototype.</p>
              </div>
            </div>
            <section className="metric-grid">
              {[
  [
    'Precision',
    lrMetrics ? lrMetrics.precision.toFixed(4) : '—',
    'Of flagged cases, the fraction that were positive',
  ],
  [
    'Recall',
    lrMetrics ? lrMetrics.recall.toFixed(4) : '—',
    'Fraction of positive cases detected',
  ],
  [
    'F1 score',
    lrMetrics ? lrMetrics.f1.toFixed(4) : '—',
    'Balance of precision and recall',
  ],
  [
    'ROC-AUC',
    lrMetrics ? lrMetrics.roc_auc.toFixed(4) : '—',
    'Ranking performance across thresholds',
  ],
  [
    'PR-AUC / AP',
    lrMetrics
      ? lrMetrics.pr_auc_average_precision.toFixed(4)
      : '—',
    'Precision-recall average precision',
  ],
  [
    'Test records',
    modelMetrics ? String(modelMetrics.test_rows) : '—',
    'Held-out test set size',
  ],
].map(([label, value, description]) => (
                <article className="metric-card detail-metric" key={label}>
                  <span>{label}</span>
                  <strong className="metric-number">{value}</strong>
                  <p className="metric-note">{description}</p>
                </article>
              ))}
            </section>
            <div className="panel info-panel">
              <h3>Evaluation notes</h3>
              <p className="chart-disclaimer">{metricsLoading
              ? 'Loading evaluation metrics from FastAPI...'
              : metricsError
              ? `Could not load metrics: ${metricsError}`
              : modelMetrics
                ? `Live API data · ${modelMetrics.dataset} · Target: ${modelMetrics.target}`
                : 'Metrics are not available yet.'}
              </p>
              <p>
                These are the current prototype metrics. Before any real-world use, the
                model needs broader validation, fairness analysis, calibration checks, and
                educator review. A score alone should never determine a student's support,
                placement, or opportunities.
              </p>
            </div>
          </section>
        )}

        {activePage === 'Pipeline Health' && (
          <section className="page-section">
            <div className="page-heading">
              <div>
                <p className="eyebrow">DATA & MLOPS</p>
                <h2>Pipeline health</h2>
                <p className="subheading">Planned architecture and current integration status.</p>
              </div>
            </div>
            <section className="panel pipeline-panel">
              {[
                ['Data ingestion', 'Python ETL scripts', 'Implemented locally', 'done'],
                ['Data validation', 'Schema and quality checks', 'Implemented locally', 'done'],
                ['Feature transformation', 'Processed and curated datasets', 'Implemented locally', 'done'],
                ['Model training', 'scikit-learn pipeline', 'Prototype available', 'done'],
                ['Azure Data Factory', 'Orchestration and scheduling', 'Integration pending', 'pending'],
                ['Azure Machine Learning', 'Experiments and model registry', 'Integration pending', 'pending'],
                ['FastAPI prediction service', 'Model serving endpoint', 'Implemented locally', 'done'],
                ['Docker / Kubernetes', 'Container deployment', 'Not built yet', 'pending'],
              ].map(([name, description, status, state]) => (
                <div className="pipeline-row" key={name}>
                  <span className={`pipeline-status ${state}`} />
                  <div className="pipeline-details">
                    <strong>{name}</strong>
                    <span>{description}</span>
                  </div>
                  <span className={`pipeline-tag ${state}`}>{status}</span>
                </div>
              ))}
            </section>
          </section>
        )}

        <footer className="footer">
          <span>ShikshaSetu · Smart Student Success Platform</span>
          <span>Prototype · Sample data · For educator support</span>
        </footer>
      </main>
    </div>
  )
}

function StudentTable({
  students,
  search,
  setSearch,
  riskFilter,
  setRiskFilter,
  compact = false,
}) {
  return (
    <>
      <div className="table-controls">
        <label className="search-box">
          <span>⌕</span>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search student or ID..."
            aria-label="Search students"
          />
        </label>
        <select
          value={riskFilter}
          onChange={(event) => setRiskFilter(event.target.value)}
          aria-label="Filter by risk category"
        >
          <option value="All">All categories</option>
          <option value="Low">Low risk</option>
          <option value="Medium">Medium risk</option>
          <option value="High">High risk</option>
        </select>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Student</th>
              <th>Study profile</th>
              <th>Attendance</th>
              <th>Demo category</th>
              {!compact && <th>Next step</th>}
            </tr>
          </thead>
          <tbody>
            {students.length > 0 ? students.map((student) => (
              <tr key={student.id}>
                <td>
                  <div className="student-cell">
                    <div className="student-avatar">{student.name.split(' ').map((part) => part[0]).join('')}</div>
                    <div>
                      <strong>{student.name}</strong>
                      <span>{student.id}</span>
                    </div>
                  </div>
                </td>
                <td>{student.study}</td>
                <td>
                  <div className="attendance-cell">
                    <span>{student.attendance}%</span>
                    <div className="attendance-track">
                      <div style={{ width: `${student.attendance}%` }} />
                    </div>
                  </div>
                </td>
                <td><span className={`risk-pill ${student.risk.toLowerCase()}`}>{student.risk}</span></td>
                {!compact && (
                  <td>
                    <span className="next-step">
                      {student.risk === 'High' ? 'Review support needs' : student.risk === 'Medium' ? 'Check in' : 'Continue support'}
                    </span>
                  </td>
                )}
              </tr>
            )) : (
              <tr>
                <td colSpan={compact ? 4 : 5} className="empty-state">No sample students match these filters.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="table-footnote">Illustrative records only. Do not use these categories for real student decisions.</p>
    </>
  )
}

export default App
          