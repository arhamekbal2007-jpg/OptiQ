import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, ReferenceLine, Legend 
} from 'recharts';
import { 
  BrainCircuit, 
  ListChecks, 
  Atom, 
  Play, 
  Settings2,
  Database,
  Cpu,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

// --- MOCK DATA ---
const rocData = [
  { fpr: 0, tpr: 0 },
  { fpr: 0.05, tpr: 0.45 },
  { fpr: 0.1, tpr: 0.65 },
  { fpr: 0.2, tpr: 0.82 },
  { fpr: 0.4, tpr: 0.91 },
  { fpr: 0.6, tpr: 0.95 },
  { fpr: 0.8, tpr: 0.98 },
  { fpr: 1, tpr: 1 }
];

const topCandidates = [
  { id: 'C0081', p: 0.824, cost: 4, cap: 1 },
  { id: 'C0031', p: 0.820, cost: 2, cap: 1 },
  { id: 'C0079', p: 0.795, cost: 3, cap: 3 },
  { id: 'C0349', p: 0.751, cost: 1, cap: 3 },
  { id: 'C0327', p: 0.710, cost: 1, cap: 4 },
  { id: 'C0170', p: 0.680, cost: 1, cap: 1 },
  { id: 'C0451', p: 0.612, cost: 5, cap: 2 },
  { id: 'C0991', p: 0.590, cost: 2, cap: 5 },
  { id: 'C0882', p: 0.550, cost: 4, cap: 2 },
  { id: 'C0112', p: 0.520, cost: 3, cap: 3 }
];

const selectedCandidates = ['C0081', 'C0031', 'C0079', 'C0349', 'C0327', 'C0170'];

const simulationData = [
  { name: 'Exact (Math)', probability: 0.2834 },
  { name: 'Quantum Circuit', probability: 0.2817 },
  { name: 'Monte Carlo', probability: 0.2832 }
];

export default function OptiQDashboard() {
  const [activeTab, setActiveTab] = useState('predict');
  
  // Stage States
  const [isPredictRunning, setIsPredictRunning] = useState(false);
  const [predictRan, setPredictRan] = useState(false);
  
  const [isSelectRunning, setIsSelectRunning] = useState(false);
  const [selectRan, setSelectRan] = useState(false);
  
  const [budget, setBudget] = useState(25);
  const [capacity, setCapacity] = useState(22);
  const [usedBudget, setUsedBudget] = useState(0);
  const [usedCapacity, setUsedCapacity] = useState(0);

  // Handlers
  const handleRunPredict = () => {
    setIsPredictRunning(true);
    setTimeout(() => {
      setIsPredictRunning(false);
      setPredictRan(true);
    }, 1200);
  };

  const handleRunSelect = () => {
    setIsSelectRunning(true);
    setTimeout(() => {
      setUsedBudget(12);
      setUsedCapacity(13);
      setIsSelectRunning(false);
      setSelectRan(true);
    }, 1500);
  };

  // UI Components
  const TabButton = ({ id, icon: Icon, label, step }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center gap-3 w-full px-6 py-4 transition-all border-l-4 ${
        activeTab === id 
          ? 'bg-slate-800/50 border-teal-400 text-teal-400' 
          : 'border-transparent text-slate-400 hover:bg-slate-800/30 hover:text-slate-200'
      }`}
    >
      <div className={`p-2 rounded-lg ${activeTab === id ? 'bg-teal-400/10' : 'bg-slate-800'}`}>
        <Icon size={20} />
      </div>
      <div className="text-left">
        <div className="text-xs uppercase tracking-wider font-semibold opacity-70">Stage {step}</div>
        <div className="font-medium">{label}</div>
      </div>
    </button>
  );

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-teal-500/30 flex flex-col md:flex-row">
      
      {/* SIDEBAR */}
      <div className="w-full md:w-72 bg-slate-900 border-r border-slate-800 flex flex-col z-10 shadow-2xl">
        <div className="p-6 border-b border-slate-800">
          <div className="flex items-center gap-3 mb-1">
            <Atom className="text-teal-400" size={28} />
            <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-teal-400 to-cyan-300 bg-clip-text text-transparent">
              OptiQ
            </h1>
          </div>
          <p className="text-xs text-slate-400 font-medium">Hybrid Quantum-Classical Term Deposit Pipeline</p>
        </div>
        
        <nav className="flex-1 py-4">
          <TabButton id="predict" step="1" icon={BrainCircuit} label="Predict" />
          <TabButton id="select" step="2" icon={ListChecks} label="Select" />
          <TabButton id="simulate" step="3" icon={Cpu} label="Simulate" />
        </nav>

        <div className="p-6 border-t border-slate-800">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Database size={14} />
            <span>Connected to Quantum Simulator (Qiskit API)</span>
          </div>
        </div>
      </div>

      {/* MAIN CONTENT */}
      <div className="flex-1 overflow-y-auto relative bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-slate-950">
        
        {/* Subtle grid background */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCI+CjxwYXRoIGQ9Ik0wIDBoNDB2NDBIMHoiIGZpbGw9Im5vbmUiLz4KPHBhdGggZD0iTTAgNDBoNDBWMEgwem0zOS0zOUgxVjM5aDM4eiIgZmlsbD0icmdiYSgyNTUsMjU1LDI1NSwwLjAyKSIvPgo8L3N2Zz4=')] opacity-50 pointer-events-none"></div>

        <div className="p-8 md:p-12 max-w-6xl mx-auto relative z-10">
          
          {/* STAGE 1: PREDICT */}
          {activeTab === 'predict' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <header className="mb-8">
                <h2 className="text-3xl font-bold text-white mb-2">Stage 1: Predictive Modeling</h2>
                <p className="text-slate-400 max-w-2xl text-lg leading-relaxed">
                  Train a hybrid probabilistic classifier to forecast individual customer subscription likelihood. 
                  Generates continuous probabilities and ranks candidates to establish the pool for quantum optimization.
                </p>
              </header>

              {!predictRan ? (
                <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-12 text-center flex flex-col items-center justify-center min-h-[400px]">
                  <BrainCircuit size={64} className="text-slate-600 mb-6" />
                  <h3 className="text-xl font-medium text-slate-300 mb-2">Ready to run prediction model</h3>
                  <p className="text-slate-500 mb-8 max-w-md">Initialize the hybrid blend (Gradient Boosting + Quantum Kernel) to process 100 test customers.</p>
                  <button 
                    onClick={handleRunPredict}
                    disabled={isPredictRunning}
                    className="flex items-center gap-2 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold py-3 px-8 rounded-full transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_20px_rgba(20,184,166,0.3)] hover:shadow-[0_0_30px_rgba(20,184,166,0.5)]"
                  >
                    {isPredictRunning ? (
                      <span className="flex items-center gap-2">
                        <div className="w-5 h-5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></div>
                        Processing...
                      </span>
                    ) : (
                      <>
                        <Play size={18} fill="currentColor" />
                        Run Hybrid Model
                      </>
                    )}
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* ROC Curve Chart */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
                    <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                      <BrainCircuit className="text-teal-400" size={20} />
                      Model Performance (ROC-AUC: 0.573)
                    </h3>
                    <div className="h-72 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={rocData} margin={{ top: 5, right: 20, bottom: 20, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis 
                            dataKey="fpr" 
                            stroke="#64748b" 
                            type="number" 
                            domain={[0, 1]} 
                            label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -10, fill: '#64748b' }} 
                          />
                          <YAxis 
                            stroke="#64748b" 
                            domain={[0, 1]}
                            label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#64748b' }}
                          />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f1f5f9' }}
                            itemStyle={{ color: '#2dd4bf' }}
                          />
                          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#334155" strokeDasharray="3 3" />
                          <Line type="monotone" dataKey="tpr" stroke="#2dd4bf" strokeWidth={3} dot={false} activeDot={{ r: 6 }} name="Hybrid Blend" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Top Candidates Table */}
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl flex flex-col">
                    <div className="p-6 border-b border-slate-800 bg-slate-900/80 backdrop-blur">
                      <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        <ListChecks className="text-teal-400" size={20} />
                        Stage 2 Candidates (Top 10)
                      </h3>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left">
                        <thead className="text-xs text-slate-400 uppercase bg-slate-950/50">
                          <tr>
                            <th className="px-6 py-4 font-medium">Rank</th>
                            <th className="px-6 py-4 font-medium">Customer ID</th>
                            <th className="px-6 py-4 font-medium">Probability</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800">
                          {topCandidates.map((candidate, index) => (
                            <tr key={candidate.id} className="hover:bg-slate-800/30 transition-colors">
                              <td className="px-6 py-3">
                                <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${index < 3 ? 'bg-teal-500/20 text-teal-400' : 'bg-slate-800 text-slate-400'}`}>
                                  {index + 1}
                                </span>
                              </td>
                              <td className="px-6 py-3 font-medium text-slate-300">{candidate.id}</td>
                              <td className="px-6 py-3">
                                <div className="flex items-center gap-3">
                                  <span className="text-teal-400 font-mono">{(candidate.p).toFixed(3)}</span>
                                  <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                    <div className="h-full bg-teal-500 rounded-full" style={{ width: `${candidate.p * 100}%` }}></div>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* STAGE 2: SELECT */}
          {activeTab === 'select' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <header className="mb-8">
                <h2 className="text-3xl font-bold text-white mb-2">Stage 2: Constrained Selection (QAOA)</h2>
                <p className="text-slate-400 max-w-2xl text-lg leading-relaxed">
                  Formulate a binary optimization problem (QUBO) to select exactly 6 customers that maximize expected net value, ensuring strict adherence to budget and capacity constraints.
                </p>
              </header>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Configuration Panel */}
                <div className="lg:col-span-1 space-y-6">
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
                    <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                      <Settings2 className="text-teal-400" size={20} />
                      Constraints Definition
                    </h3>
                    
                    <div className="space-y-5">
                      <div>
                        <label className="block text-sm font-medium text-slate-400 mb-2">Max Budget</label>
                        <div className="relative">
                          <input 
                            type="number" 
                            value={budget} 
                            onChange={(e) => setBudget(Number(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition-colors"
                          />
                          <span className="absolute right-4 top-3 text-slate-500">Units</span>
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-slate-400 mb-2">Max Capacity</label>
                        <div className="relative">
                          <input 
                            type="number" 
                            value={capacity} 
                            onChange={(e) => setCapacity(Number(e.target.value))}
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg py-3 px-4 text-white focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 transition-colors"
                          />
                          <span className="absolute right-4 top-3 text-slate-500">Hours</span>
                        </div>
                      </div>

                      <div className="pt-4 border-t border-slate-800">
                        <button 
                          onClick={handleRunSelect}
                          disabled={isSelectRunning || !predictRan}
                          className="w-full flex justify-center items-center gap-2 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold py-3 px-4 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_0_15px_rgba(20,184,166,0.2)]"
                        >
                          {isSelectRunning ? 'Optimizing Parameters...' : 'Run QAOA Optimization'}
                        </button>
                        {!predictRan && (
                          <p className="text-xs text-amber-500/80 mt-3 flex items-center gap-1">
                            <AlertCircle size={12} />
                            Must run Stage 1 Prediction first
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Results Panel */}
                <div className="lg:col-span-2">
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl h-full flex flex-col">
                    <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                      <ListChecks className="text-teal-400" size={20} />
                      Optimization Results
                    </h3>
                    
                    {!selectRan ? (
                      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 min-h-[300px]">
                        <Cpu size={48} className="mb-4 opacity-20" />
                        <p>Awaiting QAOA execution...</p>
                      </div>
                    ) : (
                      <div className="flex-1 flex flex-col justify-between animate-in fade-in duration-700">
                        
                        {/* Resource Usage Bars */}
                        <div className="space-y-6 mb-8 bg-slate-950/50 p-6 rounded-xl border border-slate-800/50">
                          <div>
                            <div className="flex justify-between text-sm font-medium mb-2">
                              <span className="text-slate-300">Budget Utilized</span>
                              <span className="text-teal-400 font-mono">{usedBudget} / {budget}</span>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                              <div className="bg-teal-500 h-2.5 rounded-full transition-all duration-1000 ease-out relative" style={{ width: `${(usedBudget / budget) * 100}%` }}>
                                <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                              </div>
                            </div>
                          </div>
                          
                          <div>
                            <div className="flex justify-between text-sm font-medium mb-2">
                              <span className="text-slate-300">Capacity Utilized</span>
                              <span className="text-cyan-400 font-mono">{usedCapacity} / {capacity}</span>
                            </div>
                            <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                              <div className="bg-cyan-500 h-2.5 rounded-full transition-all duration-1000 ease-out delay-300 relative" style={{ width: `${(usedCapacity / capacity) * 100}%` }}>
                                 <div className="absolute inset-0 bg-white/20 animate-pulse"></div>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Selected Candidates Grid */}
                        <div>
                          <h4 className="text-sm font-medium text-slate-400 mb-4 uppercase tracking-wider">Optimal Selection (Exactly 6)</h4>
                          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                            {selectedCandidates.map((id, i) => (
                              <div key={id} className="bg-slate-800 border border-slate-700 rounded-lg p-4 flex items-center gap-3 animate-in zoom-in duration-500" style={{ animationDelay: `${i * 100}ms`, animationFillMode: 'both' }}>
                                <CheckCircle2 className="text-teal-400" size={18} />
                                <span className="font-mono text-slate-200 font-medium">{id}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STAGE 3: SIMULATE */}
          {activeTab === 'simulate' && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <header className="mb-8">
                <h2 className="text-3xl font-bold text-white mb-2">Stage 3: Campaign Outcome Simulation</h2>
                <p className="text-slate-400 max-w-2xl text-lg leading-relaxed">
                  Design a 6-qubit quantum circuit to estimate the probability that at least 3 of the targeted customers will subscribe, based on their individual Stage 1 probabilities.
                </p>
              </header>

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                
                {/* Circuit Diagram */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
                   <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                    <BrainCircuit className="text-teal-400" size={20} />
                    State Preparation Circuit
                  </h3>
                  
                  <div className="bg-slate-950 border border-slate-800 rounded-xl p-6 overflow-x-auto">
                    <div className="min-w-[400px] flex flex-col gap-6 font-mono text-sm">
                      {selectedCandidates.map((id, index) => (
                        <div key={id} className="flex items-center relative">
                          {/* Qubit Label */}
                          <div className="w-16 text-slate-400 font-bold">q_{index} <span className="text-slate-600 text-xs pl-1">|0⟩</span></div>
                          
                          {/* Horizontal Wire */}
                          <div className="flex-1 h-[1px] bg-slate-700 relative">
                             {/* RY Gate */}
                            <div className="absolute top-1/2 left-12 -translate-y-1/2 bg-teal-900 border border-teal-500 text-teal-300 rounded px-3 py-1.5 shadow-[0_0_10px_rgba(20,184,166,0.2)]">
                              RY(θ<sub className="text-[10px]">{index}</sub>)
                            </div>
                            
                            {/* Measurement representation at the end */}
                            <div className="absolute top-1/2 right-4 -translate-y-1/2 bg-slate-800 border border-slate-600 rounded px-2 py-1 flex items-center justify-center">
                               <div className="w-3 h-3 border border-slate-400 rounded-full relative overflow-hidden">
                                  <div className="absolute bottom-0 right-0 w-2 h-2 border-t border-l border-slate-400 rounded-tl-full"></div>
                               </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-6 pt-4 border-t border-slate-800/50 text-xs text-slate-500">
                      * <span className="text-teal-400/80">θ = 2 * arcsin(√p)</span> where p is the Stage 1 probability.
                    </div>
                  </div>
                </div>

                {/* Results Chart */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col">
                   <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                    <BarChart className="text-teal-400" size={20} />
                    Validation: P(≥3 subscribe)
                  </h3>
                  
                  {!selectRan ? (
                     <div className="flex-1 flex flex-col items-center justify-center text-slate-500 min-h-[300px]">
                        <p className="text-center">Run Stage 2 first to select targets<br/>before running the simulation.</p>
                     </div>
                  ) : (
                    <div className="flex-1 min-h-[300px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={simulationData} margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                          <XAxis dataKey="name" stroke="#64748b" tick={{ fill: '#94a3b8' }} axisLine={{ stroke: '#334155' }} />
                          <YAxis 
                            domain={[0.27, 0.29]} 
                            stroke="#64748b" 
                            tickFormatter={(val) => val.toFixed(3)}
                            axisLine={{ stroke: '#334155' }}
                          />
                          <Tooltip 
                            cursor={{ fill: '#1e293b', opacity: 0.4 }}
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#f8fafc' }}
                            formatter={(value) => [value.toFixed(4), 'Probability']}
                          />
                          <Bar 
                            dataKey="probability" 
                            fill="#0d9488" 
                            radius={[4, 4, 0, 0]}
                            activeBar={{ fill: '#14b8a6', stroke: '#5eead4', strokeWidth: 1 }}
                            animationDuration={1500}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}