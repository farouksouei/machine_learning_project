import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { AlertCircle, TrendingUp, Users, BarChart3, Settings, Play, CheckCircle, XCircle, Loader, Zap } from 'lucide-react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, PointElement, LineElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar, Line, Doughnut } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, PointElement, LineElement, ArcElement, Title, Tooltip, Legend);

const API_BASE_URL = 'http://127.0.0.1:8000';

export default function ChurnPredictionDashboard() {
    const [activeTab, setActiveTab] = useState('predict');
    const [apiStatus, setApiStatus] = useState({ healthy: false, loading: true });
    const [prediction, setPrediction] = useState(null);
    const [models, setModels] = useState([]);
    const [availableModels, setAvailableModels] = useState([]);
    const [activeModel, setActiveModel] = useState(null);
    const [selectedModelForPrediction, setSelectedModelForPrediction] = useState('');
    const [modelComparison, setModelComparison] = useState(null);
    const [trainingStatus, setTrainingStatus] = useState(null);
    const [loading, setLoading] = useState(false);
    const [settingActiveModel, setSettingActiveModel] = useState(null);
    const [error, setError] = useState(null);

    // Customer form state
    const [customerData, setCustomerData] = useState({
        gender: 'Female',
        SeniorCitizen: 0,
        Partner: 'Yes',
        Dependents: 'No',
        tenure: 12,
        PhoneService: 'Yes',
        MultipleLines: 'No',
        InternetService: 'DSL',
        OnlineSecurity: 'No',
        OnlineBackup: 'Yes',
        DeviceProtection: 'No',
        TechSupport: 'No',
        StreamingTV: 'No',
        StreamingMovies: 'No',
        Contract: 'Month-to-month',
        PaperlessBilling: 'Yes',
        PaymentMethod: 'Electronic check',
        MonthlyCharges: 29.85,
        TotalCharges: 358.2
    });

    // Training form state
    const [trainingConfig, setTrainingConfig] = useState({
        models_to_train: null,
        optimize_hyperparameters: true,
        search_type: 'random',
        n_iter: 30,
        use_ensemble: true,
        advanced_preprocessing: true,
        feature_engineering: true,
        handle_imbalance: true,
        experiment_name: null
    });

    useEffect(() => {
        checkApiHealth();
        fetchAvailableModels();
    }, []);

    useEffect(() => {
        if (activeTab === 'models') {
            fetchModels();
            fetchModelComparison();
            fetchAvailableModels();
        }
    }, [activeTab]);

    const checkApiHealth = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/health`);
            setApiStatus({ healthy: true, loading: false, data: response.data });
        } catch (err) {
            setApiStatus({ healthy: false, loading: false, error: err.message });
        }
    };

    const fetchAvailableModels = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/models/available`);
            setAvailableModels(response.data.available_models || []);
            setActiveModel(response.data.active_model);
        } catch (err) {
            console.error('Failed to fetch available models:', err);
        }
    };

    const handleSetActiveModel = async (modelId) => {
        setSettingActiveModel(modelId);
        try {
            await axios.post(`${API_BASE_URL}/models/active/${modelId}`);
            setActiveModel(modelId);
            setError(null);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to set active model');
        } finally {
            setSettingActiveModel(null);
        }
    };

    const handlePredict = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const url = selectedModelForPrediction
                ? `${API_BASE_URL}/predict?model_id=${encodeURIComponent(selectedModelForPrediction)}`
                : `${API_BASE_URL}/predict`;
            const response = await axios.post(url, customerData);
            setPrediction(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleTrain = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const response = await axios.post(`${API_BASE_URL}/train`, trainingConfig);
            setTrainingStatus(response.data);
            startPollingTrainingStatus();
        } catch (err) {
            setError(err.response?.data?.detail || err.message);
            setLoading(false);
        }
    };

    const startPollingTrainingStatus = () => {
        const interval = setInterval(async () => {
            try {
                const response = await axios.get(`${API_BASE_URL}/train/status`);
                setTrainingStatus(response.data);

                if (response.data.status === 'completed' || response.data.status === 'failed') {
                    clearInterval(interval);
                    setLoading(false);
                    if (response.data.status === 'completed') {
                        fetchModels();
                        fetchModelComparison();
                        fetchAvailableModels();
                    }
                }
            } catch (err) {
                console.log(err, err)
                clearInterval(interval);
                setLoading(false);
            }
        }, 2000);
    };

    const fetchModels = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/models`);
            setModels(response.data.models || []);
        } catch (err) {
            console.error('Failed to fetch models:', err);
        }
    };

    const fetchModelComparison = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/models/compare`);
            setModelComparison(response.data);
        } catch (err) {
            console.error('Failed to fetch model comparison:', err);
        }
    };

    const handleInputChange = (e) => {
        const { name, value, type } = e.target;
        setCustomerData(prev => ({
            ...prev,
            [name]: type === 'number' ? parseFloat(value) || 0 : value
        }));
    };

    const handleTrainingConfigChange = (e) => {
        const { name, value, type, checked } = e.target;
        setTrainingConfig(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : type === 'number' ? parseInt(value) : value
        }));
    };

    // Chart data for model comparison
    const getComparisonChartData = () => {
        if (!modelComparison?.comparison) return null;

        return {
            labels: modelComparison.comparison.map(m => m.model_name || m.Model),
            datasets: [{
                label: 'ROC AUC Score',
                data: modelComparison.comparison.map(m => m.val_roc_auc || m.Val_ROC_AUC || 0),
                backgroundColor: 'rgba(59, 130, 246, 0.6)',
                borderColor: 'rgba(59, 130, 246, 1)',
                borderWidth: 2
            }]
        };
    };

    const getPredictionChartData = () => {
        if (!prediction) return null;

        return {
            labels: ['No Churn', 'Churn'],
            datasets: [{
                data: [1 - prediction.churn_probability, prediction.churn_probability],
                backgroundColor: ['rgba(34, 197, 94, 0.6)', 'rgba(239, 68, 68, 0.6)'],
                borderColor: ['rgba(34, 197, 94, 1)', 'rgba(239, 68, 68, 1)'],
                borderWidth: 2
            }]
        };
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
                    <div className="flex items-center justify-between">
                        <div>
                            <h1 className="text-3xl font-bold text-gray-800 flex items-center gap-3">
                                <TrendingUp className="text-blue-600" size={36} />
                                Customer Churn Prediction
                            </h1>
                            <p className="text-gray-600 mt-2">Advanced ML-powered churn prediction and analytics</p>
                        </div>
                        <div className="flex items-center gap-4">
                            {activeModel && (
                                <div className="flex items-center gap-2 px-3 py-1 bg-green-50 border border-green-200 rounded-lg">
                                    <Zap size={16} className="text-green-600" />
                                    <span className="text-sm text-green-700 font-medium">Active: {activeModel}</span>
                                </div>
                            )}
                            <div className="flex items-center gap-2">
                                {apiStatus.loading ? (
                                    <Loader className="animate-spin text-gray-400" />
                                ) : apiStatus.healthy ? (
                                    <>
                                        <CheckCircle className="text-green-500" size={20} />
                                        <span className="text-sm text-green-600 font-medium">API Connected</span>
                                    </>
                                ) : (
                                    <>
                                        <XCircle className="text-red-500" size={20} />
                                        <span className="text-sm text-red-600 font-medium">API Disconnected</span>
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Tabs */}
                <div className="bg-white rounded-lg shadow-lg mb-6">
                    <div className="flex border-b">
                        <button
                            onClick={() => setActiveTab('predict')}
                            className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
                                activeTab === 'predict'
                                    ? 'border-b-2 border-blue-600 text-blue-600'
                                    : 'text-gray-600 hover:text-gray-800'
                            }`}
                        >
                            <Users size={20} />
                            Predict Churn
                        </button>
                        <button
                            onClick={() => setActiveTab('train')}
                            className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
                                activeTab === 'train'
                                    ? 'border-b-2 border-blue-600 text-blue-600'
                                    : 'text-gray-600 hover:text-gray-800'
                            }`}
                        >
                            <Settings size={20} />
                            Train Models
                        </button>
                        <button
                            onClick={() => setActiveTab('models')}
                            className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors ${
                                activeTab === 'models'
                                    ? 'border-b-2 border-blue-600 text-blue-600'
                                    : 'text-gray-600 hover:text-gray-800'
                            }`}
                        >
                            <BarChart3 size={20} />
                            Model Analytics
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="p-6">
                        {error && (
                            <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg flex items-start gap-2">
                                <AlertCircle size={20} className="flex-shrink-0 mt-0.5" />
                                <div>
                                    <p className="font-medium">Error</p>
                                    <p className="text-sm">{error}</p>
                                </div>
                            </div>
                        )}

                        {/* Predict Tab */}
                        {activeTab === 'predict' && (
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                <div>
                                    <h3 className="text-xl font-semibold mb-4 text-gray-800">Customer Information</h3>
                                    <form onSubmit={handlePredict} className="space-y-4">
                                        {/* Model Selection */}
                                        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                            <label className="block text-sm font-medium text-blue-800 mb-2">
                                                Select Model for Prediction
                                            </label>
                                            <select
                                                value={selectedModelForPrediction}
                                                onChange={(e) => setSelectedModelForPrediction(e.target.value)}
                                                className="w-full px-3 py-2 border border-blue-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                                            >
                                                <option value="">
                                                    {activeModel ? `Use Active Model (${activeModel})` : 'Use Best Available Model'}
                                                </option>
                                                {availableModels.map((modelId) => (
                                                    <option key={modelId} value={modelId}>
                                                        {modelId} {modelId === activeModel ? '(Active)' : ''}
                                                    </option>
                                                ))}
                                            </select>
                                            <p className="text-xs text-blue-600 mt-1">
                                                {availableModels.length} model(s) available
                                            </p>
                                        </div>

                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Gender</label>
                                                <select
                                                    name="gender"
                                                    value={customerData.gender}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                >
                                                    <option>Male</option>
                                                    <option>Female</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Senior Citizen</label>
                                                <select
                                                    name="SeniorCitizen"
                                                    value={customerData.SeniorCitizen}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                >
                                                    <option value={0}>No</option>
                                                    <option value={1}>Yes</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Partner</label>
                                                <select
                                                    name="Partner"
                                                    value={customerData.Partner}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                >
                                                    <option>Yes</option>
                                                    <option>No</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Dependents</label>
                                                <select
                                                    name="Dependents"
                                                    value={customerData.Dependents}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                >
                                                    <option>Yes</option>
                                                    <option>No</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Tenure (months)</label>
                                                <input
                                                    type="number"
                                                    name="tenure"
                                                    value={customerData.tenure}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Contract</label>
                                                <select
                                                    name="Contract"
                                                    value={customerData.Contract}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                >
                                                    <option>Month-to-month</option>
                                                    <option>One year</option>
                                                    <option>Two year</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Internet Service</label>
                                                <select
                                                    name="InternetService"
                                                    value={customerData.InternetService}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                >
                                                    <option>DSL</option>
                                                    <option>Fiber optic</option>
                                                    <option>No</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Payment Method</label>
                                                <select
                                                    name="PaymentMethod"
                                                    value={customerData.PaymentMethod}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                >
                                                    <option>Electronic check</option>
                                                    <option>Mailed check</option>
                                                    <option>Bank transfer (automatic)</option>
                                                    <option>Credit card (automatic)</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Monthly Charges ($)</label>
                                                <input
                                                    type="number"
                                                    step="0.01"
                                                    name="MonthlyCharges"
                                                    value={customerData.MonthlyCharges}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Total Charges ($)</label>
                                                <input
                                                    type="number"
                                                    step="0.01"
                                                    name="TotalCharges"
                                                    value={customerData.TotalCharges}
                                                    onChange={handleInputChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                                />
                                            </div>
                                        </div>
                                        <button
                                            type="submit"
                                            disabled={loading}
                                            className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:bg-gray-400 flex items-center justify-center gap-2"
                                        >
                                            {loading ? <Loader className="animate-spin" size={20} /> : <Play size={20} />}
                                            {loading ? 'Predicting...' : 'Predict Churn'}
                                        </button>
                                    </form>
                                </div>

                                <div>
                                    <h3 className="text-xl font-semibold mb-4 text-gray-800">Prediction Results</h3>
                                    {prediction ? (
                                        <div className="space-y-4">
                                            <div className={`p-6 rounded-lg ${prediction.churn ? 'bg-red-50 border-2 border-red-200' : 'bg-green-50 border-2 border-green-200'}`}>
                                                <div className="flex items-center justify-between mb-4">
                                                    <h4 className="text-lg font-bold">
                                                        {prediction.churn ? '!! High Churn Risk' : 'Low Churn Risk'}
                                                    </h4>
                                                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                                                        prediction.confidence === 'High' ? 'bg-blue-100 text-blue-700' :
                                                            prediction.confidence === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                                                                'bg-gray-100 text-gray-700'
                                                    }`}>
                                                        {prediction.confidence} Confidence
                                                    </span>
                                                </div>
                                                <p className="text-2xl font-bold mb-2">
                                                    {(prediction.churn_probability * 100).toFixed(1)}% Churn Probability
                                                </p>
                                                <p className="text-sm text-gray-600">Model: {prediction.model_version}</p>
                                            </div>

                                            {prediction.risk_factors.length > 0 && (
                                                <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
                                                    <h4 className="font-bold mb-3 text-gray-800">Risk Factors Identified</h4>
                                                    <ul className="space-y-2">
                                                        {prediction.risk_factors.map((factor, idx) => (
                                                            <li key={idx} className="flex items-start gap-2">
                                                                <AlertCircle size={16} className="text-orange-500 mt-0.5 flex-shrink-0" />
                                                                <span className="text-gray-700">{factor}</span>
                                                            </li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}

                                            {getPredictionChartData() && (
                                                <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
                                                    <h4 className="font-bold mb-4 text-gray-800">Probability Distribution</h4>
                                                    <div className="h-64">
                                                        <Doughnut
                                                            data={getPredictionChartData()}
                                                            options={{
                                                                responsive: true,
                                                                maintainAspectRatio: false,
                                                                plugins: {
                                                                    legend: { position: 'bottom' }
                                                                }
                                                            }}
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
                                            <Users size={48} className="mx-auto text-gray-400 mb-4" />
                                            <p className="text-gray-600">Enter customer data and click predict to see results</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Train Tab */}
                        {activeTab === 'train' && (
                            <div className="max-w-4xl mx-auto">
                                <h3 className="text-xl font-semibold mb-4 text-gray-800">Training Configuration</h3>
                                <form onSubmit={handleTrain} className="space-y-6">
                                    <div className="bg-gray-50 p-6 rounded-lg space-y-4">
                                        <h4 className="font-semibold text-gray-800">Preprocessing Options</h4>
                                        <div className="grid grid-cols-2 gap-4">
                                            <label className="flex items-center gap-2">
                                                <input
                                                    type="checkbox"
                                                    name="advanced_preprocessing"
                                                    checked={trainingConfig.advanced_preprocessing}
                                                    onChange={handleTrainingConfigChange}
                                                    className="w-4 h-4 text-blue-600"
                                                />
                                                <span className="text-sm">Advanced Preprocessing</span>
                                            </label>
                                            <label className="flex items-center gap-2">
                                                <input
                                                    type="checkbox"
                                                    name="feature_engineering"
                                                    checked={trainingConfig.feature_engineering}
                                                    onChange={handleTrainingConfigChange}
                                                    className="w-4 h-4 text-blue-600"
                                                />
                                                <span className="text-sm">Feature Engineering</span>
                                            </label>
                                            <label className="flex items-center gap-2">
                                                <input
                                                    type="checkbox"
                                                    name="handle_imbalance"
                                                    checked={trainingConfig.handle_imbalance}
                                                    onChange={handleTrainingConfigChange}
                                                    className="w-4 h-4 text-blue-600"
                                                />
                                                <span className="text-sm">Handle Imbalance</span>
                                            </label>
                                            <label className="flex items-center gap-2">
                                                <input
                                                    type="checkbox"
                                                    name="use_ensemble"
                                                    checked={trainingConfig.use_ensemble}
                                                    onChange={handleTrainingConfigChange}
                                                    className="w-4 h-4 text-blue-600"
                                                />
                                                <span className="text-sm">Use Ensemble</span>
                                            </label>
                                        </div>
                                    </div>

                                    <div className="bg-gray-50 p-6 rounded-lg space-y-4">
                                        <h4 className="font-semibold text-gray-800">Optimization Settings</h4>
                                        <div className="grid grid-cols-2 gap-4">
                                            <label className="flex items-center gap-2">
                                                <input
                                                    type="checkbox"
                                                    name="optimize_hyperparameters"
                                                    checked={trainingConfig.optimize_hyperparameters}
                                                    onChange={handleTrainingConfigChange}
                                                    className="w-4 h-4 text-blue-600"
                                                />
                                                <span className="text-sm">Optimize Hyperparameters</span>
                                            </label>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Search Type</label>
                                                <select
                                                    name="search_type"
                                                    value={trainingConfig.search_type}
                                                    onChange={handleTrainingConfigChange}
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                                                >
                                                    <option value="random">Random Search</option>
                                                    <option value="grid">Grid Search</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label className="block text-sm font-medium text-gray-700 mb-1">Iterations</label>
                                                <input
                                                    type="number"
                                                    name="n_iter"
                                                    value={trainingConfig.n_iter}
                                                    onChange={handleTrainingConfigChange}
                                                    min="10"
                                                    max="100"
                                                    className="w-full px-3 py-2 border border-gray-300 rounded-lg"
                                                />
                                            </div>
                                        </div>
                                    </div>

                                    <button
                                        type="submit"
                                        disabled={loading}
                                        className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:bg-gray-400 flex items-center justify-center gap-2"
                                    >
                                        {loading ? <Loader className="animate-spin" size={20} /> : <Play size={20} />}
                                        {loading ? 'Training in Progress...' : 'Start Training'}
                                    </button>
                                </form>

                                {trainingStatus && (
                                    <div className="mt-6 bg-white p-6 rounded-lg border-2 border-gray-200">
                                        <h4 className="font-bold mb-4 text-gray-800">Training Status</h4>
                                        <div className="space-y-3">
                                            <div className="flex items-center justify-between">
                                                <span className="text-sm font-medium">Status:</span>
                                                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                                                    trainingStatus.status === 'completed' ? 'bg-green-100 text-green-700' :
                                                        trainingStatus.status === 'failed' ? 'bg-red-100 text-red-700' :
                                                            trainingStatus.status === 'running' ? 'bg-blue-100 text-blue-700' :
                                                                'bg-gray-100 text-gray-700'
                                                }`}>
                                                    {trainingStatus.status}
                                                </span>
                                            </div>
                                            {trainingStatus.progress !== undefined && (
                                                <div>
                                                    <div className="flex justify-between text-sm mb-1">
                                                        <span>Progress</span>
                                                        <span>{trainingStatus.progress}%</span>
                                                    </div>
                                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                                        <div
                                                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                                            style={{ width: `${trainingStatus.progress}%` }}
                                                        />
                                                    </div>
                                                </div>
                                            )}
                                            <p className="text-sm text-gray-600">{trainingStatus.message}</p>
                                            {trainingStatus.best_model && (
                                                <div className="pt-3 border-t">
                                                    <p className="text-sm"><strong>Best Model:</strong> {trainingStatus.best_model}</p>
                                                    <p className="text-sm"><strong>Best Score:</strong> {trainingStatus.best_score?.toFixed(4)}</p>
                                                    {trainingStatus.training_time && (
                                                        <p className="text-sm"><strong>Training Time:</strong> {trainingStatus.training_time.toFixed(2)}s</p>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Models Tab */}
                        {activeTab === 'models' && (
                            <div className="space-y-6">
                                {/* Active Model Banner */}
                                {activeModel && (
                                    <div className="bg-green-50 border-2 border-green-200 rounded-lg p-4 flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <Zap className="text-green-600" size={24} />
                                            <div>
                                                <p className="font-semibold text-green-800">Active Model</p>
                                                <p className="text-sm text-green-600">{activeModel}</p>
                                            </div>
                                        </div>
                                        <span className="px-3 py-1 bg-green-200 text-green-800 rounded-full text-sm font-medium">
                                            In Use
                                        </span>
                                    </div>
                                )}

                                <div>
                                    <h3 className="text-xl font-semibold mb-4 text-gray-800">Model Comparison</h3>
                                    {getComparisonChartData() ? (
                                        <div className="bg-white p-6 rounded-lg border-2 border-gray-200">
                                            <div className="h-96">
                                                <Bar
                                                    data={getComparisonChartData()}
                                                    options={{
                                                        responsive: true,
                                                        maintainAspectRatio: false,
                                                        plugins: {
                                                            legend: { display: false },
                                                            title: {
                                                                display: true,
                                                                text: 'Model Performance Comparison (ROC AUC)'
                                                            }
                                                        },
                                                        scales: {
                                                            y: {
                                                                beginAtZero: true,
                                                                max: 1
                                                            }
                                                        }
                                                    }}
                                                />
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
                                            <BarChart3 size={48} className="mx-auto text-gray-400 mb-4" />
                                            <p className="text-gray-600">No models available. Train models to see comparison.</p>
                                        </div>
                                    )}
                                </div>

                                {modelComparison?.summary && (
                                    <div className="grid grid-cols-3 gap-4">
                                        <div className="bg-blue-50 p-6 rounded-lg border-2 border-blue-200">
                                            <h4 className="text-sm font-medium text-blue-600 mb-2">Total Models</h4>
                                            <p className="text-3xl font-bold text-blue-900">{modelComparison.summary.total_models}</p>
                                        </div>
                                        <div className="bg-green-50 p-6 rounded-lg border-2 border-green-200">
                                            <h4 className="text-sm font-medium text-green-600 mb-2">Best Model</h4>
                                            <p className="text-lg font-bold text-green-900">{modelComparison.summary.best_model}</p>
                                        </div>
                                        <div className="bg-purple-50 p-6 rounded-lg border-2 border-purple-200">
                                            <h4 className="text-sm font-medium text-purple-600 mb-2">Best Score</h4>
                                            <p className="text-3xl font-bold text-purple-900">{modelComparison.summary.best_score?.toFixed(4)}</p>
                                        </div>
                                    </div>
                                )}

                                {models.length > 0 && (
                                    <div>
                                        <h3 className="text-xl font-semibold mb-4 text-gray-800">Model Details</h3>
                                        <div className="bg-white rounded-lg border-2 border-gray-200 overflow-hidden">
                                            <div className="overflow-x-auto">
                                                <table className="w-full">
                                                    <thead className="bg-gray-50">
                                                    <tr>
                                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Model ID
                                                        </th>
                                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Type
                                                        </th>
                                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            ROC AUC
                                                        </th>
                                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Accuracy
                                                        </th>
                                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Registered
                                                        </th>
                                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                                            Actions
                                                        </th>
                                                    </tr>
                                                    </thead>
                                                    <tbody className="bg-white divide-y divide-gray-200">
                                                    {models.map((model, idx) => (
                                                        <tr key={idx} className={`hover:bg-gray-50 transition-colors ${model.model_id === activeModel ? 'bg-green-50' : ''}`}>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className="flex items-center gap-2">
                                                                    <span className="text-sm font-medium text-gray-900">
                                                                        {model.model_id || 'N/A'}
                                                                    </span>
                                                                    {model.model_id === activeModel && (
                                                                        <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                                                                            Active
                                                                        </span>
                                                                    )}
                                                                </div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                                                    {model.model_name || 'N/A'}
                                                                </span>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className="text-sm text-gray-900">
                                                                    {(model.val_roc_auc || model.Val_ROC_AUC || 0).toFixed(4)}
                                                                </div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                <div className="text-sm text-gray-900">
                                                                    {(model.val_accuracy || model.Val_Accuracy || 0).toFixed(4)}
                                                                </div>
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                                {model.registered_at ? new Date(model.registered_at).toLocaleDateString() : 'N/A'}
                                                            </td>
                                                            <td className="px-6 py-4 whitespace-nowrap">
                                                                {model.model_id !== activeModel ? (
                                                                    <button
                                                                        onClick={() => handleSetActiveModel(model.model_id)}
                                                                        disabled={settingActiveModel === model.model_id}
                                                                        className="inline-flex items-center gap-1 px-3 py-1 text-sm font-medium text-blue-600 hover:text-blue-800 hover:bg-blue-50 rounded transition-colors disabled:opacity-50"
                                                                    >
                                                                        {settingActiveModel === model.model_id ? (
                                                                            <Loader className="animate-spin" size={14} />
                                                                        ) : (
                                                                            <Zap size={14} />
                                                                        )}
                                                                        Set Active
                                                                    </button>
                                                                ) : (
                                                                    <span className="text-sm text-green-600 font-medium">Currently Active</span>
                                                                )}
                                                            </td>
                                                        </tr>
                                                    ))}
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {modelComparison?.comparison && modelComparison.comparison.length > 0 && (
                                    <div>
                                        <h3 className="text-xl font-semibold mb-4 text-gray-800">Detailed Metrics</h3>
                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                            {modelComparison.comparison.slice(0, 6).map((model, idx) => (
                                                <div key={idx} className="bg-white p-6 rounded-lg border-2 border-gray-200 hover:shadow-lg transition-shadow">
                                                    <h4 className="font-bold text-lg mb-3 text-gray-800">
                                                        {model.model_name || model.Model}
                                                    </h4>
                                                    <div className="space-y-2">
                                                        <div className="flex justify-between items-center">
                                                            <span className="text-sm text-gray-600">ROC AUC</span>
                                                            <span className="font-semibold text-blue-600">
                                                                {(model.val_roc_auc || model.Val_ROC_AUC || 0).toFixed(4)}
                                                            </span>
                                                        </div>
                                                        <div className="flex justify-between items-center">
                                                            <span className="text-sm text-gray-600">Accuracy</span>
                                                            <span className="font-semibold text-green-600">
                                                                {(model.val_accuracy || model.Val_Accuracy || 0).toFixed(4)}
                                                            </span>
                                                        </div>
                                                        <div className="flex justify-between items-center">
                                                            <span className="text-sm text-gray-600">Precision</span>
                                                            <span className="font-semibold text-purple-600">
                                                                {(model.val_precision || model.Val_Precision || 0).toFixed(4)}
                                                            </span>
                                                        </div>
                                                        <div className="flex justify-between items-center">
                                                            <span className="text-sm text-gray-600">Recall</span>
                                                            <span className="font-semibold text-orange-600">
                                                                {(model.val_recall || model.Val_Recall || 0).toFixed(4)}
                                                            </span>
                                                        </div>
                                                        {idx === 0 && (
                                                            <div className="mt-3 pt-3 border-t">
                                                                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                                                                    Best Model
                                                                </span>
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {models.length === 0 && !modelComparison && (
                                    <div className="bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
                                        <BarChart3 size={48} className="mx-auto text-gray-400 mb-4" />
                                        <p className="text-gray-600 text-lg font-medium mb-2">No Models Available</p>
                                        <p className="text-gray-500 text-sm mb-4">
                                            Train your first model to see analytics and comparisons
                                        </p>
                                        <button
                                            onClick={() => setActiveTab('train')}
                                            className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
                                        >
                                            <Settings size={20} />
                                            Go to Training
                                        </button>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="bg-white rounded-lg shadow-lg p-4 text-center text-sm text-gray-600">
                    <p>Advanced Customer Churn Prediction API v2.0.0 | Powered by Machine Learning</p>
                    <p className="mt-1">
                        {apiStatus.healthy ? (
                            <span className="text-green-600">All systems operational</span>
                        ) : (
                            <span className="text-red-600">API connection issues</span>
                        )}
                        {activeModel && (
                            <span className="ml-3 text-blue-600">| Active Model: {activeModel}</span>
                        )}
                    </p>
                </div>
            </div>
        </div>
    );
}