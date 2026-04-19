import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';
import { finalizePackage, AdrlLensResult } from './src/lib/adrlLens';

const PIPELINE = [
    { id: 'INPUT', label: 'ВХІД' },
    { id: 'DEEP_RESEARCH', label: 'ГЛИБОКИЙ ПОШУК' },
    { id: 'CONTEXT', label: 'КОНТЕКСТ' },
    { id: 'TASK_SPLIT', label: 'ДЕКОМПОЗИЦІЯ' },
    { id: 'STRATEGY', label: 'СТРАТЕГІЯ' },
    { id: 'SIMULATION', label: 'СИМУЛЯЦІЯ' },
    { id: 'CRITIC', label: 'КРИТИКА' },
    { id: 'GATE', label: 'РІШЕННЯ (ADPE)' },
    { id: 'VERDICT', label: 'ВЕРДИКТ' }
];

const ADPE_CONFIG = {
    baseThreshold: 0.55,
    minThreshold: 0.25,
    baseTemperature: 0.4,
    minTemperature: 0.1,
    biasPower: 1.25,
    learningRate: 0.06,
    historyDecay: 0.12
};

class DecisionKernelV53 {
    static normalizeWeights(weights: Record<string, number>) {
        const sum = Object.values(weights).reduce((acc: number, val: number) => acc + Math.abs(val), 0);
        const normalized: Record<string, number> = {};
        for (const key in weights) normalized[key] = weights[key] / (sum || 1);
        return normalized;
    }

    static adaptWeights(baseWeights: Record<string, number>, history: any[]) {
        const adapted = { ...baseWeights };
        history.forEach((entry, index) => {
            const opt = entry.failedOption;
            const age: number = history.length - 1 - index;
            const decay = Math.exp(-age * ADPE_CONFIG.historyDecay);
            const lr = ADPE_CONFIG.learningRate * decay;
            if (opt.risk > 0.5) {
                adapted.risk *= (1 + lr);
                adapted.impact *= (1 - lr * 0.4);
            }
            if (opt.feasibility < 0.5) {
                adapted.feasibility *= (1 + lr);
                adapted.cost *= (1 + lr * 0.3);
            }
        });
        return this.normalizeWeights(adapted);
    }

    static calculateScore(option: any, weights: Record<string, number>) {
        const r = option.risk;
        const b = {
            ...option,
            risk: r + (1 - r) * 0.22,
            impact: option.impact * (1 - r * 0.3),
            feasibility: option.feasibility * (1 - r * 0.2)
        };
        const roiScore = Object.entries(weights).reduce((acc: number, [k, w]: [string, number]) => {
            const val = b[k] || 0;
            return acc + Math.pow(val, ADPE_CONFIG.biasPower) * w;
        }, 0);
        const stability = (b.feasibility * (1 - b.risk)) * (1 - (b.cost * 0.45 + b.time * 0.25));
        return {
            ...b,
            score: (roiScore * 0.6) + (stability * 0.4),
            stability: stability
        };
    }

    static probabilisticSelect(ranked: any[], iteration: number) {
        if (!ranked.length) return null;
        const temp = Math.max(ADPE_CONFIG.minTemperature, ADPE_CONFIG.baseTemperature - (iteration * 0.1));
        const maxScore = Math.max(...ranked.map(r => r.score));
        const exps = ranked.map(r => Math.exp((r.score - maxScore) / temp));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const probs = exps.map(e => e / sumExps);
        let rand = Math.random();
        for (let i = 0; i < probs.length; i++) {
            rand -= probs[i];
            if (rand <= 0) return { selection: ranked[i], probabilities: probs };
        }
        return { selection: ranked[0], probabilities: probs };
    }

    static computeConfidence(best: any, all: any[]) {
        if (all.length < 2) return best.stability;
        const gap = best.score - all[1].score;
        const gapFactor = Math.pow(gap * 5, 0.7);
        const confidence = Math.min(1, Math.max(0, (gapFactor * 0.7) + (best.stability * 0.3)));
        return confidence;
    }
}

type LogData = {
    reflection?: string;
    custom_payload?: string;
    strategy_A?: string;
    strategy_B?: string;
    metrics_A?: any;
    metrics_B?: any;
    cross_analysis?: string;
    fundamental_blocker?: boolean;
    research_findings?: string;
    summary: string;
    confidence: number;
    decision?: 'ACCEPT' | 'REPAIR' | 'RETHINK';
    suggestions?: string[];
    adpe_metrics?: any;
    implementation_plan?: { step: string; action: string; tools?: string }[];
};

type LogEntry = {
    stageId: string;
    data: LogData;
    isLocal?: boolean;
    iteration?: number;
};

export default function App() {
    const [input, setInput] = useState(() => localStorage.getItem('SAV_INPUT') || 'МЕТА: Заробити 500 грн за 2 години.\nОБМЕЖЕННЯ: Тільки телефон, без вкладень.\nМОВА: Українська.');
    const [isRunning, setIsRunning] = useState(false);
    const [statusText, setStatusText] = useState(() => {
        const verdict = localStorage.getItem('SAV_VERDICT');
        const logsData = JSON.parse(localStorage.getItem('SAV_LOGS') || '[]');
        const hasStarted = logsData.length > 0;
        if (hasStarted && !verdict) return 'ПАУЗА / ЗБІЙ';
        return 'ОЧІКУВАННЯ';
    });
    const [logs, setLogs] = useState<LogEntry[]>(() => JSON.parse(localStorage.getItem('SAV_LOGS') || '[]'));
    const [verdict, setVerdict] = useState<string | null>(() => localStorage.getItem('SAV_VERDICT'));
    const [loops, setLoops] = useState(() => Number(localStorage.getItem('SAV_LOOPS') || '0'));
    const [graveyardCount, setGraveyardCount] = useState(() => Number(localStorage.getItem('SAV_GRAVEYARD_COUNT') || '0'));
    const [iterationCount, setIterationCount] = useState(() => Number(localStorage.getItem('SAV_ITERATION_COUNT') || '1'));
    const [currentCursor, setCurrentCursor] = useState(() => Number(localStorage.getItem('SAV_CURSOR') || '0'));
    const [currentModelIndex, setCurrentModelIndex] = useState(() => Number(localStorage.getItem('SAV_MODEL_INDEX') || '0'));
    const [memory, setMemory] = useState<any>(() => JSON.parse(localStorage.getItem('SAV_MEMORY') || 'null'));

    const [nodeStatuses, setNodeStatuses] = useState<Record<string, string>>(() => JSON.parse(localStorage.getItem('SAV_NODE_STATUSES') || '{}'));
    const [lensResult, setLensResult] = useState<AdrlLensResult | null>(() => JSON.parse(localStorage.getItem('SAV_LENS_RESULT') || 'null'));

    const logsEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        localStorage.setItem('SAV_INPUT', input);
        localStorage.setItem('SAV_LOGS', JSON.stringify(logs));
        localStorage.setItem('SAV_LOOPS', loops.toString());
        localStorage.setItem('SAV_GRAVEYARD_COUNT', graveyardCount.toString());
        localStorage.setItem('SAV_ITERATION_COUNT', iterationCount.toString());
        localStorage.setItem('SAV_NODE_STATUSES', JSON.stringify(nodeStatuses));
        localStorage.setItem('SAV_CURSOR', currentCursor.toString());
        localStorage.setItem('SAV_MODEL_INDEX', currentModelIndex.toString());
        localStorage.setItem('SAV_MEMORY', JSON.stringify(memory));
        localStorage.setItem('SAV_LENS_RESULT', JSON.stringify(lensResult));
        if (verdict) localStorage.setItem('SAV_VERDICT', verdict);
        else localStorage.removeItem('SAV_VERDICT');
    }, [input, logs, verdict, loops, graveyardCount, iterationCount, nodeStatuses, currentCursor, currentModelIndex, memory, lensResult]);

    const scrollToBottom = () => {
        logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [logs, verdict]);

    const startCycle = async (resume = false) => {
        if (isRunning || !input.trim()) return;

        const isFollowUp = logs.length > 0 && verdict !== null;
        const currentIteration = (isFollowUp && !resume) ? iterationCount + 1 : iterationCount;

        setIsRunning(true);
        if (!isFollowUp && !resume) {
            setLogs([]);
            setLoops(0);
            setGraveyardCount(0);
            setIterationCount(1);
            setCurrentCursor(0);
        } else if (isFollowUp && !resume) {
            setIterationCount(currentIteration);
            setLogs(prev => [...prev, {
                stageId: 'SYSTEM',
                data: { summary: `🔄 ЗАПУСК ІТЕРАЦІЇ №${currentIteration}. Аналіз нових вводок у контексті попередніх результатів...`, confidence: 100 },
                isLocal: true,
                iteration: currentIteration
            }]);
            setCurrentCursor(0);
        }

        setVerdict(null);
        setStatusText('АНАЛІЗУЮ');

        let localCursor = resume ? currentCursor : 0;
        let localModelIndex = currentModelIndex;
        let currentLoops = loops;
        const maxLoops = 3;

        const localMemory = (resume && memory) ? memory : {
            input,
            history: isFollowUp ? logs.filter(l => !l.isLocal).map(l => ({ stage: l.stageId, summary: l.data.summary })) : [],
            adpeHistory: [],
            graveyard: [],
            lastPayload: isFollowUp ? `Продовження аналізу. Попередній вердикт: ${verdict}. Нова команда: ${input}` : 'Ініціалізація. Сфокусуйся на базовому розумінні задачі та виявленні прихованих обмежень.'
        };

        setMemory(localMemory);

        while (localCursor < PIPELINE.length) {
            const stage = PIPELINE[localCursor];
            setCurrentCursor(localCursor);
            setNodeStatuses(prev => ({ ...prev, [stage.id]: 'active' }));

            let systemExtra = `\n[ДИНАМІЧНИЙ МОСТ ВІД ПОПЕРЕДНЬОГО КРОКУ]: ${localMemory.lastPayload}\n`;
            if (localMemory.graveyard.length > 0) {
                systemExtra += `\n[КЛАДОВИЩЕ ГІПОТЕЗ (ЗАБОРОНЕНО ДО ВИКОРИСТАННЯ)]: ${JSON.stringify(localMemory.graveyard)}\n`;
            }

            let promptExtra = '';
            const schemaProps: any = {
                reflection: { type: Type.STRING, description: 'Мікро-цикл рефлексії: аналіз Delta між очікуванням і реальністю попереднього кроку' },
                custom_payload: { type: Type.STRING, description: 'Custom Context Payload: унікальна інструкція-фокус для НАСТУПНОГО етапу' },
                summary: { type: Type.STRING, description: 'Основний висновок поточного етапу' },
                confidence: { type: Type.NUMBER, description: 'Рівень впевненості у результаті (0-100). Будь об\'єктивним, але якщо схема робоча - став високий бал.' },
                decision: { type: Type.STRING, enum: ['ACCEPT', 'REPAIR', 'RETHINK'] }
            };

            if (stage.id === 'INPUT') {
                promptExtra = 'Проаналізуй вхідний запит. Визнач основну мету, явні та приховані обмеження. Встанови тональність аналізу.';
            } else if (stage.id === 'CONTEXT') {
                promptExtra = 'Збери контекстуальну інформацію. Які зовнішні фактори впливають на задачу? Які знання необхідні для її вирішення?';
            } else if (stage.id === 'DEEP_RESEARCH') {
                schemaProps.research_findings = { type: Type.STRING, description: 'Детальні результати глибокого дослідження: приховані ризики, неочевидні можливості, технічні нюанси, актуальні дані з мережі.' };
                promptExtra = 'Використовуй Google Search для збору найактуальнішої інформації з інтернету. Знайди реальні ціни, тренди, конкурентів або технічні вимоги, що стосуються запиту. Твої висновки мають базуватися на фактах.';
            } else if (stage.id === 'TASK_SPLIT') {
                promptExtra = 'Розбий головну задачу на логічні підзадачі. Створи ієрархію кроків для досягнення мети.';
            } else if (stage.id === 'STRATEGY') {
                const metricsSchema = {
                    type: Type.OBJECT,
                    properties: {
                        impact: { type: Type.NUMBER, description: 'Вплив (0-1)' },
                        feasibility: { type: Type.NUMBER, description: 'Здійсненність (0-1)' },
                        risk: { type: Type.NUMBER, description: 'Ризик (0-1)' },
                        cost: { type: Type.NUMBER, description: 'Вартість ресурсів (0-1)' },
                        time: { type: Type.NUMBER, description: 'Часові витрати (0-1)' }
                    },
                    required: ['impact', 'feasibility', 'risk', 'cost', 'time']
                };
                schemaProps.strategy_A = { type: Type.STRING, description: 'Ветка А: Консервативна/Пряма стратегія' };
                schemaProps.metrics_A = metricsSchema;
                schemaProps.strategy_B = { type: Type.STRING, description: 'Ветка Б: Радикальна/Альтернативна стратегія' };
                schemaProps.metrics_B = metricsSchema;
                promptExtra = 'Застосуй Multi-Path Reasoning. Згенеруй дві незалежні гілки стратегій та оціни їх за метриками (0-1).';
            } else if (stage.id === 'SIMULATION' || stage.id === 'CRITIC') {
                promptExtra = 'Обов\'язково проаналізуй ОБИДВІ гілки (А і Б), що були згенеровані на етапі STRATEGY.';
            } else if (stage.id === 'GATE') {
                schemaProps.cross_analysis = { type: Type.STRING, description: 'Cross-Analysis гілок А і Б. Пошук точок дотику та фундаментальних блокерів.' };
                schemaProps.fundamental_blocker = { type: Type.BOOLEAN, description: 'True, якщо обидві гілки буксують на одному обмеженні' };
                promptExtra = 'Виконай Cross-Analysis. Якщо є фундаментальний блокер - decision має бути RETHINK.';
            } else if (stage.id === 'VERDICT') {
                schemaProps.implementation_plan = {
                    type: Type.ARRAY,
                    items: {
                        type: Type.OBJECT,
                        properties: {
                            step: { type: Type.STRING, description: 'Назва етапу/кроку' },
                            action: { type: Type.STRING, description: 'Конкретна дія' },
                            tools: { type: Type.STRING, description: 'Інструменти або технології' }
                        },
                        required: ['step', 'action']
                    },
                    description: 'Покрокова інструкція з реалізації обраної стратегії.'
                };
                schemaProps.suggestions = {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: '3-4 конкретні варіанти для поглиблення або розширення задачі на наступній ітерації. ПИШИ ВИКЛЮЧНО УКРАЇНСЬКОЮ.'
                };
                promptExtra = 'Сформулюй фінальний вердикт. Обов\'язково надай покрокову інструкцію (implementation_plan) для реалізації обраної стратегії. УСІ ТЕКСТИ МАЮТЬ БУТИ УКРАЇНСЬКОЮ МОВОЮ.';
            }

            setStatusText(stage.id === 'DEEP_RESEARCH' ? 'ГЛИБОКЕ ДОСЛІДЖЕННЯ' : 'ЛОКАЛЬНА ОБРОБКА');
            const localMsg = stage.id === 'DEEP_RESEARCH'
                ? `🔍 Запуск протоколів глибокого пошуку та верифікації даних для етапу [${stage.label}]...`
                : `⚙️ Локальна структуризація даних для етапу [${stage.label}]...`;

            setLogs(prev => [...prev, {
                stageId: 'SYSTEM',
                data: { summary: localMsg, confidence: 100 },
                isLocal: true,
                iteration: currentIteration
            }]);

            await new Promise(r => setTimeout(r, stage.id === 'DEEP_RESEARCH' ? 4000 : 2000));
            setStatusText('АНАЛІЗУЮ');

            let result: LogData;
            if (stage.id === 'GATE') {
                const baseWeights = { impact: 0.45, feasibility: 0.25, risk: -0.20, cost: -0.05, time: -0.05 };
                const adaptedWeights = DecisionKernelV53.adaptWeights(baseWeights, localMemory.adpeHistory);

                if (!localMemory.STRATEGY?.metrics_A || !localMemory.STRATEGY?.metrics_B) {
                    result = {
                        summary: 'ADPE не може виконати вибір: відсутні метрики стратегій A/B. Повертаюсь до етапу STRATEGY для повторної генерації.',
                        confidence: 0,
                        decision: 'RETHINK'
                    };
                    localMemory.lastPayload = 'Перегенеруй стратегії A/B з валідними метриками impact/feasibility/risk/cost/time у діапазоні 0..1.';
                    localMemory.graveyard.push('ADPE guard: відсутні метрики A/B, примусовий цикл RETHINK.');
                    setGraveyardCount(localMemory.graveyard.length);
                    localMemory.adpeHistory.push({
                        attempt: currentLoops + 1,
                        failed_strategy: 'N/A',
                        reason: result.summary,
                        failedOption: { impact: 0, feasibility: 0, risk: 1, cost: 1, time: 1 }
                    });
                    setMemory({ ...localMemory });
                    setLogs(prev => [...prev, { stageId: stage.id, data: result, iteration: currentIteration }]);
                    setNodeStatuses(prev => ({ ...prev, [stage.id]: 'rethink', STRATEGY: 'rethink', SIMULATION: '', CRITIC: '' }));
                    currentLoops++;
                    setLoops(currentLoops);
                    localCursor = 4; // STRATEGY
                    await new Promise(r => setTimeout(r, 800));
                    continue;
                }

                const hypotheses = [
                    { ...localMemory.STRATEGY.metrics_A, label: 'Ветка А' },
                    { ...localMemory.STRATEGY.metrics_B, label: 'Ветка Б' }
                ];

                const evaluated = hypotheses
                    .map(h => DecisionKernelV53.calculateScore(h, adaptedWeights))
                    .sort((a, b) => b.score - a.score);

                const adpeSelection: any = DecisionKernelV53.probabilisticSelect(evaluated, currentLoops);
                const selection = adpeSelection.selection;
                selection.finalConfidence = DecisionKernelV53.computeConfidence(selection, evaluated);

                const threshold = Math.max(
                    ADPE_CONFIG.minThreshold,
                    ADPE_CONFIG.baseThreshold * (1 - currentLoops * 0.12)
                );

                const isRethink = selection.score < threshold && currentLoops < maxLoops;

                result = {
                    summary: `ADPE Вибір: ${selection.label}. Скоринг: ${selection.score.toFixed(3)}. Впевненість: ${(selection.finalConfidence * 100).toFixed(1)}%. ${isRethink ? 'Потрібне переосмислення.' : 'Рішення прийнято.'}`,
                    confidence: Math.round(selection.finalConfidence * 100),
                    decision: isRethink ? 'RETHINK' : 'ACCEPT',
                    adpe_metrics: {
                        score: selection.score,
                        confidence: selection.finalConfidence,
                        iteration: currentLoops,
                        winner: selection.label,
                        probabilities: adpeSelection.probabilities
                    }
                };
            } else {
                try {
                    const aiResponse = await askAI(stage.id, localMemory, systemExtra, promptExtra, schemaProps, 5, 10000, currentIteration, localModelIndex);
                    result = aiResponse.result;
                    localModelIndex = aiResponse.workingModelIndex;
                    setCurrentModelIndex(localModelIndex);
                } catch (error: any) {
                    setLogs(prev => [...prev, { stageId: stage.id, data: { summary: `КРИТИЧНА ПОМИЛКА API: ${error?.message || 'Невідома помилка'}`, confidence: 0 }, iteration: currentIteration }]);
                    setVerdict('Зупинено через помилку API. Ви можете спробувати продовжити цикл.');
                    setIsRunning(false);
                    setStatusText('ПОМИЛКА API');
                    setNodeStatuses(prev => ({ ...prev, [stage.id]: 'rethink' }));
                    return;
                }
            }

            if (result.custom_payload) localMemory.lastPayload = result.custom_payload;
            localMemory[stage.id] = result;
            setMemory({ ...localMemory });

            setLogs(prev => [...prev, { stageId: stage.id, data: result, iteration: currentIteration }]);
            setNodeStatuses(prev => ({ ...prev, [stage.id]: 'passed' }));

            if (stage.id === 'GATE') {
                if ((result.decision === 'RETHINK' || result.fundamental_blocker) && currentLoops < maxLoops) {
                    currentLoops++;
                    setLoops(currentLoops);
                    const failedIdea = `Ветка А: ${localMemory.STRATEGY?.strategy_A?.substring(0, 50)}... | Ветка Б: ${localMemory.STRATEGY?.strategy_B?.substring(0, 50)}...`;
                    const failedOption = result.adpe_metrics?.winner === 'Ветка А' ? localMemory.STRATEGY.metrics_A : localMemory.STRATEGY.metrics_B;
                    localMemory.graveyard.push(`Спроба ${currentLoops}: ${failedIdea} -> ${result.summary}`);
                    setGraveyardCount(localMemory.graveyard.length);
                    localMemory.adpeHistory.push({ attempt: currentLoops, failed_strategy: failedIdea, reason: result.summary, failedOption });
                    localMemory.history.push({ attempt: currentLoops, failed_strategy: failedIdea, reason: result.summary });

                    setNodeStatuses(prev => ({ ...prev, STRATEGY: 'rethink', SIMULATION: '', CRITIC: '', GATE: '' }));
                    localCursor = 3; // STRATEGY
                    await new Promise(r => setTimeout(r, 800));
                    continue;
                }
            }

            localCursor++;
            await new Promise(r => setTimeout(r, 600));
        }

        setIsRunning(false);
        setStatusText('ГОТОВО');
        setVerdict(localMemory.VERDICT?.summary || 'Аналіз завершено.');
    };

    const askAI = async (stageId: string, memory: any, systemExtra: string, promptExtra: string, schemaProperties: any, retries = 5, delay = 10000, iteration = 1, modelIndex = currentModelIndex): Promise<{ result: LogData; workingModelIndex: number }> => {
        const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
        const models = ['gemini-3.1-flash-lite-preview', 'gemini-3-flash-preview', 'gemini-flash-latest', 'gemini-3.1-pro-preview'];
        const currentModel = models[modelIndex % models.length];

        const isFlash = currentModel.includes('flash');
        const isLite = currentModel.includes('lite');

        const system = `Ти — СИСТЕМНИЙ АРХІТЕКТОР SAV V4.9.1 (УКРАЇНСЬКА АДАПТАЦІЯ).
        ВІДПОВІДАЙ ВИКЛЮЧНО УКРАЇНСЬКОЮ МОВОЮ.
        Твоя мета — бути максимально точним та використовувати ВСЮ доступну інформацію з попередніх етапів.
        ${systemExtra}`;

        const prompt = `Етап: ${stageId}. Запит: ${memory.input}.
        Пам'ять: ${JSON.stringify({
            history: memory.history,
            graveyard: memory.graveyard,
            lastPayload: memory.lastPayload,
            research: memory.DEEP_RESEARCH?.research_findings || memory.DEEP_RESEARCH?.summary
        })}.
        ${promptExtra}`;

        try {
            const config: any = {
                systemInstruction: system,
                responseMimeType: 'application/json',
                responseSchema: {
                    type: Type.OBJECT,
                    properties: schemaProperties,
                    required: ['summary', 'confidence']
                },
                thinkingConfig: (isFlash || isLite) && stageId !== 'VERDICT' ? { thinkingLevel: ThinkingLevel.LOW } : undefined
            };

            if (stageId === 'DEEP_RESEARCH') {
                config.tools = [{ googleSearch: {} }];
            }

            const response = await ai.models.generateContent({
                model: currentModel,
                contents: prompt,
                config
            });

            const text = response.text;
            if (text) return { result: JSON.parse(text), workingModelIndex: modelIndex % models.length };
            throw new Error('Empty response');
        } catch (e: any) {
            const errorMessage = e?.message || String(e);
            const isQuotaError = errorMessage.includes('429') ||
                                 errorMessage.includes('quota') ||
                                 errorMessage.includes('RESOURCE_EXHAUSTED') ||
                                 (typeof e === 'object' && (e?.error?.code === 429 || e?.status === 'RESOURCE_EXHAUSTED'));

            if (isQuotaError) {
                if (retries > 0) {
                    const nextModelIndex = modelIndex + 1;
                    const nextModel = models[nextModelIndex % models.length];
                    const nextDelay = delay + (5 - retries) * 2000;

                    setLogs(prev => [...prev, {
                        stageId: 'SYSTEM',
                        data: {
                            summary: `⚠️ Квота ${currentModel} вичерпана. Перемикаюсь на ${nextModel}. Повтор через ${nextDelay / 1000}с... (Спроб: ${retries})`,
                            confidence: 0
                        },
                        isLocal: true,
                        iteration
                    }]);

                    await new Promise(r => setTimeout(r, nextDelay));
                    return askAI(stageId, memory, systemExtra, promptExtra, schemaProperties, retries - 1, delay, iteration, nextModelIndex);
                }
                throw new Error('Вичерпано ліміти всіх доступних моделей API. Будь ласка, зачекайте.');
            }
            return { result: { summary: "Помилка зв'язку. Відновлення...", confidence: 10, decision: 'REPAIR' }, workingModelIndex: modelIndex % models.length };
        }
    };

    const resetSession = () => {
        localStorage.clear();
        setLogs([]);
        setVerdict(null);
        setLensResult(null);
        setLoops(0);
        setGraveyardCount(0);
        setIterationCount(1);
        setCurrentCursor(0);
        setCurrentModelIndex(0);
        setMemory(null);
        setNodeStatuses({});
        setStatusText('ОЧІКУВАННЯ');
    };

    const applySuggestion = (s: string) => {
        setInput(prev => `${prev}\n\nДОДАТКОВО: ${s}`);
        window.scrollTo({ top: 0, behavior: 'smooth' });
        const el = document.getElementById('taskInput');
        if (el) {
            el.focus();
            el.classList.add('highlight-flash');
            setTimeout(() => el.classList.remove('highlight-flash'), 1000);
        }
    };

    const runLens = async () => {
        if (!verdict) return;
        const result = await finalizePackage(verdict);
        setLensResult(result);
    };

    return (
        <>
            <header>
                <h3 className="engine-title">SAV CORE V4.9.1 // UA COGNITIVE</h3>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                    <div style={{ fontSize: '0.55rem', padding: '3px 8px', border: '1px solid var(--accent)', color: 'var(--accent)', backgroundColor: 'rgba(0,255,255,0.05)', borderRadius: '2px' }}>
                        ІТЕРАЦІЯ: {iterationCount}
                    </div>
                    {graveyardCount > 0 && (
                        <div style={{ fontSize: '0.55rem', padding: '3px 8px', border: '1px solid var(--error)', color: 'var(--error)', backgroundColor: '#330000', borderRadius: '2px' }}>
                            КЛАДОВИЩЕ: {graveyardCount}
                        </div>
                    )}
                    <div id="status-chip">
                        {statusText}
                        {isRunning && <span className="loading-dots"></span>}
                    </div>
                </div>
            </header>

            <div id="pipeline-ui">
                {PIPELINE.map(p => (
                    <div key={p.id} id={`node-${p.id}`} className={`node ${nodeStatuses[p.id] || ''}`}>
                        {p.label}
                        <div className="node-progress-container">
                            <div className="node-progress-bar"></div>
                        </div>
                    </div>
                ))}
            </div>

            <main>
                <textarea
                    id="taskInput"
                    rows={4}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    disabled={isRunning}
                    placeholder="Введіть ваше завдання тут..."
                />

                <div id="log-stream">
                    {logs.map((log, idx) => {
                        const isIterationStart = log.stageId === 'SYSTEM' && log.data.summary.includes('ЗАПУСК ІТЕРАЦІЇ');

                        let decisionHtml = null;
                        if (log.data.decision) {
                            const cls = log.data.decision === 'RETHINK' ? 'badge-rethink' : (log.data.decision === 'REPAIR' ? 'badge-repair' : 'badge-accept');
                            const label = log.data.decision === 'RETHINK' ? 'ПЕРЕОСМИСЛЕННЯ' : (log.data.decision === 'REPAIR' ? 'КОРИГУВАННЯ' : 'ПРИЙНЯТО');
                            decisionHtml = <div className={`decision-badge ${cls}`}>{label}</div>;
                        }

                        return (
                            <React.Fragment key={idx}>
                                {isIterationStart && (
                                    <div className="iteration-divider">
                                        ІТЕРАЦІЯ №{log.iteration}
                                    </div>
                                )}
                                <div className={log.isLocal ? 'log-local' : 'log-entry'}>
                                    <div className="log-header" style={log.isLocal ? { color: '#888' } : {}}>
                                        <span>[{log.stageId}]</span>
                                        {!log.isLocal && (
                                            <span>
                                                ДОВІРА: {log.data.confidence > 1 ? Math.round(log.data.confidence) : Math.round(log.data.confidence * 100)}%
                                            </span>
                                        )}
                                    </div>

                                    {log.data.reflection && <div className="meta-block"><strong>[РЕФЛЕКСІЯ]:</strong> {log.data.reflection}</div>}
                                    {log.data.research_findings && (
                                        <div className="meta-block research-block">
                                            <strong>[РЕЗУЛЬТАТИ ДОСЛІДЖЕННЯ]:</strong> {log.data.research_findings}
                                        </div>
                                    )}
                                    {log.data.strategy_A && <div className="meta-block branch-a"><strong>[ВЕТКА А]:</strong> {log.data.strategy_A}</div>}
                                    {log.data.strategy_B && <div className="meta-block branch-b"><strong>[ВЕТКА Б]:</strong> {log.data.strategy_B}</div>}
                                    {log.data.cross_analysis && (
                                        <div className="meta-block cross-analysis">
                                            <strong>[CROSS-ANALYSIS]:</strong> {log.data.cross_analysis}
                                            {log.data.fundamental_blocker && <span style={{ color: 'var(--error)', display: 'block', marginTop: '5px' }}>[БЛОКЕР ЗНАЙДЕНО]</span>}
                                        </div>
                                    )}

                                    {log.data.adpe_metrics && (
                                        <div className="meta-block" style={{ borderLeft: '3px solid var(--accent)', background: 'rgba(0,255,255,0.03)' }}>
                                            <strong>[ADPE METRICS]:</strong> Score: {Number(log.data.adpe_metrics.score).toFixed(3)} | Confidence: {(log.data.adpe_metrics.confidence * 100).toFixed(1)}% | Winner: {log.data.adpe_metrics.winner}
                                        </div>
                                    )}
                                    <div style={{ fontSize: '0.8rem', color: '#eee', lineHeight: 1.5, whiteSpace: 'pre-wrap', marginTop: '12px' }}>
                                        {log.data.summary}
                                    </div>

                                    {log.data.custom_payload && <div className="meta-block dynamic-bridge"><strong>[DYNAMIC BRIDGE]:</strong> {log.data.custom_payload}</div>}
                                    {decisionHtml}
                                </div>
                            </React.Fragment>
                        );
                    })}
                    <div ref={logsEndRef} />
                </div>

                {verdict && (
                    <div id="verdict-view" style={{ display: 'block' }}>
                        <div style={{ color: 'var(--accent)', fontSize: '0.9rem', fontWeight: 'bold', marginBottom: '15px', borderBottom: '1px solid var(--accent-dim)', paddingBottom: '5px' }}>
                            🏆 ФІНАЛЬНЕ РІШЕННЯ (ЦИКЛІВ: {loops})
                        </div>
                        <div style={{ fontSize: '0.85rem', color: '#fff', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                            {verdict}
                        </div>

                        {logs[logs.length - 1]?.data?.implementation_plan && (
                            <div style={{ marginTop: '25px' }}>
                                <div style={{ color: 'var(--info)', fontSize: '0.8rem', fontWeight: 'bold', marginBottom: '10px', textTransform: 'uppercase' }}>
                                    📋 ПЛАН РЕАЛІЗАЦІЇ:
                                </div>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                                    {logs[logs.length - 1].data.implementation_plan?.map((step, i) => (
                                        <div key={i} style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border)', padding: '12px', borderRadius: '4px' }}>
                                            <div style={{ color: 'var(--accent)', fontSize: '0.75rem', fontWeight: 'bold', marginBottom: '5px' }}>
                                                КРОК {i + 1}: {step.step}
                                            </div>
                                            <div style={{ fontSize: '0.8rem', color: '#eee', marginBottom: '5px' }}>
                                                {step.action}
                                            </div>
                                            {step.tools && (
                                                <div style={{ fontSize: '0.7rem', color: 'var(--info)', opacity: 0.8 }}>
                                                    🛠️ Інструменти: {step.tools}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {logs[logs.length - 1]?.data?.suggestions && (
                            <div style={{ marginTop: '20px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                                {logs[logs.length - 1].data.suggestions?.map((s, i) => (
                                    <button
                                        key={i}
                                        onClick={() => applySuggestion(s)}
                                        style={{ fontSize: '0.65rem', padding: '5px 10px', background: 'rgba(0,255,255,0.1)', border: '1px solid var(--accent-dim)', color: 'var(--accent)', cursor: 'pointer', borderRadius: '4px' }}
                                    >
                                        + {s}
                                    </button>
                                ))}
                            </div>
                        )}

                        <div style={{ marginTop: '30px', borderTop: '1px dashed var(--border)', paddingTop: '20px' }}>
                            <button
                                onClick={runLens}
                                style={{ width: '100%', background: 'var(--accent-dim)', color: 'var(--accent)', border: '1px solid var(--accent)', borderRadius: '4px', padding: '12px', fontSize: '0.7rem', fontWeight: 'bold' }}
                            >
                                🔍 ЗАПУСТИТИ ADRL-LENS (ОПТИМІЗАЦІЯ ТА АНАЛІЗ)
                            </button>

                            {lensResult && (
                                <div style={{ marginTop: '20px', background: 'rgba(0,255,255,0.02)', border: '1px solid var(--accent-dim)', padding: '15px', borderRadius: '4px', animation: 'slideIn 0.4s ease-out' }}>
                                    <div style={{ color: 'var(--accent)', fontSize: '0.75rem', fontWeight: 'bold', marginBottom: '15px', display: 'flex', justifyContent: 'space-between' }}>
                                        <span>ADRL-LENS V1.0 // АНАЛІТИЧНИЙ ЗВІТ</span>
                                        <span style={{ color: 'var(--info)' }}>Ratio: {(lensResult.compressionRatio * 100).toFixed(1)}%</span>
                                    </div>

                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
                                        {(Object.entries(lensResult.entities) as [string, string[]][]).map(([key, vals]) => (
                                            vals.length > 0 && (
                                                <div key={key}>
                                                    <div style={{ fontSize: '0.55rem', color: '#888', textTransform: 'uppercase', marginBottom: '5px' }}>{key.replace('_', ' ')}</div>
                                                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                                                        {vals.map((v, i) => (
                                                            <span key={i} style={{ fontSize: '0.6rem', padding: '2px 6px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', color: '#eee' }}>{v}</span>
                                                        ))}
                                                    </div>
                                                </div>
                                            )
                                        ))}
                                    </div>

                                    <div style={{ fontSize: '0.65rem', color: '#888', marginBottom: '10px' }}>
                                        <strong>ОПТИМІЗОВАНИЙ ТЕКСТ:</strong>
                                        <div style={{ color: '#aaa', marginTop: '5px', fontStyle: 'italic' }}>{lensResult.filteredText}</div>
                                    </div>

                                    <div style={{ fontSize: '0.55rem', color: 'var(--accent-dim)', marginTop: '10px', borderTop: '1px solid rgba(0,255,136,0.1)', paddingTop: '5px', display: 'flex', justifyContent: 'space-between' }}>
                                        <span>HASH: {lensResult.integrityHash.substring(0, 16)}...</span>
                                        <span>{new Date(lensResult.timestamp).toLocaleTimeString()}</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </main>

            <div className="controls">
                <button onClick={resetSession} disabled={isRunning}>СКИДАННЯ</button>
                {statusText === 'ПОМИЛКА API' || (logs.length > 0 && verdict === null && !isRunning) ? (
                    <button className="primary" onClick={() => startCycle(true)} disabled={isRunning}>
                        ПРОДОВЖИТИ З КРОКУ {PIPELINE[currentCursor]?.label || '...'}
                    </button>
                ) : (
                    <button id="runBtn" className="primary" onClick={() => startCycle(false)} disabled={isRunning}>
                        {logs.length > 0 && verdict !== null ? 'НАСТУПНА ІТЕРАЦІЯ' : 'ЗАПУСТИТИ ЦИКЛ'}
                    </button>
                )}
            </div>
        </>
    );
}
