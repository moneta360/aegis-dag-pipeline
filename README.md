# 🛡️ Aegis DAG Pipeline

**Aegis DAG Pipeline** — це високоperformance SDK для створення прозорих AI-інтерфейсів. Він використовує архітектуру спрямованого ациклічного графа (DAG) на базі Web Workers для паралельних обчислень.

## ✨ Ключові особливості

* 🚀 **Zero-Dependency & Blob Workers:** Воркери створюються "на льоту". Не потребує налаштування Webpack.
* ⚡ **Zero-Copy Transfer:** Використання `ArrayBuffer` для передачі даних між потоками без затримок.
* 🔋 **Eco Mode:** Автоматична оптимізація рендеру при бездіяльності.
* ⏱️ **60 FPS Guarantee:** Вся математика винесена з головного потоку.

## 📦 Як використовувати

1. Скопіюйте файл `SwarmCanvas.tsx` у свій проект.
2. Підключіть компонент у головному файлі:

```tsx
import { SwarmCanvas } from './SwarmCanvas';

function App() {
  return (
    <div style={{ background: '#020617', height: '100vh' }}>
      <SwarmCanvas threats={0} />
      <h1 style={{ color: 'white', position: 'relative', zIndex: 10 }}>
        Aegis System Active
      </h1>
    </div>
  );
}
