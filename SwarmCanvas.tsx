import React, { useEffect, useRef } from 'react';

/**
 * Aegis DAG Pipeline - High Performance Canvas Provider
 * @param cpu - Simulated CPU load percentage
 * @param threats - System threat level (changes visual state)
 */
interface SwarmProps {
  cpu?: number;
  threats?: number;
}

export const SwarmCanvas: React.FC<SwarmProps> = ({ cpu = 0, threats = 0 }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Створення Inline Web Worker через Blob (Zero-Dependency)
    const workerCode = `
      let nodes = [];
      let lastUpdate = Date.now();

      self.onmessage = (e) => {
        const { type, width, height, mouse } = e.data;
        
        if (type === 'init') {
          nodes = Array.from({ length: 3 }, (_, i) => ({
            id: i,
            x: width / 2,
            y: height / 2,
            vx: 0, vy: 0
          }));
        }

        if (type === 'update') {
          const now = Date.now();
          const dt = (now - lastUpdate) / 1000;
          lastUpdate = now;

          nodes.forEach(node => {
            const dx = mouse.x - node.x;
            const dy = mouse.y - node.y;
            const dist = Math.sqrt(dx*dx + dy*dy);
            
            // Scatter logic: вузли реагують на курсор
            if (dist > 5) {
              node.vx += dx * 0.08;
              node.vy += dy * 0.08;
            }
            
            node.x += node.vx * dt;
            node.y += node.vy * dt;
            node.vx *= 0.85; // Friction (тертя)
            node.vy *= 0.85;
          });

          self.postMessage({ nodes });
        }
      };
    `;

    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const worker = new Worker(URL.createObjectURL(blob));
    workerRef.current = worker;

    const ctx = canvas.getContext('2d');
    let animationFrame: number;
    let mouse = { x: 0, y: 0 };

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      worker.postMessage({ type: 'init', width: canvas.width, height: canvas.height });
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouse = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', handleMouseMove);

    resize();

    const renderLoop = () => {
      worker.postMessage({ type: 'update', mouse });
      
      worker.onmessage = (e) => {
        if (!ctx) return;
        const { nodes } = e.data;

        // Очищення фону з ефектом шлейфу
        ctx.fillStyle = 'rgba(2, 6, 23, 0.3)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Малювання зв'язків (Edges)
        ctx.beginPath();
        ctx.strokeStyle = threats > 50 ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 211, 238, 0.4)';
        ctx.lineWidth = 1;
        nodes.forEach((node: any) => {
          ctx.moveTo(mouse.x, mouse.y);
          ctx.lineTo(node.x, node.y);
        });
        ctx.stroke();

        // Малювання вузлів (Nodes) з неоновим свіченням
        nodes.forEach((node: any) => {
          ctx.fillStyle = threats > 50 ? '#ef4444' : '#22d3ee';
          ctx.shadowBlur = 20;
          ctx.shadowColor = ctx.fillStyle;
          ctx.beginPath();
          ctx.arc(node.x, node.y, 5, 0, Math.PI * 2);
          ctx.fill();
        });
      };

      animationFrame = requestAnimationFrame(renderLoop);
    };

    renderLoop();

    return () => {
      cancelAnimationFrame(animationFrame);
      worker.terminate();
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, [threats]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full pointer-events-none bg-slate-950"
      style={{ zIndex: 0 }}
    />
  );
};
