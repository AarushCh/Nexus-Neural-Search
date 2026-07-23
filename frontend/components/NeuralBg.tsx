"use client";

import { useEffect, useRef } from "react";

// Faithful port of the original Canvas neural-network background:
// mouse-reactive synapses, theme-aware line colour, density by viewport.
export default function NeuralBg() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const connectDist = 160;
    const mouseDist = 250;
    const mouse = { x: -1000, y: -1000 };
    let particles: {
      x: number; y: number; vx: number; vy: number; size: number; color: string;
    }[] = [];
    let raf = 0;

    const palette = [
      "hsla(0,100%,60%,1)", "hsla(30,100%,60%,1)", "hsla(60,100%,60%,1)",
      "hsla(120,100%,60%,1)", "hsla(220,100%,60%,1)", "hsla(270,100%,60%,1)",
    ];

    const init = () => {
      const area = canvas.width * canvas.height;
      const density = window.innerWidth < 768 ? 15000 : 9000;
      const count = Math.floor(area / density);
      particles = [];
      for (let i = 0; i < count; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.8,
          vy: (Math.random() - 0.5) * 0.8,
          size: Math.random() * 2.5 + 2,
          color: palette[Math.floor(Math.random() * palette.length)],
        });
      }
    };

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      init();
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const isLight = document.documentElement.classList.contains("light-mode");
      const lineBase = isLight ? "0, 0, 0" : "0, 243, 255";
      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < mouseDist) {
          ctx.beginPath();
          const opacity = 1 - dist / mouseDist;
          ctx.strokeStyle = `rgba(${lineBase}, ${opacity})`;
          ctx.lineWidth = 1.5;
          ctx.moveTo(mouse.x, mouse.y);
          ctx.lineTo(p.x, p.y);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        if (isLight) {
          ctx.fillStyle = p.color;
          ctx.shadowBlur = 5;
          ctx.shadowColor = p.color;
        } else {
          ctx.fillStyle = "#ffffff";
          ctx.shadowBlur = 10;
          ctx.shadowColor = "#ffffff";
        }
        ctx.fill();
        ctx.shadowBlur = 0;
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx2 = p.x - p2.x;
          const dy2 = p.y - p2.y;
          const dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
          if (dist2 < connectDist) {
            ctx.beginPath();
            const opacity = 1 - dist2 / connectDist;
            ctx.strokeStyle = `rgba(${lineBase}, ${opacity * 0.4})`;
            ctx.lineWidth = 1.5;
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      });
      raf = requestAnimationFrame(animate);
    };

    let resizeTimeout: ReturnType<typeof setTimeout>;
    const onResize = () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(resize, 100);
    };
    const onMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };

    resize();
    animate();
    window.addEventListener("resize", onResize);
    document.addEventListener("mousemove", onMove);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("mousemove", onMove);
    };
  }, []);

  return <canvas id="neural-canvas" ref={ref} />;
}
