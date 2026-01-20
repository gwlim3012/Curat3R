'use client';

import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stage, useGLTF, useProgress, Html } from '@react-three/drei';
import { Suspense, useState, useEffect, useRef } from 'react';
import * as THREE from 'three';

interface ModelProps {
  url: string;
  rotation: number;
}

function Model({ url, rotation }: ModelProps) {
  const { scene } = useGLTF(url);
  const modelRef = useRef<THREE.Group>(null);
  
  useEffect(() => {
    if (modelRef.current) {
      modelRef.current.rotation.z = rotation;
    }
  }, [rotation]);
  
  return <group ref={modelRef}><primitive object={scene} /></group>;
}

function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center text-slate-500 bg-white/90 backdrop-blur-md p-5 rounded-2xl shadow-xl shadow-blue-500/10 border border-white">
        <div className="w-12 h-12 border-4 border-blue-100 border-t-blue-500 rounded-full animate-spin mb-3"></div>
        <p className="font-black text-sm text-blue-600">{Math.round(progress)}%</p>
      </div>
    </Html>
  );
}

interface ModelViewerProps {
  modelUrl: string;
  backgroundColor?: string;
}

export default function ModelViewer({ modelUrl, backgroundColor = '#ffffff' }: ModelViewerProps) {
  const [intensity, setIntensity] = useState(1);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [modelRotation, setModelRotation] = useState(0);
  const [isShiftDragging, setIsShiftDragging] = useState(false);
  const dragStartX = useRef(0);
  const startRotation = useRef(0);
  const canvasRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsFullscreen(false);
      }
    };

    if (isFullscreen) {
      window.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [isFullscreen]);

  // Shift+드래그로 Z축 회전 (시계바늘 방향)
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.shiftKey) {
      setIsShiftDragging(true);
      dragStartX.current = e.clientX;
      startRotation.current = modelRotation;
      e.preventDefault();
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isShiftDragging) {
      const deltaX = e.clientX - dragStartX.current;
      const rotationDelta = -(deltaX / 200) * Math.PI; // 시계방향으로 회전
      setModelRotation(startRotation.current + rotationDelta);
    }
  };

  const handleMouseUp = () => {
    setIsShiftDragging(false);
  };

  const handleMouseLeave = () => {
    setIsShiftDragging(false);
  };

  return (
    <div 
      className={`transition-all duration-500 ease-in-out ${
        isFullscreen 
          ? 'fixed inset-0 z-[9999] flex items-center justify-center bg-slate-950/95 backdrop-blur-xl' 
          : 'relative w-full h-[500px] rounded-[2.5rem] overflow-hidden border border-blue-100/50 shadow-inner bg-transparent'
      }`}
      style={!isFullscreen ? { backgroundColor } : undefined}
    >
      
      <div 
        ref={canvasRef}
        className={isFullscreen ? 'w-full h-full' : 'w-full h-full'}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        style={{ cursor: isShiftDragging ? 'grabbing' : 'default' }}
      >
        {/* removed patterned overlay to avoid extra box appearing behind the viewer */}
        <Canvas style={{ background: backgroundColor }} camera={{ position: [0, 0, 5], fov: 45 }} shadows dpr={[1, 2]}>
          <Suspense fallback={<Loader />}>
            <Stage environment="city" intensity={intensity} adjustCamera>
              <Model url={modelUrl} rotation={modelRotation} />
            </Stage>
            <OrbitControls 
              enablePan={true} 
              enableZoom={true} 
              enableRotate={true}
              makeDefault 
              autoRotate={false}
              mouseButtons={{
                LEFT: 0,  // rotate
                MIDDLE: 1, // zoom (not used in OrbitControls)
                RIGHT: 2   // pan (not used by default)
              }}
              touches={{
                ONE: 0,  // rotate
                TWO: 2   // pan/zoom
              }}
              keys={{
                LEFT: 'ArrowLeft',
                UP: 'ArrowUp',
                RIGHT: 'ArrowRight',
                BOTTOM: 'ArrowDown'
              }}
              keyPanSpeed={7}
              panSpeed={0.5}
            />
          </Suspense>
        </Canvas>
      </div>
      
      {/* 조작 가이드 */}
      <div className="absolute top-6 left-6 pointer-events-none z-10">
        <div className={`backdrop-blur-md px-4 py-2 rounded-xl border shadow-sm text-[11px] font-bold transition-colors tracking-wide ${
           isFullscreen 
             ? 'bg-slate-900/60 border-white/10 text-cyan-100' 
             : 'bg-white/70 border-white/50 text-blue-600 shadow-blue-500/5'
        }`}>
             회전: 드래그 &nbsp;|&nbsp; 줌: 스크롤 &nbsp;|&nbsp; Z축 회전: Shift+드래그 &nbsp;|&nbsp; 이동: Ctrl+드래그
        </div>
      </div>

      {/* 전체화면 토글 버튼 */}
      <button
        onClick={() => setIsFullscreen(!isFullscreen)}
        className={`absolute top-6 right-6 p-3 rounded-full backdrop-blur-md transition-all shadow-lg border z-50 ${
           isFullscreen 
           ? 'bg-white/10 text-black border-white/20 hover:bg-blue-500/20 hover:text-black' 
           : 'bg-white/90 text-blue-600 border-white/60 hover:bg-blue-50 hover:scale-110 shadow-blue-500/10'
        }`}
        title={isFullscreen ? '닫기 (ESC)' : '전체화면'}
      >
        {isFullscreen ? (
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        )}
      </button>

      {/* 🔴 밝기 조절 슬라이더 (가시성 개선됨) */}
      <div className={`absolute bottom-6 right-6 backdrop-blur-md px-4 py-3 rounded-2xl shadow-lg border flex items-center gap-3 transition-all z-10 ${
        isFullscreen 
          ? 'bg-slate-900/80 border-white/10 text-blue-200' 
          // 👇 일반 모드: 배경은 그대로(흰색) 두고, 텍스트(아이콘) 색상을 text-blue-500 -> text-blue-700으로 진하게 변경
          : 'bg-white/90 border-blue-100/50 text-blue-700 shadow-blue-500/10'
      }`}>
        {/* 해 아이콘 */}
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
        </svg>
        
        {/* 슬라이더: accent-blue-500 -> accent-blue-600으로 더 진하게 */}
        <input
          type="range"
          min="0.2"
          max="2.5"
          step="0.1"
          value={intensity}
          onChange={(e) => setIntensity(Number(e.target.value))}
          className="w-24 h-1.5 rounded-lg appearance-none cursor-pointer bg-slate-200 accent-blue-600 hover:accent-blue-700"
        />
      </div>
    </div>
  );
}