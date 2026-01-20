'use client';

import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stage, useGLTF, Center, Html, useProgress } from '@react-three/drei';
import { useEffect, useRef, useState, Component, ReactNode, Suspense } from 'react';
import { MOUSE } from 'three'; // MOUSE 상수 import

// --- 1. Error Boundary ---
class CanvasErrorBoundary extends Component<
  { children: ReactNode; onError?: (error: Error) => void },
  { hasError: boolean }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: Error) {
    console.error('Canvas error:', error);
    this.props.onError?.(error);
  }

  render() {
    if (this.state.hasError) return null;
    return this.props.children;
  }
}

// --- 2. 3D Model Component ---
function Model({ url, rotation }: { url: string; rotation: number }) {
  const { scene } = useGLTF(url);
  const groupRef = useRef<THREE.Group>(null!);

  useEffect(() => {
    if (groupRef.current) {
      groupRef.current.rotation.z = rotation;
    }
  }, [rotation]);

  return (
    <Center>
      <group ref={groupRef}>
        <primitive object={scene} />
      </group>
    </Center>
  );
}

// --- 3. Loading Indicator ---
function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div className="flex flex-col items-center justify-center bg-white/90 backdrop-blur-md px-4 py-3 rounded-2xl shadow-xl shadow-blue-500/10 border border-white/50">
        <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin mb-2"></div>
        <p className="text-xs font-bold text-blue-600">{progress.toFixed(0)}% Loading</p>
      </div>
    </Html>
  );
}

// --- 4. Capture Helper ---
function CaptureHelper({ onReady }: { onReady: (captureFunc: () => Promise<Blob | null>) => void }) {
  const { gl, scene, camera } = useThree();

  useEffect(() => {
    const captureFunc = async (): Promise<Blob | null> => {
      return new Promise((resolve) => {
        try {
          gl.render(scene, camera);
          gl.domElement.toBlob((blob) => {
            resolve(blob);
          }, 'image/jpeg', 0.95);
        } catch (error) {
          console.error('Capture error:', error);
          resolve(null);
        }
      });
    };
    
    onReady(captureFunc);
  }, [gl, scene, camera, onReady]);

  return null;
}

interface ThumbnailCreatorProps {
  modelUrl: string;
  onCapture: (blob: Blob, backgroundColor: string) => void;
  onSkip: () => void;
}

export default function ThumbnailCreator({ modelUrl, onCapture, onSkip }: ThumbnailCreatorProps) {
  const [backgroundColor, setBackgroundColor] = useState('#f8fafc');
  const [isCapturing, setIsCapturing] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const captureFuncRef = useRef<(() => Promise<Blob | null>) | null>(null);
  
  // OrbitControls 제어를 위한 Ref
  const controlsRef = useRef<any>(null);

  // Z축 회전을 위한 상태
  const [modelRotation, setModelRotation] = useState(0);
  const [isShiftDragging, setIsShiftDragging] = useState(false);
  const lastMouseXRef = useRef(0);

  const presetColors = [
    { name: '스튜디오 그레이', value: '#f8fafc' },
    { name: '오션 폼', value: '#ecfeff' },
    { name: '스카이 미스트', value: '#f0f9ff' },
    { name: '클린 화이트', value: '#ffffff' },
    { name: '쿨 슬레이트', value: '#f1f5f9' },
    { name: '소프트 인디고', value: '#eef2ff' },
  ];

  // Shift 키 이벤트 리스너: Shift를 누르면 마우스 왼쪽 클릭이 PAN(이동)으로 변경됨
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Shift' && controlsRef.current) {
        controlsRef.current.mouseButtons.LEFT = MOUSE.PAN;
        // 커서 스타일 변경 (선택사항)
        document.body.style.cursor = 'move';
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Shift' && controlsRef.current) {
        controlsRef.current.mouseButtons.LEFT = MOUSE.ROTATE;
        document.body.style.cursor = 'default';
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // Z축 회전을 위한 마우스 이벤트 핸들러
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.shiftKey) {
      setIsShiftDragging(true);
      lastMouseXRef.current = e.clientX;
      if (controlsRef.current) {
        controlsRef.current.enabled = false;
      }
      e.preventDefault();
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isShiftDragging) {
      const deltaX = e.clientX - lastMouseXRef.current;
      setModelRotation((prev) => prev + deltaX * 0.01);
      lastMouseXRef.current = e.clientX;
      e.preventDefault();
    }
  };

  const handleMouseUp = () => {
    if (isShiftDragging) {
      setIsShiftDragging(false);
      if (controlsRef.current) {
        controlsRef.current.enabled = true;
      }
    }
  };

  const handleMouseLeave = () => {
    if (isShiftDragging) {
      setIsShiftDragging(false);
      if (controlsRef.current) {
        controlsRef.current.enabled = true;
      }
    }
  };

  const handleCapture = async () => {
    if (!captureFuncRef.current || isCapturing) return;
    
    setIsCapturing(true);
    
    setTimeout(async () => {
      try {
        if (captureFuncRef.current) {
          const blob = await captureFuncRef.current();
          if (blob) {
            onCapture(blob, backgroundColor);
          } else {
            throw new Error('Blob creation failed');
          }
        }
      } catch (error) {
        console.error('Capture failed:', error);
        alert('캡처 중 오류가 발생했습니다.');
        setIsCapturing(false);
      }
    }, 100);
  };

  return (
    <div className="fixed inset-0 z-[100] bg-slate-900/60 backdrop-blur-md flex items-center justify-center p-4">
      {/* 📐 크기 조정 부분:
        1. max-w-5xl -> max-w-7xl (너비 대폭 확장)
        2. max-h-[90vh] -> h-[85vh] (높이를 화면의 85%로 고정하여 시원하게 만듦)
      */}
      <div className="bg-white rounded-[2rem] shadow-2xl w-full max-w-7xl h-[85vh] overflow-hidden border border-white/50 flex flex-col md:flex-row relative">
        
        {/* 닫기 버튼 */}
        <button 
          onClick={onSkip}
          className="absolute top-6 right-6 z-50 p-2.5 rounded-full bg-white/90 hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-colors shadow-sm border border-slate-100"
          title="닫기"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* 왼쪽: 3D 캔버스 영역 (높이를 100% 채움) */}
        <div 
          className="flex-1 relative bg-slate-50 h-full"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        >
          {/* 배경색 적용 */}
          <div className="absolute inset-0 transition-colors duration-500" style={{ backgroundColor }} />
          
          <CanvasErrorBoundary onError={() => setLoadError(true)}>
            <Canvas 
              camera={{ position: [0, 0, 4.5], fov: 45 }} 
              gl={{ preserveDrawingBuffer: true, antialias: true }}
              shadows
              className="h-full w-full"
            >
              <color attach="background" args={[backgroundColor]} />
              
              <Suspense fallback={<Loader />}>
                <Stage environment="city" intensity={0.6} adjustCamera>
                  <Model url={modelUrl} rotation={modelRotation} />
                </Stage>
                {/* enablePan={true}로 변경하고 ref를 연결했습니다.
                   이제 위에서 정의한 useEffect가 Shift 키에 따라 동작을 제어합니다.
                */}
                <OrbitControls 
                  ref={controlsRef}
                  makeDefault 
                  enablePan={true} 
                  minPolarAngle={0} 
                  maxPolarAngle={Math.PI / 1.5} 
                />
                <CaptureHelper onReady={(func) => (captureFuncRef.current = func)} />
              </Suspense>
            </Canvas>
          </CanvasErrorBoundary>

          {/* 오버레이 가이드 */}
          <div className="absolute top-6 left-6 pointer-events-none">
             <div className="flex flex-col gap-2">
               <div className="bg-white/80 backdrop-blur-md px-4 py-2 rounded-2xl border border-white/50 shadow-sm text-sm font-bold text-blue-500 tracking-wide flex items-center gap-2">
                 <span>🖱️</span> 드래그하여 회전
               </div>
               <div className="bg-white/80 backdrop-blur-md px-4 py-2 rounded-2xl border border-white/50 shadow-sm text-sm font-bold text-slate-500 tracking-wide flex items-center gap-2">
                 <span>✋</span> Ctrl + 드래그하여 이동
               </div>
               <div className="bg-white/80 backdrop-blur-md px-4 py-2 rounded-2xl border border-white/50 shadow-sm text-sm font-bold text-purple-500 tracking-wide flex items-center gap-2">
                 <span>🔄</span> Z축 회전: Shift+드래그
               </div>
             </div>
          </div>

          {/* 에러 상태 */}
          {loadError && (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-50/90 z-10">
              <div className="text-center p-6">
                <div className="text-4xl mb-2">😢</div>
                <h3 className="font-bold text-slate-700">모델을 불러올 수 없어요</h3>
                <button onClick={onSkip} className="mt-4 text-sm text-blue-600 font-bold hover:underline">
                  기본 썸네일 사용하기
                </button>
              </div>
            </div>
          )}
        </div>

        {/* 오른쪽: 컨트롤 패널 (너비 고정, 세로 스크롤 가능) */}
        <div className="w-full md:w-96 bg-white p-8 flex flex-col justify-between border-l border-slate-100 overflow-y-auto">
          <div>
            <div className="mb-8">
              <h2 className="text-2xl font-black bg-gradient-to-r from-slate-800 to-blue-900 bg-clip-text text-transparent mb-2">
                썸네일 스튜디오
              </h2>
              <p className="text-slate-400 font-medium">
                원하는 각도로 돌리고 배경을 골라<br/>나만의 썸네일을 만들어보세요.
              </p>
            </div>

            {/* 배경색 선택 */}
            <div className="mb-8">
              <label className="block text-xs font-black text-slate-400 uppercase tracking-wider mb-4">
                배경 컬러
              </label>
              <div className="grid grid-cols-2 gap-3">
                {presetColors.map((color) => (
                  <button
                    key={color.value}
                    onClick={() => setBackgroundColor(color.value)}
                    className={`h-12 rounded-xl border transition-all relative group flex items-center justify-center ${
                      backgroundColor === color.value
                        ? 'border-blue-500 ring-2 ring-blue-100'
                        : 'border-slate-100 hover:border-blue-200'
                    }`}
                    style={{ backgroundColor: color.value }}
                    title={color.name}
                  >
                    {backgroundColor === color.value && (
                        <span className="text-slate-700">
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" /></svg>
                        </span>
                    )}
                  </button>
                ))}
              </div>
              
              {/* 커스텀 컬러 */}
              <div className="mt-4 pt-4 border-t border-slate-50 flex items-center gap-3">
                <div className="relative">
                   <input
                    type="color"
                    value={backgroundColor}
                    onChange={(e) => setBackgroundColor(e.target.value)}
                    className="w-10 h-10 rounded-full cursor-pointer border-2 border-slate-200 p-0 overflow-hidden shadow-sm hover:scale-105 transition-transform"
                   />
                </div>
                <div className="flex flex-col">
                    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">CUSTOM</span>
                    <span className="text-sm font-bold text-slate-600 font-mono uppercase">{backgroundColor}</span>
                </div>
              </div>
            </div>
          </div>

          {/* 하단 액션 버튼 */}
          <div className="space-y-3 pt-6 border-t border-slate-100">
            <button
              onClick={handleCapture}
              disabled={isCapturing || loadError}
              className="w-full py-5 bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 text-white rounded-2xl font-bold text-lg shadow-xl shadow-blue-500/20 hover:shadow-blue-500/30 active:scale-95 transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isCapturing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>캡처 중...</span>
                </>
              ) : (
                <>
                  <span>📸 이 각도로 저장</span>
                </>
              )}
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}