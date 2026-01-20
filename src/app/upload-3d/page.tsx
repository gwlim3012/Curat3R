'use client';

import Link from 'next/link'; // 로고 누르면 초기 화면 복귀
import { useState, ChangeEvent, FormEvent, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { archiveService } from '@/services/archiveService';
import { pipelineService, FilterResult, ProcessResponse } from '@/services/pipelineService';
import ThumbnailCreator from '@/components/ThumbnailCreator';
import AutoThumbnailGenerator from '@/components/AutoThumbnailGenerator';

type ProcessingStage = 'idle' | 'filtering' | 'reconstruction' | 'saving' | 'completed';

export default function UploadWithPipelinePage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // ✅ 취소 기능을 위한 Ref (타이머 & 네트워크 요청 제어)
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  
  // State definitions
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [processingStage, setProcessingStage] = useState<ProcessingStage>('idle');
  const [filterResult, setFilterResult] = useState<FilterResult | null>(null);
  const [processResult, setProcessResult] = useState<ProcessResponse | null>(null);
  const [progressMessage, setProgressMessage] = useState('');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<'fast' | 'quality'>('fast');
  const [modelBlob, setModelBlob] = useState<Blob | null>(null);
  const [thumbnailBlob, setThumbnailBlob] = useState<Blob | null>(null);
  const [showThumbnailCreator, setShowThumbnailCreator] = useState(false);
  const [isGeneratingThumbnail, setIsGeneratingThumbnail] = useState(false);
  
  // 강제 진행 여부 상태
  const [forceProceed, setForceProceed] = useState(false);
  
  const [blobValidated, setBlobValidated] = useState(false);

  // -- Handlers --
  const handleImageChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;
    processFile(selectedFile);
  };

  const processFile = (file: File | null) => {
    // 파일 변경 시 기존 작업 취소
    handleCancel();

    setImageFile(file);
    setImagePreview('');
    setFilterResult(null);
    setProcessResult(null);
    setTaskId(null);
    setModelBlob(null);
    setThumbnailBlob(null);
    setProcessingStage('idle');
    setForceProceed(false);

    if (!file) return;

    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // ✅ [복구됨] 작업 취소 함수
  const handleCancel = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setProcessingStage('idle');
    setProgressMessage('');
  };

  const handleFilterCheck = async () => {
    if (!imageFile) return;
    setProcessingStage('filtering');
    setProgressMessage('이미지 품질을 꼼꼼히 검사하고 있어요...');
    setForceProceed(false);

    try {
      const result = await pipelineService.filterImage(imageFile);
      setFilterResult(result.filter_result);
      setTaskId(result.task_id);
      setProcessingStage('idle');
      setProgressMessage('');
    } catch (error) {
      console.error('Filter check failed:', error);
      alert('필터링 체크 실패: ' + (error instanceof Error ? error.message : '알 수 없는 오류'));
      setProcessingStage('idle');
    }
  };

  const handleProcess3D = async () => {
    if (!taskId) return;
    setProcessingStage('reconstruction');
    
    // 취소 컨트롤러 생성
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const modelName = selectedModel === 'quality' ? 'Trellis (Quality)' : 'SPAR3D (Fast)';
    const estimatedTime = selectedModel === 'quality' ? '5-10분' : '1-3분';
    setProgressMessage(`${modelName} 모델로 3D 모델 생성 중...`);
    
    const startTime = Date.now();
    
    try {
      try {
        const response = await fetch(`/api/pipeline/reconstruct/${taskId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: selectedModel }),
          signal: controller.signal, // 취소 신호 연결
        });

        if (!response.ok) throw new Error('서버응답 오류. 폴링 모드로 전환합니다.');

        const result: ProcessResponse = await response.json();
        setProcessResult(result);

        if (result.stage === 'completed' && result.task_id) {
          setProgressMessage('완성된 3D 모델을 가져오는 중...');
          const modelBlob = await pipelineService.downloadModel(result.task_id);
          setModelBlob(modelBlob);
          pipelineService.cleanup(result.task_id).catch(console.error);
        }
      } catch (fetchError: any) {
        if (fetchError.name === 'AbortError') return;

        setProgressMessage(`열심히 만드는 중... (${estimatedTime} 소요예상)`);
        
        // 폴링 로직 시작
        intervalRef.current = setInterval(async () => {
          try {
            const modelBlob = await pipelineService.downloadModel(taskId);
            if (!modelBlob || modelBlob.size === 0) return;
            
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
                intervalRef.current = null;
            }
            setModelBlob(modelBlob);
            setProcessingStage('idle');
            pipelineService.cleanup(taskId).catch(console.error);
          } catch (e) {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            setProgressMessage(`제작 중... (${elapsed}초 경과)`);
            
            if (elapsed > 1800) {
                handleCancel();
                alert('시간이 너무 오래 걸려 중단되었습니다.');
            }
          }
        }, 10000);
        return;
      }
      setProcessingStage('idle');
    } catch (error: any) {
      if (error.name === 'AbortError') return;
      handleCancel();
    }
  };

  const handleManualThumbnail = () => { if (modelBlob) setShowThumbnailCreator(true); };

  const handleThumbnailCapture = (blob: Blob) => {
    setThumbnailBlob(blob);
    setShowThumbnailCreator(false);
    setIsGeneratingThumbnail(false);
  };

  const handleAutoThumbnailGenerated = (blob: Blob) => {
    setThumbnailBlob(blob);
    setIsGeneratingThumbnail(false);
    setProgressMessage('');
  };

  const handleSkipManualThumbnail = () => {
    setShowThumbnailCreator(false);
    setIsGeneratingThumbnail(true);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !modelBlob) return;
    setProcessingStage('saving');
    try {
      const modelFile = new File([modelBlob], 'model.glb', { type: 'model/gltf-binary' });
      
      const modelTag = selectedModel === 'quality' 
        ? '\n\n[Model: Trellis]' 
        : '\n\n[Model: Stable]';
      
      const finalContent = content.trim() + modelTag;

      await archiveService.addArchive({
        title: title.trim(), 
        content: finalContent, 
        fileType: 'model',
        fileBlob: modelFile, 
        thumbnailBlob: thumbnailBlob || undefined,
      });
      alert('✨ 새로운 추억이 기록되었습니다!');
      router.push('/');
    } catch (error) {
      alert('저장에 실패했습니다.');
    } finally {
      setProcessingStage('idle');
    }
  };

  const filterStatusInfo = filterResult ? pipelineService.getFilterStatusMessage(filterResult) : null;
  const isProcessing = processingStage !== 'idle';

  // 컴포넌트 언마운트 시 타이머 정리
  useEffect(() => {
    return () => handleCancel();
  }, []);

  useEffect(() => {
    if (showThumbnailCreator && modelBlob && !blobValidated) {
      (async () => {
        try {
          const arrayBuffer = await modelBlob.slice(0, 4).arrayBuffer();
          const header = new TextDecoder().decode(arrayBuffer);
          if (!header.startsWith('glTF')) throw new Error('Invalid format');
          setBlobValidated(true);
        } catch (error) {
          setShowThumbnailCreator(false);
        }
      })();
    }
  }, [showThumbnailCreator, modelBlob, blobValidated]);

  if (showThumbnailCreator && modelBlob && blobValidated) {
    return (
      <ThumbnailCreator
        modelUrl={URL.createObjectURL(modelBlob)}
        onCapture={handleThumbnailCapture}
        onSkip={handleSkipManualThumbnail}
      />
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 relative selection:bg-blue-200 selection:text-blue-900 pb-20">
      
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-blue-200/30 rounded-full blur-[100px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-cyan-100/40 rounded-full blur-[120px]" />
      </div>

      <div className="relative max-w-4xl mx-auto px-4 py-12 z-10">
        
      <div className="text-center mb-10">
        <Link href="/" className="relative inline-block hover:scale-105 transition-transform duration-300 cursor-pointer group">
          <h1 className="text-5xl font-black tracking-tighter bg-gradient-to-r from-blue-600 via-sky-500 to-cyan-400 bg-clip-text text-transparent mb-3 pb-1">
            Curat3R
          </h1>
          <span className="absolute -top-1 -right-5 text-2xl animate-pulse">🪄</span>
        </Link>
        <p className="text-slate-600 text-lg font-medium">
          이미지에서 <span className="text-blue-600 font-bold">3D 모델</span>을 자동으로 생성합니다
        </p>
        <div className="inline-flex items-center gap-1.5 px-4 py-1.5 bg-blue-50/50 border border-blue-100 rounded-full text-sm text-blue-500 shadow-sm">
          <span>배경이 복잡하지 않은 <span className="font-bold text-blue-700">단일 객체</span> 사진이 가장 잘 나와요!</span>
        </div>
      </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          
          <section className="bg-white/80 backdrop-blur-sm rounded-3xl p-6 shadow-xl shadow-blue-500/5 border border-white/50">
             <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                   <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 text-sm">1</span>이미지 선택
                </h2>
                {imageFile && <button type="button" onClick={() => processFile(null)} className="text-sm text-slate-400 hover:text-red-500" disabled={isProcessing}>초기화</button>}
             </div>

             {!imagePreview ? (
                <div onDragOver={handleDragOver} onDrop={handleDrop} onClick={() => fileInputRef.current?.click()} className="w-full h-64 border-3 border-dashed border-blue-200 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:bg-blue-50 hover:border-blue-400 transition-all group">
                   <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-sm mb-4 group-hover:scale-110 transition-transform"><span className="text-3xl">📷</span></div>
                   <p className="text-slate-600 font-medium text-lg">클릭하거나 이미지를 드래그하세요</p>
                   <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageChange} className="hidden" />
                </div>
             ) : (
                <div className="relative rounded-2xl overflow-hidden bg-slate-100 border border-slate-200 group">
                   <img src={imagePreview} alt="Preview" className="w-full max-h-96 object-contain mx-auto" />
                   {!filterResult && !isProcessing && (
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                         <button type="button" onClick={handleFilterCheck} className="bg-white text-blue-900 px-6 py-3 rounded-full font-bold shadow-lg">🔍 품질 검사하기</button>
                      </div>
                   )}
                   {processingStage === 'filtering' && (
                      <div className="absolute inset-0 bg-white/80 flex flex-col items-center justify-center backdrop-blur-sm z-10">
                         <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-3"></div>
                         <p className="text-blue-600 font-bold animate-pulse">AI가 이미지를 분석 중입니다...</p>
                      </div>
                   )}
                </div>
             )}
          </section>

          {filterResult && !modelBlob && (
             <section className="bg-white/80 backdrop-blur-sm rounded-3xl p-6 shadow-xl border border-white/50 animate-fade-in-up">
                
                <div className={`rounded-xl p-5 mb-6 border-l-4 ${filterStatusInfo?.canProceed ? 'bg-cyan-50 border-cyan-500' : 'bg-red-50 border-red-500'}`}>
                   <div className="flex gap-4">
                      <span className="text-4xl flex-shrink-0">
                        {filterStatusInfo?.canProceed ? filterStatusInfo.emoji : '🤔'}
                      </span>
                      
                      <div className="flex-1">
                         <h3 className={`font-bold text-lg mb-1 ${filterStatusInfo?.canProceed ? 'text-cyan-800' : 'text-red-800'}`}>
                            {filterStatusInfo?.canProceed 
                              ? filterStatusInfo.title 
                              : (filterResult.reason ? `AI 의견: ${filterResult.reason}` : 'AI가 확신하지 못하네요')}
                         </h3>
                         <p className={`text-base leading-relaxed ${filterStatusInfo?.canProceed ? 'text-cyan-700' : 'text-red-700'}`}>
                            {!filterStatusInfo?.canProceed && (filterResult as any).guide ? (filterResult as any).guide : filterStatusInfo?.description}
                         </p>

                         {/* 강제 진행 버튼 (왼쪽 정렬) */}
                         {!filterStatusInfo?.canProceed && !forceProceed && (
                           <div className="mt-6 flex flex-col items-start">
                             <p className="text-sm text-red-500 font-bold mb-3">
                               * 큐레이터님의 판단이 맞다면 아래 버튼을 눌러주세요!
                             </p>
                             <button 
                               type="button"
                               onClick={() => setForceProceed(true)}
                               className="py-2 px-4 bg-white border-2 border-red-200 text-red-500 hover:bg-red-50 hover:text-red-600 hover:border-red-300 text-sm font-bold rounded-lg transition-all shadow-sm hover:shadow-md flex items-center gap-2 group"
                             >
                               <span className="group-hover:scale-125 transition-transform text-lg">✨</span>
                               <span>그래도 한번 만들어볼래요</span>
                             </button>
                           </div>
                         )}
                         
                         {/* 강제 진행 활성화 메시지 (가로 한 줄) */}
                         {forceProceed && (
                           <div className="mt-4 flex flex-row items-center gap-2 bg-blue-50 px-4 py-3 rounded-xl border border-blue-100 animate-fade-in w-fit">
                           <svg className="w-5 h-5 text-blue-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                             <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                           </svg>
                           <span className="text-blue-700 font-bold text-base whitespace-nowrap">
                             큐레이터님의 판단을 믿고 진행합니다!
                           </span>
                         </div>
                         )}
                      </div>
                   </div>
                </div>

                {(filterStatusInfo?.canProceed || forceProceed) && (
                   <div className="space-y-4 animate-fade-in">
                      <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
                        <span className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 text-sm">2</span>재구성 모델 선택
                      </h2>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                         <label className={`relative p-5 rounded-2xl border-2 cursor-pointer transition-all ${selectedModel === 'fast' ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-100' : 'border-slate-100 bg-white hover:border-blue-200'}`}>
                            <input type="radio" name="model" value="fast" checked={selectedModel === 'fast'} onChange={(e) => setSelectedModel(e.target.value as any)} className="hidden" />
                            <div className="flex items-center justify-between mb-2"><span className="text-3xl">⚡</span></div>
                            <div className="font-bold text-slate-800 text-xl">빠른 생성 (Fast)</div>
                            <p className="text-slate-500 text-base mt-1">1-3분 소요 • 일반 품질</p>
                         </label>
                         <label className={`relative p-5 rounded-2xl border-2 cursor-pointer transition-all ${selectedModel === 'quality' ? 'border-indigo-500 bg-indigo-50 ring-2 ring-indigo-100' : 'border-slate-100 bg-white hover:border-indigo-200'}`}>
                            <input type="radio" name="model" value="quality" checked={selectedModel === 'quality'} onChange={(e) => setSelectedModel(e.target.value as any)} className="hidden" />
                            <div className="flex items-center justify-between mb-2"><span className="text-3xl">💎</span></div>
                            <div className="font-bold text-slate-800 text-xl">고품질 (Quality)</div>
                            <p className="text-slate-500 text-base mt-1">5-10분 소요 • 최고 품질</p>
                         </label>
                      </div>
                      
                      <button type="button" onClick={handleProcess3D} disabled={isProcessing} className="w-full mt-6 bg-gradient-to-r from-blue-600 to-indigo-700 text-white text-xl font-bold py-5 rounded-2xl shadow-lg shadow-blue-500/20 hover:-translate-y-1 transition-all">
                         {isProcessing ? '🎨 3D 재구성 진행 중...' : '🌊 3D 재구성 시작'}
                      </button>
                      
                      {progressMessage && (
                        <div className="text-center mt-6">
                            <p className="text-lg font-bold text-blue-600 mb-3 flex items-center justify-center gap-3">
                                <span className="inline-block w-5 h-5 border-3 border-blue-600 border-t-transparent rounded-full animate-spin"></span>
                                {progressMessage}
                            </p>
                            
                            {/* ✅ [수정됨] 취소 버튼 (문구: 생성 취소) */}
                            <button 
                              type="button"
                              onClick={handleCancel}
                              className="px-4 py-2 text-slate-400 hover:text-red-500 text-sm font-medium transition-colors border border-transparent hover:border-red-100 hover:bg-red-50 rounded-lg"
                            >
                              생성 취소
                            </button>
                        </div>
                      )}
                   </div>
                )}
             </section>
          )}

          {modelBlob && (
             <section className="bg-white/80 backdrop-blur-sm rounded-3xl p-6 shadow-xl border border-white/50 animate-fade-in-up">
                <div className="text-center mb-8"><h2 className="text-2xl font-bold text-slate-800">3D 생성이 완료되었습니다!</h2></div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
                   <div className="bg-slate-50 rounded-2xl p-4 border border-slate-200 flex flex-col items-center justify-center min-h-[200px]">
                      {thumbnailBlob ? <img src={URL.createObjectURL(thumbnailBlob)} alt="Thumbnail" className="rounded-lg shadow-md max-h-48 object-cover" /> : <div className="text-slate-400 text-sm">썸네일을 만들어보세요</div>}
                      <button type="button" onClick={handleManualThumbnail} className="mt-4 text-sm text-blue-600 font-medium hover:underline">📸 썸네일 직접 만들기</button>
                      {isGeneratingThumbnail && (
                         <div className="hidden">
                            <AutoThumbnailGenerator modelUrl={URL.createObjectURL(modelBlob)} onThumbnailGenerated={handleAutoThumbnailGenerated} />
                         </div>
                      )}
                   </div>
                   <div className="space-y-4">
                      <div><label className="block text-sm font-black text-slate-900 mb-2">제목</label><input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="멋진 이름을 지어주세요" className="w-full px-4 py-3 border border-slate-300 rounded-xl outline-none focus:ring-2 focus:ring-blue-400 bg-white text-slate-950 font-medium placeholder-slate-500 shadow-sm" /></div>
                      <div><label className="block text-sm font-black text-slate-900 mb-2">내용</label><textarea value={content} onChange={(e) => setContent(e.target.value)} placeholder="설명을 적어주세요." rows={4} className="w-full px-4 py-3 border border-slate-300 rounded-xl outline-none resize-none focus:ring-2 focus:ring-blue-400 bg-white text-slate-950 font-medium placeholder-slate-500 shadow-sm" /></div>
                   </div>
                </div>
                <div className="flex gap-3">
                   <button type="button" onClick={() => router.push('/')} className="flex-1 py-4 rounded-xl font-bold text-slate-600 bg-slate-100 hover:bg-slate-200">취소</button>
                   <button type="submit" disabled={isProcessing || isGeneratingThumbnail} className="flex-[2] py-4 rounded-xl font-bold text-white bg-gradient-to-r from-blue-500 to-sky-600 hover:shadow-lg transition-all shadow-blue-500/20">{processingStage === 'saving' ? '저장 중...' : '💾 저장하기'}</button>
                </div>
             </section>
          )}
        </form>
      </div>
    </div>
  );
}