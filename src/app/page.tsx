'use client';

import { useEffect, useState } from 'react';
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd';
import { Archive, Folder } from '@/lib/db';
import { archiveService } from '@/services/archiveService';
import ArchiveCard from '@/components/ArchiveCard';
import FolderStackCard from '@/components/FolderStackCard';
import Link from 'next/link';
import UserProfileCard from '@/components/UserProfileCard';

// 폴더 타입에 fileCount(파일 개수) 추가
interface FolderWithCount extends Omit<Folder, 'id' | 'name'> {
  id: number;
  name: string;
  fileCount?: number;
  thumbnails?: string[]; // 썸네일 url 최대 3개
}

// 그리드 아이템 타입
type GridItem = 
  | { type: 'FOLDER'; data: FolderWithCount }
  | { type: 'ARCHIVE'; data: Archive };

export default function HomePage() {
  const [archives, setArchives] = useState<Archive[]>([]);
  const [folders, setFolders] = useState<FolderWithCount[]>([]);
  const [currentFolderId, setCurrentFolderId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [currentFolderTitle, setCurrentFolderTitle] = useState<string>('');
  const [currentFolderCount, setCurrentFolderCount] = useState<number>(0);

  useEffect(() => {
    loadData();
  }, [currentFolderId]);

  useEffect(() => {
    // Fetch folder title and count when entering a folder
    if (currentFolderId !== null) {
      (async () => {
        try {
          const all = await archiveService.getAllFolders();
          const f = all.find(x => x.id === currentFolderId);
          setCurrentFolderTitle(f?.name ?? '');
          const folderArchives = await archiveService.getAllArchives(currentFolderId);
          setCurrentFolderCount(folderArchives.length);
        } catch (e) {
          setCurrentFolderTitle('');
          setCurrentFolderCount(0);
        }
      })();
    } else {
      setCurrentFolderTitle('');
      setCurrentFolderCount(0);
    }
  }, [currentFolderId]);

  const loadData = async () => {
    setLoading(true);
    try {
      // 1. 현재 파일 로드
      const archiveData = await archiveService.getAllArchives(currentFolderId);
      setArchives(archiveData);

      // 2. 루트일 때만 폴더 로드
      if (currentFolderId === null) {
        const folderData = await archiveService.getAllFolders();
        // 각 폴더별로 최대 3개 썸네일 추출
        const foldersWithCount = await Promise.all(folderData.map(async (f) => {
          const folderArchives = await archiveService.getAllArchives(f.id);
          // 썸네일 url 생성 (최대 3개)
          const thumbnails: string[] = folderArchives.slice(0, 3).map(a => {
            if (a.fileType === 'image') {
              return URL.createObjectURL(a.fileBlob);
            } else if (a.fileType === 'model' && a.thumbnailBlob) {
              return URL.createObjectURL(a.thumbnailBlob);
            }
            return '';
          }).filter(Boolean);
          // Ensure id and name are always present
          return {
            id: f.id!,
            name: f.name!,
            fileCount: folderArchives.length,
            thumbnails,
            createdAt: f.createdAt,
            updatedAt: f.updatedAt,
          };
        }));
        setFolders(foldersWithCount);
      } else {
        setFolders([]);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  // 통합 그리드 아이템
  const gridItems: GridItem[] = [
    ...folders.map(f => ({ type: 'FOLDER' as const, data: f })),
    ...archives.map(a => ({ type: 'ARCHIVE' as const, data: a })),
  ];

  // --- 드래그 핸들러 ---
  const onDragEnd = async (result: DropResult) => {
    const { source, destination, combine, draggableId } = result;

    // If combine result is provided (older behavior), handle it
    if (combine) {
      const targetIdStr = combine.draggableId; // 폴더 ID (target)
      const sourceIdStr = draggableId;         // 파일 ID (source)

      if (targetIdStr.startsWith('folder-') && sourceIdStr.startsWith('archive-')) {
        const targetFolderId = Number(targetIdStr.split('-')[1]);
        const sourceArchiveId = Number(sourceIdStr.split('-')[1]);

        // 확인 메시지
        const targetFolder = folders.find(f => f.id === targetFolderId);
        if (window.confirm(`'${targetFolder?.name}' 폴더로 이동하시겠습니까?`)) {
          // [수정] UI에서 즉시 제거 (복사처럼 보이는 현상 방지)
          setArchives(prev => prev.filter(a => a.id !== sourceArchiveId));
          
          try {
            await archiveService.updateArchive(sourceArchiveId, { folderId: targetFolderId });
            // 이동 후 전체 데이터 새로고침
            await loadData();
          } catch (error) {
            console.error('Move failed', error);
            loadData(); // 실패 시 롤백을 위해 다시 로드
          }
        }
      }
      return;
    }

    // 2. Drop handling for archives into folder positions
    if (!destination) return;

    // If the destination droppable is a folder drop area, handle move directly
    if (destination.droppableId && destination.droppableId.startsWith('folder-drop-') && draggableId.startsWith('archive-')) {
      const targetFolderId = Number(destination.droppableId.replace('folder-drop-', ''));
      const sourceArchiveId = Number(draggableId.split('-')[1]);
      const targetFolder = folders.find(f => f.id === targetFolderId);
      
      if (window.confirm(`'${targetFolder?.name}' 폴더로 이동하시겠습니까?`)) {
        // [수정] UI에서 즉시 제거 (복사처럼 보이는 현상 방지)
        setArchives(prev => prev.filter(a => a.id !== sourceArchiveId));

        try {
          await archiveService.updateArchive(sourceArchiveId, { folderId: targetFolderId });
          // 이동 후 전체 데이터 새로고침
          await loadData();
        } catch (err) {
          console.error('Move failed', err);
          loadData(); // 실패 시 롤백을 위해 다시 로드
        }
      }
      return;
    }

    // previous heuristic fallback: only handle archive draggables if needed
    if (!draggableId.startsWith('archive-')) return;
  };

  // 폴더 추가
  const handleAddFolder = async () => {
    const name = prompt('새 폴더 이름');
    if (name) {
      await archiveService.addFolder(name);
      loadData();
    }
  };

  const currentFolderName = currentFolderId !== null
    ? (folders.find(f => f.id === currentFolderId)?.name || '')
    : '';

  // 정렬 상태 및 핸들러 추가
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');

  const handleSort = (order: 'desc' | 'asc') => {
    setSortOrder(order);
    setArchives(prev => {
      const sorted = [...prev].sort((a, b) => {
        const aTime = new Date(a.createdAt).getTime();
        const bTime = new Date(b.createdAt).getTime();
        return order === 'desc' ? bTime - aTime : aTime - bTime;
      });
      return sorted;
    });
  };

  return (
    <div className="min-h-screen bg-slate-50 selection:bg-blue-100 selection:text-blue-900 pb-20">
      
      {/* 헤더 */}
      <div className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {currentFolderId !== null ? (
              <button 
                onClick={() => setCurrentFolderId(null)}
                className="flex items-center gap-1 text-slate-500 hover:text-blue-600 font-bold transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" /></svg>
                BACK
              </button>
            ) : (
              <div 
                className="text-2xl font-black bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-transparent cursor-pointer" 
                onClick={() => setCurrentFolderId(null)}
              >
                Curat3R
              </div>
            )}
          </div>
          {/* 왼쪽: 편집모드/폴더추가, 오른쪽: 정렬/업로드 */}
          <div className="flex flex-1 items-center justify-between">
            {/* 왼쪽 버튼 그룹 - 비워두기 */}
            <div></div>
            {/* 오른쪽 버튼 그룹 - 업로드 버튼만 */}
            <div className="flex items-center gap-4">
              <div className="inline-flex items-center gap-2 bg-transparent p-1 rounded-xl">
                <Link href="/upload-3d" aria-label="3D 모델 생성" className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-extrabold text-white shadow-sm bg-gradient-to-r from-cyan-500 to-blue-600 hover:scale-105 transition-transform duration-150 cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-cyan-400">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" xmlns="http://www.w3.org/2000/svg"><path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M12 2l8 4v8l-8 4-8-4V6l8-4zM12 8v8" /></svg>
                  <span>3D 모델 생성</span>
                </Link>

                <div className="w-px bg-slate-100 h-8 mx-2" />

                <Link href="/upload" aria-label="일반 업로드" className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold text-indigo-600 bg-white border border-indigo-100 shadow-sm hover:scale-105 hover:shadow-md transition-transform duration-150 cursor-pointer focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-indigo-200">
                  <svg className="w-4 h-4 text-indigo-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" xmlns="http://www.w3.org/2000/svg"><path strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" d="M12 3v12m0 0l4-4m-4 4l-4-4M21 12v6a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3v-6"/></svg>
                  <span>일반 업로드</span>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 메인 컨텐츠 */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
           <UserProfileCard />
           {/* 버튼 그룹: 편집모드/폴더추가/정렬 */}
           <div className="mt-6 flex items-center justify-between">
             {/* 왼쪽: 폴더추가 */}
             <div className="flex gap-2">
               {currentFolderId === null && (
                 <button onClick={handleAddFolder} className="p-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition font-bold text-sm">
                   + 폴더 추가
                 </button>
               )}
             </div>
             {/* 오른쪽: 정렬 버튼 */}
             <div className="flex gap-2">
               <button
                 onClick={() => handleSort('desc')}
                 className={`px-4 py-2 rounded-lg text-sm font-bold transition ${sortOrder === 'desc' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-blue-100'}`}
               >
                 최신순 정렬
               </button>
               <button
                 onClick={() => handleSort('asc')}
                 className={`px-4 py-2 rounded-lg text-sm font-bold transition ${sortOrder === 'asc' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-blue-100'}`}
               >
                 오래된순 정렬
               </button>
              {/* 편집 모드 제거: 드래그로 정렬하지 않음 (폴더로 이동만 허용) */}
             </div>
           </div>
        </div>
        {currentFolderId !== null && (
          <div className="mb-6 flex items-baseline gap-3">
            <h2 className="text-2xl font-extrabold text-slate-800">{currentFolderTitle || currentFolderName}</h2>
            <span className="text-sm text-slate-400">{currentFolderCount}개</span>
          </div>
        )}
        <DragDropContext 
          onDragEnd={onDragEnd}
          enableDefaultSensors={true}
        >
          <Droppable 
            droppableId="main-grid" 
            isCombineEnabled={true}
            ignoreContainerClipping={true}
            mode="standard"
            renderClone={(providedClone, snapshotClone, rubric) => {
              const id = rubric.draggableId;
              const isFolder = id.startsWith('folder-');
              if (isFolder) {
                const fid = Number(id.split('-')[1]);
                const f = folders.find(x => x.id === fid);
                if (!f) return null;
                return (
                  <div
                    ref={providedClone.innerRef}
                    {...providedClone.draggableProps}
                    {...providedClone.dragHandleProps}
                    style={providedClone.draggableProps.style}
                    className="w-[260px]"
                  >
                    <FolderStackCard folder={f} />
                  </div>
                );
              }
              const aid = Number(id.split('-')[1]);
              const a = archives.find(x => x.id === aid);
              if (!a) return null;
              return (
                <div
                  ref={providedClone.innerRef}
                  {...providedClone.draggableProps}
                  {...providedClone.dragHandleProps}
                  style={providedClone.draggableProps.style}
                  className="w-[260px]"
                >
                  <ArchiveCard archive={a} />
                </div>
              );
            }}
          >
            {(provided) => (
              <div 
                ref={provided.innerRef} 
                {...provided.droppableProps}
                className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 relative min-h-[200px]"
                style={{ gridAutoRows: 'minmax(320px, auto)' }}
              >
                {gridItems.map((item, index) => {
                  const uniqueId = item.type === 'FOLDER' 
                    ? `folder-${item.data.id}` 
                    : `archive-${item.data.id}`;

                  // Type guard: only render FolderStackCard if id and name are defined
                  if (item.type === 'FOLDER') {
                    const folder = item.data;
                    if (typeof folder.id !== 'number' || typeof folder.name !== 'string') return null;
                    // Make each folder its own Droppable so archives can be dropped into it
                    return (
                      <Droppable key={`folder-drop-${folder.id}`} droppableId={`folder-drop-${folder.id}`} isDropDisabled={false}>
                        {(providedFolder, snapshotFolder) => (
                          <div
                            ref={providedFolder.innerRef}
                            {...providedFolder.droppableProps}
                            className={`h-full transition-shadow duration-150 ease-linear opacity-100 relative overflow-hidden ${snapshotFolder.isDraggingOver ? 'ring-1 ring-blue-200 shadow-md' : ''}`}
                          >
                            <FolderStackCard
                              folder={folder}
                              onClick={() => setCurrentFolderId(folder.id)}
                              onDelete={async (id) => {
                                if (window.confirm('폴더를 삭제할까요? 폴더 내 파일은 전체보기로 이동합니다.')) {
                                  await archiveService.deleteFolder(id);
                                  loadData();
                                }
                              }}
                              onRename={async (id, name) => {
                                try {
                                  await archiveService.renameFolder(id, name);
                                  loadData();
                                } catch (err) {
                                  console.error('Rename failed', err);
                                  alert('폴더명 변경에 실패했습니다.');
                                }
                              }}
                            />
                            {snapshotFolder.isDraggingOver && <div className="absolute inset-0 pointer-events-none rounded-2xl bg-blue-50/40" />}
                            <div className="absolute inset-0 pointer-events-none flex items-center justify-center opacity-0">
                              {providedFolder.placeholder}
                            </div>
                          </div>
                        )}
                      </Droppable>
                    );
                  }
                  // Archive
                  return (
                    <Draggable 
                      key={uniqueId} 
                      draggableId={uniqueId} 
                      index={index}
                      isDragDisabled={false} 
                    >
                      {(provided, snapshot) => (
                        <div
                          ref={provided.innerRef}
                          {...provided.draggableProps}
                          {...provided.dragHandleProps}
                          style={{
                            ...provided.draggableProps.style,
                          }}
                          className={`h-full transform transition-all duration-150 ease-linear transform-gpu ${
                            snapshot.isDragging 
                              ? 'opacity-95 shadow-md z-50' 
                              : 'opacity-100'
                          }`}
                        >
                          <div className="transform transition-transform hover:-translate-y-1">
                            <ArchiveCard archive={item.data} />
                          </div>
                        </div>
                      )}
                    </Draggable>
                  );
                })}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        </DragDropContext>

        {!loading && gridItems.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20 text-slate-400">
             <p>보관함이 비어있습니다.</p>
          </div>
        )}
      </div>
    </div>
  );
}