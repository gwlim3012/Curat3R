/**
 * 파이프라인 서비스: YOLO 필터링 + StableFast3D 3D 재구성
 */

const API_BASE_URL = '/api/pipeline';

export interface FilterResult {
  status: 'early_reject' | 'reject' | 'major_revision' | 'minor_revision' | 'accept' | 'error';
  reasons?: string[];
}

export interface ProcessResponse {
  task_id: string;
  stage: 'filtering' | 'reconstruction' | 'completed';
  filter_result: FilterResult;
  mesh_path?: string;
  reconstruction_error?: string;
  message?: string;
  error?: string;
}

class PipelineService {
  /**
   * 헬스 체크
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch (error) {
      console.error('Pipeline service health check failed:', error);
      return false;
    }
  }

  /**
   * 이미지 필터링만 수행
   */
  async filterImage(imageFile: File): Promise<{ task_id: string; filter_result: FilterResult }> {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/filter`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || '필터링 실패');
    }

    return await response.json();
  }

  /**
   * 필터링 후 3D 재구성만 수행 (필터링 건너뛰기) - 폴링 방식
   */
  async reconstructOnly(
    taskId: string,
    modelType: 'fast' | 'quality' = 'fast',
    onProgress?: (stage: string, message: string) => void
  ): Promise<ProcessResponse> {
    const modelName = modelType === 'quality' ? 'Trellis (Quality)' : 'StableFast3D (Fast)';
    const estimatedTime = modelType === 'quality' ? '5-15분' : '1-3분';
    onProgress?.('reconstruction', `${modelName} 3D 재구성 중... (${estimatedTime} 소요)`);

    const response = await fetch(`${API_BASE_URL}/reconstruct/${taskId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model: modelType }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || '재구성 실패');
    }

    const result: ProcessResponse = await response.json();

    if (result.stage === 'completed') {
      onProgress?.('completed', '3D 재구성 완료!');
    }

    return result;
  }

  /**
   * 전체 파이프라인 수행: 필터링 + 3D 재구성 (한 번에)
   */
  async processImage(
    imageFile: File,
    onProgress?: (stage: string, message: string) => void
  ): Promise<ProcessResponse> {
    onProgress?.('upload', '이미지 업로드 중...');

    const formData = new FormData();
    formData.append('image', imageFile);

    onProgress?.('filtering', 'YOLO 필터링 중...');

    const response = await fetch(`${API_BASE_URL}/process`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || '처리 실패');
    }

    const result: ProcessResponse = await response.json();

    if (result.stage === 'completed') {
      onProgress?.('completed', '3D 재구성 완료!');
    } else if (result.stage === 'filtering') {
      onProgress?.('filtering', '필터링 단계에서 중단됨');
    }

    return result;
  }

  /**
   * 생성된 3D 모델 다운로드
   */
  async downloadModel(taskId: string): Promise<Blob> {
    const response = await fetch(`${API_BASE_URL}/download/${taskId}`);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || '다운로드 실패');
    }

    return await response.blob();
  }

  /**
   * 작업 디렉토리 정리
   */
  async cleanup(taskId: string): Promise<void> {
    const response = await fetch(`${API_BASE_URL}/cleanup/${taskId}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      console.error('Cleanup failed for task:', taskId);
    }
  }

  /**
   * 필터 결과 메시지를 사용자 친화적으로 변환
   */
  getFilterStatusMessage(filterResult: FilterResult): { emoji: string; title: string; description: string; canProceed: boolean } {
    switch (filterResult.status) {
      case 'accept':
        return {
          emoji: '✅',
          title: '완벽합니다!',
          description: '3D 재구성에 적합한 이미지입니다.',
          canProceed: true,
        };
      case 'minor_revision':
        return {
          emoji: '⚠️',
          title: '주의사항이 있습니다',
          description: filterResult.reasons?.join(', ') || '일부 개선이 필요합니다.',
          canProceed: true,
        };
      case 'major_revision':
        return {
          emoji: '🔴',
          title: '주요 문제가 발견되었습니다',
          description: filterResult.reasons?.join(', ') || '여러 개선이 필요합니다.',
          canProceed: true,
        };
      case 'reject':
        return {
          emoji: '❌',
          title: '3D 재구성 불가',
          description: filterResult.reasons?.join(', ') || '이 이미지는 3D 재구성에 적합하지 않습니다.',
          canProceed: false,
        };
      case 'early_reject':
        return {
          emoji: '❌',
          title: '처리할 수 없습니다',
          description: filterResult.reasons?.join(', ') || '이미지를 인식할 수 없습니다.',
          canProceed: false,
        };
      case 'error':
        return {
          emoji: '⚠️',
          title: '오류 발생',
          description: filterResult.reasons?.join(', ') || '처리 중 오류가 발생했습니다.',
          canProceed: false,
        };
      default:
        return {
          emoji: '❓',
          title: '알 수 없는 상태',
          description: '상태를 확인할 수 없습니다.',
          canProceed: false,
        };
    }
  }
}

export const pipelineService = new PipelineService();
