import { Archive, UserProfile, Folder } from '@/lib/db';
import { db } from '@/lib/dbClient';

export const archiveService = {
  async addArchive(data: Omit<Archive, 'id' | 'createdAt' | 'updatedAt'>): Promise<number> {
    const id = await db.archives.add({
      ...data,
      createdAt: new Date(),
    });
    return id;
  },

  async updateArchive(id: number, data: Partial<Omit<Archive, 'id' | 'createdAt'>>): Promise<void> {
    await db.archives.update(id, {
      ...data,
      updatedAt: new Date(),
    });
  },

  async getArchive(id: number): Promise<Archive | undefined> {
    return await db.archives.get(id);
  },

  async getAllArchives(folderId?: number | null): Promise<Archive[]> {
    if (typeof folderId !== 'undefined' && folderId !== null) {
      // 특정 폴더의 파일들만 반환
      const arr = await db.archives.where('folderId').equals(folderId).sortBy('createdAt');
      return arr.reverse();
    } else {
      // 루트 화면: folderId가 null 또는 undefined인 파일들만 반환
      const allArchives = await db.archives.orderBy('createdAt').reverse().toArray();
      return allArchives.filter(a => a.folderId === null || a.folderId === undefined);
    }
  },
  // 폴더 관련 CRUD
  async getAllFolders(): Promise<Folder[]> {
    return await db.folders.orderBy('createdAt').toArray();
  },

  async addFolder(name: string): Promise<number> {
    return await db.folders.add({ name, createdAt: new Date() });
  },

  async renameFolder(id: number, name: string): Promise<void> {
    await db.folders.update(id, { name, updatedAt: new Date() });
  },

  async deleteFolder(id: number): Promise<void> {
    await db.folders.delete(id);
    // 폴더 삭제 시 해당 폴더의 아카이브는 folderId를 null로 변경
    await db.archives.where('folderId').equals(id).modify({ folderId: null });
  },

  async deleteArchive(id: number): Promise<void> {
    await db.archives.delete(id);
  },
};

export const profileService = {
  async getProfile(): Promise<UserProfile | undefined> {
    const profiles = await db.userProfile.toArray();
    return profiles[0];
  },

  async saveProfile(data: Omit<UserProfile, 'id' | 'updatedAt'>): Promise<void> {
    const existing = await this.getProfile();
    
    if (existing?.id) {
      await db.userProfile.update(existing.id, {
        ...data,
        updatedAt: new Date(),
      });
    } else {
      await db.userProfile.add({
        ...data,
        updatedAt: new Date(),
      });
    }
  },
};