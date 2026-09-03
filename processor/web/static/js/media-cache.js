/* Persistent browser cache for review-side derived data.
 *
 * The source videos/parquet files remain server-side. This store only keeps
 * transport responses for 2D keypoints, 3D landmarks and glove frames. The
 * cache is best-effort: private/incognito browsers or quota pressure simply
 * fall back to the normal network path.
 */
(function () {
    'use strict';

    const DB_NAME = 'egodata-review-cache';
    const DB_VERSION = 1;
    const STORE = 'payloads';
    let dbPromise = null;

    function open() {
        if (dbPromise) return dbPromise;
        if (!window.indexedDB) return Promise.resolve(null);
        dbPromise = new Promise(resolve => {
            try {
                const request = indexedDB.open(DB_NAME, DB_VERSION);
                request.onupgradeneeded = () => {
                    const db = request.result;
                    if (!db.objectStoreNames.contains(STORE)) {
                        db.createObjectStore(STORE, { keyPath: 'key' });
                    }
                };
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => resolve(null);
                request.onblocked = () => resolve(null);
            } catch (_) { resolve(null); }
        });
        return dbPromise;
    }

    async function get(key) {
        const db = await open();
        if (!db || !key) return null;
        return new Promise(resolve => {
            try {
                const tx = db.transaction(STORE, 'readonly');
                const req = tx.objectStore(STORE).get(key);
                req.onsuccess = () => resolve(req.result || null);
                req.onerror = () => resolve(null);
            } catch (_) { resolve(null); }
        });
    }

    async function put(key, value) {
        const db = await open();
        if (!db || !key) return false;
        return new Promise(resolve => {
            try {
                const tx = db.transaction(STORE, 'readwrite');
                tx.objectStore(STORE).put({ key, value, savedAt: Date.now() });
                tx.oncomplete = () => resolve(true);
                tx.onerror = () => resolve(false);
                tx.onabort = () => resolve(false);
            } catch (_) { resolve(false); }
        });
    }

    async function removeEpisode(episodeId) {
        const db = await open();
        if (!db || !episodeId) return;
        return new Promise(resolve => {
            try {
                const tx = db.transaction(STORE, 'readwrite');
                const store = tx.objectStore(STORE);
                const req = store.openCursor();
                req.onsuccess = () => {
                    const cursor = req.result;
                    if (!cursor) return;
                    if (String(cursor.key).startsWith(`${episodeId}:`)) cursor.delete();
                    cursor.continue();
                };
                tx.oncomplete = () => resolve();
                tx.onerror = () => resolve();
                tx.onabort = () => resolve();
            } catch (_) { resolve(); }
        });
    }

    window.EgoMediaCache = { get, put, removeEpisode };
})();
