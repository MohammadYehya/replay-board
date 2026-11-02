// Run with `npm test`. The Ask Muse tray: adding, removing, no duplicates, at most 12 items, and keeping it per
// lecture in sessionStorage (with junk in storage ignored).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  addEntry, cleanItem, hasEntry, itemKey, loadTray, openKey, removeEntry, saveTray, shortLabel, TRAY_MAX, trayKey,
  type TrayEntry,
} from '../src/ask/tray.ts';

const piece = (id: number): TrayEntry => ({ item: { kind: 'piece', id }, label: `piece ${id}`, t: id });

class MemoryStorage {
  map = new Map<string, string>();
  getItem(k: string) {
    return this.map.has(k) ? this.map.get(k)! : null;
  }
  setItem(k: string, v: string) {
    this.map.set(k, v);
  }
  removeItem(k: string) {
    this.map.delete(k);
  }
}

test('add, dedupe, remove', () => {
  let tray: TrayEntry[] = [];
  let r = addEntry(tray, piece(1));
  assert.equal(r.status, 'added');
  tray = r.tray;
  r = addEntry(tray, { ...piece(1), label: 'other label' });
  assert.equal(r.status, 'duplicate');
  assert.equal(r.tray, tray); // unchanged, same array
  tray = addEntry(tray, { item: { kind: 'line', id: 'L3' }, label: 'line', t: 5 }).tray;
  tray = addEntry(tray, { item: { kind: 'element', id: 's0e1' }, label: 'title', t: 0 }).tray;
  tray = addEntry(tray, { item: { kind: 'span', t0: 61.2, t1: 75 }, label: 'words', t: 61.2 }).tray;
  tray = addEntry(tray, { item: { kind: 'moment', t: 123.4 }, label: 'Moment', t: 123.4 }).tray;
  assert.equal(tray.length, 5);
  // the same moment to the second, or the same span, is the same item
  assert.equal(addEntry(tray, { item: { kind: 'moment', t: 123.1 }, label: 'Moment', t: 123.1 }).status, 'duplicate');
  assert.equal(addEntry(tray, { item: { kind: 'span', t0: 61.2, t1: 75.0 }, label: 'x', t: 0 }).status, 'duplicate');
  assert.equal(addEntry(tray, { item: { kind: 'span', t0: 61.2, t1: 76 }, label: 'x', t: 0 }).status, 'added');
  assert.ok(hasEntry(tray, { kind: 'line', id: 'L3' }));
  tray = removeEntry(tray, itemKey({ kind: 'line', id: 'L3' }));
  assert.ok(!hasEntry(tray, { kind: 'line', id: 'L3' }));
  assert.equal(tray.length, 4);
});

test('at most 12 items', () => {
  let tray: TrayEntry[] = [];
  for (let i = 0; i < TRAY_MAX; i++) tray = addEntry(tray, piece(i)).tray;
  assert.equal(TRAY_MAX, 12);
  assert.equal(tray.length, 12);
  const r = addEntry(tray, piece(99));
  assert.equal(r.status, 'full');
  assert.equal(r.tray.length, 12);
  // a duplicate is reported as such even when full
  assert.equal(addEntry(tray, piece(3)).status, 'duplicate');
});

test('saved per lecture in sessionStorage, read back checked', () => {
  const store = new MemoryStorage();
  assert.equal(trayKey('job-1'), 'replay-board:ask-tray:job-1');
  assert.notEqual(trayKey('job-1'), trayKey('job-2'));
  assert.notEqual(trayKey('job-1'), openKey('job-1'));
  const tray = [piece(4), { item: { kind: 'span' as const, t0: 1.234, t1: 5.678 }, label: 'said', t: 1.234 }];
  saveTray(store, 'job-1', tray);
  assert.ok(store.getItem('replay-board:ask-tray:job-1'));
  assert.deepEqual(loadTray(store, 'job-2'), []);
  const back = loadTray(store, 'job-1');
  assert.equal(back.length, 2);
  assert.deepEqual(back[0], piece(4));
  assert.deepEqual(back[1].item, { kind: 'span', t0: 1.23, t1: 5.68 });
  // an empty tray removes the key
  saveTray(store, 'job-1', []);
  assert.equal(store.getItem(trayKey('job-1')), null);
});

test('junk in storage, blocked storage and bad items are ignored', () => {
  const store = new MemoryStorage();
  store.setItem(trayKey('j'), 'not json');
  assert.deepEqual(loadTray(store, 'j'), []);
  store.setItem(
    trayKey('j'),
    JSON.stringify([
      { item: { kind: 'piece', id: 2 }, label: 'ok', t: 3 },
      { item: { kind: 'piece', id: 2 }, label: 'dup', t: 3 },
      { item: { kind: 'piece', id: 'x' } },
      { item: { kind: 'line', id: '3' } },
      { item: { kind: 'span', t0: 5, t1: 2 } },
      { item: { kind: 'virus' } },
      null,
    ]),
  );
  assert.deepEqual(loadTray(store, 'j'), [{ item: { kind: 'piece', id: 2 }, label: 'ok', t: 3 }]);
  const blocked = {
    getItem(): string | null {
      throw new Error('SecurityError');
    },
    setItem() {
      throw new Error('QuotaExceededError');
    },
    removeItem() {
      throw new Error('SecurityError');
    },
  };
  assert.deepEqual(loadTray(blocked, 'j'), []);
  assert.doesNotThrow(() => saveTray(blocked, 'j', [piece(1)]));
  assert.deepEqual(loadTray(null, 'j'), []);
  assert.equal(cleanItem({ kind: 'element', id: 's12e3' })?.kind, 'element');
  assert.equal(cleanItem({ kind: 'element', id: 'e3' }), null);
  assert.deepEqual(cleanItem({ kind: 'moment', t: 12.345 }), { kind: 'moment', t: 12.3 });
});

test('labels are cut on a word boundary', () => {
  assert.equal(shortLabel('  short   label '), 'short label');
  const long = 'the compound interest formula applied to a deposit over several years at a fixed rate';
  const cut = shortLabel(long);
  assert.ok(cut.length <= 60, cut);
  assert.ok(cut.endsWith('…'));
  assert.ok(long.startsWith(cut.slice(0, -1)));
});
