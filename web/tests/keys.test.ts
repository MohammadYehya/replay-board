// Which keys a focused field keeps for itself (the player's keys and the 3D view's [ ] are left alone for those).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fieldTakesKey } from '../src/player/keys.ts';

const slider = { tagName: 'INPUT', type: 'range' };

test('a Try-it slider keeps only the keys that move it: Esc and [ ] still work while it has the focus', () => {
  for (const k of ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']) {
    assert.equal(fieldTakesKey(slider, k), true, k);
  }
  for (const k of ['Escape', '[', ']', ' ']) assert.equal(fieldTakesKey(slider, k), false, k);
});

test('text fields, text areas, selects and editable text keep every key', () => {
  for (const el of [
    { tagName: 'INPUT', type: 'text' },
    { tagName: 'INPUT' },
    { tagName: 'input', type: 'search' },
    { tagName: 'TEXTAREA' },
    { tagName: 'SELECT' },
    { tagName: 'DIV', isContentEditable: true },
  ]) {
    for (const k of ['Escape', '[', ' ', 'ArrowLeft', 'a']) assert.equal(fieldTakesKey(el, k), true, `${JSON.stringify(el)} ${k}`);
  }
});

test('anything else keeps no key', () => {
  for (const el of [null, undefined, { tagName: 'BUTTON' }, { tagName: 'DIV' }]) {
    for (const k of ['Escape', ']', ' ', 'ArrowRight']) assert.equal(fieldTakesKey(el, k), false);
  }
});
