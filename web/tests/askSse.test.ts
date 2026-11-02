// Run with `npm test`. The answer streams in as server-sent events read from fetch(): chunks can end anywhere,
// even mid-line or between "\r" and "\n"; deltas arrive in order; "done" carries the answer; "error" is one line.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SseParser, type SseEvent } from '../src/ask/sse.ts';
import {
  cleanChat, cleanDone, costLabel, historyFor, readAnswer, requestSelection, StreamError,
} from '../src/ask/protocol.ts';

const ev = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const STREAM =
  ev('delta', { text: 'The area ' }) +
  ev('delta', { text: 'is $A = \\pi r^2$ [#12] at [1:05]. Ünïcode ✓' }) +
  ': keep-alive\n\n' +
  ev('done', {
    id: 't1',
    answer: 'The area is $A = \\pi r^2$ [#12] at [1:05]. Ünïcode ✓',
    items: [{ ref: '#12', kind: 'piece', id: 12, t: 61.5, label: 'A = pi r^2' }, { ref: 'bad' }],
    times: [65, 'x'],
    cost_usd: 0.0042,
    cached: false,
  });

const all = (p: SseParser, chunks: string[]) => {
  const out: SseEvent[] = [];
  for (const c of chunks) out.push(...p.push(c));
  out.push(...p.end());
  return out;
};

test('events come out whole however the text is split', () => {
  const whole = all(new SseParser(), [STREAM]);
  assert.deepEqual(
    whole.map((e) => e.event),
    ['delta', 'delta', 'done'],
  );
  for (let cut = 1; cut < STREAM.length; cut++) {
    const got = all(new SseParser(), [STREAM.slice(0, cut), STREAM.slice(cut)]);
    assert.deepEqual(got, whole, `split at ${cut}`);
  }
  // one character at a time
  assert.deepEqual(all(new SseParser(), [...STREAM]), whole);
});

test('CRLF and CR line ends, split between \\r and \\n; multi-line data; no space after the colon', () => {
  const crlf = 'event: delta\r\ndata: {"text":"a"}\r\n\r\nevent:done\r\ndata:{"x":1,\r\ndata: "y":2}\r\n\r\n';
  const want = [
    { event: 'delta', data: '{"text":"a"}' },
    { event: 'done', data: '{"x":1,\n"y":2}' },
  ];
  for (let cut = 1; cut < crlf.length; cut++) {
    assert.deepEqual(all(new SseParser(), [crlf.slice(0, cut), crlf.slice(cut)]), want, `split at ${cut}`);
  }
  assert.deepEqual(all(new SseParser(), [crlf.replace(/\r\n/g, '\r')]), want);
  // an event with no name is a "message"; an event left open when the stream ends is still delivered
  assert.deepEqual(all(new SseParser(), ['data: [DONE]']), [{ event: 'message', data: '[DONE]' }]);
});

/** A fetch body that delivers the given byte chunks. */
const body = (chunks: Uint8Array[]) =>
  new ReadableStream<Uint8Array>({
    start(c) {
      for (const x of chunks) c.enqueue(x);
      c.close();
    },
  });

const bytes = (s: string) => new TextEncoder().encode(s);

test('readAnswer: deltas in order (UTF-8 split mid-character), then the checked done event', async () => {
  const all = bytes(STREAM);
  for (const size of [1, 3, 7, 64, all.length]) {
    const chunks: Uint8Array[] = [];
    for (let i = 0; i < all.length; i += size) chunks.push(all.slice(i, i + size));
    const deltas: string[] = [];
    const done = await readAnswer(body(chunks), (d) => deltas.push(d));
    assert.equal(deltas.join(''), 'The area is $A = \\pi r^2$ [#12] at [1:05]. Ünïcode ✓');
    assert.equal(done.id, 't1');
    assert.equal(done.answer, deltas.join(''));
    assert.deepEqual(done.items, [{ ref: '#12', kind: 'piece', id: 12, t: 61.5, label: 'A = pi r^2' }]);
    assert.deepEqual(done.times, [65]);
    assert.equal(done.cost_usd, 0.0042);
    assert.equal(done.cached, false);
  }
});

test('readAnswer: an error event is one plain line; a stream cut short says so', async () => {
  const partial: string[] = [];
  await assert.rejects(
    readAnswer(body([bytes(ev('delta', { text: 'Half' }) + ev('error', { error: 'Muse Spark is busy.\nTry again.' }))]), (d) =>
      partial.push(d),
    ),
    (e: Error) => e instanceof StreamError && e.message === 'Muse Spark is busy. Try again.',
  );
  assert.deepEqual(partial, ['Half']);
  await assert.rejects(
    readAnswer(body([bytes(ev('delta', { text: 'Half' }))]), () => {}),
    /stopped before it finished/,
  );
  await assert.rejects(readAnswer(body([bytes(ev('error', {}))]), () => {}), /could not answer/);
});

test('done and chat payloads are checked field by field', () => {
  assert.deepEqual(cleanDone(null), { id: '', answer: '', items: [], times: [], cost_usd: null, cached: false, saved: true });
  assert.equal(cleanDone({ cached: true, cost_usd: 'x' }).cost_usd, null);
  const turns = cleanChat({
    turns: [
      { id: 'a', q: 'Why?', answer: 'Because.', selection: [{ kind: 'piece', id: 3 }, { kind: 'nope' }], t: 12, cost_usd: 0.01 },
      { q: 42, answer: 'no' },
      'junk',
    ],
  });
  assert.equal(turns.length, 1);
  assert.deepEqual(turns[0].selection, [{ kind: 'piece', id: 3 }]);
  assert.equal(turns[0].t, 12);
  assert.deepEqual(cleanChat({}), []);
});

test('history: the last six finished turns, answers cut to 2000 characters', () => {
  const turns = Array.from({ length: 9 }, (_, i) => ({ q: `q${i}`, answer: i === 8 ? 'x'.repeat(2500) : `a${i}`, ok: i !== 7 }));
  const h = historyFor(turns);
  assert.deepEqual(
    h.map((x) => x.q),
    ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q8'].slice(-6),
  );
  assert.equal(h.at(-1)!.a.length, 2000);
});

test('cost label', () => {
  assert.equal(costLabel(0.0042), '$0.004');
  assert.equal(costLabel(0.00004), '<$0.001');
  assert.equal(costLabel(0.25), '$0.25');
  assert.equal(costLabel(null), '');
  assert.equal(costLabel(0.004, true), 'cached');
});

test('readAnswer: once "done" arrives the answer is kept, whatever follows it', async () => {
  const s = ev('delta', { text: 'All of it.' }) + ev('done', { id: 't9', answer: 'All of it.' }) + ev('error', { error: 'late' });
  for (const chunks of [[bytes(s)], [...bytes(s)].map((b) => Uint8Array.of(b))]) {
    const done = await readAnswer(body(chunks), () => {});
    assert.equal(done.id, 't9');
    assert.equal(done.answer, 'All of it.');
  }
});

test('done says whether the turn was saved (a server without the field saved it)', () => {
  assert.equal(cleanDone({ id: 'a' }).saved, true);
  assert.equal(cleanDone({ id: 'a', saved: false }).saved, false);
  assert.equal(cleanDone({ id: 'a', saved: 'no' }).saved, true);
});

test('saved turns keep the labels of what was attached; a span sends the words the student selected', () => {
  const [turn] = cleanChat({
    turns: [
      {
        id: 'a',
        q: 'Why?',
        answer: 'Because.',
        selection: [
          { kind: 'span', t0: 61.2, t1: 75, label: 'so the interest is compounded', t: 61.2 },
          { kind: 'moment', t: 123.4, label: 'The board at 2:03' },
          { kind: 'piece', id: 3, label: '   ' },
          { kind: 'line', id: 'L2', label: 'x'.repeat(200) },
        ],
      },
    ],
  });
  assert.deepEqual(turn.selection.slice(0, 3), [
    { kind: 'span', t0: 61.2, t1: 75, label: 'so the interest is compounded' },
    { kind: 'moment', t: 123.4, label: 'The board at 2:03' },
    { kind: 'piece', id: 3 },
  ]);
  assert.ok((turn.selection[3] as { label: string }).label.length <= 60);
  assert.deepEqual(
    requestSelection([
      { item: { kind: 'span', t0: 1, t1: 2 }, label: 'the words', t: 1 },
      { item: { kind: 'span', t0: 3, t1: 4 }, label: ' ', t: 3 },
      { item: { kind: 'piece', id: 3 }, label: 'x = 3', t: 25 },
      { item: { kind: 'moment', t: 9 }, label: 'Moment', t: 9 },
    ]),
    [
      { kind: 'span', t0: 1, t1: 2, label: 'the words' },
      { kind: 'span', t0: 3, t1: 4 },
      { kind: 'piece', id: 3 },
      { kind: 'moment', t: 9 },
    ],
  );
});
