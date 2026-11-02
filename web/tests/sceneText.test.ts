// How the 3D view shows scene text and values: Greek names as letters, "degrees" as a sign, values rounded.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { angleLabel, displayName, formatDegrees, formatValue, formulaText, labelText, roundText, sliderStep, viewTitle } from '../src/player/sceneText.ts';

test('Greek names become letters, as whole words in any case', () => {
  assert.equal(displayName('alpha'), 'α');
  assert.equal(displayName('Beta is not 90, pushed over'), 'β is not 90, pushed over');
  assert.equal(displayName('Alpha and gamma stay at 90'), 'α and γ stay at 90');
  assert.equal(displayName('ALPHA BETA GAMMA DELTA THETA PHI PSI OMEGA LAMBDA MU SIGMA RHO TAU'), 'α β γ δ θ φ ψ ω λ μ σ ρ τ');
  assert.equal(displayName('angle theta'), 'angle θ');
  assert.equal(displayName('(beta), beta. beta: "gamma"'), '(β), β. β: "γ"');
  assert.equal(displayName('alpha-beta'), 'α-β');
});

test('never inside other words or names', () => {
  assert.equal(displayName('alphabet'), 'alphabet');
  assert.equal(displayName('betamax and gammas'), 'betamax and gammas');
  assert.equal(displayName('beta_arc mu2 xalpha'), 'beta_arc mu2 xalpha');
  assert.equal(displayName('mutation of the tautology'), 'mutation of the tautology');
  assert.equal(displayName('Muse Spark'), 'Muse Spark');
});

test('"<number> degrees" becomes "<number>°"', () => {
  assert.equal(displayName('all angles 90 degrees'), 'all angles 90°');
  assert.equal(displayName('Gamma 120 degrees, the internal hexagon angle'), 'γ 120°, the internal hexagon angle');
  assert.equal(displayName('1 degree more'), '1° more');
  assert.equal(displayName('12.5 Degrees'), '12.5°');
  assert.equal(displayName('beta is not 90'), 'β is not 90');
  assert.equal(displayName('degrees of freedom'), 'degrees of freedom');
  assert.equal(displayName('90 degreesish'), '90 degreesish');
  assert.equal(displayName(''), '');
});

test('values: degrees with one decimal and a sign, the rest with at most two decimals', () => {
  assert.equal(formatDegrees(75), '75°');
  assert.equal(formatDegrees(75.04), '75°');
  assert.equal(formatDegrees(109.47), '109.5°');
  assert.equal(formatDegrees(89.96), '90°');
  assert.equal(formatValue(90, 'deg'), '90°');
  assert.equal(formatValue(3, ''), '3');
  assert.equal(formatValue(2.756, ''), '2.76');
  assert.equal(formatValue(2.5, ''), '2.5');
  assert.equal(formatValue(-0.001, ''), '0');
  assert.equal(formatValue(250.4, ''), '250');
  assert.equal(roundText(NaN, 1), '–');
});

test('an angle\'s label: its name as a letter and its value', () => {
  assert.equal(angleLabel('beta', 75), 'β 75°');
  assert.equal(angleLabel('angle ABC', 90.04), 'angle ABC 90°');
  assert.equal(angleLabel('', 120), '120°');
  assert.equal(angleLabel('theta', 33.333), 'θ 33.3°');
});

test("a slider's step: whole degrees, or a round hundredth of the range", () => {
  assert.equal(sliderStep(60, 120, 'deg'), 1);
  assert.equal(sliderStep(85, 92, 'deg'), 0.1);
  assert.equal(sliderStep(1, 6, ''), 0.01);
  assert.equal(sliderStep(0, 100, ''), 1);
  assert.equal(sliderStep(-0.5, 0.5, ''), 0.01);
  assert.equal(sliderStep(2, 2, ''), 1);
});

test("the view's title: the scene's with letters for names; the board's own reading as it is", () => {
  assert.equal(viewTitle('Angle beta of the cell', 'beta = 75 degrees'), 'Angle β of the cell');
  // no scene title (still loading, or none): the lecture's own text is never changed
  assert.equal(viewTitle('', 'beta = 75 degrees'), 'beta = 75 degrees');
  assert.equal(viewTitle(undefined, '  alpha  '), 'alpha');
  assert.equal(viewTitle(undefined, undefined), 'Drawing from the board');
});

test('labels and titles show formulas the model wrote in words as symbols', () => {
  assert.equal(labelText('z equals minus y squared'), 'z = −y²');
  assert.equal(labelText('Saddle surface z equals x squared minus y squared'), 'Saddle surface z = x² − y²');
  assert.equal(labelText('Graph of y equals ln of x squared'), 'Graph of y = ln of x²');
  assert.equal(labelText('f(x) equals minus x cubed'), 'f(x) = −x³');
  assert.equal(labelText('angle beta'), 'angle β');
  assert.equal(labelText('saddle point'), 'saddle point'); // nothing else changes
  assert.equal(formulaText('the squared-off minus-sign'), 'the squared-off minus-sign'); // whole words only
  assert.equal(viewTitle('y equals x squared', 'board'), 'y = x²');
  assert.equal(displayName('z equals x squared'), 'z equals x squared'); // captions are sentences: left as they are
});
