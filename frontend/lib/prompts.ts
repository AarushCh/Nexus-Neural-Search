// Prompt pool for the 🎲 button.
//
// The old version was six hardcoded strings, so the dice repeated constantly and
// every roll produced the same handful of searches. This is a curated pool plus a
// combinatorial generator, drawn without replacement until the pool is exhausted.

const VIBE = [
  "rain-soaked melancholy",
  "cozy autumn evening",
  "loud dumb fun",
  "quietly devastating",
  "beautiful and mean",
  "hopeful sci-fi that isn't naive",
  "warm and low stakes",
  "elegantly bleak",
  "unbearably tense",
  "strange and tender",
  "lush and slow",
  "grimy and electric",
  "sad in a good way",
  "joyful chaos",
  "dread that builds quietly",
  "wistful and sunlit",
  "cold, clean, precise",
  "messy human feelings",
  "euphoric and loud",
  "deeply uncomfortable in a smart way",
];

const SCENARIO = [
  "something to watch at 3am",
  "a first date film that isn't a rom-com",
  "hungover Sunday viewing",
  "background noise while I work",
  "something to watch with my mum",
  "a film to cry to on purpose",
  "comfort rewatch energy",
  "something to put on with friends who talk over films",
  "a show I can binge in one weekend",
  "something short enough for a weeknight",
  "for when I can't concentrate",
  "to watch with the sound up loud",
  "a film that will ruin my evening in a good way",
  "something to fall asleep to",
  "a series to get lost in for a month",
  "for when I want to feel something again",
  "to watch on a long flight",
  "something my flatmates won't complain about",
];

const STRUCTURAL = [
  "one-location thriller",
  "single-take filmmaking",
  "unreliable narrator",
  "non-linear timeline",
  "no dialogue for the first twenty minutes",
  "told entirely in flashback",
  "one continuous night",
  "a story that starts at the ending",
  "multiple perspectives on one event",
  "a film that changes genre halfway through",
  "anthology of connected stories",
  "real time, no cuts away",
  "framed as a documentary but isn't",
  "the narrator is lying to you",
  "two timelines that converge",
];

const AESTHETIC = [
  "neon-drenched",
  "sun-bleached western",
  "brutalist and cold",
  "shot on film, feels like memory",
  "handheld and claustrophobic",
  "impossibly beautiful cinematography",
  "grainy seventies texture",
  "clinical white rooms",
  "saturated and dreamlike",
  "muted greys and rain",
  "practical effects, no CGI",
  "analogue horror aesthetic",
  "sweeping landscapes, tiny people",
  "all shot at golden hour",
  "cramped interiors and bad lighting",
];

const BENDING = [
  "horror that's secretly a comedy",
  "sci-fi that's actually about grief",
  "sports anime for people who hate sports",
  "a war film that's really a love story",
  "a heist that's actually about friendship",
  "a musical that isn't twee",
  "fantasy with no chosen one",
  "a romance with a body count",
  "a comedy that turns devastating",
  "a thriller where nobody is armed",
  "a courtroom drama that's secretly horror",
  "a coming-of-age story about old people",
  "a road movie that never leaves one town",
  "a zombie film about the living",
];

const HUNGERS = [
  "found family",
  "the villain is right",
  "slow-burn romance with actual tension",
  "competence porn",
  "a heist where the plan goes wrong",
  "a mentor who is wrong about everything",
  "two people arguing brilliantly",
  "an ensemble that actually feels like friends",
  "a protagonist who never wins",
  "impossible odds and a stupid plan",
  "quiet people doing difficult things",
  "revenge that costs too much",
  "someone rebuilding their life",
  "a friendship that survives the plot",
  "morally grey everyone",
  "the system is the antagonist",
  "an ending that earns itself",
  "a character study with nowhere to hide",
];

const CLASSIC = [
  "cyberpunk anime about identity",
  "80s horror",
  "deep space sci-fi",
  "noir mystery",
  "cozy slice of life",
  "mind-bending thriller",
  "psychological horror",
  "space opera",
  "courtroom drama",
  "heist thriller",
  "post-apocalyptic survival",
  "time loop story",
  "haunted house",
  "spy thriller with real tradecraft",
  "sword and sorcery",
  "small town secrets",
  "creature feature",
  "political thriller",
  "medical drama that isn't soapy",
  "nature documentary that feels like a thriller",
  "true crime that respects the victims",
  "dystopian satire",
  "folk horror",
  "samurai epic",
  "prison drama",
  "submarine thriller",
  "mockumentary comedy",
  "body horror",
  "coming-of-age in the suburbs",
  "gangster saga across decades",
];

export const CURATED = [
  ...VIBE, ...SCENARIO, ...STRUCTURAL, ...AESTHETIC, ...BENDING, ...HUNGERS, ...CLASSIC,
];

// Combinatorial layer — mixes a mood, an origin/era and a genre so the dice keep
// producing fresh phrasings once the curated pool has been walked through.
const MOODS = ["bleak", "cosy", "frantic", "dreamlike", "tense", "tender", "absurd",
  "melancholic", "hopeful", "brutal", "wry", "haunting"];
const ORIGINS = ["Korean", "Japanese", "French", "Nordic", "Italian", "Spanish", "British",
  "Australian", "Brazilian", "Indian", "90s", "80s", "70s", "modern", "near-future"];
const GENRES = ["thriller", "horror", "sci-fi", "romance", "crime drama", "comedy",
  "fantasy", "mystery", "anime", "documentary", "western", "noir"];

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function generated(): string {
  return `${pick(MOODS)} ${pick(ORIGINS)} ${pick(GENRES)}`;
}

// Session memory so the same prompt doesn't come up twice in a row, or twice at all
// until everything else has had a turn.
const seen = new Set<string>();

export function randomPrompt(): string {
  const fresh = CURATED.filter((p) => !seen.has(p));
  if (fresh.length === 0) seen.clear();

  // Mostly curated (they read better); occasionally a generated combination.
  const prompt =
    Math.random() < 0.8 ? pick(fresh.length ? fresh : CURATED) : generated();

  seen.add(prompt);
  return prompt;
}
