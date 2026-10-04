// Turns recognised ISL signs (gloss, in signing order) into English, and keeps a running summary.
// Works offline with rules; the server can upgrade the wording with a local LLM if one is set up.
//
// ISL is topic-first and has no articles or "to be"; the dataset's vocabulary has no verbs or
// question words. So the rules: greetings become their own sentences, a pronoun or person is the
// subject (default "I": the signer), adjectives describe the subject, things/people/jobs are what
// the subject needs, places become "at/to the …", and times go at the end.

const SUBJECT_FORMS = {
  i: ["I", "am", "my"], you: ["you", "are", "your"], youplural: ["you all", "are", "your"],
  he: ["he", "is", "his"], she: ["she", "is", "her"], we: ["we", "are", "our"],
  they: ["they", "are", "their"], it: ["it", "is", "its"],
};
const GREETING = {
  hello: "Hello.", thankyou: "Thank you.", goodmorning: "Good morning.", goodafternoon: "Good afternoon.",
  goodevening: "Good evening.", goodnight: "Good night.", howareyou: "How are you?",
  pleased: "Pleased to meet you.", alright: "All right.",
};
const MASS = new Set(["money", "medicine", "energy", "paper", "soap", "peace", "science", "technology", "time",
  "transportation", "clothing", "exercise", "sport", "religion", "war", "death", "marriage", "paint", "dream"]);
const NO_ARTICLE = new Set(["india", "god"]);
const THE = new Set(["police", "king", "queen", "president", "sun", "court"]);
const TIME_PHRASE = (w, d) => {
  if (["today", "tomorrow", "yesterday"].includes(w)) return d;
  if (["morning", "afternoon", "evening"].includes(w)) return `in the ${d}`;
  if (w === "night") return "at night";
  if (/day$/.test(w)) return `on ${d}`;
  if (["week", "month", "year"].includes(w)) return `this ${d}`;
  return d;
};
const CAT = { pron: "pronouns", greet: "greetings", adj: "adjectives", col: "colours", time: "days and time",
  season: "seasons", place: "places", people: "people", job: "jobs" };

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const article = (w, d) => {
  if (THE.has(w)) return "the " + d;
  if (MASS.has(w) || NO_ARTICLE.has(w) || /s$/.test(w) && !/ss$/.test(w)) return d;
  return (/^[aeiou]/i.test(d) ? "an " : "a ") + d;
};
const list = (xs) => xs.length <= 1 ? (xs[0] || "") : xs.slice(0, -1).join(", ") + " and " + xs[xs.length - 1];

/** tokens: [{label, text, cat}] in signing order -> { english, parts } */
export function toEnglish(tokens) {
  const out = [];
  const rest = [];
  for (const t of tokens) {
    if (t.cat === CAT.greet && GREETING[t.label]) out.push(GREETING[t.label]);
    else rest.push(t);
  }
  // collapse immediate repeats (signers repeat for emphasis)
  const toks = rest.filter((t, i) => i === 0 || t.label !== rest[i - 1].label);
  if (!toks.length) return { english: out.join(" "), parts: {} };

  let subj = null;
  const adjs = [], needs = [], places = [], times = [], people = [], other = [];
  for (const t of toks) {
    if (t.cat === CAT.pron && !subj) subj = t;
    else if (t.cat === CAT.adj || t.cat === CAT.col) adjs.push(t);
    else if (t.cat === CAT.time || t.cat === CAT.season) times.push(t);
    else if (t.cat === CAT.place) places.push(t);
    else if ((t.cat === CAT.people) && !subj && !adjs.length && !needs.length) subj = t;
    else if (t.cat === CAT.people || t.cat === CAT.job) { people.push(t); needs.push(t); }
    else needs.push(t);
  }
  // "car red": a thing followed by a description, with no one named, describes the thing
  if (!subj && adjs.length && needs.length && toks.indexOf(needs[0]) < toks.indexOf(adjs[0])) {
    subj = needs.shift();
    subj = { ...subj, thing: true };
  }
  let S, be;
  if (!subj) [S, be] = ["I", "am"];
  else if (SUBJECT_FORMS[subj.label]) [S, be] = SUBJECT_FORMS[subj.label];
  else if (subj.thing) [S, be] = [`the ${subj.text}`, "is"];
  else [S, be] = [`my ${subj.text}`, "is"];

  const clauses = [];
  if (adjs.length) clauses.push(`${S} ${be} ${list(adjs.map((a) => a.text))}`);
  // a person + a job/person after it reads as identity: "father doctor" -> "my father is a doctor"
  if (subj && !SUBJECT_FORMS[subj.label] && !subj.thing && people.length && needs.length === people.length && !adjs.length) {
    clauses.push(`${S} ${be} ${list(people.map((p) => article(p.label, p.text)))}`);
    needs.length = 0;
  }
  const wants = needs.map((n) => article(n.label, n.text));
  const third = be === "is";
  if (wants.length) clauses.push(`${clauses.length ? "" : S + " "}${third ? "needs" : "need"} ${list(wants)}`.trim());
  let sentence = clauses.join(" and ");
  if (places.length) {
    const p = list(places.map((p) => NO_ARTICLE.has(p.label) ? p.text : `the ${p.text}`));
    sentence = sentence ? `${sentence} at ${p}` : `${S} ${be} going to ${p}`;
  }
  if (!sentence) sentence = subj ? (SUBJECT_FORMS[subj.label] ? S : cap(S)) : "";
  if (times.length) {
    // "tomorrow morning", not "tomorrow in the morning"
    const day = times.find((t) => ["today", "tomorrow", "yesterday"].includes(t.label));
    const part = times.find((t) => ["morning", "afternoon", "evening", "night"].includes(t.label));
    let phrase;
    if (day && part) phrase = [day.text, part.text, ...times.filter((t) => t !== day && t !== part).map((t) => TIME_PHRASE(t.label, t.text))].join(" ");
    else phrase = times.map((t) => TIME_PHRASE(t.label, t.text)).join(" ");
    sentence = `${sentence} ${phrase}`.trim();
  }
  if (sentence) out.push(cap(sentence.trim()) + ".");
  return { english: out.join(" "), parts: { subject: S, adjs, needs: wants, places, times } };
}

/** Third-person summary of the whole conversation: [{english, tokens}] -> text */
export function summarize(history) {
  if (!history.length) return "";
  const greet = history.some((h) => h.tokens.some((t) => t.cat === CAT.greet));
  const thanks = history.some((h) => h.tokens.some((t) => t.label === "thankyou"));
  const lines = [];
  if (greet) lines.push("They greeted you" + (thanks ? " and said thank you." : "."));
  for (const h of history) {
    const body = h.english.replace(/^(Hello|Thank you|Good (morning|afternoon|evening|night)|All right|Pleased to meet you)\.\s*/g, "")
      .replace(/How are you\?\s*/, "");
    if (!body) continue;
    lines.push(body
      .replace(/\bI am\b/g, "they are").replace(/\bI need\b/g, "they need").replace(/\bI\b/g, "they")
      .replace(/\b[Mm]y\b/g, "their").replace(/^(they|their)/, (m) => cap(m)));
  }
  const topics = new Set();
  for (const h of history) for (const t of h.tokens) {
    if (["sick", "doctor", "hospital", "medicine", "patient", "healthy", "blind", "deaf"].includes(t.label)) topics.add("health");
    if (t.cat === "means of transportation" || ["trainstation", "trainticket"].includes(t.label)) topics.add("travel");
    if (["money", "bank", "price", "bill", "cheap", "expensive", "market", "storeorshop"].includes(t.label)) topics.add("money and shopping");
    if (["police", "court", "lawyer", "attack", "gun"].includes(t.label)) topics.add("safety");
    if (t.cat === "people") topics.add("family and people");
  }
  const urgent = history.some((h) => h.tokens.some((t) => ["attack", "gun", "dead", "death", "sick", "police"].includes(t.label)));
  return { text: lines.join(" "), topics: [...topics], urgent };
}
