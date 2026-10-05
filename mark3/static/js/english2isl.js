// English -> ISL gloss (the signs to show, in ISL order).
//  - drops words ISL does not sign: articles, forms of "to be", "to", "of" ...
//  - maps English words and phrases to the signs the model knows (synonyms included)
//  - ISL order: greetings first, then time ("tomorrow"), then the rest; adjectives follow the noun
//  - a word with no sign is kept as a "spell" item, shown as text (or teach it in Teach mode)

const SYNONYMS = {
  "thank you": "thankyou", thanks: "thankyou", "thank u": "thankyou",
  "good morning": "goodmorning", "good afternoon": "goodafternoon", "good evening": "goodevening", "good night": "goodnight",
  "how are you": "howareyou", "how r u": "howareyou", "nice to meet you": "pleased", "pleased to meet you": "pleased",
  hi: "hello", hey: "hello", namaste: "hello", ok: "alright", okay: "alright", "all right": "alright", fine: "alright",
  "train ticket": "trainticket", ticket: "trainticket", tickets: "trainticket",
  "train station": "trainstation", "railway station": "trainstation", station: "trainstation",
  store: "storeorshop", shop: "storeorshop", "street": "streetorroad", road: "streetorroad",
  big: "biglarge", large: "biglarge", huge: "biglarge", small: "smalllittle", little: "smalllittle", tiny: "smalllittle",
  phone: "cellphone", mobile: "cellphone", "cell phone": "cellphone", "mobile phone": "cellphone", tv: "television",
  "t-shirt": "tshirt", "t shirt": "tshirt", pants: "pant", trousers: "pant", shoe: "shoes",
  "you all": "youplural", "all of you": "youplural", me: "i", my: "i", mine: "i", myself: "i",
  your: "you", yours: "you", him: "he", his: "he", her: "she", hers: "she", us: "we", our: "we",
  them: "they", their: "they", its: "it",
  mom: "mother", mum: "mother", mummy: "mother", amma: "mother", dad: "father", papa: "father", appa: "father",
  kid: "child", kids: "child", children: "child", grandpa: "grandfather", grandma: "grandmother",
  ill: "sick", unwell: "sick", tablet: "medicine", tablets: "medicine", pills: "medicine", medicines: "medicine",
  cash: "money", rupees: "money", rupee: "money", cost: "price", costly: "expensive",
  taxi: "car", cab: "car", bike: "bicycle", cycle: "bicycle", aeroplane: "plane", airplane: "plane", flight: "plane",
  rain: "exmonsoon", monsoon: "exmonsoon", autumn: "fall", color: "colour", gray: "grey", photo: "photograph",
  cop: "police", "police officer": "police", toilet: "bathroom", restroom: "bathroom", washroom: "bathroom",
  road: "streetorroad", lady: "woman", gentleman: "man", buddy: "friend", pal: "friend",
  race: "raceethnicity", ethnicity: "raceethnicity", "store or shop": "storeorshop",
};

const QUESTION = new Set(["what", "where", "when", "who", "why", "how", "which", "whose"]);
const NEGATION = new Set(["not", "no", "never", "dont", "doesnt", "didnt", "cant", "cannot", "wont", "isnt", "arent", "wasnt"]);
const DROP = new Set(("a an the is am are was were be been being to of do does did will would shall should can could may " +
  "might must has have had and or but so very just really please much many lot lots too also quite  that this these those there here it's i'm you're " +
  "he's she's we're they're im youre for with at in on into from by as about want wants wanted need needs needed go goes going went gone get gets got getting").split(" "));
// "want/need/go" have no sign in the dataset: dropping them keeps the meaning in ISL's topic-first style
// (teach your own sign for them in Teach mode and they will be used instead).

export class EnglishToISL {
  constructor(labels, display, category) {
    this.known = new Set(labels.filter((l) => l !== "none"));
    this.display = display; this.category = category;
    this.phrases = new Map();
    for (const l of this.known) {
      this.phrases.set(l, l);
      const d = (display[l] || l).toLowerCase();
      if (d) this.phrases.set(d, l);
    }
    for (const [k, v] of Object.entries(SYNONYMS)) if (this.known.has(v)) this.phrases.set(k, v);
  }

  /** add personal signs (Teach mode) so they are used even before retraining */
  addWords(words) { for (const w of words) { this.known.add(w.label); this.phrases.set(w.label, w.label); if (w.display) this.phrases.set(w.display.toLowerCase(), w.label); } }

  _lookup(word) {
    if (this.phrases.has(word)) return this.phrases.get(word);
    if (DROP.has(word) || word.length <= 3) return null;      // "is" must never become "i"
    const tries = [];
    if (word.endsWith("ies")) tries.push(word.slice(0, -3) + "y");
    if (word.endsWith("es")) tries.push(word.slice(0, -2));
    if (word.endsWith("s")) tries.push(word.slice(0, -1));
    if (word.endsWith("ing")) tries.push(word.slice(0, -3), word.slice(0, -3) + "e");
    if (word.endsWith("ed")) tries.push(word.slice(0, -2), word.slice(0, -1));
    for (const t of tries) if (this.phrases.has(t)) return this.phrases.get(t);
    return null;
  }

  /** "Good morning. I need a train ticket for tomorrow" -> [{label, text, kind, clause}] in ISL order.
   *  ISL grammar, per clause: greetings first, then TIME, SUBJECT, OBJECTS/PLACES, DESCRIPTION,
   *  then the negative (NOT) and finally the question word (WHERE, WHAT ...). */
  translate(text) {
    const clauses = String(text).toLowerCase().replace(/[’']/g, "'")
      .split(/[.!?;,]+|\s+(?:and|but|because|so|then)\s+/).map((c) => c.trim()).filter(Boolean);
    const out = [];
    clauses.forEach((c, ci) => { for (const t of this._clause(c)) out.push({ ...t, clause: ci }); });
    // greetings always open the conversation
    const greet = out.filter((t) => t.cat === "greetings");
    return [...greet, ...out.filter((t) => t.cat !== "greetings")];
  }

  _clause(clause) {
    const words = clause.replace(/[^a-z0-9' -]+/g, " ").split(/\s+/).filter(Boolean);
    const items = [];
    const POSSESSIVE = new Set(["my", "your", "his", "her", "our", "their", "its"]);
    for (let i = 0; i < words.length;) {
      // "my mother" -> MOTHER: the possessive is not signed before a noun
      if (POSSESSIVE.has(words[i]) && i + 1 < words.length && (this._lookup(words[i + 1]) || this.phrases.get(words.slice(i + 1, i + 3).join(" ")))) { i++; continue; }
      let hit = null, len = 0;
      for (let n = Math.min(4, words.length - i); n >= 1; n--) {          // longest phrase first
        const phrase = words.slice(i, i + n).join(" ");
        const l = n === 1 ? this._lookup(phrase) : this.phrases.get(phrase);
        if (l) { hit = l; len = n; break; }
      }
      if (hit) { items.push({ label: hit, text: this.display[hit] || hit, kind: "sign", cat: this.category[hit] || "" }); i += len; continue; }
      const w = words[i++].replace(/'/g, "");
      if (NEGATION.has(w)) items.push({ label: null, text: "not", kind: "spell", cat: "negation" });
      else if (!DROP.has(w) && w.length > 0) items.push({ label: null, text: w, kind: "spell", cat: QUESTION.has(w) ? "question" : "" });
    }
    // one subject per clause ("I ... my" -> I), no repeated signs
    const seen = new Set();
    const toks = items.filter((t) => { const k = t.label || t.text; if (seen.has(k)) return false; seen.add(k); return true; });
    const isAdj = (t) => t.cat === "adjectives" || t.cat === "colours";
    const greet = toks.filter((t) => t.cat === "greetings");
    const time = toks.filter((t) => t.cat === "days and time" || t.cat === "seasons");
    let subj = toks.find((t) => t.cat === "pronouns") || null;
    if (subj && subj.label === "it" && toks.filter((t) => t.kind === "sign").length > 1) { toks.splice(toks.indexOf(subj), 1); subj = null; }   // dummy "it"
    const neg = toks.filter((t) => t.cat === "negation");
    const q = toks.filter((t) => t.cat === "question");
    // words without a sign are mostly verbs ("buy", "call", "open"): ISL puts the action last
    const verbs = toks.filter((t) => t.kind === "spell" && !t.cat);
    const rest = toks.filter((t) => !greet.includes(t) && !time.includes(t) && t !== subj && !neg.includes(t) && !q.includes(t) && !verbs.includes(t));
    // a description follows what it describes: "red car" -> CAR RED; "I am sick" -> I SICK
    const body = [];
    for (let i = 0; i < rest.length; i++) {
      const a = rest[i], b = rest[i + 1];
      if (isAdj(a) && b && !isAdj(b) && b.kind === "sign") { body.push(b, a); i++; } else body.push(a);
    }
    const desc = body.filter(isAdj);
    // keep adjective-after-noun pairs together, other descriptions after the objects
    const ordered = [];
    for (const t of body) if (!isAdj(t) || ordered.length && !isAdj(ordered[ordered.length - 1]) && body.indexOf(t) === body.indexOf(ordered[ordered.length - 1]) + 1) ordered.push(t);
    for (const t of desc) if (!ordered.includes(t)) ordered.push(t);
    return [...greet, ...time, ...(subj ? [subj] : []), ...ordered, ...verbs, ...neg, ...q];
  }
}
