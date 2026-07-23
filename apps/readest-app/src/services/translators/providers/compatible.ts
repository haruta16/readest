import { stubTranslation as _ } from '@/utils/misc';
import { fetch as tauriFetch } from '@tauri-apps/plugin-http';
import { isTauriAppPlatform } from '@/services/environment';
import { TranslationProvider } from '../types';

// ── Prompt Templates ──────────────────────────────────────────────────────────
//
// Optimised for DeepSeek V4 and OpenAI-compatible LLMs as of mid-2026.
//
// Research foundation:
//   • AIware 2025: detailed prompts consistently outperform concise ones (13–15%
//     gains). Put the most important constraints at the TOP (positional bias).
//   • Du et al. MTSummit 2025: "Translate creatively" + temp≈1 for literary;
//     but for general reading, structured prompts + temp 0.3 is more reliable.
//   • DeepSeek-specific: the model tends to inject internet slang ("yyds",
//     "绝绝子") and add explanatory chatter — must explicitly ban both.
//   • Community-validated prompt from linux.do (533862) — the most-upvoted
//     DeepSeek translation template as of 2025.
//
// Design decisions:
//   • English system prompt (even for CN→CN cases).  AIware 2025 shows English
//     prompts yield 13–15% higher CodeBLEU than non-English across all tested
//     models including DeepSeek.
//   • "Only output the translation" repeated at BOTH top AND bottom — DeepSeek
//     often misses trailing constraints.
//   • Internet-slang ban is DeepSeek-specific but harmless for other models.
//   • Literary-quality pair of examples (Tagore + Tolkien) gives the model a
//     concrete quality bar for Chinese output.

const SYSTEM_PROMPT = `You are a professional literary translator. Translate each [N] segment from {sourceLang} to {targetLang}.

## Translation Standards ("信达雅")
1. Faithfulness (信): preserve the exact meaning and intent — no additions, no omissions.
2. Expressiveness (达): read as if originally written in {targetLang}. Natural, fluent, idiomatic.
3. Elegance (雅): match the register, tone, and literary quality of the source.

## Rules (most important first)
- Output ONLY the translations with [N] markers. NO explanations, notes, or commentary.
- Preserve all [N] markers exactly as provided.
- Translate meaning, not word-for-word. Find natural equivalents for idioms.
- Keep proper names in original form unless a canonical translation exists.
- NEVER use internet slang, colloquialisms, or trendy phrases unless the source explicitly does so.
- Dialogue must sound like real people speaking naturally in {targetLang}.

## Quality Reference
Example of literary translation quality:
Source: "In my solitude of heart I feel the sigh of this widowed evening veiled with mist and rain."
→ "这寡独的黄昏，幕着雾与雨，我在我的心的孤寂里，感觉到它的叹息。"

Source: "The world is indeed full of peril, and in it there are many dark places; but still there is much that is fair."
→ "世间确多险恶，遍地暗影幢幢；然美好之物亦存。"

Only output the translations with [N] markers. No other text.`;

// ── Language name helper ──────────────────────────────────────────────────────

const LANG_NAME: Record<string, string> = {
  'zh-cn': 'Simplified Chinese',
  'zh-tw': 'Traditional Chinese',
  'zh-hk': 'Traditional Chinese (Hong Kong)',
  ja: 'Japanese',
  ko: 'Korean',
  en: 'English',
  fr: 'French',
  de: 'German',
  es: 'Spanish',
  it: 'Italian',
  pt: 'Portuguese',
  ru: 'Russian',
  ar: 'Arabic',
  hi: 'Hindi',
  th: 'Thai',
  vi: 'Vietnamese',
  id: 'Indonesian',
  tr: 'Turkish',
  nl: 'Dutch',
  pl: 'Polish',
  sv: 'Swedish',
  no: 'Norwegian',
  da: 'Danish',
  fi: 'Finnish',
  cs: 'Czech',
  ro: 'Romanian',
  hu: 'Hungarian',
  el: 'Greek',
  he: 'Hebrew',
  uk: 'Ukrainian',
};

function langName(code: string): string {
  const short = code.toLowerCase().split('-')[0]!;
  return LANG_NAME[code.toLowerCase()] || LANG_NAME[short] || code;
}

function buildSystemPrompt(sourceLang: string, targetLang: string, customPrompt?: string): string {
  // When the user provides a custom prompt, use it as-is with placeholder substitution.
  // Otherwise use the curated system prompt above.
  if (customPrompt?.trim()) {
    return customPrompt
      .replace(/\{sourceLang\}/g, langName(sourceLang))
      .replace(/\{targetLang\}/g, langName(targetLang));
  }
  return SYSTEM_PROMPT.replace(/\{sourceLang\}/g, langName(sourceLang)).replace(
    /\{targetLang\}/g,
    langName(targetLang),
  );
}

// ── Batch translation with [N] markers ───────────────────────────────────────
//
// Multiple texts are packed into a single API call:
//   [0] First paragraph text
//   [1] Second paragraph text
//   ...
//
// Benefits:
//   • Single round-trip → lower latency
//   • LLM sees all texts as mutual context → better consistency
//   • Shared system prompt → lower token cost

function packTexts(texts: string[]): string {
  return texts.map((t, i) => `[${i}] ${t}`).join('\n\n');
}

function unpackTexts(raw: string, count: number): string[] {
  const results: string[] = new Array(count).fill('');
  // Split on lines that look like the start of a new [N] segment.
  // We use a lookahead split: split before every occurrence of [digits] at line start.
  const parts = raw.split(/(?=^\s*\[\d+\]\s)/gm);
  for (const part of parts) {
    const match = /^\s*\[(\d+)\]\s*/.exec(part);
    if (match) {
      const idx = parseInt(match[1]!, 10);
      const text = part.slice(match[0].length).trim();
      if (idx >= 0 && idx < count) {
        results[idx] = text;
      }
    }
  }
  return results;
}

// ── Provider ──────────────────────────────────────────────────────────────────

export const compatibleProvider: TranslationProvider = {
  name: 'compatible',
  label: _('Custom LLM Translator'),
  authRequired: true,
  translate: async (
    texts: string[],
    sourceLang: string,
    targetLang: string,
    token?: string | null,
    _useCache?: boolean,
  ): Promise<string[]> => {
    if (!texts.length) return [];

    // token carries the API key for this provider
    const apiKey = token;
    if (!apiKey) {
      throw new Error('API key is required for custom LLM translation');
    }

    // Read config from global settings (dynamic import avoids circular deps).
    const { useSettingsStore } = await import('@/store/settingsStore');
    const settings = useSettingsStore.getState().settings;
    const cfg = settings?.compatibleTranslator;

    const baseUrl = cfg?.apiBaseUrl?.replace(/\/+$/, '') || 'https://api.deepseek.com';
    const model = cfg?.model || 'deepseek-v4-flash';
    // DeepSeek official recommendation: 0.3 for translation.
    // Lower (0.0–0.1) yields overly literal output; higher (>0.5) introduces
    // unwanted creative reformulation.
    const temperature = cfg?.temperature ?? 0.3;

    // Filter empty texts
    const nonEmptyIndices: number[] = [];
    const nonEmptyTexts: string[] = [];
    texts.forEach((t, i) => {
      if (t?.trim()) {
        nonEmptyIndices.push(i);
        nonEmptyTexts.push(t);
      }
    });
    if (nonEmptyTexts.length === 0) return texts;

    const systemPrompt = buildSystemPrompt(sourceLang, targetLang, cfg?.systemPrompt);
    const userMessage = packTexts(nonEmptyTexts);

    const fetchImpl = isTauriAppPlatform() ? tauriFetch : window.fetch;

    const response = await fetchImpl(`${baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userMessage },
        ],
        temperature,
        max_tokens: 4096,
      }),
    });

    if (!response.ok) {
      let errorDetail = '';
      try {
        const errBody = await response.json();
        errorDetail = errBody?.error?.message || JSON.stringify(errBody);
      } catch {
        errorDetail = await response.text().catch(() => '');
      }
      if (response.status === 401 || response.status === 403) {
        throw new Error(`Authentication failed: ${errorDetail || response.statusText}`);
      }
      throw new Error(
        `LLM translation failed (${response.status}): ${errorDetail || response.statusText}`,
      );
    }

    const data = await response.json();
    const rawOutput: string = data?.choices?.[0]?.message?.content || '';

    if (!rawOutput.trim()) {
      throw new Error('LLM returned an empty response');
    }

    const translated = unpackTexts(rawOutput, nonEmptyTexts.length);

    // Rebuild full result array matching input order
    const results = [...texts];
    nonEmptyIndices.forEach((originalIdx, batchIdx) => {
      const translatedText = translated[batchIdx]?.trim();
      results[originalIdx] = translatedText || nonEmptyTexts[batchIdx]!;
    });

    return results;
  },
};
