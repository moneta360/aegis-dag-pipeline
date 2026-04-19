/**
 * ADRL-LENS Module
 * Optimized text processing pipeline for entity extraction and redundancy filtering.
 */

const PATTERNS = {
    CURRENCY: /\b(?:\$|€|£|¥|грн|UAH|USD|EUR)\s?\d+(?:[.,]\d+)?\b|\b\d+(?:[.,]\d+)?\s?(?:\$|€|£|¥|грн|UAH|USD|EUR)\b/g,
    DATE: /\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b/g,
    TECH_PARAM: /\b\d+(?:\.\d+)?\s?(?:kb|mb|gb|tb|hz|mhz|ghz|px|rem|em|vh|vw|ms|s|min|h|кг|г|км|м|см|мм)\b/gi,
    POTENTIAL_NAME: /\b[А-ЯІЇЄ][а-яіїє']+\s[А-ЯІЇЄ][а-яіїє']+(?:\s[А-ЯІЇЄ][а-яіїє']+)?\b/g
};

const STOP_WORDS = new Set([
    'і', 'та', 'але', 'чи', 'як', 'що', 'це', 'на', 'в', 'у', 'до', 'за', 'про', 'від', 'для', 'без',
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how'
]);

export interface AdrlLensResult {
    entities: Record<string, string[]>;
    filteredText: string;
    compressionRatio: number;
    integrityHash: string;
    timestamp: string;
}

export const extractEntities = (text: string): Map<string, string[]> => {
    const entities = new Map<string, string[]>();
    entities.set('currencies', text.match(PATTERNS.CURRENCY) || []);
    entities.set('dates', text.match(PATTERNS.DATE) || []);
    entities.set('technical_params', text.match(PATTERNS.TECH_PARAM) || []);
    entities.set('potential_names', text.match(PATTERNS.POTENTIAL_NAME) || []);
    return entities;
};

export const filterRedundancy = (text: string): { filtered: string; ratio: number } => {
    const originalLength = text.length;
    const tokens = text.split(/\s+/);
    const uniqueTokens = Array.from(new Set(
        tokens.filter(token => {
            const cleanToken = token.toLowerCase().replace(/[.,!?;:]/g, '');
            return !STOP_WORDS.has(cleanToken) && cleanToken.length > 1;
        })
    ));
    const filtered = uniqueTokens.join(' ');
    const ratio = originalLength > 0 ? Number((filtered.length / originalLength).toFixed(4)) : 1;
    return { filtered, ratio };
};

export const finalizePackage = async (text: string): Promise<AdrlLensResult> => {
    const entitiesMap = extractEntities(text);
    const { filtered, ratio } = filterRedundancy(text);
    const entities: Record<string, string[]> = {};
    entitiesMap.forEach((val, key) => {
        entities[key] = Array.from(new Set(val));
    });
    const msgUint8 = new TextEncoder().encode(filtered);
    const hashBuffer = await crypto.subtle.digest('SHA-256', msgUint8);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return {
        entities,
        filteredText: filtered,
        compressionRatio: ratio,
        integrityHash: hashHex,
        timestamp: new Date().toISOString()
    };
};
